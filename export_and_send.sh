#!/usr/bin/env bash
# export_and_send.sh — export the phone snapshot, push it over Taildrop from
# a user-owned relay, and delete the local copy once the push succeeds.
#
# Usage: ./export_and_send.sh [tailnet-device]   (default: ios-app)
#
# Rocky is a tagged Tailscale node, and Taildrop cannot send from tagged nodes.
# The relay is a separate, user-owned Tailscale node in a short-lived rootless
# Podman container. Its state volume survives each transfer, so it reconnects
# as the same node without interactive authentication; the container and its
# copy of the export are removed after every run. The relay receives the export
# through `podman cp`, never a mount of this checkout.
#
# Bootstrap the relay once by generating an untagged, non-ephemeral auth key
# in the Tailscale admin console, then run:
#
#   TAILSCALE_AUTHKEY=tskey-auth-... ./export_and_send.sh
#
# Disable key expiry for the resulting `wordle-taildrop` machine in the
# Tailscale admin console. Subsequent runs need no auth key. Do not put an auth
# key in this repository or in the persistent state volume.
#
# The export snapshot is transitory: its job ends when the bytes reach the
# phone. `tailscale file cp` blocks until the peer has received the file and
# exits nonzero otherwise, so its exit status is the handoff signal — the local
# copy is deleted only after a successful push. The phone deletes its copy
# after a successful merge (import_cache.py's default), which ends the cycle
# with no snapshot left on either device.
#
# The watermark file (next to this script) holds the unix time of the start
# of the last successfully pushed delta. Its absence means bootstrap: a full
# export, unchanged from the original whole-file behavior. Every export
# after that carries only rows updated since the watermark, via
# export_cache.py --since.
#
# next_watermark is captured *before* exporting, so a row that commits
# during this run's WAL-snapshot read is re-sent (not skipped) by the next
# delta — an overlap window absorbs clock skew and in-flight writes at the
# cost of some harmlessly re-sent rows (import_cache.py's merge is
# idempotent). The watermark file is updated only after the push succeeds:
# if the export or the push fails, the watermark is left untouched so the
# next run re-covers the same span.

set -euo pipefail

if [ $# -gt 1 ]; then
    echo "Usage: $0 [tailnet-device]   (default: ios-app)" >&2
    exit 2
fi
tailnet_device="${1:-ios-app}"

cd "$(dirname "$0")"

export_file="wordle_erd_export.sqlite3"
watermark_file="wordle_export_watermark"
overlap_seconds=3600
taildrop_state_volume="${WORDLE_TAILDROP_STATE_VOLUME:-wordle-taildrop-state}"
taildrop_hostname="${WORDLE_TAILDROP_HOSTNAME:-wordle-taildrop}"
taildrop_image="${WORDLE_TAILDROP_IMAGE:-docker.io/tailscale/tailscale:stable}"
taildrop_container="wordle-taildrop"
taildrop_lock_file="${WORDLE_TAILDROP_LOCK_FILE:-${TMPDIR:-/tmp}/wordle-taildrop-export.lock}"
taildrop_started=false
taildrop_temporary_output=""

if ! command -v flock >/dev/null; then
    echo "flock is required to serialize Taildrop exports." >&2
    exit 1
fi
exec {taildrop_lock_fd}>"$taildrop_lock_file"
if ! flock -n "$taildrop_lock_fd"; then
    echo "Another Taildrop export is already running." >&2
    exit 1
fi

cleanup_taildrop_container() {
    if [ -n "$taildrop_temporary_output" ]; then
        rm -f "$taildrop_temporary_output"
    fi
    if "$taildrop_started"; then
        podman rm --force "$taildrop_container" >/dev/null 2>&1 || true
    fi
}
trap cleanup_taildrop_container EXIT
trap 'exit 130' INT TERM

start_taildrop_relay() {
    local -a podman_args=(
        run --detach
        --name "$taildrop_container"
        --hostname "$taildrop_hostname"
        --security-opt no-new-privileges
        --volume "$taildrop_state_volume:/var/lib/tailscale"
        --tmpfs /exports:rw,noexec,nosuid,nodev
        --env TS_AUTH_ONCE=true
        --env TS_STATE_DIR=/var/lib/tailscale
        --env TS_USERSPACE=true
    )
    if [ -n "${TAILSCALE_AUTHKEY:-}" ]; then
        podman_args+=(--env TS_AUTHKEY)
    fi
    if podman container exists "$taildrop_container"; then
        if [ "$(podman container inspect --format '{{.State.Running}}' \
                "$taildrop_container" 2>/dev/null)" = true ]; then
            echo "A Taildrop relay is already running: $taildrop_container." >&2
            return 1
        fi
        echo "Removing a leftover Taildrop relay from an interrupted run:" >&2
        podman logs "$taildrop_container" >&2 || true
        podman rm --force "$taildrop_container" >/dev/null
    fi
    podman_args+=("$taildrop_image")
    podman "${podman_args[@]}" >/dev/null
    taildrop_started=true

    local attempt status_json relay_running
    for attempt in {1..30}; do
        status_json="$(podman exec "$taildrop_container" \
            tailscale status --json --peers=false 2>&1 || true)"
        if [[ "$status_json" =~ \"BackendState\"[[:space:]]*:[[:space:]]*\"Running\" ]]; then
            return
        fi
        relay_running="$(podman container inspect --format '{{.State.Running}}' \
            "$taildrop_container" 2>/dev/null || true)"
        if [ "$relay_running" != true ]; then
            echo "The Taildrop relay stopped before it connected." >&2
            podman logs "$taildrop_container" >&2 || true
            return 1
        fi
        sleep 1
    done

    echo "The Taildrop relay did not connect within 30 seconds." >&2
    if [ -z "${TAILSCALE_AUTHKEY:-}" ]; then
        echo "For first-time setup, run with TAILSCALE_AUTHKEY=tskey-auth-...." >&2
    fi
    podman logs "$taildrop_container" >&2 || true
    return 1
}

run_taildrop_command() {
    local headline="$1" retry_instructions="$2" command_status
    shift 2
    taildrop_temporary_output="$(mktemp)"
    if "$@" \
            >"$taildrop_temporary_output" 2>&1; then
        rm -f "$taildrop_temporary_output"
        taildrop_temporary_output=""
        return
    else
        command_status=$?
    fi

    echo "$headline" >&2
    echo "The export was kept at $export_file; retrying will resend it." >&2
    if "$retry_instructions"; then
        echo "On the receiving device, open Tailscale and wait for it to report synchronized, then retry:" >&2
        echo "  ./export_and_send.sh" >&2
    fi
    if [ -s "$taildrop_temporary_output" ]; then
        printf 'Details: %s\n' "$(<"$taildrop_temporary_output")" >&2
    fi
    rm -f "$taildrop_temporary_output"
    taildrop_temporary_output=""
    return "$command_status"
}

since_args=()
if [ -f "$watermark_file" ]; then
    watermark="$(cat "$watermark_file")"
    since_args=(--since "$watermark")
fi
next_watermark=$(( $(date +%s) - overlap_seconds ))

start_taildrop_relay
python3.13 export_cache.py "${since_args[@]}"
run_taildrop_command "The Taildrop relay could not prepare the export." false \
    podman cp "$export_file" "$taildrop_container:/exports/$export_file"
run_taildrop_command "Taildrop could not reach $tailnet_device." true \
    podman exec "$taildrop_container" \
    tailscale file cp "/exports/$export_file" "${tailnet_device}:"
echo "$next_watermark" > "$watermark_file"
rm -f "$export_file" "$export_file-wal" "$export_file-shm"
echo "Sent $export_file to $tailnet_device through the Taildrop relay and deleted the local copy."
