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
taildrop_container="wordle-taildrop-${RANDOM}-${RANDOM}"
taildrop_started=false

cleanup_taildrop_container() {
    if "$taildrop_started"; then
        podman rm --force "$taildrop_container" >/dev/null 2>&1 || true
    fi
}
trap cleanup_taildrop_container EXIT INT TERM

start_taildrop_relay() {
    local -a podman_args=(
        run --detach --rm
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
        podman_args+=(--env "TS_AUTHKEY=$TAILSCALE_AUTHKEY")
    fi
    podman volume create "$taildrop_state_volume" >/dev/null
    podman_args+=("$taildrop_image")
    podman "${podman_args[@]}" >/dev/null
    taildrop_started=true

    local attempt
    for attempt in {1..30}; do
        if podman exec "$taildrop_container" tailscale status --json 2>/dev/null \
                | grep -Eq '"BackendState"[[:space:]]*:[[:space:]]*"Running"'; then
            return
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

send_export_through_taildrop() {
    local transfer_output transfer_status
    transfer_output="$(mktemp)"
    if podman exec "$taildrop_container" \
            tailscale file cp "/exports/$export_file" "${tailnet_device}:" \
            >"$transfer_output" 2>&1; then
        rm -f "$transfer_output"
        return
    else
        transfer_status=$?
    fi

    echo "Taildrop could not reach $tailnet_device." >&2
    echo "The export was kept at $export_file; retrying will resend it." >&2
    echo "On the receiving device, open Tailscale and wait for it to report synchronized, then retry:" >&2
    echo "  ./export_and_send.sh" >&2
    if [ -s "$transfer_output" ]; then
        printf 'Details: %s\n' "$(<"$transfer_output")" >&2
    fi
    rm -f "$transfer_output"
    return "$transfer_status"
}

since_args=()
if [ -f "$watermark_file" ]; then
    watermark="$(cat "$watermark_file")"
    since_args=(--since "$watermark")
fi
next_watermark=$(( $(date +%s) - overlap_seconds ))

start_taildrop_relay
python3.13 export_cache.py "${since_args[@]}"
podman cp "$export_file" "$taildrop_container:/exports/$export_file"
send_export_through_taildrop
echo "$next_watermark" > "$watermark_file"
rm -f "$export_file" "$export_file-wal" "$export_file-shm"
echo "Sent $export_file to $tailnet_device through the Taildrop relay and deleted the local copy."
