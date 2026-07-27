#!/usr/bin/env bash
# export_and_send.sh — export the phone snapshot, push it over Taildrop, and
# delete the local copy once the push succeeds.
#
# Usage: ./export_and_send.sh [tailnet-device]   (default: ios-app)
#
# The export snapshot is transitory: its job ends when the bytes reach the
# phone. `tailscale file cp` blocks until the peer has received the file and
# exits nonzero otherwise, so its exit status is the handoff signal — the
# local copy is deleted only after a successful push. The phone deletes its
# copy after a successful merge (import_cache.py's default), which ends the
# cycle with no snapshot left on either device.
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

since_args=()
if [ -f "$watermark_file" ]; then
    watermark="$(cat "$watermark_file")"
    since_args=(--since "$watermark")
fi
next_watermark=$(( $(date +%s) - overlap_seconds ))

python3.13 export_cache.py "${since_args[@]}"
tailscale file cp "$export_file" "${tailnet_device}:"
echo "$next_watermark" > "$watermark_file"
rm -f "$export_file" "$export_file-wal" "$export_file-shm"
echo "Sent $export_file to $tailnet_device and deleted the local copy."
