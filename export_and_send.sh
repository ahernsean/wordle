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

set -euo pipefail

if [ $# -gt 1 ]; then
    echo "Usage: $0 [tailnet-device]   (default: ios-app)" >&2
    exit 2
fi
tailnet_device="${1:-ios-app}"

cd "$(dirname "$0")"

export_file="wordle_erd_export.sqlite3"

python3.13 export_cache.py
tailscale file cp "$export_file" "${tailnet_device}:"
rm -f "$export_file" "$export_file-wal" "$export_file-shm"
echo "Sent $export_file to $tailnet_device and deleted the local copy."
