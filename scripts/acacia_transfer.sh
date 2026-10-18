#!/bin/bash
# Upload + verify + read-back for raw archives (Acacia, S3 via rclone).
# Never reads or prints credentials. Requires a user-created rclone remote:
#   export RCLONE_CONFIG=/scratch/pawsey1257/quanla/.rclone/rclone.conf   (mode 600, NOT in git, NOT in /home)
#   remote name defaults to "acacia"; bucket must already exist.
# usage: acacia_transfer.sh <bucket> <prefix> <file-or-dir>...    (dry run unless APPLY=1)
set -euo pipefail
BUCKET="$1"; PREFIX="$2"; shift 2
REMOTE="${ACACIA_REMOTE:-acacia}"
module load rclone/1.68.1 >/dev/null 2>&1 || true
: "${RCLONE_CONFIG:?set RCLONE_CONFIG to a private path outside git}"
[ -f "$RCLONE_CONFIG" ] || { echo "no rclone config at $RCLONE_CONFIG"; exit 2; }
DEST="$REMOTE:$BUCKET/$PREFIX"
FLAGS="--local-no-set-modtime --s3-no-check-bucket --checksum"
for src in "$@"; do
  if [ "${APPLY:-0}" != 1 ]; then echo "[dry-run] would copy $src -> $DEST"; continue; fi
  if [ -d "$src" ]; then rclone copy $FLAGS "$src" "$DEST/$(basename "$src")"; else rclone copyto $FLAGS "$src" "$DEST/$(basename "$src")"; fi
done
[ "${APPLY:-0}" = 1 ] || exit 0
# 1) sizes + hashes on the remote against the local source
for src in "$@"; do
  if [ -d "$src" ]; then rclone check "$src" "$DEST/$(basename "$src")" --download=false --one-way 2>&1 | tail -3
  else rclone check "$(dirname "$src")" "$DEST" --include "$(basename "$src")" --one-way 2>&1 | tail -3; fi
done
# 2) read-back: download to scratch and compare sha256 with the manifest
RB=/scratch/pawsey1257/quanla/acacia_readback_$$; mkdir -p "$RB"
for src in "$@"; do [ -f "$src" ] && rclone copyto "$DEST/$(basename "$src")" "$RB/$(basename "$src")" --local-no-set-modtime && \
  [ "$(sha256sum "$src" | cut -d' ' -f1)" = "$(sha256sum "$RB/$(basename "$src")" | cut -d' ' -f1)" ] && echo "READBACK_OK $(basename "$src")"; done
echo "read-back dir $RB (remove after review)"
