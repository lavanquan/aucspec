#!/bin/bash
# Idempotent, non-destructive creation of the Triton shim for runtime "V1-batchwall-tritonfa33".
# The shim is one symlink: <SHIM>/triton -> torch's own Triton 3.3.1 (inside the pytorch module container).
# Placing <SHIM> FIRST on PYTHONPATH makes `import triton` resolve to 3.3.1 instead of the user-site 3.8.0.
# Nothing is installed, removed or edited; an existing different link is never overwritten.
set -euo pipefail
SHIM="${1:-/software/projects/pawsey1257/quanla/aucspec_runtime/tritonfa33/shim}"
TARGET="/usr/local/lib/python3.12/dist-packages/triton"
mkdir -p "$SHIM"
if [ -L "$SHIM/triton" ]; then
  have="$(readlink "$SHIM/triton")"
  [ "$have" = "$TARGET" ] || { echo "existing shim link points to $have, not $TARGET; refusing to overwrite" >&2; exit 2; }
elif [ -e "$SHIM/triton" ]; then
  echo "$SHIM/triton exists and is not a symlink; refusing" >&2; exit 2
else
  ln -s "$TARGET" "$SHIM/triton"
fi
echo "shim ready: $SHIM/triton -> $(readlink "$SHIM/triton")"
