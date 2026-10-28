#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"
require_linux
"$MODEL_PYTHON" "$VLACAL_ROOT/scripts/prepare_checkpoint.py" \
    --output "${VLACAL_CHECKPOINT:-$VLACAL_DATA_DIR/checkpoints/pi05_libero_pytorch}" "$@"
