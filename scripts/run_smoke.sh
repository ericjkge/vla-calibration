#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"
require_linux
"$SIM_PYTHON" "$VLACAL_ROOT/scripts/configure_libero.py"
"$MODEL_PYTHON" "$VLACAL_ROOT/scripts/run_smoke.py" \
    --checkpoint "${VLACAL_CHECKPOINT:-$VLACAL_DATA_DIR/checkpoints/pi05_libero_pytorch}" \
    --output-dir "$VLACAL_DATA_DIR/results/smoke-$(date -u +%Y%m%dT%H%M%SZ)-$$" "$@"
