#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"
require_linux
require_uv
cd "$VLACAL_ROOT"
git submodule update --init --depth 1 third_party/openpi
# Reuse upstream's resolved dependency graph, including its pinned LeRobot source.
# Keep this separate from the lightweight root development environment.
UV_PROJECT_ENVIRONMENT="$VLACAL_ROOT/.venv-model" uv sync \
    --project third_party/openpi --python 3.11 --frozen --no-dev
uv pip install --python "$MODEL_PYTHON" --no-deps -e "$VLACAL_ROOT"
"$MODEL_PYTHON" "$VLACAL_ROOT/scripts/patch_transformers.py"
"$MODEL_PYTHON" -c 'import torch, transformers, openpi; print("torch", torch.__version__, "transformers", transformers.__version__, "CUDA", torch.cuda.is_available())'
