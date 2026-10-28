#!/usr/bin/env bash
# Source this file; paths are independent of the caller's working directory.
VLACAL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export VLACAL_DATA_DIR="${VLACAL_DATA_DIR:-$VLACAL_ROOT/data}"
mkdir -p "$VLACAL_DATA_DIR"
VLACAL_DATA_DIR="$(cd "$VLACAL_DATA_DIR" && pwd)"
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-$VLACAL_DATA_DIR/cache/openpi}"
export HF_HOME="${HF_HOME:-$VLACAL_DATA_DIR/cache/huggingface}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$VLACAL_DATA_DIR/cache/uv}"
# Patching hardlinked transformers files would also modify uv's shared cache.
export UV_LINK_MODE=copy
export GIT_LFS_SKIP_SMUDGE=1
export PYTHONUNBUFFERED=1
export JAX_PLATFORMS=cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export LIBERO_CONFIG_PATH="$VLACAL_DATA_DIR/libero-config"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-$MUJOCO_GL}"
MODEL_PYTHON="$VLACAL_ROOT/.venv-model/bin/python"
SIM_PYTHON="$VLACAL_ROOT/sim/.venv/bin/python"

require_linux() {
    if [[ "$(uname -s)" != Linux || "$(uname -m)" != x86_64 ]]; then
        echo "The model/simulator runtime requires Linux x86_64; use a Cannon compute node." >&2
        exit 1
    fi
}

require_uv() {
    command -v uv >/dev/null || { echo "Install uv first: https://docs.astral.sh/uv/" >&2; exit 1; }
}
