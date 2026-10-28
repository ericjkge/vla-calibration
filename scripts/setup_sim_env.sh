#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"
require_linux
require_uv
cd "$VLACAL_ROOT"
git submodule update --init --depth 1 third_party/openpi
# Initialize only the original ID benchmark, not unrelated robot dependencies.
git -C third_party/openpi submodule update --init --depth 1 third_party/libero
uv sync --project sim --python 3.8 --frozen
uv pip install --python "$SIM_PYTHON" --no-deps --no-build-isolation \
    -e "$VLACAL_ROOT/third_party/openpi/third_party/libero"
"$SIM_PYTHON" "$VLACAL_ROOT/scripts/configure_libero.py"
"$SIM_PYTHON" -c 'import mujoco, robosuite; from libero.libero import benchmark; print("MuJoCo", mujoco.__version__, "suites", list(benchmark.get_benchmark_dict()))'
