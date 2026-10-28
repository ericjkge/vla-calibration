#!/usr/bin/env bash
#SBATCH --job-name=vlacal-smoke
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:45:00
#SBATCH --output=data/logs/smoke-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit this job from the repository root}"
export PATH="$PWD/data/tools:$HOME/.local/bin:$PATH"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
# Do not overwrite CUDA_VISIBLE_DEVICES: Slurm owns GPU selection.
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
nvidia-smi
bash scripts/run_smoke.sh --require-success "$@"
