#!/usr/bin/env bash
#SBATCH --job-name=vlacal-setup
#SBATCH --partition=test
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=data/logs/setup-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit this job from the repository root}"
export PATH="$PWD/data/tools:$HOME/.local/bin:$PATH"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
bash scripts/setup_model_env.sh
bash scripts/setup_sim_env.sh
bash scripts/prepare_checkpoint.sh
