#!/usr/bin/env bash
#SBATCH --job-name=tsforecast
#SBATCH --output=logs/slurm_%A_%a.out
#SBATCH --error=logs/slurm_%A_%a.err
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1  

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p logs

source ~/miniconda3/etc/profile.d/conda.sh
conda activate tsforecast-env

GRID="${GRID_FILE:-scripts/grid.jsonl}"
LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$GRID")
echo "Task $SLURM_ARRAY_TASK_ID: $LINE"

python scripts/run_config.py "$LINE"
