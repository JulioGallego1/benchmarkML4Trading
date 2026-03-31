#!/usr/bin/env bash
# Usage:  bash scripts/slurm/run_one.sh <model> <regime> <L> <H> [extra args...]
# Slurm:  mkdir -p logs && sbatch scripts/slurm/run_one.sh rf bear 96 21
#         mkdir -p logs && sbatch scripts/slurm/run_one.sh lstm bull 48 63 --strategy recursive --step 16
#SBATCH --job-name=tsforecast
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1  

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p logs

source ~/miniconda3/etc/profile.d/conda.sh
conda activate tsforecast-env

python -m tsforecast.cli.train \
    --model   "$1" \
    --regime  "$2" \
    --L       "$3" \
    --H       "$4" \
    "${@:5}"
