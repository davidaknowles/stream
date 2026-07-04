#!/bin/bash
#SBATCH --job-name=stream_eval
#SBATCH --partition=gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=logs/stream_eval_%x_%j.out
#SBATCH --error=logs/stream_eval_%x_%j.err

set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-/gpfs/commons/home/daknowles/projects/stream}"
VARIANT="${VARIANT:-film}"
BATCHES="${BATCHES:-20}"
cd "$PROJECT_ROOT"
mkdir -p logs
module purge
module load PyTorch/2.1.2-foss-2023b
export PYTHONPATH="$HOME/.local/lib/python3.11/site-packages:$PROJECT_ROOT/src:$PROJECT_ROOT/.venv-jax-adata-eda/lib/python3.11/site-packages:${PYTHONPATH:-}"
python scripts/evaluate_stream.py --config configs/stream_mouse_dev.yaml --variant "$VARIANT" --batches "$BATCHES" --device cuda
