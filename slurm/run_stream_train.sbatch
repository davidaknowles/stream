#!/bin/bash
#SBATCH --job-name=stream_train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=logs/stream_train_%x_%j.out
#SBATCH --error=logs/stream_train_%x_%j.err

set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-/gpfs/commons/home/daknowles/projects/stream}"
VARIANT="${VARIANT:-film}"
STEPS_PER_EPOCH="${STEPS_PER_EPOCH:-100}"
cd "$PROJECT_ROOT"
mkdir -p logs
mkdir -p outputs/stream/wandb
module purge
module load PyTorch/2.1.2-foss-2023b
export PYTHONPATH="$HOME/.local/lib/python3.11/site-packages:$PROJECT_ROOT/src:$PROJECT_ROOT/.venv-jax-adata-eda/lib/python3.11/site-packages:${PYTHONPATH:-}"
export WANDB_DIR="$PROJECT_ROOT/outputs/stream/wandb"
export WANDB_PROJECT="${WANDB_PROJECT:-stream}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
python scripts/train_stream.py --config configs/stream_mouse_dev.yaml --variant "$VARIANT" --steps-per-epoch "$STEPS_PER_EPOCH" --device cuda
