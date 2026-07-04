#!/bin/bash
#SBATCH --job-name=stream_embed_cre
#SBATCH --partition=gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=logs/stream_embed_cre_%j.out
#SBATCH --error=logs/stream_embed_cre_%j.err

set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-/gpfs/commons/home/daknowles/projects/stream}"
cd "$PROJECT_ROOT"
mkdir -p logs
module purge
module load PyTorch/2.1.2-foss-2023b
export PYTHONPATH="$PROJECT_ROOT/src:$PROJECT_ROOT/.venv-jax-adata-eda/lib/python3.11/site-packages:/gpfs/commons/home/daknowles/projects/alphagenome-pytorch/src:${PYTHONPATH:-}"
python scripts/embed_cre_alphagenome.py --config configs/stream_mouse_dev.yaml --device cuda
