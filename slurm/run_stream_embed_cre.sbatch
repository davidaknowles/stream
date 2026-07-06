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
CONFIG="${CONFIG:-configs/stream_mouse_dev.yaml}"
HVG_CSV="${HVG_CSV:-}"
N_HVG="${N_HVG:-}"
OUT_DIR="${OUT_DIR:-}"
STREAM_PYTHON="${STREAM_PYTHON:-$HOME/venv/torchfix/bin/python}"
cd "$PROJECT_ROOT"
mkdir -p logs
module purge
if [[ ! -x "$STREAM_PYTHON" ]]; then
  module load PyTorch/2.1.2-foss-2023b
  STREAM_PYTHON=python
  export PYTHONPATH="$HOME/.local/lib/python3.11/site-packages:$PROJECT_ROOT/src:$PROJECT_ROOT/.venv-jax-adata-eda/lib/python3.11/site-packages:/gpfs/commons/home/daknowles/projects/alphagenome-pytorch/src:${PYTHONPATH:-}"
else
  export PYTHONPATH="$PROJECT_ROOT/src:/gpfs/commons/home/daknowles/projects/alphagenome-pytorch/src:${PYTHONPATH:-}"
fi
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
args=(--config "$CONFIG" --device cuda)
if [[ -n "$HVG_CSV" ]]; then args+=(--hvg-csv "$HVG_CSV"); fi
if [[ -n "$N_HVG" ]]; then args+=(--n-hvg "$N_HVG"); fi
if [[ -n "$OUT_DIR" ]]; then args+=(--out-dir "$OUT_DIR"); fi
"$STREAM_PYTHON" scripts/embed_cre_alphagenome.py "${args[@]}"
