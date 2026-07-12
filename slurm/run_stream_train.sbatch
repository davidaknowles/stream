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
CONFIG="${CONFIG:-configs/stream_mouse_dev.yaml}"
VARIANT="${VARIANT:-film}"
STEPS_PER_EPOCH="${STEPS_PER_EPOCH:-100}"
HVG_CSV="${HVG_CSV:-}"
N_HVG="${N_HVG:-}"
OUT_DIR="${OUT_DIR:-}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-}"
BATCH_SIZE="${BATCH_SIZE:-}"
GENE_CHUNK_SIZE="${GENE_CHUNK_SIZE:-}"
CELL_STATE="${CELL_STATE:-}"
UCE_EMBEDDING_DIR="${UCE_EMBEDDING_DIR:-}"
WANDB_MODE="${WANDB_MODE:-}"
EXPERIMENT_LABEL="${EXPERIMENT_LABEL:-}"
INIT_CHECKPOINT="${INIT_CHECKPOINT:-}"
STREAM_PYTHON="${STREAM_PYTHON:-$HOME/venv/torchfix/bin/python}"
cd "$PROJECT_ROOT"
mkdir -p logs
mkdir -p outputs/stream/wandb
module purge
if [[ ! -x "$STREAM_PYTHON" ]]; then
  module load PyTorch/2.1.2-foss-2023b
  STREAM_PYTHON=python
  export PYTHONPATH="$HOME/.local/lib/python3.11/site-packages:$PROJECT_ROOT/src:$PROJECT_ROOT/.venv-jax-adata-eda/lib/python3.11/site-packages:${PYTHONPATH:-}"
else
  export PYTHONPATH="$PROJECT_ROOT/src:${PYTHONPATH:-}"
fi
export WANDB_DIR="$PROJECT_ROOT/outputs/stream/wandb"
export WANDB_PROJECT="${WANDB_PROJECT:-stream}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
args=(--config "$CONFIG" --variant "$VARIANT" --steps-per-epoch "$STEPS_PER_EPOCH" --device cuda)
if [[ -n "$HVG_CSV" ]]; then args+=(--hvg-csv "$HVG_CSV"); fi
if [[ -n "$N_HVG" ]]; then args+=(--n-hvg "$N_HVG"); fi
if [[ -n "$OUT_DIR" ]]; then args+=(--out-dir "$OUT_DIR"); fi
if [[ -n "$WANDB_RUN_NAME" ]]; then args+=(--wandb-run-name "$WANDB_RUN_NAME"); fi
if [[ -n "$BATCH_SIZE" ]]; then args+=(--batch-size "$BATCH_SIZE"); fi
if [[ -n "$GENE_CHUNK_SIZE" ]]; then args+=(--gene-chunk-size "$GENE_CHUNK_SIZE"); fi
if [[ -n "$CELL_STATE" ]]; then args+=(--cell-state "$CELL_STATE"); fi
if [[ -n "$UCE_EMBEDDING_DIR" ]]; then args+=(--uce-embedding-dir "$UCE_EMBEDDING_DIR"); fi
if [[ -n "$WANDB_MODE" ]]; then args+=(--wandb-mode "$WANDB_MODE"); fi
if [[ -n "$EXPERIMENT_LABEL" ]]; then args+=(--experiment-label "$EXPERIMENT_LABEL"); fi
if [[ -n "$INIT_CHECKPOINT" ]]; then args+=(--init-checkpoint "$INIT_CHECKPOINT"); fi
"$STREAM_PYTHON" scripts/train_stream.py "${args[@]}"
