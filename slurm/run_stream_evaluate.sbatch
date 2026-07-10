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
CONFIG="${CONFIG:-configs/stream_mouse_dev.yaml}"
VARIANT="${VARIANT:-film}"
BATCHES="${BATCHES:-20}"
HVG_CSV="${HVG_CSV:-}"
N_HVG="${N_HVG:-}"
OUT_DIR="${OUT_DIR:-}"
EVAL_GENE_SUBSET="${EVAL_GENE_SUBSET:-}"
BATCH_SIZE="${BATCH_SIZE:-}"
GENE_CHUNK_SIZE="${GENE_CHUNK_SIZE:-}"
CELL_STATE="${CELL_STATE:-}"
UCE_EMBEDDING_DIR="${UCE_EMBEDDING_DIR:-}"
STREAM_PYTHON="${STREAM_PYTHON:-$HOME/venv/torchfix/bin/python}"
cd "$PROJECT_ROOT"
mkdir -p logs
module purge
if [[ ! -x "$STREAM_PYTHON" ]]; then
  module load PyTorch/2.1.2-foss-2023b
  STREAM_PYTHON=python
  export PYTHONPATH="$HOME/.local/lib/python3.11/site-packages:$PROJECT_ROOT/src:$PROJECT_ROOT/.venv-jax-adata-eda/lib/python3.11/site-packages:${PYTHONPATH:-}"
else
  export PYTHONPATH="$PROJECT_ROOT/src:${PYTHONPATH:-}"
fi
args=(--config "$CONFIG" --variant "$VARIANT" --batches "$BATCHES" --device cuda)
if [[ -n "$HVG_CSV" ]]; then args+=(--hvg-csv "$HVG_CSV"); fi
if [[ -n "$N_HVG" ]]; then args+=(--n-hvg "$N_HVG"); fi
if [[ -n "$OUT_DIR" ]]; then args+=(--out-dir "$OUT_DIR"); fi
if [[ -n "$EVAL_GENE_SUBSET" ]]; then args+=(--eval-gene-subset "$EVAL_GENE_SUBSET"); fi
if [[ -n "$BATCH_SIZE" ]]; then args+=(--batch-size "$BATCH_SIZE"); fi
if [[ -n "$GENE_CHUNK_SIZE" ]]; then args+=(--gene-chunk-size "$GENE_CHUNK_SIZE"); fi
if [[ -n "$CELL_STATE" ]]; then args+=(--cell-state "$CELL_STATE"); fi
if [[ -n "$UCE_EMBEDDING_DIR" ]]; then args+=(--uce-embedding-dir "$UCE_EMBEDDING_DIR"); fi
"$STREAM_PYTHON" scripts/evaluate_stream.py "${args[@]}"
