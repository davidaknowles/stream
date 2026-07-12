#!/bin/bash
#SBATCH --job-name=stream_cache_eval
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/stream_cache_eval_%j.out
#SBATCH --error=logs/stream_cache_eval_%j.err

set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-/gpfs/commons/home/daknowles/projects/stream}"
CONFIG="${CONFIG:?CONFIG is required}"
OUT_DIR="${OUT_DIR:?OUT_DIR is required}"
EVAL_CACHE="${EVAL_CACHE:?EVAL_CACHE is required}"
N_HVG="${N_HVG:?N_HVG is required}"
BATCHES="${BATCHES:-20}"
CELL_STATE="${CELL_STATE:-uce}"
UCE_EMBEDDING_DIR="${UCE_EMBEDDING_DIR:-}"
STREAM_PYTHON="${STREAM_PYTHON:-$HOME/venv/torchfix/bin/python}"
cd "$PROJECT_ROOT"
mkdir -p logs
export PYTHONPATH="$PROJECT_ROOT/src:${PYTHONPATH:-}"
args=(--config "$CONFIG" --out-dir "$OUT_DIR" --cache "$EVAL_CACHE" --n-hvg "$N_HVG" --batches "$BATCHES" --cell-state "$CELL_STATE")
if [[ -n "$UCE_EMBEDDING_DIR" ]]; then args+=(--uce-embedding-dir "$UCE_EMBEDDING_DIR"); fi
"$STREAM_PYTHON" scripts/cache_stream_eval_batches.py "${args[@]}"
