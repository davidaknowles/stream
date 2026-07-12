#!/bin/bash
#SBATCH --job-name=stream_hvg
#SBATCH --partition=bigmem
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=logs/stream_hvg_%j.out
#SBATCH --error=logs/stream_hvg_%j.err

set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-/gpfs/commons/home/daknowles/projects/stream}"
CONFIG="${CONFIG:-configs/stream_mouse_dev.yaml}"
HVG_OUTPUT="${HVG_OUTPUT:-outputs/jax_adata_eda/streaming_gene_variances.csv}"
CHUNK_CELLS="${CHUNK_CELLS:-5000}"
N_GENES="${N_GENES:-0}"
STREAM_DATA_PYTHON="${STREAM_DATA_PYTHON:-$HOME/venv/torchfix/bin/python}"
cd "$PROJECT_ROOT"
mkdir -p logs
export PYTHONPATH="$PROJECT_ROOT/src:${PYTHONPATH:-}"
"$STREAM_DATA_PYTHON" scripts/select_streaming_hvgs.py \
  --config "$CONFIG" \
  --output "$HVG_OUTPUT" \
  --chunk-cells "$CHUNK_CELLS" \
  --n-genes "$N_GENES"
