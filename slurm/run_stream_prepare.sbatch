#!/bin/bash
#SBATCH --job-name=stream_prepare
#SBATCH --partition=bigmem
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/stream_prepare_%j.out
#SBATCH --error=logs/stream_prepare_%j.err

set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-/gpfs/commons/home/daknowles/projects/stream}"
CONFIG="${CONFIG:-configs/stream_mouse_dev.yaml}"
HVG_CSV="${HVG_CSV:-}"
N_HVG="${N_HVG:-}"
OUT_DIR="${OUT_DIR:-}"
STREAM_DATA_PYTHON="${STREAM_DATA_PYTHON:-$HOME/venv/torchfix/bin/python}"
cd "$PROJECT_ROOT"
mkdir -p logs
export PYTHONPATH="$PROJECT_ROOT/src:${PYTHONPATH:-}"
args=(--config "$CONFIG")
if [[ -n "$HVG_CSV" ]]; then args+=(--hvg-csv "$HVG_CSV"); fi
if [[ -n "$N_HVG" ]]; then args+=(--n-hvg "$N_HVG"); fi
if [[ -n "$OUT_DIR" ]]; then args+=(--out-dir "$OUT_DIR"); fi
"$STREAM_DATA_PYTHON" scripts/prepare_stream_data.py "${args[@]}"
