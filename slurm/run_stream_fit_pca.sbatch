#!/bin/bash
#SBATCH --job-name=stream_fit_pca
#SBATCH --partition=bigmem
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=12:00:00
#SBATCH --output=logs/stream_fit_pca_%x_%j.out
#SBATCH --error=logs/stream_fit_pca_%x_%j.err

set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-/gpfs/commons/home/daknowles/projects/stream}"
STREAM_PYTHON="${STREAM_PYTHON:-$HOME/venv/torchfix/bin/python}"
CONFIG="${CONFIG:-configs/stream_mouse_dev.yaml}"
OUT_DIR="${OUT_DIR:-outputs/stream_hvg10000}"
N_HVG="${N_HVG:-10000}"
OUTPUT="${OUTPUT:-${OUT_DIR}/pca_denoiser_100pc.npz}"
PCA_COMPONENTS="${PCA_COMPONENTS:-100}"
PCA_FIT_CELLS="${PCA_FIT_CELLS:-10000}"
cd "$PROJECT_ROOT"
mkdir -p logs
export PYTHONPATH="$PROJECT_ROOT/src:${PYTHONPATH:-}"
"$STREAM_PYTHON" scripts/fit_stream_pca.py \
  --config "$CONFIG" \
  --out-dir "$OUT_DIR" \
  --n-hvg "$N_HVG" \
  --output "$OUTPUT" \
  --components "$PCA_COMPONENTS" \
  --fit-cells "$PCA_FIT_CELLS"
