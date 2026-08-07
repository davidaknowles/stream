#!/bin/bash
#SBATCH --job-name=stream_subtype_centroids
#SBATCH --partition=bigmem
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=12:00:00
#SBATCH --output=logs/stream_subtype_centroids_%j.out
#SBATCH --error=logs/stream_subtype_centroids_%j.err

set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-/gpfs/commons/home/daknowles/projects/stream}"
STREAM_PYTHON="${STREAM_PYTHON:-$HOME/venv/torchfix/bin/python}"
cd "$PROJECT_ROOT"
mkdir -p logs
export PYTHONPATH="$PROJECT_ROOT/src:${PYTHONPATH:-}"

"$STREAM_PYTHON" scripts/fit_subtype_centroids.py \
  --config "${CONFIG:-configs/stream_mouse_dev.yaml}" \
  --out-dir "${OUT_DIR:-outputs/stream_hvg10000}" \
  --timepoint-split "${TIMEPOINT_SPLIT:?}" --pca-artifact "${PCA_ARTIFACT:?}" \
  --output "${SUBTYPE_CENTROIDS:?}" --label-column "${LABEL_COLUMN:-celltype_update}" \
  --train-per-subtype "${TRAIN_PER_SUBTYPE:-512}" \
  --validation-per-subtype "${VALIDATION_PER_SUBTYPE:-128}"
