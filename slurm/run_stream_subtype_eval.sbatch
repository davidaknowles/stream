#!/bin/bash
#SBATCH --job-name=stream_subtype_eval
#SBATCH --partition=gpu
#SBATCH --gres=gpu:b6k:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=30:00:00
#SBATCH --output=logs/stream_subtype_eval_%x_%j.out
#SBATCH --error=logs/stream_subtype_eval_%x_%j.err

set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-/gpfs/commons/home/daknowles/projects/stream}"
STREAM_PYTHON="${STREAM_PYTHON:-$HOME/venv/torchfix/bin/python}"
cd "$PROJECT_ROOT"
mkdir -p logs
export PYTHONPATH="$PROJECT_ROOT/src:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

gene_subset_args=()
IFS=';' read -ra gene_subsets <<< "${EVAL_GENE_SUBSETS:-}"
for subset in "${gene_subsets[@]}"; do
  [[ -z "$subset" ]] || gene_subset_args+=(--eval-gene-subset "$subset")
done

"$STREAM_PYTHON" scripts/evaluate_subtype_dynamics.py \
  --config "${CONFIG:-configs/stream_mouse_dev.yaml}" \
  --out-dir "${OUT_DIR:-outputs/stream_hvg10000}" --checkpoint "${CHECKPOINT:?}" \
  --pca-artifact "${PCA_ARTIFACT:?}" --subtype-centroids "${SUBTYPE_CENTROIDS:?}" \
  --cells-per-batch "${CELLS_PER_BATCH:-128}" \
  --batches-per-interval "${BATCHES_PER_INTERVAL:-32}" \
  --minimum-subtype-cells "${MINIMUM_SUBTYPE_CELLS:-10}" \
  --steps "${INTEGRATION_STEPS:-16}" --growth-control "${GROWTH_CONTROL:-none}" \
  "${gene_subset_args[@]}" --device cuda
