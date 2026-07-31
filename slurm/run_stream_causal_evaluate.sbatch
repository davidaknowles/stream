#!/bin/bash
#SBATCH --job-name=stream_causal_eval
#SBATCH --partition=gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=logs/stream_causal_eval_%x_%j.out
#SBATCH --error=logs/stream_causal_eval_%x_%j.err

set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-/gpfs/commons/home/daknowles/projects/stream}"
STREAM_PYTHON="${STREAM_PYTHON:-$HOME/venv/torchfix/bin/python}"
CONFIG="${CONFIG:-configs/stream_mouse_dev.yaml}"
OUT_DIR="${OUT_DIR:?OUT_DIR is required}"
EXPERIMENT_LABEL="${EXPERIMENT_LABEL:?EXPERIMENT_LABEL is required}"
EVAL_GENE_SUBSET="${EVAL_GENE_SUBSET:-}"
CELLS_PER_INTERVAL="${CELLS_PER_INTERVAL:-128}"
INTEGRATION_STEPS="${INTEGRATION_STEPS:-4,8,16}"
PCA_CELLS="${PCA_CELLS:-10000}"
cd "$PROJECT_ROOT"
mkdir -p logs
export PYTHONPATH="$PROJECT_ROOT/src:${PYTHONPATH:-}"
args=(--config "$CONFIG" --out-dir "$OUT_DIR" --experiment-label "$EXPERIMENT_LABEL" --cells-per-interval "$CELLS_PER_INTERVAL" --integration-steps "$INTEGRATION_STEPS" --pca-cells "$PCA_CELLS")
if [[ -n "$EVAL_GENE_SUBSET" ]]; then
  eval_gene_subset_list="${EVAL_GENE_SUBSET//;/,}"
  IFS=',' read -r -a eval_gene_subsets <<< "$eval_gene_subset_list"
  for subset in "${eval_gene_subsets[@]}"; do
    if [[ -n "$subset" ]]; then args+=(--eval-gene-subset "$subset"); fi
  done
fi
"$STREAM_PYTHON" scripts/evaluate_causal_stream.py "${args[@]}"
