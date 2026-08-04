#!/bin/bash
#SBATCH --job-name=stream_score_flow
#SBATCH --partition=gpu
#SBATCH --gres=gpu:b6k:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=30:00:00
#SBATCH --output=logs/stream_score_flow_%x_%j.out
#SBATCH --error=logs/stream_score_flow_%x_%j.err

set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-/gpfs/commons/home/daknowles/projects/stream}"
STREAM_PYTHON="${STREAM_PYTHON:-$HOME/venv/torchfix/bin/python}"
MODE="${MODE:-train}"
OUT_DIR="${OUT_DIR:-outputs/stream_hvg10000}"
cd "$PROJECT_ROOT"
mkdir -p logs
export PYTHONPATH="$PROJECT_ROOT/src:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

if [[ "$MODE" == "train" ]]; then
  "$STREAM_PYTHON" scripts/train_score_flow.py \
    --config "${CONFIG:-configs/stream_mouse_dev.yaml}" \
    --timepoint-split "${TIMEPOINT_SPLIT:?}" --out-dir "$OUT_DIR" \
    --variant "${VARIANT:-cross_attention}" --endpoint-denoising "${ENDPOINT_DENOISING:-none}" \
    --pca-artifact "${PCA_ARTIFACT:?}" --cre-token-arrays "${CRE_TOKEN_ARRAYS:?}" \
    --experiment-label "${EXPERIMENT_LABEL:?}" --ot-pool-size "${OT_POOL_SIZE:-16384}" \
    --ot-pairs-per-pool "${OT_PAIRS_PER_POOL:-1024}" --batch-size "${BATCH_SIZE:-8}" \
    --gene-chunk-size "${GENE_CHUNK_SIZE:-256}" --epochs "${EPOCHS:-50}" \
    --steps-per-epoch "${STEPS_PER_EPOCH:-100}" --noise-scale "${NOISE_SCALE:-0.2}" \
    --limit-intervals "${LIMIT_INTERVALS:-0}" --device cuda
elif [[ "$MODE" == "population_finetune" ]]; then
  "$STREAM_PYTHON" scripts/finetune_population_rollout.py \
    --config "${CONFIG:-configs/stream_mouse_dev.yaml}" --out-dir "$OUT_DIR" \
    --checkpoint "${CHECKPOINT:?}" --pca-artifact "${PCA_ARTIFACT:?}" \
    --experiment-label "${EXPERIMENT_LABEL:?}" --source-batch-size "${SOURCE_BATCH_SIZE:-8}" \
    --target-batch-size "${TARGET_BATCH_SIZE:-32}" --particles "${PARTICLES:-2}" \
    --rollout-steps "${INTEGRATION_STEPS:-2}" --diffusions "${DIFFUSIONS:-0.001;0.01}" \
    --max-updates "${MAX_UPDATES:-200}" --validation-every "${VALIDATION_EVERY:-20}" \
    --validation-intervals "${VALIDATION_INTERVALS:-4}" --patience "${PATIENCE:-5}" \
    --learning-rate "${LEARNING_RATE:-1e-5}" --gene-chunk-size "${GENE_CHUNK_SIZE:-256}" \
    --limit-intervals "${LIMIT_INTERVALS:-0}" --device cuda
elif [[ "$MODE" == "evaluate" ]]; then
  "$STREAM_PYTHON" scripts/evaluate_score_flow.py \
    --config "${CONFIG:-configs/stream_mouse_dev.yaml}" --out-dir "$OUT_DIR" \
    --checkpoint "${CHECKPOINT:?}" --pca-artifact "${PCA_ARTIFACT:?}" \
    --cells-per-interval "${CELLS_PER_INTERVAL:-128}" --steps "${INTEGRATION_STEPS:-16}" \
    --diffusions "${DIFFUSIONS:-0;0.001;0.01;0.05}" --stochastic-replicates "${STOCHASTIC_REPLICATES:-3}" --device cuda
else
  echo "Unknown MODE=$MODE" >&2
  exit 2
fi
