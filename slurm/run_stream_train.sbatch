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
EPOCHS="${EPOCHS:-}"
HVG_CSV="${HVG_CSV:-}"
N_HVG="${N_HVG:-}"
OUT_DIR="${OUT_DIR:-}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-}"
BATCH_SIZE="${BATCH_SIZE:-}"
GENE_CHUNK_SIZE="${GENE_CHUNK_SIZE:-}"
CELL_STATE="${CELL_STATE:-}"
UCE_EMBEDDING_DIR="${UCE_EMBEDDING_DIR:-}"
UCE_MODE="${UCE_MODE:-}"
WANDB_MODE="${WANDB_MODE:-}"
EXPERIMENT_LABEL="${EXPERIMENT_LABEL:-}"
INIT_CHECKPOINT="${INIT_CHECKPOINT:-}"
LOSS_GENE_SUBSET="${LOSS_GENE_SUBSET:-}"
CRE_TOKEN_ARRAYS="${CRE_TOKEN_ARRAYS:-}"
VALIDATION_FRACTION="${VALIDATION_FRACTION:-0.1}"
VALIDATION_BATCHES_PER_INTERVAL="${VALIDATION_BATCHES_PER_INTERVAL:-1}"
VALIDATION_EVERY_EPOCHS="${VALIDATION_EVERY_EPOCHS:-1}"
EARLY_STOPPING_PATIENCE="${EARLY_STOPPING_PATIENCE:-12}"
EARLY_STOPPING_MIN_DELTA="${EARLY_STOPPING_MIN_DELTA:-0.005}"
VALIDATION_EMA_ALPHA="${VALIDATION_EMA_ALPHA:-0.3}"
ROLLOUT_EVERY_VALIDATIONS="${ROLLOUT_EVERY_VALIDATIONS:-5}"
VALIDATION_ROLLOUT_STEPS="${VALIDATION_ROLLOUT_STEPS:-4}"
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
args=(--config "$CONFIG" --variant "$VARIANT" --steps-per-epoch "$STEPS_PER_EPOCH" --device cuda \
  --validation-fraction "$VALIDATION_FRACTION" \
  --validation-batches-per-interval "$VALIDATION_BATCHES_PER_INTERVAL" \
  --validation-every-epochs "$VALIDATION_EVERY_EPOCHS" \
  --early-stopping-patience "$EARLY_STOPPING_PATIENCE" \
  --early-stopping-min-delta "$EARLY_STOPPING_MIN_DELTA" \
  --validation-ema-alpha "$VALIDATION_EMA_ALPHA" \
  --rollout-every-validations "$ROLLOUT_EVERY_VALIDATIONS" \
  --validation-rollout-steps "$VALIDATION_ROLLOUT_STEPS")
if [[ -n "$EPOCHS" ]]; then args+=(--epochs "$EPOCHS"); fi
if [[ -n "$HVG_CSV" ]]; then args+=(--hvg-csv "$HVG_CSV"); fi
if [[ -n "$N_HVG" ]]; then args+=(--n-hvg "$N_HVG"); fi
if [[ -n "$OUT_DIR" ]]; then args+=(--out-dir "$OUT_DIR"); fi
if [[ -n "$WANDB_RUN_NAME" ]]; then args+=(--wandb-run-name "$WANDB_RUN_NAME"); fi
if [[ -n "$BATCH_SIZE" ]]; then args+=(--batch-size "$BATCH_SIZE"); fi
if [[ -n "$GENE_CHUNK_SIZE" ]]; then args+=(--gene-chunk-size "$GENE_CHUNK_SIZE"); fi
if [[ -n "$CELL_STATE" ]]; then args+=(--cell-state "$CELL_STATE"); fi
if [[ -n "$UCE_EMBEDDING_DIR" ]]; then args+=(--uce-embedding-dir "$UCE_EMBEDDING_DIR"); fi
if [[ -n "$UCE_MODE" ]]; then args+=(--uce-mode "$UCE_MODE"); fi
if [[ -n "$WANDB_MODE" ]]; then args+=(--wandb-mode "$WANDB_MODE"); fi
if [[ -n "$EXPERIMENT_LABEL" ]]; then args+=(--experiment-label "$EXPERIMENT_LABEL"); fi
if [[ -n "$INIT_CHECKPOINT" ]]; then args+=(--init-checkpoint "$INIT_CHECKPOINT"); fi
if [[ -n "$LOSS_GENE_SUBSET" ]]; then args+=(--loss-gene-subset "$LOSS_GENE_SUBSET"); fi
if [[ -n "$CRE_TOKEN_ARRAYS" ]]; then args+=(--cre-token-arrays "$CRE_TOKEN_ARRAYS"); fi
"$STREAM_PYTHON" scripts/train_stream.py "${args[@]}"
