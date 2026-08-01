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
OT_METHOD="${OT_METHOD:-}"
OT_PARTIAL_MASS="${OT_PARTIAL_MASS:-}"
OT_MARGINAL_RELAXATION="${OT_MARGINAL_RELAXATION:-}"
OT_POOL_SIZE="${OT_POOL_SIZE:-}"
OT_PAIRS_PER_POOL="${OT_PAIRS_PER_POOL:-}"
OT_PAIR_BANK_MODE="${OT_PAIR_BANK_MODE:-}"
OT_COST_SPACE="${OT_COST_SPACE:-}"
ENDPOINT_DENOISING="${ENDPOINT_DENOISING:-}"
DENOISING_NEIGHBORS="${DENOISING_NEIGHBORS:-}"
DENOISING_METACELLS="${DENOISING_METACELLS:-}"
UCE_EXPRESSION_PREPROCESSING="${UCE_EXPRESSION_PREPROCESSING:-}"
PCA_ARTIFACT="${PCA_ARTIFACT:-}"
MAX_INTERVAL_SKIP="${MAX_INTERVAL_SKIP:-}"
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
if [[ -n "$OT_METHOD" ]]; then args+=(--ot-method "$OT_METHOD"); fi
if [[ -n "$OT_PARTIAL_MASS" ]]; then args+=(--ot-partial-mass "$OT_PARTIAL_MASS"); fi
if [[ -n "$OT_MARGINAL_RELAXATION" ]]; then args+=(--ot-marginal-relaxation "$OT_MARGINAL_RELAXATION"); fi
if [[ -n "$OT_POOL_SIZE" ]]; then args+=(--ot-pool-size "$OT_POOL_SIZE"); fi
if [[ -n "$OT_PAIRS_PER_POOL" ]]; then args+=(--ot-pairs-per-pool "$OT_PAIRS_PER_POOL"); fi
if [[ -n "$OT_PAIR_BANK_MODE" ]]; then args+=(--ot-pair-bank-mode "$OT_PAIR_BANK_MODE"); fi
if [[ -n "$OT_COST_SPACE" ]]; then args+=(--ot-cost-space "$OT_COST_SPACE"); fi
if [[ -n "$ENDPOINT_DENOISING" ]]; then args+=(--endpoint-denoising "$ENDPOINT_DENOISING"); fi
if [[ -n "$DENOISING_NEIGHBORS" ]]; then args+=(--denoising-neighbors "$DENOISING_NEIGHBORS"); fi
if [[ -n "$DENOISING_METACELLS" ]]; then args+=(--denoising-metacells "$DENOISING_METACELLS"); fi
if [[ -n "$UCE_EXPRESSION_PREPROCESSING" ]]; then args+=(--uce-expression-preprocessing "$UCE_EXPRESSION_PREPROCESSING"); fi
if [[ -n "$PCA_ARTIFACT" ]]; then args+=(--pca-artifact "$PCA_ARTIFACT"); fi
if [[ -n "$MAX_INTERVAL_SKIP" ]]; then args+=(--max-interval-skip "$MAX_INTERVAL_SKIP"); fi
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
