#!/bin/bash
# Rerun the autonomous online-UCE benchmark with validation-based early stopping.

set -euo pipefail

project_root="${PROJECT_ROOT:-/gpfs/commons/home/daknowles/projects/stream}"
python_bin="${STREAM_PYTHON:-$HOME/venv/torchfix/bin/python}"
out_dir="${OUT_DIR:-outputs/stream_hvg10000}"
max_epochs="${EPOCHS:-50}"
cd "$project_root"

submit_version() {
  local suffix="$1"
  local cre_tokens="$2"
  local label="gene_holdout_seed1337_frac20_causal_${suffix}earlystop"
  local train_job
  local eval_job

  train_job="$(sbatch --parsable --job-name="stream_${suffix}earlystop" \
    --export="ALL,CONFIG=configs/stream_mouse_dev.yaml,VARIANT=cross_attention,OUT_DIR=${out_dir},N_HVG=10000,CELL_STATE=uce,UCE_MODE=online,BATCH_SIZE=8,GENE_CHUNK_SIZE=256,EPOCHS=${max_epochs},LOSS_GENE_SUBSET=${out_dir}/gene_holdout_seed1337_frac20_train.csv,EXPERIMENT_LABEL=${label},CRE_TOKEN_ARRAYS=${cre_tokens},STREAM_PYTHON=${python_bin},WANDB_MODE=online,WANDB_RUN_NAME=stream_${suffix}earlystop" \
    slurm/run_stream_train.sbatch)"

  eval_job="$(sbatch --parsable --dependency="afterok:${train_job}" --job-name="stream_${suffix}earlystop_eval" \
    --export="ALL,CONFIG=configs/stream_mouse_dev.yaml,OUT_DIR=${out_dir},EXPERIMENT_LABEL=${label},CRE_TOKEN_ARRAYS=${cre_tokens},EVAL_GENE_SUBSET=train_genes:${out_dir}/gene_holdout_seed1337_frac20_train.csv;heldout_genes:${out_dir}/gene_holdout_seed1337_frac20_heldout.csv,INTEGRATION_STEPS=4;8;16,STREAM_PYTHON=${python_bin}" \
    slurm/run_stream_causal_evaluate.sbatch)"
  echo "${suffix:-independent}: train=${train_job}; evaluation=${eval_job}"
}

submit_version "" "${out_dir}/cre_token_arrays.npz"
submit_version "tss_context_" "${out_dir}/cre_token_arrays_tss_context.npz"
