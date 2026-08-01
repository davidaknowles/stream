#!/bin/bash
# Submit controlled OT-bank, PCA-denoising, and interval-skip ablations.

set -euo pipefail

project_root="${PROJECT_ROOT:-/gpfs/commons/home/daknowles/projects/stream}"
python_bin="${STREAM_PYTHON:-$HOME/venv/torchfix/bin/python}"
out_dir="${OUT_DIR:-outputs/stream_hvg10000}"
pca_artifact="${PCA_ARTIFACT:-${out_dir}/pca_denoiser_100pc.npz}"
pca_job="${PCA_JOB:-}"
pool_size="${OT_POOL_SIZE:?Set OT_POOL_SIZE from the completed pool benchmark}"
pairs_per_pool="${OT_PAIRS_PER_POOL:-1024}"
gpu_type="${GPU_TYPE:-b6k}"
cd "$project_root"

submit_experiment() {
  local variant="$1"
  local name="$2"
  local method="$3"
  local cost_space="$4"
  local denoising="$5"
  local max_skip="$6"
  local dependency=()
  local pca_value=""
  if [[ "$cost_space" == "pca" || "$denoising" == "pca" ]]; then
    [[ -f "$pca_artifact" || -n "$pca_job" ]] || { echo "Missing PCA artifact and PCA_JOB" >&2; exit 1; }
    if [[ -n "$pca_job" ]]; then dependency=(--dependency="afterok:${pca_job}"); fi
    pca_value="$pca_artifact"
  fi
  local label="gene_holdout_seed1337_frac20_causal_${name}_pool${pool_size}_bank"
  local job_name="stream_${variant}_${name}_p${pool_size}"
  local train_job
  local eval_job
  train_job="$(sbatch --parsable --gres="gpu:${gpu_type}:1" "${dependency[@]}" --job-name="$job_name" \
    --export="ALL,CONFIG=configs/stream_mouse_dev.yaml,VARIANT=${variant},OUT_DIR=${out_dir},N_HVG=10000,CELL_STATE=uce,UCE_MODE=online,BATCH_SIZE=8,GENE_CHUNK_SIZE=256,OT_POOL_SIZE=${pool_size},OT_PAIRS_PER_POOL=${pairs_per_pool},OT_PAIR_BANK_MODE=interval,OT_METHOD=${method},OT_PARTIAL_MASS=0.95,OT_MARGINAL_RELAXATION=0.1,OT_COST_SPACE=${cost_space},ENDPOINT_DENOISING=${denoising},PCA_ARTIFACT=${pca_value},MAX_INTERVAL_SKIP=${max_skip},EPOCHS=50,STEPS_PER_EPOCH=100,LOSS_GENE_SUBSET=${out_dir}/gene_holdout_seed1337_frac20_train.csv,EXPERIMENT_LABEL=${label},CRE_TOKEN_ARRAYS=${out_dir}/cre_token_arrays.npz,STREAM_PYTHON=${python_bin},WANDB_MODE=online,WANDB_RUN_NAME=${job_name}" \
    slurm/run_stream_train.sbatch)"
  eval_job="$(sbatch --parsable --gres="gpu:${gpu_type}:1" --dependency="afterok:${train_job}" --job-name="${job_name}_eval" \
    --export="ALL,CONFIG=configs/stream_mouse_dev.yaml,VARIANT=${variant},OUT_DIR=${out_dir},EXPERIMENT_LABEL=${label},CRE_TOKEN_ARRAYS=${out_dir}/cre_token_arrays.npz,EVAL_GENE_SUBSET=train_genes:${out_dir}/gene_holdout_seed1337_frac20_train.csv;heldout_genes:${out_dir}/gene_holdout_seed1337_frac20_heldout.csv,INTEGRATION_STEPS=4;8;16,STREAM_PYTHON=${python_bin}" \
    slurm/run_stream_causal_evaluate.sbatch)"
  echo "${variant} ${name}: train=${train_job}; evaluation=${eval_job}"
}

for variant in film cross_attention; do
  submit_experiment "$variant" bank_balanced_raw_adj balanced expression none 0
  submit_experiment "$variant" bank_partial_raw_adj partial expression none 0
  submit_experiment "$variant" bank_uot_raw_adj unbalanced expression none 0
  submit_experiment "$variant" bank_partial_pcacost_adj partial pca none 0
  submit_experiment "$variant" bank_partial_pcacost_denoise_adj partial pca pca 0
  submit_experiment "$variant" bank_partial_pcacost_denoise_skip1 partial pca pca 1
  submit_experiment "$variant" bank_partial_pcacost_denoise_skip2 partial pca pca 2
done
