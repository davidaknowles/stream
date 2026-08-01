#!/bin/bash
# Submit PCA-smoothed-expression UCE conditioning ablations.

set -euo pipefail

project_root="${PROJECT_ROOT:-/gpfs/commons/home/daknowles/projects/stream}"
python_bin="${STREAM_PYTHON:-$HOME/venv/torchfix/bin/python}"
out_dir="${OUT_DIR:-outputs/stream_hvg10000}"
pca_artifact="${PCA_ARTIFACT:-${out_dir}/pca_denoiser_100pc.npz}"
pool_size="${OT_POOL_SIZE:-16384}"
pairs_per_pool="${OT_PAIRS_PER_POOL:-1024}"
gpu_type="${GPU_TYPE:-b6k}"
smoke_job="${UCE_SMOOTHING_SMOKE_JOB:-}"
cd "$project_root"

[[ -f "$pca_artifact" ]] || { echo "Missing PCA artifact: $pca_artifact" >&2; exit 1; }
dependency=()
if [[ -n "$smoke_job" ]]; then
  dependency=(--dependency="afterok:${smoke_job}")
fi

for variant in film cross_attention; do
  name="bank_partial_pcacost_denoise_pca_uce_adj"
  label="gene_holdout_seed1337_frac20_causal_${name}_pool${pool_size}_bank"
  job_name="stream_${variant}_${name}_p${pool_size}"
  train_job="$(sbatch --parsable --gres="gpu:${gpu_type}:1" "${dependency[@]}" --job-name="$job_name" \
    --export="ALL,CONFIG=configs/stream_mouse_dev.yaml,VARIANT=${variant},OUT_DIR=${out_dir},N_HVG=10000,CELL_STATE=uce,UCE_MODE=online,BATCH_SIZE=8,GENE_CHUNK_SIZE=256,OT_POOL_SIZE=${pool_size},OT_PAIRS_PER_POOL=${pairs_per_pool},OT_PAIR_BANK_MODE=interval,OT_METHOD=partial,OT_PARTIAL_MASS=0.95,OT_COST_SPACE=pca,ENDPOINT_DENOISING=pca,UCE_EXPRESSION_PREPROCESSING=pca,PCA_ARTIFACT=${pca_artifact},MAX_INTERVAL_SKIP=0,EPOCHS=50,STEPS_PER_EPOCH=100,LOSS_GENE_SUBSET=${out_dir}/gene_holdout_seed1337_frac20_train.csv,EXPERIMENT_LABEL=${label},CRE_TOKEN_ARRAYS=${out_dir}/cre_token_arrays.npz,STREAM_PYTHON=${python_bin},WANDB_MODE=online,WANDB_RUN_NAME=${job_name}" \
    slurm/run_stream_train.sbatch)"
  eval_job="$(sbatch --parsable --gres="gpu:${gpu_type}:1" --dependency="afterok:${train_job}" --job-name="${job_name}_eval" \
    --export="ALL,CONFIG=configs/stream_mouse_dev.yaml,VARIANT=${variant},OUT_DIR=${out_dir},EXPERIMENT_LABEL=${label},CRE_TOKEN_ARRAYS=${out_dir}/cre_token_arrays.npz,EVAL_GENE_SUBSET=train_genes:${out_dir}/gene_holdout_seed1337_frac20_train.csv;heldout_genes:${out_dir}/gene_holdout_seed1337_frac20_heldout.csv,INTEGRATION_STEPS=4;8;16,STREAM_PYTHON=${python_bin}" \
    slurm/run_stream_causal_evaluate.sbatch)"
  echo "${variant}: train=${train_job}; evaluation=${eval_job}"
done
