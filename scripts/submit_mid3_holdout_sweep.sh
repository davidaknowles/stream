#!/bin/bash
# Submit adjacent/skip-1/skip-2 forecasts across the middle three-stage block.

set -euo pipefail

project_root="${PROJECT_ROOT:-/gpfs/commons/home/daknowles/projects/stream}"
python_bin="${STREAM_PYTHON:-$HOME/venv/torchfix/bin/python}"
out_dir="${OUT_DIR:-outputs/stream_hvg10000}"
split_path="${TIMEPOINT_SPLIT:-${out_dir}/timepoint_split_mid3.json}"
pca_artifact="${PCA_ARTIFACT:-${out_dir}/pca_denoiser_mid3_100pc.npz}"
pca_job="${PCA_JOB:?Set PCA_JOB to the train-only middle-block PCA job}"
pool_size="${OT_POOL_SIZE:-16384}"
pairs_per_pool="${OT_PAIRS_PER_POOL:-1024}"
gpu_type="${GPU_TYPE:-b6k}"
cd "$project_root"

[[ -f "$split_path" ]] || { echo "Missing split: $split_path" >&2; exit 1; }

for max_skip in 0 1 2; do
  name="mid3_partial_pcacost_denoise_pca_uce_skip${max_skip}"
  label="gene_holdout_seed1337_frac20_causal_${name}_pool${pool_size}_bank"
  job_name="stream_cross_attention_${name}_p${pool_size}"
  train_job="$(sbatch --parsable --gres="gpu:${gpu_type}:1" --dependency="afterok:${pca_job}" --job-name="$job_name" \
    --export="ALL,CONFIG=configs/stream_mouse_dev.yaml,TIMEPOINT_SPLIT=${split_path},VARIANT=cross_attention,OUT_DIR=${out_dir},N_HVG=10000,CELL_STATE=uce,UCE_MODE=online,BATCH_SIZE=8,GENE_CHUNK_SIZE=256,OT_POOL_SIZE=${pool_size},OT_PAIRS_PER_POOL=${pairs_per_pool},OT_PAIR_BANK_MODE=interval,OT_METHOD=partial,OT_PARTIAL_MASS=0.95,OT_COST_SPACE=pca,ENDPOINT_DENOISING=pca,UCE_EXPRESSION_PREPROCESSING=pca,PCA_ARTIFACT=${pca_artifact},MAX_INTERVAL_SKIP=${max_skip},EPOCHS=50,STEPS_PER_EPOCH=100,LOSS_GENE_SUBSET=${out_dir}/gene_holdout_seed1337_frac20_train.csv,EXPERIMENT_LABEL=${label},CRE_TOKEN_ARRAYS=${out_dir}/cre_token_arrays.npz,STREAM_PYTHON=${python_bin},WANDB_MODE=online,WANDB_RUN_NAME=${job_name}" \
    slurm/run_stream_train.sbatch)"
  eval_job="$(sbatch --parsable --gres="gpu:${gpu_type}:1" --dependency="afterok:${train_job}" --job-name="${job_name}_eval" \
    --export="ALL,CONFIG=configs/stream_mouse_dev.yaml,TIMEPOINT_SPLIT=${split_path},VARIANT=cross_attention,OUT_DIR=${out_dir},EXPERIMENT_LABEL=${label},CRE_TOKEN_ARRAYS=${out_dir}/cre_token_arrays.npz,EVAL_GENE_SUBSET=train_genes:${out_dir}/gene_holdout_seed1337_frac20_train.csv;heldout_genes:${out_dir}/gene_holdout_seed1337_frac20_heldout.csv,INTEGRATION_STEPS=4;8;16,STREAM_PYTHON=${python_bin}" \
    slurm/run_stream_causal_evaluate.sbatch)"
  echo "skip${max_skip}: train=${train_job}; evaluation=${eval_job}"
done
