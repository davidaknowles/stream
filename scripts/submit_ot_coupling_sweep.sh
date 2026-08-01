#!/bin/bash
# Compare balanced, partial, and KL-unbalanced OT across STREAM variants.

set -euo pipefail

project_root="${PROJECT_ROOT:-/gpfs/commons/home/daknowles/projects/stream}"
python_bin="${STREAM_PYTHON:-$HOME/venv/torchfix/bin/python}"
out_dir="${OUT_DIR:-outputs/stream_hvg10000}"
batch_size="${BATCH_SIZE:-8}"
gene_chunk_size="${GENE_CHUNK_SIZE:-256}"
ot_pool_size="${OT_POOL_SIZE:-1024}"
ot_pairs_per_pool="${OT_PAIRS_PER_POOL:-64}"
gpu_type="${GPU_TYPE:-b6k}"
max_epochs="${EPOCHS:-50}"
steps_per_epoch="${STEPS_PER_EPOCH:-100}"
partial_mass="${OT_PARTIAL_MASS:-0.95}"
uot_rho="${OT_MARGINAL_RELAXATION:-0.1}"
cd "$project_root"

for variant in film cross_attention; do
  for cre_version in independent tss_context; do
    if [[ "$cre_version" == "independent" ]]; then
      cre_tokens="${out_dir}/cre_token_arrays.npz"
      cre_label=""
    else
      cre_tokens="${out_dir}/cre_token_arrays_tss_context.npz"
      cre_label="tss_context_"
    fi
    for method in balanced partial unbalanced; do
      case "$method" in
        balanced) method_label="balanced" ;;
        partial) method_label="partial95" ;;
        unbalanced) method_label="uot_rho0p1" ;;
      esac
      label="gene_holdout_seed1337_frac20_causal_${cre_label}${method_label}_pool${ot_pool_size}_earlystop"
      name="stream_${variant}_${cre_version}_${method_label}_p${ot_pool_size}"
      train_job="$(sbatch --parsable --gres="gpu:${gpu_type}:1" --job-name="$name" \
        --export="ALL,CONFIG=configs/stream_mouse_dev.yaml,VARIANT=${variant},OUT_DIR=${out_dir},N_HVG=10000,CELL_STATE=uce,UCE_MODE=online,BATCH_SIZE=${batch_size},GENE_CHUNK_SIZE=${gene_chunk_size},OT_POOL_SIZE=${ot_pool_size},OT_PAIRS_PER_POOL=${ot_pairs_per_pool},EPOCHS=${max_epochs},STEPS_PER_EPOCH=${steps_per_epoch},OT_METHOD=${method},OT_PARTIAL_MASS=${partial_mass},OT_MARGINAL_RELAXATION=${uot_rho},LOSS_GENE_SUBSET=${out_dir}/gene_holdout_seed1337_frac20_train.csv,EXPERIMENT_LABEL=${label},CRE_TOKEN_ARRAYS=${cre_tokens},STREAM_PYTHON=${python_bin},WANDB_MODE=online,WANDB_RUN_NAME=${name}" \
        slurm/run_stream_train.sbatch)"
      eval_job="$(sbatch --parsable --gres="gpu:${gpu_type}:1" --dependency="afterok:${train_job}" --job-name="${name}_eval" \
        --export="ALL,CONFIG=configs/stream_mouse_dev.yaml,VARIANT=${variant},OUT_DIR=${out_dir},EXPERIMENT_LABEL=${label},CRE_TOKEN_ARRAYS=${cre_tokens},EVAL_GENE_SUBSET=train_genes:${out_dir}/gene_holdout_seed1337_frac20_train.csv;heldout_genes:${out_dir}/gene_holdout_seed1337_frac20_heldout.csv,INTEGRATION_STEPS=4;8;16,STREAM_PYTHON=${python_bin}" \
        slurm/run_stream_causal_evaluate.sbatch)"
      echo "${variant} ${cre_version} ${method}: train=${train_job}; evaluation=${eval_job}"
    done
  done
done
