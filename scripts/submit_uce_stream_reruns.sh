#!/bin/bash
# Submit UCE cache generation followed by 5k/10k STREAM comparison reruns.

set -euo pipefail

project_root="${PROJECT_ROOT:-/gpfs/commons/home/daknowles/projects/stream}"
cd "$project_root"

if [[ "${SKIP_UCE_EMBED:-0}" == "1" ]]; then
  test -f outputs/uce/embeddings/manifest.csv
  uce_job=""
  echo "Reusing completed UCE embedding cache."
else
  uce_job="$(sbatch --parsable --job-name=uce_atlas_embed_bf16 slurm/run_uce_atlas_embed.sbatch)"
  echo "Submitted UCE embedding job: $uce_job"
fi

for n_hvg in 5000 10000; do
  out_dir="outputs/stream_hvg${n_hvg}"
  for variant in standard_cfm film cross_attention; do
    train_name="stream_uce_${n_hvg}_${variant}"
    dependency=()
    if [[ -n "$uce_job" ]]; then dependency=(--dependency="afterok:${uce_job}"); fi
    train_job="$(sbatch --parsable "${dependency[@]}" --job-name="$train_name" \
      --export="ALL,VARIANT=${variant},OUT_DIR=${out_dir},N_HVG=${n_hvg},CELL_STATE=uce,UCE_EMBEDDING_DIR=outputs/uce/embeddings,BATCH_SIZE=8,GENE_CHUNK_SIZE=256,STREAM_PYTHON=${HOME}/venv/torchfix/bin/python,WANDB_MODE=online,WANDB_RUN_NAME=${train_name}" \
      slurm/run_stream_train.sbatch)"
    eval_name="stream_uce_eval_${n_hvg}_${variant}"
    eval_job="$(sbatch --parsable --dependency="afterok:${train_job}" --job-name="$eval_name" \
      --export="ALL,VARIANT=${variant},OUT_DIR=${out_dir},N_HVG=${n_hvg},CELL_STATE=uce,UCE_EMBEDDING_DIR=outputs/uce/embeddings,BATCH_SIZE=8,GENE_CHUNK_SIZE=256,STREAM_PYTHON=${HOME}/venv/torchfix/bin/python,EVAL_GENE_SUBSET=legacy_1984:outputs/stream/selected_genes.csv" \
      slurm/run_stream_evaluate.sbatch)"
    echo "Submitted ${train_name}: ${train_job}; ${eval_name}: ${eval_job}"
  done
done
