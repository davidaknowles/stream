#!/bin/bash
# Submit raw-endpoint growth-only and joint growth/velocity comparisons.

set -euo pipefail
project_root="${PROJECT_ROOT:-/gpfs/commons/home/daknowles/projects/stream}"
out_dir="${OUT_DIR:-outputs/stream_hvg10000}"
pca="${PCA_ARTIFACT:-${out_dir}/pca_denoiser_mid3_100pc.npz}"
python_bin="${STREAM_PYTHON:-$HOME/venv/torchfix/bin/python}"
gpu_type="${GPU_TYPE:-b6k}"
growth_smoke="${GROWTH_SMOKE_JOB:-19533138}"
joint_smoke="${JOINT_SMOKE_JOB:-19533139}"
cd "$project_root"
evaluation_jobs=()

for variant in film cross_attention; do
  parent="${out_dir}/model_score_flow_${variant}_mid3_partial_pcacost_none_systematic_gene_scaled_v3.pt"
  for mode in growth_only joint; do
    label="mid3_growth_${mode}_raw_v5"
    if [[ "$mode" == "growth_only" ]]; then
      smoke_job="$growth_smoke"
      source_batch=64
      target_batch=128
      particles=1
    else
      smoke_job="$joint_smoke"
      source_batch=8
      target_batch=32
      particles=2
    fi
    train_job="$(sbatch --parsable --gres="gpu:${gpu_type}:1" \
      --dependency="afterok:${smoke_job}" --job-name="sfg_${variant}_${mode}" \
      --export="ALL,MODE=population_finetune,OUT_DIR=${out_dir},CHECKPOINT=${parent},PCA_ARTIFACT=${pca},EXPERIMENT_LABEL=${label},GROWTH_MODE=${mode},SOURCE_BATCH_SIZE=${source_batch},TARGET_BATCH_SIZE=${target_batch},PARTICLES=${particles},INTEGRATION_STEPS=2,DIFFUSIONS=0.001;0.01,MAX_UPDATES=200,VALIDATION_EVERY=20,VALIDATION_INTERVALS=4,PATIENCE=5,GENE_CHUNK_SIZE=256,STREAM_PYTHON=${python_bin}" \
      slurm/run_score_flow.sbatch)"
    checkpoint="${out_dir}/model_score_flow_${variant}_${label}.pt"
    eval_job="$(sbatch --parsable --gres="gpu:${gpu_type}:1" \
      --dependency="afterok:${train_job}" --job-name="sfg_${variant}_${mode}_eval" \
      --export="ALL,MODE=evaluate,OUT_DIR=${out_dir},CHECKPOINT=${checkpoint},PCA_ARTIFACT=${pca},CELLS_PER_INTERVAL=128,INTEGRATION_STEPS=16,DIFFUSIONS=0;0.001;0.01;0.05,STOCHASTIC_REPLICATES=3,STREAM_PYTHON=${python_bin}" \
      slurm/run_score_flow.sbatch)"
    evaluation_jobs+=("$eval_job")
    echo "${variant} ${mode}: train=${train_job}; eval=${eval_job}"
  done
done

dependency="$(IFS=:; echo "${evaluation_jobs[*]}")"
summary_job="$(sbatch --parsable --partition=cpu --cpus-per-task=1 --mem=8G --time=00:30:00 \
  --dependency="afterok:${dependency}" --job-name=sfg_summary \
  --output=logs/score_flow_growth_summary_%j.out --error=logs/score_flow_growth_summary_%j.err \
  --wrap="bash -lc 'cd ${project_root} && source ~/.bashrc && ${python_bin} scripts/summarize_score_flow_ablation.py --out-dir ${out_dir} --pattern \"causal_eval_model_score_flow_*_mid3_growth_*_raw_v5.csv\" --output score_flow_growth_raw_v5_summary.csv'")"
echo "summary=${summary_job}"
