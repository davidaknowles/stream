#!/bin/bash
# Submit matched coupled score-flow and autonomous-control experiments.

set -euo pipefail
project_root="${PROJECT_ROOT:-/gpfs/commons/home/daknowles/projects/stream}"
out_dir="${OUT_DIR:-outputs/stream_hvg10000}"
split="${TIMEPOINT_SPLIT:-${out_dir}/timepoint_split_mid3.json}"
pca="${PCA_ARTIFACT:-${out_dir}/pca_denoiser_mid3_100pc.npz}"
cre="${CRE_TOKEN_ARRAYS:-${out_dir}/cre_token_arrays.npz}"
python_bin="${STREAM_PYTHON:-$HOME/venv/torchfix/bin/python}"
gpu_type="${GPU_TYPE:-b6k}"
cd "$project_root"
evaluation_jobs=()

for variant in film cross_attention; do
  for denoising in none knn metacell; do
    label="mid3_partial_pcacost_${denoising}_systematic_coupled_v2"
    stem="score_flow_${variant}_${label}"
    train_job="$(sbatch --parsable --gres="gpu:${gpu_type}:1" --job-name="sfc_${variant}_${denoising}" \
      --export="ALL,MODE=train,OUT_DIR=${out_dir},TIMEPOINT_SPLIT=${split},PCA_ARTIFACT=${pca},CRE_TOKEN_ARRAYS=${cre},VARIANT=${variant},ENDPOINT_DENOISING=${denoising},EXPERIMENT_LABEL=${label},OT_POOL_SIZE=16384,OT_PAIRS_PER_POOL=1024,BATCH_SIZE=8,GENE_CHUNK_SIZE=256,EPOCHS=50,STEPS_PER_EPOCH=100,STREAM_PYTHON=${python_bin}" \
      slurm/run_score_flow.sbatch)"
    eval_job="$(sbatch --parsable --gres="gpu:${gpu_type}:1" --dependency="afterok:${train_job}" \
      --job-name="sfc_${variant}_${denoising}_eval" \
      --export="ALL,MODE=evaluate,OUT_DIR=${out_dir},PCA_ARTIFACT=${pca},CHECKPOINT=${out_dir}/model_${stem}.pt,DIFFUSIONS=0;0.001;0.01;0.05,STREAM_PYTHON=${python_bin}" \
      slurm/run_score_flow.sbatch)"
    evaluation_jobs+=("$eval_job")
    echo "${variant} ${denoising}: train=${train_job}; eval=${eval_job}"
  done
done

dependency="$(IFS=:; echo "${evaluation_jobs[*]}")"
summary_job="$(sbatch --parsable --partition=cpu --cpus-per-task=1 --mem=8G --time=00:30:00 \
  --dependency="afterok:${dependency}" --job-name=sfc_summary \
  --output=logs/score_flow_coupled_summary_%j.out --error=logs/score_flow_coupled_summary_%j.err \
  --wrap="bash -lc 'cd ${project_root} && source ~/.bashrc && ${python_bin} scripts/summarize_score_flow_ablation.py --out-dir ${out_dir}'")"
echo "summary=${summary_job}"
