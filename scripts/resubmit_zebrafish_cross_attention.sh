#!/bin/bash
# Resume the retained zebrafish transfer benchmark after target CRE embedding.

set -euo pipefail

project_root="${PROJECT_ROOT:-/gpfs/commons/home/daknowles/projects/stream}"
python_bin="${STREAM_PYTHON:-$HOME/venv/torchfix/bin/python}"
cd "$project_root"

submit() {
  sbatch --parsable "$@"
}

for coordinate in relative days; do
  config="configs/stream_zebrafish_${coordinate}.yaml"
  out_dir="outputs/zscape_stream_hvg10000_${coordinate}"
  source_dir="outputs/stream_mouse_hvg10000_${coordinate}"
  source_checkpoint="${source_dir}/model_cross_attention_uce_source.pt"
  cache="${out_dir}/eval_batches.npz"

  [[ -f "$source_checkpoint" ]] || { echo "Missing source checkpoint: $source_checkpoint" >&2; exit 1; }
  [[ -f "$cache" ]] || { echo "Missing evaluation cache: $cache" >&2; exit 1; }

  cre_job="$(submit --job-name="zfish_${coordinate}_10000_cross_cre_controls" \
    --export="ALL,CONFIG=${config},N_HVG=10000,OUT_DIR=${out_dir},STREAM_PYTHON=${python_bin}" \
    slurm/run_stream_embed_cre.sbatch)"

  zero_job="$(submit --dependency="afterok:${cre_job}" --job-name="zfish_${coordinate}_10000_cross_zero_shot_controls" \
    --export="ALL,CONFIG=${config},VARIANT=cross_attention,N_HVG=10000,OUT_DIR=${out_dir},CELL_STATE=uce,UCE_EMBEDDING_DIR=outputs/zscape_uce_full_controls/embeddings,EXPERIMENT_LABEL=zero_shot,CHECKPOINT=${source_checkpoint},EVAL_CACHE=${cache},STREAM_PYTHON=${python_bin}" \
    slurm/run_stream_evaluate.sbatch)"
  fine_job="$(submit --dependency="afterok:${cre_job}" --job-name="zfish_${coordinate}_10000_cross_fine_controls" \
    --export="ALL,CONFIG=${config},VARIANT=cross_attention,N_HVG=10000,OUT_DIR=${out_dir},CELL_STATE=uce,UCE_EMBEDDING_DIR=outputs/zscape_uce_full_controls/embeddings,EXPERIMENT_LABEL=fine_tuned,INIT_CHECKPOINT=${source_checkpoint},STREAM_PYTHON=${python_bin},WANDB_MODE=online,WANDB_RUN_NAME=zfish_${coordinate}_10000_cross_fine_controls" \
    slurm/run_stream_train.sbatch)"
  scratch_job="$(submit --dependency="afterok:${cre_job}" --job-name="zfish_${coordinate}_10000_cross_scratch_controls" \
    --export="ALL,CONFIG=${config},VARIANT=cross_attention,N_HVG=10000,OUT_DIR=${out_dir},CELL_STATE=uce,UCE_EMBEDDING_DIR=outputs/zscape_uce_full_controls/embeddings,EXPERIMENT_LABEL=zebrafish_only,STREAM_PYTHON=${python_bin},WANDB_MODE=online,WANDB_RUN_NAME=zfish_${coordinate}_10000_cross_scratch_controls" \
    slurm/run_stream_train.sbatch)"
  fine_eval="$(submit --dependency="afterok:${fine_job}" --job-name="zfish_${coordinate}_10000_cross_fine_eval_controls" \
    --export="ALL,CONFIG=${config},VARIANT=cross_attention,N_HVG=10000,OUT_DIR=${out_dir},CELL_STATE=uce,UCE_EMBEDDING_DIR=outputs/zscape_uce_full_controls/embeddings,EXPERIMENT_LABEL=fine_tuned,EVAL_CACHE=${cache},STREAM_PYTHON=${python_bin}" \
    slurm/run_stream_evaluate.sbatch)"
  scratch_eval="$(submit --dependency="afterok:${scratch_job}" --job-name="zfish_${coordinate}_10000_cross_scratch_eval_controls" \
    --export="ALL,CONFIG=${config},VARIANT=cross_attention,N_HVG=10000,OUT_DIR=${out_dir},CELL_STATE=uce,UCE_EMBEDDING_DIR=outputs/zscape_uce_full_controls/embeddings,EXPERIMENT_LABEL=zebrafish_only,EVAL_CACHE=${cache},STREAM_PYTHON=${python_bin}" \
    slurm/run_stream_evaluate.sbatch)"

  echo "${coordinate}: CRE=${cre_job}, zero-shot=${zero_job}, fine=${fine_job}, scratch=${scratch_job}, fine-eval=${fine_eval}, scratch-eval=${scratch_eval}"
done
