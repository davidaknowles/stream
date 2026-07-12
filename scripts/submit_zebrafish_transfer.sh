#!/bin/bash
# Submit the complete ZSCAPE UCE transfer benchmark with two time-coordinate regimes.

set -euo pipefail

project_root="${PROJECT_ROOT:-/gpfs/commons/home/daknowles/projects/stream}"
python_bin="${STREAM_PYTHON:-$HOME/venv/torchfix/bin/python}"
cd "$project_root"

submit() {
  sbatch --parsable "$@"
}

zscape_prepare="$(submit slurm/run_zscape_prepare.sbatch)"
zscape_hvg="$(submit --dependency="afterok:${zscape_prepare}" \
  --export="ALL,CONFIG=configs/stream_zebrafish_relative.yaml,HVG_OUTPUT=outputs/zscape/streaming_gene_variances.csv,N_GENES=20000,STREAM_DATA_PYTHON=${python_bin}" \
  slurm/run_stream_select_hvgs.sbatch)"
zscape_uce="$(submit --dependency="afterok:${zscape_prepare}" \
  --export="ALL,ADATA_DIR=downloads/zscape/adata,OUT_DIR=outputs/zscape_uce/embeddings,SPECIES=zebrafish,UCE_PYTHON=${python_bin}" \
  slurm/run_uce_atlas_embed.sbatch)"
echo "ZSCAPE preparation=${zscape_prepare} HVG=${zscape_hvg} UCE=${zscape_uce}"

for coordinate in relative days; do
  z_config="configs/stream_zebrafish_${coordinate}.yaml"
  mouse_config="configs/stream_mouse_dev_${coordinate}.yaml"
  for n_hvg in 5000 10000; do
    z_out="outputs/zscape_stream_hvg${n_hvg}_${coordinate}"
    mouse_out="outputs/stream_mouse_hvg${n_hvg}_${coordinate}"

    z_prepare="$(submit --dependency="afterok:${zscape_hvg}" \
      --export="ALL,CONFIG=${z_config},N_HVG=${n_hvg},OUT_DIR=${z_out},STREAM_DATA_PYTHON=${python_bin}" \
      slurm/run_stream_prepare.sbatch)"
    z_cre="$(submit --dependency="afterok:${z_prepare}" \
      --export="ALL,CONFIG=${z_config},N_HVG=${n_hvg},OUT_DIR=${z_out},STREAM_PYTHON=${python_bin}" \
      slurm/run_stream_embed_cre.sbatch)"
    z_cache="$(submit --dependency="afterok:${z_prepare}:${zscape_uce}" \
      --export="ALL,CONFIG=${z_config},N_HVG=${n_hvg},OUT_DIR=${z_out},EVAL_CACHE=${z_out}/eval_batches.npz,CELL_STATE=uce,UCE_EMBEDDING_DIR=outputs/zscape_uce/embeddings,STREAM_PYTHON=${python_bin}" \
      slurm/run_stream_cache_eval.sbatch)"

    mouse_prepare="$(submit --export="ALL,CONFIG=${mouse_config},N_HVG=${n_hvg},OUT_DIR=${mouse_out},STREAM_DATA_PYTHON=${python_bin}" \
      slurm/run_stream_prepare.sbatch)"
    mouse_cre="$(submit --dependency="afterok:${mouse_prepare}" \
      --export="ALL,CONFIG=${mouse_config},N_HVG=${n_hvg},OUT_DIR=${mouse_out},STREAM_PYTHON=${python_bin}" \
      slurm/run_stream_embed_cre.sbatch)"

    common_dep="afterok:${z_prepare}:${z_cre}:${zscape_uce}:${mouse_cre}"
    for variant in film cross_attention; do
      source_name="mouse_${coordinate}_${n_hvg}_${variant}"
      source_train="$(submit --dependency="$common_dep" --job-name="$source_name" \
        --export="ALL,CONFIG=${mouse_config},VARIANT=${variant},N_HVG=${n_hvg},OUT_DIR=${mouse_out},CELL_STATE=uce,UCE_EMBEDDING_DIR=outputs/uce/embeddings,EXPERIMENT_LABEL=source,STREAM_PYTHON=${python_bin},WANDB_MODE=online,WANDB_RUN_NAME=${source_name}" \
        slurm/run_stream_train.sbatch)"
      source_ckpt="${mouse_out}/model_${variant}_uce_source.pt"

      zero_name="zfish_${coordinate}_${n_hvg}_${variant}_zero_shot"
      zero_eval="$(submit --dependency="afterok:${source_train}:${z_cache}" --job-name="$zero_name" \
        --export="ALL,CONFIG=${z_config},VARIANT=${variant},N_HVG=${n_hvg},OUT_DIR=${z_out},CELL_STATE=uce,UCE_EMBEDDING_DIR=outputs/zscape_uce/embeddings,EXPERIMENT_LABEL=zero_shot,CHECKPOINT=${source_ckpt},EVAL_CACHE=${z_out}/eval_batches.npz,STREAM_PYTHON=${python_bin}" \
        slurm/run_stream_evaluate.sbatch)"

      fine_name="zfish_${coordinate}_${n_hvg}_${variant}_fine_tuned"
      fine_train="$(submit --dependency="afterok:${source_train}:${z_cre}:${zscape_uce}" --job-name="$fine_name" \
        --export="ALL,CONFIG=${z_config},VARIANT=${variant},N_HVG=${n_hvg},OUT_DIR=${z_out},CELL_STATE=uce,UCE_EMBEDDING_DIR=outputs/zscape_uce/embeddings,EXPERIMENT_LABEL=fine_tuned,INIT_CHECKPOINT=${source_ckpt},STREAM_PYTHON=${python_bin},WANDB_MODE=online,WANDB_RUN_NAME=${fine_name}" \
        slurm/run_stream_train.sbatch)"
      scratch_name="zfish_${coordinate}_${n_hvg}_${variant}_zebrafish_only"
      scratch_train="$(submit --dependency="afterok:${z_cre}:${zscape_uce}" --job-name="$scratch_name" \
        --export="ALL,CONFIG=${z_config},VARIANT=${variant},N_HVG=${n_hvg},OUT_DIR=${z_out},CELL_STATE=uce,UCE_EMBEDDING_DIR=outputs/zscape_uce/embeddings,EXPERIMENT_LABEL=zebrafish_only,STREAM_PYTHON=${python_bin},WANDB_MODE=online,WANDB_RUN_NAME=${scratch_name}" \
        slurm/run_stream_train.sbatch)"

      for label in fine_tuned zebrafish_only; do
        train_job="$fine_train"
        [[ "$label" == "zebrafish_only" ]] && train_job="$scratch_train"
        submit --dependency="afterok:${train_job}:${z_cache}" --job-name="zfish_eval_${coordinate}_${n_hvg}_${variant}_${label}" \
          --export="ALL,CONFIG=${z_config},VARIANT=${variant},N_HVG=${n_hvg},OUT_DIR=${z_out},CELL_STATE=uce,UCE_EMBEDDING_DIR=outputs/zscape_uce/embeddings,EXPERIMENT_LABEL=${label},EVAL_CACHE=${z_out}/eval_batches.npz,STREAM_PYTHON=${python_bin}" \
          slurm/run_stream_evaluate.sbatch >/dev/null
      done
    done

    baseline_name="zfish_${coordinate}_${n_hvg}_standard_cfm_zebrafish_only"
    baseline_train="$(submit --dependency="afterok:${z_prepare}:${zscape_uce}" --job-name="$baseline_name" \
      --export="ALL,CONFIG=${z_config},VARIANT=standard_cfm,N_HVG=${n_hvg},OUT_DIR=${z_out},CELL_STATE=uce,UCE_EMBEDDING_DIR=outputs/zscape_uce/embeddings,EXPERIMENT_LABEL=zebrafish_only,STREAM_PYTHON=${python_bin},WANDB_MODE=online,WANDB_RUN_NAME=${baseline_name}" \
      slurm/run_stream_train.sbatch)"
    submit --dependency="afterok:${baseline_train}:${z_cache}" --job-name="zfish_eval_${coordinate}_${n_hvg}_standard_cfm" \
      --export="ALL,CONFIG=${z_config},VARIANT=standard_cfm,N_HVG=${n_hvg},OUT_DIR=${z_out},CELL_STATE=uce,UCE_EMBEDDING_DIR=outputs/zscape_uce/embeddings,EXPERIMENT_LABEL=zebrafish_only,EVAL_CACHE=${z_out}/eval_batches.npz,STREAM_PYTHON=${python_bin}" \
      slurm/run_stream_evaluate.sbatch >/dev/null
    echo "Submitted ${coordinate} ${n_hvg}: ZSCAPE prep=${z_prepare}, CRE=${z_cre}, source prep=${mouse_prepare}, CRE=${mouse_cre}"
  done
done
