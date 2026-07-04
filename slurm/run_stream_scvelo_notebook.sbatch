#!/bin/bash
#SBATCH --job-name=stream_scvelo
#SBATCH --partition=gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=logs/stream_scvelo_%j.out
#SBATCH --error=logs/stream_scvelo_%j.err

set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-/gpfs/commons/home/daknowles/projects/stream}"
cd "$PROJECT_ROOT"
mkdir -p logs outputs/stream/scvelo_stream
module purge
module load Python/3.11.5-GCCcore-13.2.0
export PROJECT_ROOT="$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT/src:$PROJECT_ROOT/.venv-jax-adata-eda/lib/python3.11/site-packages:$HOME/.local/lib/python3.11/site-packages:${PYTHONPATH:-}"
export STREAM_SCVELO_CELLS="${STREAM_SCVELO_CELLS:-2500}"
export STREAM_SCVELO_BATCH="${STREAM_SCVELO_BATCH:-32}"
.venv-jax-adata-eda/bin/jupyter nbconvert \
  --to notebook \
  --execute notebooks/stream_scvelo_velocity_stream.ipynb \
  --output stream_scvelo_velocity_stream_executed.ipynb \
  --output-dir notebooks
