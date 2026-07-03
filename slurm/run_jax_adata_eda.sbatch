#!/usr/bin/env bash
#SBATCH --job-name=jax-adata-eda
#SBATCH --partition=bigmem
#SBATCH --cpus-per-task=16
#SBATCH --mem=512G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/jax_adata_eda_%j.out
#SBATCH --error=logs/jax_adata_eda_%j.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/gpfs/commons/home/daknowles/projects/stream}"
cd "$PROJECT_ROOT"

module load Python/3.11.5-GCCcore-13.2.0
source .venv-jax-adata-eda/bin/activate

export PROJECT_ROOT="$PROJECT_ROOT"
export ADATA_DIR="$PROJECT_ROOT/downloads/adata"
export EDA_OUT_DIR="$PROJECT_ROOT/outputs/jax_adata_eda"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"

mkdir -p logs outputs/jax_adata_eda

jupyter nbconvert \
  --to notebook \
  --execute notebooks/jax_adata_eda.ipynb \
  --output jax_adata_eda_executed.ipynb \
  --output-dir notebooks \
  --ExecutePreprocessor.timeout=-1
