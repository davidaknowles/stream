#!/bin/bash
#SBATCH --job-name=stream_prepare
#SBATCH --partition=bigmem
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/stream_prepare_%j.out
#SBATCH --error=logs/stream_prepare_%j.err

set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-/gpfs/commons/home/daknowles/projects/stream}"
cd "$PROJECT_ROOT"
mkdir -p logs
module purge
module load Python/3.11.5-GCCcore-13.2.0
export PYTHONPATH="$PROJECT_ROOT/src:${PYTHONPATH:-}"
python scripts/prepare_stream_data.py --config configs/stream_mouse_dev.yaml
