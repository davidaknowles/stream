#!/bin/bash
#SBATCH --job-name=zscape_prepare
#SBATCH --partition=bigmem
#SBATCH --cpus-per-task=12
#SBATCH --mem=512G
#SBATCH --time=24:00:00
#SBATCH --output=logs/zscape_prepare_%j.out
#SBATCH --error=logs/zscape_prepare_%j.err

set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-/gpfs/commons/home/daknowles/projects/stream}"
ZSCAPE_PYTHON="${ZSCAPE_PYTHON:-$HOME/venv/torchfix/bin/python}"
SHARD_CELLS="${SHARD_CELLS:-100000}"
cd "$PROJECT_ROOT"
mkdir -p logs
export PYTHONPATH="$PROJECT_ROOT/src:${PYTHONPATH:-}"
"$ZSCAPE_PYTHON" scripts/prepare_zscape_data.py --shard-cells "$SHARD_CELLS"
