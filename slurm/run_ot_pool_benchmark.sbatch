#!/bin/bash
#SBATCH --job-name=stream_ot_pool_bench
#SBATCH --partition=gpu
#SBATCH --gres=gpu:b6k:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=logs/stream_ot_pool_bench_%j.out
#SBATCH --error=logs/stream_ot_pool_bench_%j.err

set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-/gpfs/commons/home/daknowles/projects/stream}"
STREAM_PYTHON="${STREAM_PYTHON:-$HOME/venv/torchfix/bin/python}"
POOL_SIZES="${POOL_SIZES:-1024;2048;4096;8192}"
cd "$PROJECT_ROOT"
mkdir -p logs
export PYTHONPATH="$PROJECT_ROOT/src:${PYTHONPATH:-}"
pool_sizes_arg="${POOL_SIZES//;/,}"
"$STREAM_PYTHON" scripts/benchmark_ot_pool_size.py --pool-sizes "$pool_sizes_arg"
