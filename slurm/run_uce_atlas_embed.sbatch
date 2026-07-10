#!/bin/bash
#SBATCH --job-name=uce_atlas_embed
#SBATCH --partition=gpu
#SBATCH --gres=gpu:b6k:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=30:00:00
#SBATCH --output=logs/uce_atlas_embed_%j.out
#SBATCH --error=logs/uce_atlas_embed_%j.err

set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-/gpfs/commons/home/daknowles/projects/stream}"
UCE_PYTHON="${UCE_PYTHON:-$HOME/venv/torch/bin/python}"
UCE_DIR="${UCE_DIR:-$PROJECT_ROOT/outputs/uce/vendor/UCE}"
ASSETS_DIR="${ASSETS_DIR:-$PROJECT_ROOT/outputs/uce/assets}"
OUT_DIR="${OUT_DIR:-$PROJECT_ROOT/outputs/uce/embeddings}"
BATCH_SIZE="${BATCH_SIZE:-50}"
ROW_CHUNK_SIZE="${ROW_CHUNK_SIZE:-4096}"

cd "$PROJECT_ROOT"
mkdir -p logs "$ASSETS_DIR" "$OUT_DIR"
if [[ ! -d "$UCE_DIR/.git" ]]; then
  mkdir -p "$(dirname "$UCE_DIR")"
  git clone --depth 1 git@github.com:snap-stanford/UCE.git "$UCE_DIR"
fi
checkpoint="$ASSETS_DIR/uce_33layer.torch"
species_chrom="$ASSETS_DIR/species_chrom.csv"
species_offsets="$ASSETS_DIR/species_offsets.pkl"
protein_tar="$ASSETS_DIR/protein_embeddings.tar.gz"
if [[ ! -s "$checkpoint" ]]; then
  curl -L --fail --retry 5 --retry-delay 10 -o "$checkpoint" \
    'https://huggingface.co/minwoosun/uce-650m/resolve/main/pytorch_model.bin?download=true'
fi
if [[ ! -s "$species_chrom" ]]; then curl -L --fail --retry 5 -o "$species_chrom" https://ndownloader.figshare.com/files/42706558; fi
if [[ ! -s "$species_offsets" ]]; then curl -L --fail --retry 5 -o "$species_offsets" https://ndownloader.figshare.com/files/42706555; fi
if [[ ! -s "$protein_tar" ]]; then curl -L --fail --retry 5 -o "$protein_tar" https://ndownloader.figshare.com/files/42715213; fi
mouse_protein="$(find "$ASSETS_DIR" -name 'Mus_musculus.GRCm39.gene_symbol_to_embedding_ESM2.pt' -print -quit)"
if [[ -z "$mouse_protein" ]]; then
  tar -xzf "$protein_tar" -C "$ASSETS_DIR"
  mouse_protein="$(find "$ASSETS_DIR" -name 'Mus_musculus.GRCm39.gene_symbol_to_embedding_ESM2.pt' -print -quit)"
fi
if [[ -z "$mouse_protein" ]]; then
  echo 'Mouse UCE protein embeddings were not found after extraction.' >&2
  exit 1
fi
export PYTHONPATH="$PROJECT_ROOT/src:${PYTHONPATH:-}"
"$UCE_PYTHON" scripts/embed_uce_atlas.py \
  --adata-dir downloads/adata \
  --output-dir "$OUT_DIR" \
  --uce-dir "$UCE_DIR" \
  --checkpoint "$checkpoint" \
  --mouse-protein-embeddings "$mouse_protein" \
  --species-chrom "$species_chrom" \
  --species-offsets "$species_offsets" \
  --batch-size "$BATCH_SIZE" \
  --row-chunk-size "$ROW_CHUNK_SIZE"
