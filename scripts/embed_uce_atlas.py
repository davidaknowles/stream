#!/usr/bin/env python
"""Stream 33-layer UCE embeddings for each JAX atlas AnnData file."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import torch

from stream_model.uce import UCE_MODEL_DIM, embed_uce_sentences, load_uce_gene_metadata, load_uce_model, sample_uce_sentence


def prepare_chunk(atlas, start: int, stop: int, metadata, rng: np.random.Generator, sample_size: int):
    """Read and tokenize one sparse chunk while the preceding chunk is on GPU."""

    buckets: dict[int, list[tuple[int, np.ndarray]]] = {}
    empty_cells = 0
    for local_index, row in enumerate(atlas.X[start:stop]):
        sentence = sample_uce_sentence(row, metadata, rng, sample_size=sample_size)
        absolute_index = start + local_index
        if sentence is None:
            buckets.setdefault(0, []).append((absolute_index, np.empty(0, dtype=np.int64)))
            empty_cells += 1
        else:
            buckets.setdefault(len(sentence), []).append((absolute_index, sentence))
    return buckets, empty_cells


def embed_file(args, model, path: Path, device: torch.device) -> dict[str, object]:
    output_path = args.output_dir / f"{path.stem}.npy"
    if output_path.exists() and not args.overwrite:
        existing = np.load(output_path, mmap_mode="r")
        if existing.shape[1] == UCE_MODEL_DIM:
            return {"file": path.name, "cells": int(existing.shape[0]), "status": "reused", "output": str(output_path)}
        raise ValueError(f"Unexpected existing embedding dimension in {output_path}")

    atlas = ad.read_h5ad(path, backed="r")
    metadata = load_uce_gene_metadata(
        atlas.var,
        args.mouse_protein_embeddings,
        args.species_chrom,
        args.species_offsets,
        gene_symbol_column=args.gene_symbol_column,
    )
    n_cells = atlas.n_obs
    output = np.lib.format.open_memmap(output_path, mode="w+", dtype=np.float16, shape=(n_cells, UCE_MODEL_DIM))
    rng = np.random.default_rng(args.seed)
    empty_cells = 0
    chunk_starts = list(range(0, n_cells, args.row_chunk_size))
    with ThreadPoolExecutor(max_workers=1) as executor:
        next_start = chunk_starts[0]
        next_stop = min(next_start + args.row_chunk_size, n_cells)
        future = executor.submit(prepare_chunk, atlas, next_start, next_stop, metadata, rng, args.sample_size)
        for chunk_index, start in enumerate(chunk_starts):
            stop = min(start + args.row_chunk_size, n_cells)
            buckets, chunk_empty = future.result()
            empty_cells += chunk_empty
            if chunk_index + 1 < len(chunk_starts):
                next_start = chunk_starts[chunk_index + 1]
                next_stop = min(next_start + args.row_chunk_size, n_cells)
                future = executor.submit(prepare_chunk, atlas, next_start, next_stop, metadata, rng, args.sample_size)
            for length, entries in buckets.items():
                if length == 0:
                    output[np.asarray([index for index, _ in entries])] = 0
                    continue
                for batch_start in range(0, len(entries), args.batch_size):
                    batch = entries[batch_start : batch_start + args.batch_size]
                    indices, sentences = zip(*batch, strict=True)
                    output[np.asarray(indices)] = embed_uce_sentences(model, np.stack(sentences), device).astype(np.float16)
            output.flush()
            print(f"{path.name}: {stop:,}/{n_cells:,} cells", flush=True)
    atlas.file.close()
    return {
        "file": path.name,
        "cells": int(n_cells),
        "status": "embedded",
        "output": str(output_path),
        "mapped_genes": int(metadata.valid.sum()),
        "empty_cells": int(empty_cells),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adata-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--uce-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--mouse-protein-embeddings", type=Path, required=True)
    parser.add_argument("--species-chrom", type=Path, required=True)
    parser.add_argument("--species-offsets", type=Path, required=True)
    parser.add_argument("--gene-symbol-column", default="gene_short_name")
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--row-chunk-size", type=int, default=4096)
    parser.add_argument("--sample-size", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("UCE atlas embedding requires CUDA")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    torch.backends.cuda.enable_flash_sdp(True)
    model = load_uce_model(args.uce_dir, args.checkpoint, device)
    rows = [embed_file(args, model, path, device) for path in sorted(args.adata_dir.glob("*.h5ad"))]
    pd.DataFrame(rows).to_csv(args.output_dir / "manifest.csv", index=False)
    (args.output_dir / "config.json").write_text(json.dumps(vars(args), default=str, indent=2) + "\n")
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
