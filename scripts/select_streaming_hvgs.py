#!/usr/bin/env python
"""Compute streaming gene variances for STREAM gene-panel selection."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from jax_adata_streaming import _chunk_looks_like_counts, _maybe_normalize
from stream_model.config import StreamConfig


def _gene_ids(adata) -> pd.Index:
    return pd.Index(adata.var["gene_id"] if "gene_id" in adata.var else adata.var_names).astype(str)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/stream_mouse_dev.yaml")
    parser.add_argument("--output", default="outputs/jax_adata_eda/streaming_gene_variances.csv")
    parser.add_argument("--chunk-cells", type=int, default=5000)
    parser.add_argument("--n-genes", type=int, default=0, help="Write this many top genes; 0 writes all genes.")
    parser.add_argument("--normalize-log1p", choices=["auto", "0", "1"], default="auto")
    args = parser.parse_args()

    import anndata as ad

    cfg = StreamConfig.from_yaml(args.config)
    paths = sorted(Path(cfg.adata_dir).glob("*.h5ad"))
    if not paths:
        raise SystemExit(f"No .h5ad files found in {cfg.adata_dir}")

    first = ad.read_h5ad(paths[0], backed="r")
    genes = _gene_ids(first)
    n_vars = len(genes)
    first_chunk = first.X[: min(args.chunk_cells, first.n_obs), :]
    do_normalize = _chunk_looks_like_counts(first_chunk) if args.normalize_log1p == "auto" else args.normalize_log1p == "1"
    first.file.close()

    sum_x = np.zeros(n_vars, dtype=np.float64)
    sum_x2 = np.zeros(n_vars, dtype=np.float64)
    seen = 0
    for path in paths:
        adata = ad.read_h5ad(path, backed="r")
        path_genes = _gene_ids(adata)
        if not path_genes.equals(genes):
            raise ValueError(f"{path} has a different var gene_id order than {paths[0]}")
        for start in range(0, adata.n_obs, args.chunk_cells):
            end = min(start + args.chunk_cells, adata.n_obs)
            x = _maybe_normalize(adata.X[start:end, :], do_normalize)
            sum_x += np.asarray(x.sum(axis=0)).ravel()
            sum_x2 += (
                np.asarray(x.power(2).sum(axis=0)).ravel()
                if sparse.issparse(x)
                else np.square(x).sum(axis=0)
            )
            seen += end - start
            if seen % 500_000 == 0:
                print(f"Processed {seen:,} cells", flush=True)
        adata.file.close()

    mean_x = sum_x / seen
    variance = np.maximum(sum_x2 / seen - mean_x**2, 0)
    order = np.argsort(variance)[::-1]
    if args.n_genes > 0:
        order = order[: args.n_genes]
    out = cfg.resolve_path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"gene": genes[order].to_numpy(dtype=str), "variance": variance[order]}).to_csv(out, index=False)
    print(f"Wrote {len(order):,} genes to {out}")


if __name__ == "__main__":
    main()
