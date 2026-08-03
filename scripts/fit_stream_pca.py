#!/usr/bin/env python
"""Fit a train-only PCA artifact for STREAM coupling and endpoint smoothing."""

from __future__ import annotations

import argparse
import hashlib
import json

import pandas as pd

from stream_model.config import StreamConfig, apply_config_overrides
from stream_model.data import H5adIntervalSampler
from stream_model.denoise import fit_pca_denoiser


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/stream_mouse_dev.yaml")
    parser.add_argument("--timepoint-split", default=None)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--n-hvg", type=int, default=10_000)
    parser.add_argument("--output", required=True)
    parser.add_argument("--components", type=int, default=100)
    parser.add_argument("--fit-cells", type=int, default=10_000)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    args = parser.parse_args()

    cfg = StreamConfig.from_yaml(args.config)
    apply_config_overrides(cfg, out_dir=args.out_dir, n_hvg=args.n_hvg)
    selected = pd.read_csv(cfg.out_dir / "selected_genes.csv")
    gene_ids = selected["gene_id"].astype(str).tolist()
    split_path = cfg.resolve_path(args.timepoint_split) if args.timepoint_split else cfg.out_dir / "timepoint_split.json"
    with split_path.open() as handle:
        split = json.load(handle)
    intervals = [tuple(interval) for interval in split["train_intervals"]]
    cells = pd.read_csv(cfg.cell_metadata_csv, index_col=0)
    full_sampler = H5adIntervalSampler.from_adata_dir(
        cfg.adata_dir,
        cells,
        gene_ids,
        intervals,
        batch_size=cfg.batch_size,
        seed=cfg.seed,
        time_coordinates=split.get("time_coordinates"),
    )
    train_sampler, _validation_sampler = full_sampler.split_validation(args.validation_fraction, cfg.seed + 17)
    train_days = sorted({day for interval in intervals for day in interval}, key=split["all_days"].index)
    pca = fit_pca_denoiser(
        train_sampler,
        train_days,
        total_cells=args.fit_cells,
        n_components=args.components,
        seed=cfg.seed,
    )
    output = cfg.resolve_path(args.output)
    pca.save(
        output,
        metadata={
            "n_components": args.components,
            "fit_cells": args.fit_cells,
            "validation_fraction": args.validation_fraction,
            "split_seed": cfg.seed + 17,
            "gene_ids_sha256": hashlib.sha256("\n".join(gene_ids).encode()).hexdigest(),
            "n_genes": len(gene_ids),
            "timepoint_split": str(split_path),
        },
    )
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
