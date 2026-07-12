#!/usr/bin/env python
"""Materialize deterministic held-out endpoint batches shared by model evaluations."""

from __future__ import annotations

import argparse
import json

import pandas as pd

from stream_model.config import StreamConfig, apply_config_overrides
from stream_model.data import H5adIntervalSampler
from stream_model.evaluate import _load_or_create_batches


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--n-hvg", type=int, required=True)
    parser.add_argument("--batches", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--cell-state", choices=["expression", "uce"], default=None)
    parser.add_argument("--uce-embedding-dir", default=None)
    args = parser.parse_args()
    cfg = StreamConfig.from_yaml(args.config)
    apply_config_overrides(cfg, out_dir=args.out_dir, n_hvg=args.n_hvg, cell_state=args.cell_state, uce_embedding_dir=args.uce_embedding_dir)
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    selected = pd.read_csv(cfg.out_dir / "selected_genes.csv")
    with (cfg.out_dir / "timepoint_split.json").open() as handle:
        split = json.load(handle)
    cells = pd.read_csv(cfg.cell_metadata_csv, index_col=0)
    sampler = H5adIntervalSampler.from_adata_dir(
        cfg.adata_dir,
        cells,
        selected["gene_id"].tolist(),
        [tuple(interval) for interval in split["heldout_touching_intervals"] or split["train_intervals"]],
        batch_size=cfg.batch_size,
        seed=cfg.seed + 1,
        state_embeddings_dir=cfg.uce_embedding_dir if cfg.cell_state == "uce" else None,
        state_dim=cfg.uce_embedding_dim if cfg.cell_state == "uce" else None,
        time_coordinates=split.get("time_coordinates"),
    )
    _load_or_create_batches(sampler, args.batches, cfg.resolve_path(args.cache))
    print(f"Wrote evaluation cache to {cfg.resolve_path(args.cache)}")


if __name__ == "__main__":
    main()
