#!/usr/bin/env python
"""Evaluate trained STREAM/CFM variants on held-out timepoint intervals."""

from __future__ import annotations

import argparse
import json

import pandas as pd
import torch

from stream_model.config import StreamConfig, apply_config_overrides
from stream_model.data import H5adIntervalSampler
from stream_model.evaluate import evaluate_intervals
from stream_model.train import build_model, load_cre_npz


def _load_eval_gene_sets(gene_ids: list[str], specs: list[str], cfg: StreamConfig) -> dict[str, list[int] | None]:
    gene_index = pd.Index(gene_ids)
    eval_gene_sets: dict[str, list[int] | None] = {"full": None}
    for spec in specs:
        if ":" not in spec:
            raise ValueError(f"Expected --eval-gene-subset NAME:CSV, got {spec!r}")
        name, raw_path = spec.split(":", 1)
        path = cfg.resolve_path(raw_path)
        reference = pd.read_csv(path)
        if "gene_id" not in reference.columns:
            raise ValueError(f"{path} must contain a gene_id column")
        ref_ids = reference["gene_id"].astype(str).drop_duplicates().tolist()
        indices = gene_index.get_indexer(ref_ids)
        if (indices < 0).any():
            missing = pd.Index(ref_ids)[indices < 0][:10].tolist()
            raise ValueError(f"{path} contains genes absent from this model panel: {missing}")
        eval_gene_sets[name] = indices.tolist()
    return eval_gene_sets


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/stream_mouse_dev.yaml")
    parser.add_argument("--variant", choices=["standard_cfm", "film", "cross_attention"], required=True)
    parser.add_argument("--hvg-csv", default=None)
    parser.add_argument("--n-hvg", type=int, default=None)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument(
        "--eval-gene-subset",
        action="append",
        default=[],
        help="Additional metric panel as NAME:CSV with a gene_id column.",
    )
    parser.add_argument("--batches", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--gene-chunk-size", type=int, default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    cfg = StreamConfig.from_yaml(args.config)
    apply_config_overrides(cfg, hvg_csv=args.hvg_csv, n_hvg=args.n_hvg, out_dir=args.out_dir)
    cfg.model_variant = args.variant
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.gene_chunk_size is not None:
        cfg.gene_chunk_size = args.gene_chunk_size
    device = torch.device(args.device or cfg.device if torch.cuda.is_available() else "cpu")
    selected = pd.read_csv(cfg.out_dir / "selected_genes.csv")
    gene_ids = selected["gene_id"].tolist()
    with (cfg.out_dir / "timepoint_split.json").open() as handle:
        split = json.load(handle)
    eval_intervals = split["heldout_touching_intervals"] or split["train_intervals"]
    cells = pd.read_csv(cfg.cell_metadata_csv, index_col=0)
    sampler = H5adIntervalSampler.from_adata_dir(
        cfg.adata_dir,
        cells,
        gene_ids,
        [tuple(interval) for interval in eval_intervals],
        batch_size=cfg.batch_size,
        seed=cfg.seed + 1,
    )

    cre_inputs = None
    cre_dim = None
    if cfg.model_variant != "standard_cfm":
        cre_inputs = load_cre_npz(cfg.out_dir / "cre_token_arrays.npz", device)
        cre_dim = int(cre_inputs["cre_embeddings"].shape[-1])
    model = build_model(cfg, n_genes=len(gene_ids), cre_dim=cre_dim).to(device)
    ckpt = torch.load(cfg.out_dir / f"model_{cfg.model_variant}.pt", map_location=device)
    model.load_state_dict(ckpt["model"])
    eval_gene_sets = _load_eval_gene_sets(gene_ids, args.eval_gene_subset, cfg)
    metrics = evaluate_intervals(
        cfg,
        sampler,
        model,
        cre_inputs=cre_inputs,
        n_batches=args.batches,
        eval_gene_sets=eval_gene_sets,
    )
    out = cfg.out_dir / f"eval_metrics_{cfg.model_variant}.csv"
    metrics.to_csv(out, index=False)
    print(metrics.groupby(["eval_gene_set", "day0", "day1"]).mean(numeric_only=True))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
