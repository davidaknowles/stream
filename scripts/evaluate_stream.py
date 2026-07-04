#!/usr/bin/env python
"""Evaluate trained STREAM/CFM variants on held-out timepoint intervals."""

from __future__ import annotations

import argparse
import json

import pandas as pd
import torch

from stream_model.config import StreamConfig
from stream_model.data import H5adIntervalSampler
from stream_model.evaluate import evaluate_intervals
from stream_model.train import build_model, load_cre_npz


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/stream_mouse_dev.yaml")
    parser.add_argument("--variant", choices=["standard_cfm", "film", "cross_attention"], required=True)
    parser.add_argument("--batches", type=int, default=20)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    cfg = StreamConfig.from_yaml(args.config)
    cfg.model_variant = args.variant
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
    metrics = evaluate_intervals(cfg, sampler, model, cre_inputs=cre_inputs, n_batches=args.batches)
    out = cfg.out_dir / f"eval_metrics_{cfg.model_variant}.csv"
    metrics.to_csv(out, index=False)
    print(metrics.groupby(["day0", "day1"]).mean(numeric_only=True))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
