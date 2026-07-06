#!/usr/bin/env python
"""Prepare selected genes and CRE-gene links for STREAM."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from stream_model.config import StreamConfig, apply_config_overrides
from stream_model.data import adjacent_intervals, load_selected_genes, ordered_days
from stream_model.genome import link_cres_to_genes, parse_gtf_tss, read_ccre_bed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/stream_mouse_dev.yaml")
    parser.add_argument("--hvg-csv", default=None)
    parser.add_argument("--n-hvg", type=int, default=None)
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()
    cfg = StreamConfig.from_yaml(args.config)
    apply_config_overrides(cfg, hvg_csv=args.hvg_csv, n_hvg=args.n_hvg, out_dir=args.out_dir)
    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    selected_raw = load_selected_genes(cfg.gene_metadata_csv, cfg.hvg_csv, cfg.n_hvg + 1000)

    tss = parse_gtf_tss(cfg.gtf, gene_ids=selected_raw["gene_id"], gene_type="protein_coding")
    tss = selected_raw[["gene_id", "gene_short_name", "gene_type", "chr", "variance"]].merge(
        tss, on="gene_id", how="inner", suffixes=("", "_gtf")
    )
    tss = tss.sort_values("variance", ascending=False).head(cfg.n_hvg).copy()
    tss["gene_name"] = tss["gene_short_name"].fillna(tss["gene_name"])
    selected = tss[["gene_id", "gene_short_name", "gene_type", "chr", "variance"]].copy()
    selected.to_csv(cfg.out_dir / "selected_genes.csv", index=False)
    tss.to_csv(cfg.out_dir / "gene_tss.csv", index=False)

    cres = read_ccre_bed(cfg.ccre_bed)
    links = link_cres_to_genes(
        tss=tss[["gene_id", "gene_name", "chrom", "strand", "tss0"]],
        cres=cres,
        window_bp=cfg.cre_window_bp,
        promoter_window_bp=cfg.promoter_window_bp,
        synthetic_promoter_bp=cfg.synthetic_promoter_bp,
    )
    links = links[links["token_rank"] < cfg.max_cres_per_gene].copy()
    links.to_csv(cfg.out_dir / "cre_gene_links.csv", index=False)

    cells = pd.read_csv(cfg.cell_metadata_csv, index_col=0)
    days = ordered_days(cells["day"].dropna().astype(str).to_numpy())
    heldout = set(str(day) for day in cfg.heldout_days)
    split = {
        "all_days": days,
        "heldout_days": sorted(heldout, key=lambda x: days.index(x) if x in days else len(days)),
        "train_intervals": adjacent_intervals(days, heldout),
        "heldout_touching_intervals": [
            (a, b) for a, b in zip(days[:-1], days[1:], strict=True) if a in heldout or b in heldout
        ],
    }
    with (cfg.out_dir / "timepoint_split.json").open("w") as handle:
        json.dump(split, handle, indent=2)

    np.save(cfg.out_dir / "selected_gene_ids.npy", selected["gene_id"].to_numpy(dtype=str))
    print(f"Wrote STREAM prep outputs to {cfg.out_dir}")
    print(f"Selected genes: {len(selected):,}")
    print(f"CRE-gene tokens: {len(links):,}")


if __name__ == "__main__":
    main()
