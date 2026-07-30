#!/usr/bin/env python
"""Create reproducible train/held-out gene panels for dynamics prediction tests."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selected-genes", required=True, help="CSV with at least gene_id; usually OUT_DIR/selected_genes.csv")
    parser.add_argument("--out-prefix", required=True, help="Output prefix for *_train.csv and *_heldout.csv")
    parser.add_argument("--heldout-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    if not 0 < args.heldout_fraction < 1:
        raise ValueError("--heldout-fraction must be between 0 and 1")
    selected = pd.read_csv(args.selected_genes)
    if "gene_id" not in selected.columns:
        raise ValueError("--selected-genes must contain a gene_id column")
    selected = selected.drop_duplicates("gene_id").reset_index(drop=True)
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(selected))
    n_heldout = max(1, int(round(len(selected) * args.heldout_fraction)))
    heldout_idx = np.sort(order[:n_heldout])
    is_heldout = np.zeros(len(selected), dtype=bool)
    is_heldout[heldout_idx] = True

    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    train = selected.loc[~is_heldout].copy()
    heldout = selected.loc[is_heldout].copy()
    train.to_csv(out_prefix.with_name(out_prefix.name + "_train.csv"), index=False)
    heldout.to_csv(out_prefix.with_name(out_prefix.name + "_heldout.csv"), index=False)
    pd.DataFrame(
        [
            {
                "selected_genes": str(args.selected_genes),
                "n_total": len(selected),
                "n_train": len(train),
                "n_heldout": len(heldout),
                "heldout_fraction": args.heldout_fraction,
                "seed": args.seed,
            }
        ]
    ).to_csv(out_prefix.with_name(out_prefix.name + "_summary.csv"), index=False)
    print(f"Wrote {len(train)} train and {len(heldout)} held-out genes with prefix {out_prefix}")


if __name__ == "__main__":
    main()
