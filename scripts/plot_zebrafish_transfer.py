#!/usr/bin/env python
"""Summarize cached zebrafish transfer evaluations with plotnine."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from plotnine import aes, facet_grid, geom_col, ggplot, labs, position_dodge, scale_fill_brewer, theme_bw


def collect_metrics(root: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(root.glob("zscape_stream_hvg*_*/*eval_metrics_*.csv")):
        frame = pd.read_csv(path)
        frame["artifact"] = path.stem.removeprefix("eval_metrics_")
        frame["result_dir"] = path.parent.name
        rows.append(frame)
    if not rows:
        raise FileNotFoundError(f"No zebrafish evaluation metrics found under {root}")
    combined = pd.concat(rows, ignore_index=True)
    combined["time_scale"] = combined["result_dir"].str.extract(r"_(relative|days)$", expand=False)
    combined["panel"] = combined["result_dir"].str.extract(r"hvg(\\d+)", expand=False).astype(int)
    combined["variant"] = combined["artifact"].str.extract(r"(standard_cfm|film|cross_attention)", expand=False)
    combined["training"] = combined["artifact"].str.extract(r"(zero_shot|fine_tuned|zebrafish_only)", expand=False).fillna("zebrafish_only")
    return combined


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs", default="outputs")
    parser.add_argument("--figure", default="figures/zebrafish_transfer_displacement_mae.png")
    args = parser.parse_args()
    metrics = collect_metrics(Path(args.outputs))
    summary = (
        metrics.groupby(["time_scale", "panel", "variant", "training", "eval_gene_set"], as_index=False)
        .agg(displacement_mae=("displacement_mae", "mean"), displacement_mse=("displacement_mse", "mean"), velocity_mae=("velocity_mae", "mean"))
    )
    figure = Path(args.figure)
    figure.parent.mkdir(parents=True, exist_ok=True)
    plot = (
        ggplot(summary, aes("training", "displacement_mae", fill="variant"))
        + geom_col(position=position_dodge(width=0.8))
        + facet_grid("time_scale ~ panel")
        + scale_fill_brewer(type="qual", palette="Set2")
        + labs(x="Training regime", y="Held-out expression displacement MAE", fill="Model")
        + theme_bw()
    )
    plot.save(figure, width=10, height=6, dpi=180)
    summary.to_csv(figure.with_suffix(".csv"), index=False)
    print(f"Wrote {figure}")


if __name__ == "__main__":
    main()
