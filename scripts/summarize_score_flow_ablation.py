#!/usr/bin/env python
"""Combine completed score-flow causal evaluations into one ablation table."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def checkpoint_training_mode(path: Path) -> str:
    """Identify checkpoint families that must not be averaged together."""
    name = path.stem
    if "_growth_growth_only_" in name:
        return "growth_only"
    if "_growth_joint_" in name:
        return "joint"
    if "_population_" in name:
        return "population"
    return "score_flow"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="outputs/stream_hvg10000")
    parser.add_argument("--pattern", default="causal_eval_model_score_flow_*_coupled_v2.csv")
    parser.add_argument("--output", default="score_flow_coupled_ablation_summary.csv")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    paths = sorted(out_dir.glob(args.pattern))
    if not paths:
        raise FileNotFoundError(f"No score-flow evaluations match {out_dir / args.pattern}")
    frame = pd.concat(
        [
            pd.read_csv(path).assign(
                source_file=path.name,
                checkpoint_training_mode=checkpoint_training_mode(path),
            )
            for path in paths
        ],
        ignore_index=True,
    )
    metrics = [
        "sinkhorn_skill",
        "endpoint_sinkhorn",
        "mean_shift_r2",
        "mean_shift_mae",
        "mean_gene_wasserstein1",
    ]
    group_columns = [
        "checkpoint_training_mode",
        "variant",
        "endpoint_denoising",
        "dynamics_mode",
        "score_control",
        "diffusion",
    ]
    if "growth_control" in frame:
        group_columns.append("growth_control")
    metrics.extend(
        column
        for column in (
            "growth_effective_sample_size",
            "growth_weight_kl",
            "growth_rate_rms",
        )
        if column in frame
    )
    summary = (
        frame.groupby(
            group_columns,
            as_index=False,
        )[metrics]
        .mean()
        .sort_values("sinkhorn_skill", ascending=False)
    )
    output = out_dir / args.output
    summary.to_csv(output, index=False)
    print(summary.to_string(index=False))
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
