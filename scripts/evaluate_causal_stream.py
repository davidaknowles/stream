#!/usr/bin/env python
"""Roll an online-UCE STREAM field into held-out mouse stages and score endpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA

from stream_model.config import StreamConfig, apply_config_overrides
from stream_model.data import H5adIntervalSampler, heldout_block_forecast_intervals
from stream_model.denoise import PCADenoiser
from stream_model.rollout import mean_shift_metrics, projected_euler_rollout, sinkhorn_divergence
from stream_model.train import (
    artifact_stem,
    build_model,
    load_cre_npz,
    predict_stream_chunked,
    uce_input_expression,
)
from stream_model.uce import build_online_uce_encoder


def _load_gene_sets(gene_ids: list[str], specs: list[str], cfg: StreamConfig):
    gene_index = pd.Index(gene_ids)
    result: dict[str, np.ndarray | None] = {"full": None}
    for spec in specs:
        name, raw_path = spec.split(":", 1)
        reference = pd.read_csv(cfg.resolve_path(raw_path))
        indices = gene_index.get_indexer(reference["gene_id"].astype(str).drop_duplicates())
        if (indices < 0).any():
            raise ValueError(f"{raw_path} includes genes outside the modeled panel")
        result[name] = indices
    return result


def _fit_pca(sampler, train_days: list[str], total_cells: int, seed: int):
    per_day = int(np.ceil(total_cells / len(train_days)))
    old_size = sampler.batch_size
    sampler.batch_size = per_day
    try:
        samples = [np.asarray(sampler.sample_day(day), dtype=np.float32) for day in train_days]
    finally:
        sampler.batch_size = old_size
    matrix = np.vstack(samples)[:total_cells]
    pca = PCA(n_components=50, whiten=True, svd_solver="randomized", random_state=seed)
    pca.fit(np.log1p(matrix))
    reference = pca.transform(np.log1p(matrix[: min(2048, len(matrix))]))
    distance = np.square(reference[:1024, None, :] - reference[None, :1024, :]).sum(axis=2) / 50
    epsilon = 0.05 * float(np.median(distance[distance > 0]))
    return pca, epsilon


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/stream_mouse_dev.yaml")
    parser.add_argument("--timepoint-split", default=None)
    parser.add_argument("--variant", choices=["film", "cross_attention"], default="cross_attention")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--n-hvg", type=int, default=10000)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--experiment-label", required=True)
    parser.add_argument("--eval-gene-subset", action="append", default=[])
    parser.add_argument("--cells-per-interval", type=int, default=128)
    parser.add_argument("--integration-steps", default="4,8,16")
    parser.add_argument("--pca-cells", type=int, default=10000)
    parser.add_argument("--gene-chunk-size", type=int, default=256)
    parser.add_argument("--cre-token-arrays", default=None)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    cfg = StreamConfig.from_yaml(args.config)
    apply_config_overrides(
        cfg,
        out_dir=args.out_dir,
        n_hvg=args.n_hvg,
        cell_state="uce",
        uce_mode="online",
        experiment_label=args.experiment_label,
    )
    cfg.model_variant = args.variant
    cfg.batch_size = args.cells_per_interval
    cfg.gene_chunk_size = args.gene_chunk_size
    device = torch.device(args.device)
    selected = pd.read_csv(cfg.out_dir / "selected_genes.csv")
    gene_ids = selected["gene_id"].astype(str).tolist()
    gene_sets = _load_gene_sets(gene_ids, args.eval_gene_subset, cfg)
    split_path = cfg.resolve_path(args.timepoint_split) if args.timepoint_split else cfg.out_dir / "timepoint_split.json"
    with split_path.open() as handle:
        split = json.load(handle)
    heldout = set(split["heldout_days"])
    intervals = heldout_block_forecast_intervals(split["all_days"], heldout)
    if not intervals:
        raise ValueError("No observed-to-held-out intervals were found")
    cells = pd.read_csv(cfg.cell_metadata_csv, index_col=0)
    sampler = H5adIntervalSampler.from_adata_dir(
        cfg.adata_dir,
        cells,
        gene_ids,
        intervals,
        batch_size=cfg.batch_size,
        seed=cfg.seed + 1,
        time_coordinates=split.get("time_coordinates"),
    )

    cre_token_path = cfg.resolve_path(args.cre_token_arrays) if args.cre_token_arrays else cfg.out_dir / "cre_token_arrays.npz"
    cre_inputs = load_cre_npz(cre_token_path, device)
    model = build_model(cfg, len(gene_ids), int(cre_inputs["cre_embeddings"].shape[-1])).to(device)
    checkpoint_path = cfg.resolve_path(args.checkpoint) if args.checkpoint else cfg.out_dir / f"model_{artifact_stem(cfg)}.pt"
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if checkpoint.get("model_contract") != "online_uce_autonomous_v1":
        raise ValueError("Causal rollout requires an online_uce_autonomous_v1 checkpoint")
    if checkpoint.get("gene_ids") != gene_ids:
        raise ValueError("Checkpoint gene panel does not match selected_genes.csv")
    if Path(checkpoint.get("cre_token_arrays", cre_token_path)).resolve() != cre_token_path.resolve():
        raise ValueError("Checkpoint and evaluation CRE token arrays do not match")
    checkpoint_split = checkpoint.get("validation_config", {}).get("timepoint_split")
    if checkpoint_split and cfg.resolve_path(checkpoint_split).resolve() != split_path.resolve():
        raise ValueError("Checkpoint and evaluation timepoint splits do not match")
    checkpoint_config = checkpoint.get("config", {})
    cfg.uce_expression_preprocessing = checkpoint_config.get("uce_expression_preprocessing", "raw")
    # Checkpoints predating systematic resampling retain their original multinomial UCE contract.
    cfg.uce_sampling = checkpoint_config.get("uce_sampling", "multinomial")
    uce_pca = None
    if cfg.uce_expression_preprocessing == "pca":
        pca_artifact = checkpoint.get("pca_artifact")
        if not pca_artifact:
            raise ValueError("Checkpoint requests PCA preprocessing before UCE but has no PCA artifact")
        uce_pca = PCADenoiser.load(cfg.resolve_path(pca_artifact))
    model.load_state_dict(checkpoint["model"])
    model.eval()
    encoder = build_online_uce_encoder(cfg, selected, device)

    train_days = [day for day in split["all_days"] if day not in heldout]
    pca, sinkhorn_epsilon = _fit_pca(sampler, train_days, args.pca_cells, cfg.seed)
    pca_path = cfg.out_dir / f"causal_pca_{artifact_stem(cfg)}.npz"
    np.savez_compressed(
        pca_path,
        components=pca.components_,
        mean=pca.mean_,
        explained_variance=pca.explained_variance_,
        sinkhorn_epsilon=sinkhorn_epsilon,
    )

    rows = []
    prediction_payload = {}
    for interval_index, (day0, day1) in enumerate(intervals):
        batch = sampler.sample_interval(day0, day1)
        x0 = torch.as_tensor(batch.x0, device=device)
        observed_x1 = np.asarray(batch.x1, dtype=np.float32)
        source = np.asarray(batch.x0, dtype=np.float32)
        observed_pca = pca.transform(np.log1p(observed_x1))
        source_pca = pca.transform(np.log1p(source))
        persistence_sinkhorn = sinkhorn_divergence(source_pca, observed_pca, sinkhorn_epsilon)

        def velocity_fn(current_x, seed):
            state = encoder.encode(uce_input_expression(cfg, current_x, uce_pca), seed)
            return predict_stream_chunked(model, state, cre_inputs, cfg.gene_chunk_size)

        for steps in [int(value) for value in args.integration_steps.split(",")]:
            seed = cfg.seed + interval_index * 100_000
            predicted = projected_euler_rollout(x0, batch.t0, batch.t1, velocity_fn, steps, seed)
            predicted_np = predicted.cpu().numpy()
            predicted_pca = pca.transform(np.log1p(predicted_np))
            endpoint_sinkhorn = sinkhorn_divergence(predicted_pca, observed_pca, sinkhorn_epsilon)
            for name, indices in gene_sets.items():
                row = {
                    "day0": day0,
                    "day1": day1,
                    "integration_steps": steps,
                    "eval_gene_set": name,
                    "n_eval_genes": len(gene_ids) if indices is None else len(indices),
                    "uce_expression_preprocessing": cfg.uce_expression_preprocessing,
                    "uce_rollout_expression_preprocessing": (
                        "pca" if cfg.uce_expression_preprocessing == "pca" else "raw"
                    ),
                    **mean_shift_metrics(source, observed_x1, predicted_np, indices),
                    "endpoint_sinkhorn": endpoint_sinkhorn if indices is None else np.nan,
                    "persistence_sinkhorn": persistence_sinkhorn if indices is None else np.nan,
                    "sinkhorn_skill": (
                        1.0 - endpoint_sinkhorn / persistence_sinkhorn
                        if indices is None and persistence_sinkhorn > 0
                        else np.nan
                    ),
                }
                rows.append(row)
            prediction_payload[f"{day0}_{day1}_steps{steps}"] = predicted_np
        prediction_payload[f"{day0}_{day1}_source"] = source
        prediction_payload[f"{day0}_{day1}_observed"] = observed_x1

    metrics_path = cfg.out_dir / f"causal_eval_metrics_{artifact_stem(cfg)}.csv"
    predictions_path = cfg.out_dir / f"causal_eval_predictions_{artifact_stem(cfg)}.npz"
    pd.DataFrame(rows).to_csv(metrics_path, index=False)
    np.savez_compressed(predictions_path, **prediction_payload)
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"Wrote {metrics_path}")
    print(f"Wrote {predictions_path}")


if __name__ == "__main__":
    main()
