#!/usr/bin/env python
"""Evaluate subtype composition and within-subtype gene dynamics."""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
import torch

from stream_model.config import StreamConfig, apply_config_overrides
from stream_model.coordinates import coordinates_from_checkpoint
from stream_model.data import H5adIntervalSampler, heldout_block_forecast_intervals
from stream_model.denoise import PCADenoiser
from stream_model.models import GrowthRateHead, ScoreFlowStreamModel
from stream_model.population_finetune import differentiable_growth_rollout
from stream_model.score_flow import predict_score_flow_chunked, score_flow_rollout
from stream_model.subtype import (
    SubtypeCentroidClassifier,
    SubtypeStatistics,
    subtype_dynamics_metrics,
)
from stream_model.train import load_cre_npz
from stream_model.uce import build_online_uce_encoder


def load_gene_sets(gene_ids: list[str], specs: list[str], cfg: StreamConfig):
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/stream_mouse_dev.yaml")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--pca-artifact", required=True)
    parser.add_argument("--subtype-centroids", required=True)
    parser.add_argument("--eval-gene-subset", action="append", default=[])
    parser.add_argument("--cells-per-batch", type=int, default=128)
    parser.add_argument("--batches-per-interval", type=int, default=32)
    parser.add_argument("--minimum-subtype-cells", type=float, default=10.0)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--growth-control", choices=["none", "learned"], default="none")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    checkpoint_path = StreamConfig().resolve_path(args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    supported_contracts = {
        "online_uce_gene_scaled_population_score_flow_v4",
        "online_uce_gene_scaled_growth_population_score_flow_v5",
    }
    if checkpoint.get("model_contract") not in supported_contracts:
        raise ValueError(f"Expected one of {sorted(supported_contracts)}")
    saved = checkpoint["config"]
    cfg = StreamConfig.from_yaml(args.config)
    apply_config_overrides(
        cfg,
        out_dir=args.out_dir,
        n_hvg=len(checkpoint["gene_ids"]),
        cell_state="uce",
        uce_mode="online",
    )
    for name in (
        "model_variant", "d_model", "n_heads", "n_layers", "dropout", "n_context_tokens",
        "positional_encoding", "score_flow_time_dim", "score_flow_noise_scale", "uce_sampling",
    ):
        setattr(cfg, name, saved[name])
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    selected = pd.read_csv(cfg.out_dir / "selected_genes.csv")
    gene_ids = selected["gene_id"].astype(str).tolist()
    if gene_ids != checkpoint["gene_ids"]:
        raise ValueError("Checkpoint gene panel does not match selected_genes.csv")
    gene_sets = load_gene_sets(gene_ids, args.eval_gene_subset, cfg)
    with cfg.resolve_path(checkpoint["timepoint_split"]).open() as handle:
        split = json.load(handle)
    intervals = heldout_block_forecast_intervals(split["all_days"], set(split["heldout_days"]))
    cells = pd.read_csv(cfg.cell_metadata_csv, index_col=0)
    sampler = H5adIntervalSampler.from_adata_dir(
        cfg.adata_dir,
        cells,
        gene_ids,
        intervals,
        batch_size=args.cells_per_batch,
        seed=cfg.seed + 91,
        time_coordinates=split.get("time_coordinates"),
    )
    pca = PCADenoiser.load(cfg.resolve_path(args.pca_artifact))
    classifier = SubtypeCentroidClassifier.load(cfg.resolve_path(args.subtype_centroids))
    if classifier.centroids.shape[1] != pca.n_components:
        raise ValueError("Subtype centroids and PCA artifact have different dimensions")
    cre_inputs = load_cre_npz(cfg.resolve_path(checkpoint["cre_token_arrays"]), device)
    model = ScoreFlowStreamModel(
        n_genes=len(gene_ids),
        cre_dim=int(cre_inputs["cre_embeddings"].shape[-1]),
        d_model=cfg.d_model,
        n_heads=cfg.n_heads,
        n_layers=cfg.n_layers,
        dropout=cfg.dropout,
        variant=cfg.model_variant,
        positional_encoding=cfg.positional_encoding,
        n_context_tokens=cfg.n_context_tokens,
        state_dim=cfg.uce_embedding_dim,
        time_dim=cfg.score_flow_time_dim,
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    growth_config = checkpoint.get("population_finetuning", {})
    growth_head = None
    if args.growth_control == "learned":
        if "growth_head" not in checkpoint:
            raise ValueError("Learned growth evaluation requires a growth checkpoint")
        growth_head = GrowthRateHead(
            cfg.uce_embedding_dim,
            hidden_dim=int(growth_config.get("growth_hidden_dim", 256)),
        ).to(device)
        growth_head.load_state_dict(checkpoint["growth_head"])
        growth_head.eval()
    encoder = build_online_uce_encoder(cfg, selected, device)
    coordinates = coordinates_from_checkpoint(checkpoint, device)
    gene_scale = checkpoint["gene_scale"].to(device)
    perturbation_scale = coordinates.perturbation_scale(gene_scale)
    rows = []

    for interval_index, (day0, day1) in enumerate(intervals):
        source_stats = SubtypeStatistics.empty(len(classifier.labels), len(gene_ids))
        observed_stats = SubtypeStatistics.empty(len(classifier.labels), len(gene_ids))
        predicted_stats = SubtypeStatistics.empty(len(classifier.labels), len(gene_ids))
        for batch_index in range(args.batches_per_interval):
            batch = sampler.sample_interval(day0, day1)
            source = np.asarray(batch.x0, dtype=np.float32)
            observed = np.asarray(batch.x1, dtype=np.float32)

            def predict(current_y, tau, seed):
                state = encoder.encode(coordinates.to_expression(current_y), seed)
                return predict_score_flow_chunked(model, state, tau, cre_inputs, cfg.gene_chunk_size)

            seed = cfg.seed + interval_index * 100_000 + batch_index
            if growth_head is None:
                predicted_y = score_flow_rollout(
                    coordinates.to_model(torch.as_tensor(source, device=device)),
                    batch.t0,
                    batch.t1,
                    predict,
                    perturbation_scale,
                    args.steps,
                    seed,
                    diffusion=0.0,
                    noise_amplitude=cfg.score_flow_noise_scale,
                    dynamics_mode="coupled",
                    score_control="learned",
                )
                predicted_weights = None
            else:
                def predict_growth(current_y, tau, current_seed):
                    state = encoder.encode(coordinates.to_expression(current_y), current_seed)
                    return (
                        predict_score_flow_chunked(model, state, tau, cre_inputs, cfg.gene_chunk_size),
                        growth_head(state),
                    )

                with torch.no_grad():
                    result = differentiable_growth_rollout(
                        coordinates.to_model(torch.as_tensor(source, device=device)),
                        batch.t0,
                        batch.t1,
                        predict_growth,
                        perturbation_scale,
                        args.steps,
                        seed,
                        diffusion=0.0,
                        noise_amplitude=cfg.score_flow_noise_scale,
                        particles=1,
                        dynamics_mode="coupled",
                        score_control="learned",
                        max_growth_rate=float(growth_config.get("max_relative_growth_rate", 4.0)),
                    )
                predicted_y = result.state
                predicted_weights = result.weights.detach().cpu().numpy()
            predicted = coordinates.to_expression(predicted_y).detach().cpu().numpy()
            source_assignment = classifier.predict_coordinates(pca.transform_counts(source))
            observed_assignment = classifier.predict_coordinates(pca.transform_counts(observed))
            predicted_assignment = classifier.predict_coordinates(pca.transform_counts(predicted))
            source_stats.update(source, source_assignment)
            observed_stats.update(observed, observed_assignment)
            predicted_stats.update(predicted, predicted_assignment, predicted_weights)

        for gene_set, gene_indices in gene_sets.items():
            metrics = subtype_dynamics_metrics(
                source_stats,
                observed_stats,
                predicted_stats,
                gene_indices=gene_indices,
                min_cells=args.minimum_subtype_cells,
            )
            rows.append({
                "day0": day0,
                "day1": day1,
                "variant": cfg.model_variant,
                "growth_control": args.growth_control,
                "eval_gene_set": gene_set,
                "n_eval_genes": len(gene_ids) if gene_indices is None else len(gene_indices),
                "n_cells": args.cells_per_batch * args.batches_per_interval,
                "minimum_subtype_cells": args.minimum_subtype_cells,
                "classifier_validation_accuracy": classifier.metadata.get("validation_accuracy"),
                "classifier_validation_balanced_accuracy": classifier.metadata.get(
                    "validation_balanced_accuracy"
                ),
                **metrics,
            })
    frame = pd.DataFrame(rows)
    output = cfg.out_dir / f"subtype_eval_{checkpoint_path.stem}.csv"
    frame.to_csv(output, index=False)
    print(frame.to_string(index=False))
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
