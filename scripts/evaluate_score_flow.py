#!/usr/bin/env python
"""Evaluate deterministic-flow and score-corrected stochastic STREAM rollouts."""

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
from stream_model.rollout import mean_shift_metrics, sinkhorn_divergence
from stream_model.score_flow import predict_score_flow_chunked, score_flow_rollout
from stream_model.train import load_cre_npz
from stream_model.uce import build_online_uce_encoder


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/stream_mouse_dev.yaml")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--pca-artifact", required=True)
    parser.add_argument("--cells-per-interval", type=int, default=128)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--diffusions", default="0,0.001,0.01,0.05")
    parser.add_argument("--stochastic-replicates", type=int, default=3)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    checkpoint_path = StreamConfig().resolve_path(args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    supported_contracts = {
        "online_uce_coupled_score_flow_v2",
        "online_uce_population_finetuned_score_flow_v3",
        "online_uce_gene_scaled_score_flow_v3",
        "online_uce_gene_scaled_population_score_flow_v4",
        "online_uce_growth_population_score_flow_v4",
        "online_uce_gene_scaled_growth_population_score_flow_v5",
    }
    if checkpoint.get("model_contract") not in supported_contracts:
        raise ValueError(f"Expected one of {sorted(supported_contracts)}")
    saved = checkpoint["config"]
    cfg = StreamConfig.from_yaml(args.config)
    apply_config_overrides(cfg, out_dir=args.out_dir, n_hvg=len(checkpoint["gene_ids"]), cell_state="uce", uce_mode="online")
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
    split_path = cfg.resolve_path(checkpoint["timepoint_split"])
    with split_path.open() as handle:
        split = json.load(handle)
    intervals = heldout_block_forecast_intervals(split["all_days"], set(split["heldout_days"]))
    cells = pd.read_csv(cfg.cell_metadata_csv, index_col=0)
    sampler = H5adIntervalSampler.from_adata_dir(
        cfg.adata_dir,
        cells,
        gene_ids,
        intervals,
        batch_size=args.cells_per_interval,
        seed=cfg.seed + 91,
        time_coordinates=split.get("time_coordinates"),
    )
    cre_path = cfg.resolve_path(checkpoint["cre_token_arrays"])
    cre_inputs = load_cre_npz(cre_path, device)
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
    if "growth_head" in checkpoint:
        growth_head = GrowthRateHead(
            cfg.uce_embedding_dim,
            hidden_dim=int(growth_config.get("growth_hidden_dim", 256)),
        ).to(device)
        growth_head.load_state_dict(checkpoint["growth_head"])
        growth_head.eval()
    encoder = build_online_uce_encoder(cfg, selected, device)
    pca = PCADenoiser.load(cfg.resolve_path(args.pca_artifact))
    gene_scale = checkpoint["gene_scale"].to(device)
    coordinates = coordinates_from_checkpoint(checkpoint, device)
    perturbation_scale = coordinates.perturbation_scale(gene_scale)
    diffusions = [float(value) for value in args.diffusions.replace(";", ",").split(",")]
    rows = []

    for interval_index, (day0, day1) in enumerate(intervals):
        batch = sampler.sample_interval(day0, day1)
        source = np.asarray(batch.x0, dtype=np.float32)
        observed = np.asarray(batch.x1, dtype=np.float32)
        source_pca = pca.transform_counts(source)
        observed_pca = pca.transform_counts(observed)
        persistence = sinkhorn_divergence(source_pca, observed_pca, epsilon=0.05)

        def predict(current_y, tau, seed):
            # KNN/metacell references are unavailable causally; rollout always encodes the current raw state.
            state = encoder.encode(coordinates.to_expression(current_y), seed)
            return predict_score_flow_chunked(model, state, tau, cre_inputs, cfg.gene_chunk_size)

        def predict_growth(current_y, tau, seed):
            state = encoder.encode(coordinates.to_expression(current_y), seed)
            prediction = predict_score_flow_chunked(
                model, state, tau, cre_inputs, cfg.gene_chunk_size
            )
            return prediction, growth_head(state)

        def predict_growth_only(current_y, tau, seed):
            del tau
            state = encoder.encode(coordinates.to_expression(current_y), seed)
            return None, growth_head(state)

        controls = [
            ("autonomous", "not_used", 0.0, "none"),
            ("coupled", "not_used", 0.0, "none"),
        ]
        controls.extend(
            ("coupled", score_control, diffusion, "none")
            for diffusion in diffusions
            if diffusion > 0
            for score_control in ("learned", "zero")
        )
        if growth_head is not None:
            controls.extend([
                ("none", "not_used", 0.0, "learned"),
                ("coupled", "not_used", 0.0, "learned"),
            ])
            controls.extend(
                ("coupled", score_control, diffusion, "learned")
                for diffusion in diffusions
                if diffusion > 0
                for score_control in ("learned", "zero")
            )
        for dynamics_mode, score_control, diffusion, growth_control in controls:
            replicates = 1 if diffusion == 0 else args.stochastic_replicates
            replicate_rows = []
            for replicate in range(replicates):
                seed = cfg.seed + interval_index * 10_000 + replicate
                if growth_control == "learned":
                    result = differentiable_growth_rollout(
                        coordinates.to_model(torch.as_tensor(source, device=device)),
                        batch.t0,
                        batch.t1,
                        predict_growth_only if dynamics_mode == "none" else predict_growth,
                        perturbation_scale,
                        args.steps,
                        seed,
                        diffusion=diffusion,
                        noise_amplitude=cfg.score_flow_noise_scale,
                        particles=1,
                        dynamics_mode=dynamics_mode,
                        score_control=(
                            "learned" if score_control == "not_used" else score_control
                        ),
                        max_growth_rate=float(
                            growth_config.get("max_relative_growth_rate", 4.0)
                        ),
                    )
                    predicted_y = result.state
                    predicted_weights = result.weights.detach().cpu().numpy()
                    growth_metrics = {
                        "growth_effective_sample_size": float(
                            result.effective_sample_size.detach().cpu()
                        ),
                        "growth_weight_kl": float(result.weight_kl.detach().cpu()),
                        "growth_rate_rms": float(result.growth_rate_rms.detach().cpu()),
                    }
                else:
                    predicted_y = score_flow_rollout(
                        coordinates.to_model(torch.as_tensor(source, device=device)),
                        batch.t0,
                        batch.t1,
                        predict,
                        perturbation_scale,
                        args.steps,
                        seed,
                        diffusion=diffusion,
                        noise_amplitude=cfg.score_flow_noise_scale,
                        dynamics_mode=dynamics_mode,
                        score_control=(
                            "learned" if score_control == "not_used" else score_control
                        ),
                    )
                    predicted_weights = None
                    growth_metrics = {
                        "growth_effective_sample_size": float(len(source)),
                        "growth_weight_kl": 0.0,
                        "growth_rate_rms": 0.0,
                    }
                predicted = coordinates.to_expression(predicted_y).detach().cpu().numpy()
                endpoint = sinkhorn_divergence(
                    pca.transform_counts(predicted),
                    observed_pca,
                    epsilon=0.05,
                    x_weights=predicted_weights,
                )
                replicate_rows.append(
                    {
                        **mean_shift_metrics(
                            source,
                            observed,
                            predicted,
                            predicted_weights=predicted_weights,
                        ),
                        "endpoint_sinkhorn": endpoint,
                        "sinkhorn_skill": 1.0 - endpoint / persistence if persistence > 0 else np.nan,
                        **growth_metrics,
                    }
                )
            row = {
                "day0": day0,
                "day1": day1,
                "variant": cfg.model_variant,
                "endpoint_denoising": saved["endpoint_denoising"],
                "dynamics_mode": "growth_only" if dynamics_mode == "none" else dynamics_mode,
                "score_control": score_control,
                "growth_control": growth_control,
                "diffusion": diffusion,
                "integration_steps": args.steps,
                "n_replicates": replicates,
                "persistence_sinkhorn": persistence,
            }
            for key in replicate_rows[0]:
                row[key] = float(np.mean([result[key] for result in replicate_rows]))
            rows.append(row)
    output = cfg.out_dir / f"causal_eval_{checkpoint_path.stem}.csv"
    frame = pd.DataFrame(rows)
    frame.to_csv(output, index=False)
    print(frame.to_string(index=False))
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
