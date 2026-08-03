#!/usr/bin/env python
"""Evaluate deterministic-flow and score-corrected stochastic STREAM rollouts."""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
import torch

from stream_model.config import StreamConfig, apply_config_overrides
from stream_model.data import H5adIntervalSampler, heldout_block_forecast_intervals
from stream_model.denoise import PCADenoiser
from stream_model.models import ScoreFlowStreamModel
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
    if checkpoint.get("model_contract") != "online_uce_coupled_score_flow_v2":
        raise ValueError("Expected online_uce_coupled_score_flow_v2 checkpoint")
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
    encoder = build_online_uce_encoder(cfg, selected, device)
    pca = PCADenoiser.load(cfg.resolve_path(args.pca_artifact))
    gene_scale = checkpoint["gene_scale"].to(device)
    diffusions = [float(value) for value in args.diffusions.replace(";", ",").split(",")]
    rows = []

    for interval_index, (day0, day1) in enumerate(intervals):
        batch = sampler.sample_interval(day0, day1)
        source = np.asarray(batch.x0, dtype=np.float32)
        observed = np.asarray(batch.x1, dtype=np.float32)
        source_pca = pca.transform_counts(source)
        observed_pca = pca.transform_counts(observed)
        persistence = sinkhorn_divergence(source_pca, observed_pca, epsilon=0.05)

        def predict(current_x, tau, seed):
            # KNN/metacell references are unavailable causally; rollout always encodes the current raw state.
            state = encoder.encode(current_x, seed)
            return predict_score_flow_chunked(model, state, tau, cre_inputs, cfg.gene_chunk_size)

        controls = [("autonomous", "not_used", 0.0), ("coupled", "not_used", 0.0)]
        controls.extend(
            ("coupled", score_control, diffusion)
            for diffusion in diffusions
            if diffusion > 0
            for score_control in ("learned", "zero")
        )
        for dynamics_mode, score_control, diffusion in controls:
            replicates = 1 if diffusion == 0 else args.stochastic_replicates
            replicate_rows = []
            for replicate in range(replicates):
                seed = cfg.seed + interval_index * 10_000 + replicate
                predicted = score_flow_rollout(
                    torch.as_tensor(source, device=device),
                    batch.t0,
                    batch.t1,
                    predict,
                    gene_scale,
                    args.steps,
                    seed,
                    diffusion=diffusion,
                    noise_amplitude=cfg.score_flow_noise_scale,
                    dynamics_mode=dynamics_mode,
                    score_control="learned" if score_control == "not_used" else score_control,
                ).cpu().numpy()
                endpoint = sinkhorn_divergence(pca.transform_counts(predicted), observed_pca, epsilon=0.05)
                replicate_rows.append(
                    {
                        **mean_shift_metrics(source, observed, predicted),
                        "endpoint_sinkhorn": endpoint,
                        "sinkhorn_skill": 1.0 - endpoint / persistence if persistence > 0 else np.nan,
                    }
                )
            row = {
                "day0": day0,
                "day1": day1,
                "variant": cfg.model_variant,
                "endpoint_denoising": saved["endpoint_denoising"],
                "dynamics_mode": dynamics_mode,
                "score_control": score_control,
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
