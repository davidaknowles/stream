#!/usr/bin/env python
"""Fine-tune a coupled score-flow model using unpaired endpoint populations."""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
import torch

from stream_model.config import StreamConfig, apply_config_overrides
from stream_model.data import H5adIntervalSampler, intervals_with_skips
from stream_model.denoise import PCADenoiser
from stream_model.models import ScoreFlowStreamModel
from stream_model.population_finetune import (
    configure_population_finetuning,
    differentiable_score_flow_rollout,
    parameter_anchor_loss,
    population_endpoint_loss,
    snapshot_parameters,
)
from stream_model.score_flow import predict_score_flow_chunked
from stream_model.train import load_cre_npz
from stream_model.uce import build_online_uce_encoder


def parse_float_list(value: str) -> list[float]:
    values = [float(item) for item in value.replace(";", ",").split(",")]
    if not values or any(item <= 0 for item in values):
        raise ValueError("Diffusions must be a nonempty list of positive values")
    return values


def select_validation_intervals(intervals: list[tuple[str, str]], count: int) -> list[tuple[str, str]]:
    if count <= 0 or count >= len(intervals):
        return intervals
    indices = np.linspace(0, len(intervals) - 1, count).round().astype(int)
    return [intervals[index] for index in np.unique(indices)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/stream_mouse_dev.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--pca-artifact", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--experiment-label", required=True)
    parser.add_argument("--source-batch-size", type=int, default=8)
    parser.add_argument("--target-batch-size", type=int, default=32)
    parser.add_argument("--particles", type=int, default=2)
    parser.add_argument("--rollout-steps", type=int, default=2)
    parser.add_argument("--diffusions", default="0.001,0.01")
    parser.add_argument("--max-updates", type=int, default=200)
    parser.add_argument("--validation-every", type=int, default=20)
    parser.add_argument("--validation-intervals", type=int, default=4)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--min-delta", type=float, default=0.002)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--anchor-weight", type=float, default=1e-3)
    parser.add_argument("--mean-weight", type=float, default=0.1)
    parser.add_argument("--covariance-weight", type=float, default=0.01)
    parser.add_argument("--sinkhorn-epsilon", type=float, default=0.05)
    parser.add_argument("--sinkhorn-iterations", type=int, default=80)
    parser.add_argument("--gene-chunk-size", type=int, default=256)
    parser.add_argument("--limit-intervals", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.source_batch_size <= 0 or args.target_batch_size <= 0:
        raise ValueError("Batch sizes must be positive")

    checkpoint_path = StreamConfig().resolve_path(args.checkpoint)
    parent = torch.load(checkpoint_path, map_location="cpu")
    if parent.get("model_contract") != "online_uce_coupled_score_flow_v2":
        raise ValueError("Population fine-tuning requires an online_uce_coupled_score_flow_v2 checkpoint")
    saved = parent["config"]
    cfg = StreamConfig.from_yaml(args.config)
    apply_config_overrides(
        cfg,
        out_dir=args.out_dir,
        n_hvg=len(parent["gene_ids"]),
        cell_state="uce",
        uce_mode="online",
        experiment_label=args.experiment_label,
    )
    for name in (
        "model_variant", "d_model", "n_heads", "n_layers", "dropout", "n_context_tokens",
        "positional_encoding", "score_flow_time_dim", "score_flow_noise_scale", "uce_sampling",
        "endpoint_denoising",
    ):
        setattr(cfg, name, saved[name])
    cfg.uce_expression_preprocessing = "raw"
    cfg.gene_chunk_size = args.gene_chunk_size
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    selected = pd.read_csv(cfg.out_dir / "selected_genes.csv")
    gene_ids = selected["gene_id"].astype(str).tolist()
    if gene_ids != parent["gene_ids"]:
        raise ValueError("Checkpoint gene panel does not match selected_genes.csv")
    split_path = cfg.resolve_path(parent["timepoint_split"])
    with split_path.open() as handle:
        split = json.load(handle)
    intervals = intervals_with_skips(split["all_days"], set(split["heldout_days"]), max_skip=0)
    if args.limit_intervals > 0:
        intervals = intervals[: args.limit_intervals]
    cells = pd.read_csv(cfg.cell_metadata_csv, index_col=0)
    full_sampler = H5adIntervalSampler.from_adata_dir(
        cfg.adata_dir,
        cells,
        gene_ids,
        intervals,
        batch_size=max(args.source_batch_size, args.target_batch_size),
        seed=cfg.seed + 701,
        time_coordinates=split.get("time_coordinates"),
    )
    sampler, validation_sampler = full_sampler.split_validation(0.1, cfg.seed + 702)
    validation_data = []
    for interval in select_validation_intervals(intervals, args.validation_intervals):
        batch = validation_sampler.sample_interval(*interval)
        validation_data.append(
            (batch.x0[: args.source_batch_size], batch.x1[: args.target_batch_size], batch.t0, batch.t1, *interval)
        )

    pca = PCADenoiser.load(cfg.resolve_path(args.pca_artifact))
    cre_path = cfg.resolve_path(parent["cre_token_arrays"])
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
    model.load_state_dict(parent["model"])
    trainable = configure_population_finetuning(model)
    references = snapshot_parameters(trainable)
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate)
    encoder = build_online_uce_encoder(cfg, selected, device)
    gene_scale = parent["gene_scale"].to(device)
    diffusions = parse_float_list(args.diffusions)
    output_path = cfg.out_dir / f"model_score_flow_{cfg.model_variant}_{args.experiment_label}.pt"
    metrics_path = cfg.out_dir / f"train_metrics_score_flow_{cfg.model_variant}_{args.experiment_label}.csv"
    metrics: list[dict[str, float | int | str]] = []

    def predict(current_x, tau, seed):
        # UCE tokenization is discrete: use its current-state value but truncate df/dx through UCE.
        state = encoder.encode(current_x, seed)
        return predict_score_flow_chunked(model, state, tau, cre_inputs, cfg.gene_chunk_size)

    def endpoint_objective(source, observed, t0, t1, diffusion, seed):
        deterministic = differentiable_score_flow_rollout(
            source, t0, t1, predict, gene_scale, args.rollout_steps, seed, 0.0,
            cfg.score_flow_noise_scale, args.particles,
        )
        stochastic = differentiable_score_flow_rollout(
            source, t0, t1, predict, gene_scale, args.rollout_steps, seed, diffusion,
            cfg.score_flow_noise_scale, args.particles,
        )
        repeated_source = source.repeat_interleave(args.particles, dim=0)
        deterministic_loss, deterministic_metrics = population_endpoint_loss(
            deterministic, observed, repeated_source, pca, gene_scale,
            args.sinkhorn_epsilon, args.sinkhorn_iterations, args.mean_weight, args.covariance_weight,
        )
        stochastic_loss, stochastic_metrics = population_endpoint_loss(
            stochastic, observed, repeated_source, pca, gene_scale,
            args.sinkhorn_epsilon, args.sinkhorn_iterations, args.mean_weight, args.covariance_weight,
        )
        return 0.5 * (deterministic_loss + stochastic_loss), deterministic_metrics, stochastic_metrics

    def validate() -> dict[str, float]:
        model.eval()
        rows = []
        with torch.no_grad():
            for index, (source_np, target_np, t0, t1, _day0, _day1) in enumerate(validation_data):
                diffusion = diffusions[index % len(diffusions)]
                loss, ode, sde = endpoint_objective(
                    torch.as_tensor(source_np, device=device),
                    torch.as_tensor(target_np, device=device),
                    t0, t1, diffusion, cfg.seed + 90_000 + index,
                )
                rows.append({
                    "validation_loss": float(loss.cpu()),
                    "validation_ode_sinkhorn": float(ode["sinkhorn"].cpu()),
                    "validation_sde_sinkhorn": float(sde["sinkhorn"].cpu()),
                    "validation_ode_skill": float(ode["sinkhorn_skill"].cpu()),
                    "validation_sde_skill": float(sde["sinkhorn_skill"].cpu()),
                })
        return {key: float(np.mean([row[key] for row in rows])) for key in rows[0]}

    def save_checkpoint(selection: dict[str, float | int]) -> None:
        payload = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": cfg.to_dict(),
            "model_contract": "online_uce_population_finetuned_score_flow_v3",
            "gene_ids": gene_ids,
            "cre_token_arrays": str(cre_path),
            "timepoint_split": str(split_path),
            "selection": selection,
            "gene_scale": parent["gene_scale"],
            "flow_scale": parent["flow_scale"],
            "parent_checkpoint": str(checkpoint_path),
            "population_finetuning": vars(args),
        }
        temporary = output_path.with_suffix(".pt.tmp")
        torch.save(payload, temporary)
        temporary.replace(output_path)
        pd.DataFrame(metrics).to_csv(metrics_path, index=False)

    initial = validate()
    ema = initial["validation_loss"]
    best_ema = ema
    baseline_validation_loss = ema
    metrics.append({"update": 0, "phase": "validation", **initial, "validation_loss_ema": ema})
    save_checkpoint(metrics[-1])
    checks_without_improvement = 0
    rng = np.random.default_rng(cfg.seed + 703)
    schedule = list(intervals)
    rng.shuffle(schedule)
    model.eval()
    for update in range(1, args.max_updates + 1):
        if (update - 1) % len(schedule) == 0 and update > 1:
            rng.shuffle(schedule)
        day0, day1 = schedule[(update - 1) % len(schedule)]
        batch = sampler.sample_interval(day0, day1)
        source = torch.as_tensor(batch.x0[: args.source_batch_size], device=device)
        observed = torch.as_tensor(batch.x1[: args.target_batch_size], device=device)
        diffusion = diffusions[(update - 1) % len(diffusions)]
        optimizer.zero_grad(set_to_none=True)
        endpoint_loss, ode, sde = endpoint_objective(
            source, observed, batch.t0, batch.t1, diffusion, cfg.seed + update * 101
        )
        anchor = parameter_anchor_loss(trainable, references)
        loss = endpoint_loss + args.anchor_weight * anchor
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite population loss at update {update}")
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        metrics.append({
            "update": update,
            "phase": "train",
            "day0": day0,
            "day1": day1,
            "diffusion": diffusion,
            "loss": float(loss.detach().cpu()),
            "endpoint_loss": float(endpoint_loss.detach().cpu()),
            "ode_sinkhorn": float(ode["sinkhorn"].detach().cpu()),
            "sde_sinkhorn": float(sde["sinkhorn"].detach().cpu()),
            "ode_skill": float(ode["sinkhorn_skill"].detach().cpu()),
            "sde_skill": float(sde["sinkhorn_skill"].detach().cpu()),
            "anchor_loss": float(anchor.detach().cpu()),
            "gradient_norm": float(gradient_norm.detach().cpu()),
        })
        if update % args.validation_every != 0 and update != args.max_updates:
            continue
        validation = validate()
        ema = 0.3 * validation["validation_loss"] + 0.7 * ema
        row = {"update": update, "phase": "validation", **validation, "validation_loss_ema": ema}
        metrics.append(row)
        pd.DataFrame(metrics).to_csv(metrics_path, index=False)
        if ema < best_ema * (1.0 - args.min_delta):
            best_ema = ema
            checks_without_improvement = 0
            save_checkpoint(row)
        else:
            checks_without_improvement += 1
        print(json.dumps(row, sort_keys=True), flush=True)
        if checks_without_improvement >= args.patience:
            break

    pd.DataFrame(metrics).to_csv(metrics_path, index=False)
    print(f"Wrote {output_path}")
    print(f"Wrote {metrics_path}")
    print(f"Best validation loss: {best_ema:.6g} (pretrained baseline {baseline_validation_loss:.6g})")


if __name__ == "__main__":
    main()
