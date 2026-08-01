#!/usr/bin/env python
"""Train standard CFM or STREAM model variants."""

from __future__ import annotations

import argparse
import hashlib
import json

import pandas as pd
import torch

from stream_model.config import StreamConfig, apply_config_overrides
from stream_model.data import H5adIntervalSampler, intervals_with_skips
from stream_model.denoise import PCADenoiser
from stream_model.train import (
    artifact_stem,
    build_fixed_validation_batches,
    build_model,
    load_cre_npz,
    train_steps,
)
from stream_model.uce import build_online_uce_encoder


def _load_gene_subset_indices(gene_ids: list[str], subset_csv: str | None, cfg: StreamConfig) -> list[int] | None:
    if subset_csv is None:
        return None
    path = cfg.resolve_path(subset_csv)
    reference = pd.read_csv(path)
    if "gene_id" not in reference.columns:
        raise ValueError(f"{path} must contain a gene_id column")
    ref_ids = reference["gene_id"].astype(str).drop_duplicates().tolist()
    indices = pd.Index(gene_ids).get_indexer(ref_ids)
    if (indices < 0).any():
        missing = pd.Index(ref_ids)[indices < 0][:10].tolist()
        raise ValueError(f"{path} contains genes absent from this model panel: {missing}")
    return indices.tolist()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/stream_mouse_dev.yaml")
    parser.add_argument("--variant", choices=["standard_cfm", "film", "cross_attention"], default=None)
    parser.add_argument("--hvg-csv", default=None)
    parser.add_argument("--n-hvg", type=int, default=None)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--gene-chunk-size", type=int, default=None)
    parser.add_argument("--cell-state", choices=["expression", "uce"], default=None)
    parser.add_argument("--uce-embedding-dir", default=None)
    parser.add_argument("--uce-mode", choices=["cached", "online"], default=None)
    parser.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default=None)
    parser.add_argument("--experiment-label", default=None)
    parser.add_argument("--init-checkpoint", default=None)
    parser.add_argument(
        "--loss-gene-subset",
        default=None,
        help="Optional CSV with a gene_id column. Training loss is restricted to these genes, while OT and state still use the full panel.",
    )
    parser.add_argument("--cre-token-arrays", default=None)
    parser.add_argument("--steps-per-epoch", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--validation-batches-per-interval", type=int, default=1)
    parser.add_argument("--validation-every-epochs", type=int, default=1)
    parser.add_argument("--early-stopping-patience", type=int, default=12)
    parser.add_argument("--early-stopping-min-delta", type=float, default=0.005)
    parser.add_argument("--validation-ema-alpha", type=float, default=0.3)
    parser.add_argument("--rollout-every-validations", type=int, default=5)
    parser.add_argument("--validation-rollout-steps", type=int, default=4)
    parser.add_argument("--ot-method", choices=["balanced", "partial", "unbalanced"], default=None)
    parser.add_argument("--ot-partial-mass", type=float, default=None)
    parser.add_argument("--ot-marginal-relaxation", type=float, default=None)
    parser.add_argument("--ot-pool-size", type=int, default=None)
    parser.add_argument("--ot-pairs-per-pool", type=int, default=None)
    parser.add_argument("--ot-pair-bank-mode", choices=["sequential", "interval"], default=None)
    parser.add_argument("--ot-cost-space", choices=["expression", "pca"], default=None)
    parser.add_argument("--endpoint-denoising", choices=["none", "pca"], default=None)
    parser.add_argument("--pca-artifact", default=None)
    parser.add_argument("--max-interval-skip", type=int, choices=[0, 1, 2], default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    cfg = StreamConfig.from_yaml(args.config)
    apply_config_overrides(
        cfg,
        hvg_csv=args.hvg_csv,
        n_hvg=args.n_hvg,
        out_dir=args.out_dir,
        wandb_run_name=args.wandb_run_name,
        cell_state=args.cell_state,
        uce_mode=args.uce_mode,
        uce_embedding_dir=args.uce_embedding_dir,
        wandb_mode=args.wandb_mode,
        experiment_label=args.experiment_label,
    )
    if args.variant is not None:
        cfg.model_variant = args.variant
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.gene_chunk_size is not None:
        cfg.gene_chunk_size = args.gene_chunk_size
    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.ot_method is not None:
        cfg.ot_method = args.ot_method
    if args.ot_partial_mass is not None:
        cfg.ot_partial_mass = args.ot_partial_mass
    if args.ot_marginal_relaxation is not None:
        cfg.ot_marginal_relaxation = args.ot_marginal_relaxation
    if args.ot_pool_size is not None:
        cfg.ot_pool_size = args.ot_pool_size
    if args.ot_pairs_per_pool is not None:
        cfg.ot_pairs_per_pool = args.ot_pairs_per_pool
    if args.ot_pair_bank_mode is not None:
        cfg.ot_pair_bank_mode = args.ot_pair_bank_mode
    if args.ot_cost_space is not None:
        cfg.ot_cost_space = args.ot_cost_space
    if args.endpoint_denoising is not None:
        cfg.endpoint_denoising = args.endpoint_denoising
    if args.pca_artifact is not None:
        cfg.pca_artifact = cfg.resolve_path(args.pca_artifact)
    if args.max_interval_skip is not None:
        cfg.max_interval_skip = args.max_interval_skip
    if (cfg.ot_cost_space == "pca" or cfg.endpoint_denoising == "pca") and cfg.pca_artifact is None:
        parser.error("PCA coupling or denoising requires --pca-artifact")
    if cfg.endpoint_denoising != "none" and cfg.ot_pair_bank_mode != "interval":
        parser.error("Endpoint denoising requires --ot-pair-bank-mode interval")
    device = torch.device(args.device or cfg.device if torch.cuda.is_available() else "cpu")
    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    selected = pd.read_csv(cfg.out_dir / "selected_genes.csv")
    gene_ids = selected["gene_id"].tolist()
    loss_gene_indices = _load_gene_subset_indices(gene_ids, args.loss_gene_subset, cfg)
    with (cfg.out_dir / "timepoint_split.json").open() as handle:
        split = json.load(handle)
    adjacent_train_intervals = [tuple(interval) for interval in split["train_intervals"]]
    train_intervals = intervals_with_skips(
        split["all_days"], set(split["heldout_days"]), max_skip=cfg.max_interval_skip
    )
    cells = pd.read_csv(cfg.cell_metadata_csv, index_col=0)
    full_sampler = H5adIntervalSampler.from_adata_dir(
        cfg.adata_dir,
        cells,
        gene_ids,
        train_intervals,
        batch_size=cfg.ot_pool_size if cfg.ot_pool_size > 0 else cfg.batch_size,
        seed=cfg.seed,
        state_embeddings_dir=cfg.uce_embedding_dir if cfg.cell_state == "uce" and cfg.uce_mode == "cached" else None,
        state_dim=cfg.uce_embedding_dim if cfg.cell_state == "uce" else None,
        time_coordinates=split.get("time_coordinates"),
    )
    if args.validation_fraction > 0:
        sampler, validation_sampler = full_sampler.split_validation(args.validation_fraction, cfg.seed + 17)
        validation_sampler.intervals = adjacent_train_intervals
    else:
        sampler, validation_sampler = full_sampler, None
    pca = PCADenoiser.load(cfg.pca_artifact) if cfg.pca_artifact is not None else None
    if pca is not None:
        expected_hash = hashlib.sha256("\n".join(gene_ids).encode()).hexdigest()
        if pca.components.shape[1] != len(gene_ids) or pca.metadata.get("gene_ids_sha256") != expected_hash:
            raise ValueError("PCA artifact gene panel does not match selected_genes.csv")

    cre_inputs = None
    cre_dim = None
    cre_token_path = cfg.resolve_path(args.cre_token_arrays) if args.cre_token_arrays else cfg.out_dir / "cre_token_arrays.npz"
    if cfg.model_variant != "standard_cfm":
        cre_inputs = load_cre_npz(cre_token_path, device)
        cre_dim = int(cre_inputs["cre_embeddings"].shape[-1])
    model = build_model(cfg, n_genes=len(gene_ids), cre_dim=cre_dim).to(device)
    state_encoder = (
        build_online_uce_encoder(cfg, selected, device)
        if cfg.cell_state == "uce" and cfg.uce_mode == "online"
        else None
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)
    if args.init_checkpoint is not None:
        checkpoint = torch.load(cfg.resolve_path(args.init_checkpoint), map_location=device)
        model.load_state_dict(checkpoint["model"], strict=True)
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
    validation_batches = (
        build_fixed_validation_batches(
            cfg,
            validation_sampler,
            device,
            batches_per_interval=args.validation_batches_per_interval,
            state_encoder=state_encoder,
            seed=cfg.seed + 23,
            pca=pca,
        )
        if validation_sampler is not None
        else None
    )
    wandb_run = None
    if cfg.use_wandb:
        import wandb

        run_name = cfg.wandb_run_name or f"{artifact_stem(cfg)}_{len(gene_ids)}genes_heldout_timepoints"
        wandb_run = wandb.init(
            project=cfg.wandb_project,
            entity=cfg.wandb_entity,
            mode=cfg.wandb_mode,
            name=run_name,
            config=cfg.to_dict(),
            tags=["stream", cfg.model_variant, cfg.cell_state, cfg.dataset_name, cfg.time_coordinate, cfg.ot_method],
        )
        if loss_gene_indices is not None:
            wandb_run.config["loss_gene_subset"] = args.loss_gene_subset
            wandb_run.config["n_loss_genes"] = len(loss_gene_indices)
    stem = artifact_stem(cfg)
    metrics_path = cfg.out_dir / f"train_metrics_{stem}.csv"
    validation_metrics_path = cfg.out_dir / f"validation_metrics_{stem}.csv"
    ckpt_path = cfg.out_dir / f"model_{stem}.pt"
    validation_config = {
        "fraction": args.validation_fraction,
        "batches_per_interval": args.validation_batches_per_interval,
        "every_epochs": args.validation_every_epochs,
        "patience": args.early_stopping_patience,
        "min_delta": args.early_stopping_min_delta,
        "ema_alpha": args.validation_ema_alpha,
        "rollout_every_validations": args.rollout_every_validations,
        "rollout_steps": args.validation_rollout_steps,
        "n_train_cells": len(sampler.manifest),
        "n_validation_cells": 0 if validation_sampler is None else len(validation_sampler.manifest),
        "ot_method": cfg.ot_method,
        "ot_partial_mass": cfg.ot_partial_mass,
        "ot_marginal_relaxation": cfg.ot_marginal_relaxation,
        "ot_pool_size": cfg.ot_pool_size if cfg.ot_pool_size > 0 else cfg.batch_size,
        "ot_pairs_per_pool": cfg.ot_pairs_per_pool if cfg.ot_pairs_per_pool > 0 else cfg.batch_size,
        "ot_pair_bank_mode": cfg.ot_pair_bank_mode,
        "ot_cost_space": cfg.ot_cost_space,
        "endpoint_denoising": cfg.endpoint_denoising,
        "pca_artifact": None if cfg.pca_artifact is None else str(cfg.pca_artifact),
        "max_interval_skip": cfg.max_interval_skip,
        "n_train_intervals": len(train_intervals),
        "n_validation_intervals": len(adjacent_train_intervals),
    }

    def save_best_checkpoint(current_model, current_optimizer, validation_row, train_rows, validation_rows):
        payload = {
            "model": current_model.state_dict(),
            "optimizer": current_optimizer.state_dict(),
            "config": cfg.to_dict(),
            "model_contract": "online_uce_autonomous_v1" if state_encoder is not None else "legacy_v1",
            "gene_ids": gene_ids,
            "cre_token_arrays": str(cre_token_path),
            "selection": validation_row,
            "validation_config": validation_config,
            "pca_artifact": None if cfg.pca_artifact is None else str(cfg.pca_artifact),
        }
        temporary_path = ckpt_path.with_suffix(ckpt_path.suffix + ".tmp")
        torch.save(payload, temporary_path)
        temporary_path.replace(ckpt_path)
        pd.DataFrame(train_rows).to_csv(metrics_path, index=False)
        pd.DataFrame(validation_rows).to_csv(validation_metrics_path, index=False)

    result = train_steps(
        cfg,
        sampler,
        model,
        optimizer,
        cre_inputs=cre_inputs,
        steps_per_epoch=args.steps_per_epoch,
        wandb_run=wandb_run,
        loss_gene_indices=loss_gene_indices,
        state_encoder=state_encoder,
        validation_batches=validation_batches,
        validation_every_epochs=args.validation_every_epochs,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_min_delta=args.early_stopping_min_delta,
        validation_ema_alpha=args.validation_ema_alpha,
        rollout_every_validations=args.rollout_every_validations,
        validation_rollout_steps=args.validation_rollout_steps,
        checkpoint_callback=save_best_checkpoint,
        pca=pca,
    )

    pd.DataFrame(result.train_metrics).to_csv(metrics_path, index=False)
    if result.validation_metrics:
        pd.DataFrame(result.validation_metrics).to_csv(validation_metrics_path, index=False)
    else:
        save_best_checkpoint(model, optimizer, {}, result.train_metrics, [])
    if wandb_run is not None:
        wandb_run.summary["final_train_loss"] = float(result.train_metrics[-1]["loss"]) if result.train_metrics else None
        wandb_run.summary["best_validation_loss"] = result.best_validation_loss
        wandb_run.summary["best_epoch"] = result.best_epoch
        wandb_run.summary["best_global_step"] = result.best_global_step
        wandb_run.summary["stopped_early"] = result.stopped_early
        wandb_run.summary["checkpoint_path"] = str(ckpt_path)
        wandb_run.finish()
    print(f"Wrote {metrics_path}")
    if result.validation_metrics:
        print(f"Wrote {validation_metrics_path}")
    print(f"Wrote {ckpt_path}")
    print(
        f"Best normalized validation EMA={result.best_validation_loss} at step={result.best_global_step}; "
        f"stopped_early={result.stopped_early}"
    )


if __name__ == "__main__":
    main()
