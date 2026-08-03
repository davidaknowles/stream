#!/usr/bin/env python
"""Train stochastic-interpolant STREAM flow and score heads."""

from __future__ import annotations

import argparse
import hashlib
import json

import numpy as np
import pandas as pd
import torch

from stream_model.config import StreamConfig, apply_config_overrides
from stream_model.data import H5adIntervalSampler, intervals_with_skips
from stream_model.denoise import PCADenoiser
from stream_model.models import ScoreFlowStreamModel
from stream_model.score_flow import train_score_flow_steps
from stream_model.train import load_cre_npz
from stream_model.uce import build_online_uce_encoder


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/stream_mouse_dev.yaml")
    parser.add_argument("--timepoint-split", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--variant", choices=["film", "cross_attention"], default="cross_attention")
    parser.add_argument("--endpoint-denoising", choices=["none", "knn", "metacell"], default="none")
    parser.add_argument("--pca-artifact", required=True)
    parser.add_argument("--cre-token-arrays", default=None)
    parser.add_argument("--experiment-label", required=True)
    parser.add_argument("--ot-pool-size", type=int, default=16384)
    parser.add_argument("--ot-pairs-per-pool", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--gene-chunk-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--steps-per-epoch", type=int, default=100)
    parser.add_argument("--noise-scale", type=float, default=0.2)
    parser.add_argument("--limit-intervals", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    cfg = StreamConfig.from_yaml(args.config)
    apply_config_overrides(
        cfg,
        out_dir=args.out_dir,
        n_hvg=10000,
        cell_state="uce",
        uce_mode="online",
        experiment_label=args.experiment_label,
    )
    cfg.model_variant = args.variant
    cfg.endpoint_denoising = args.endpoint_denoising
    cfg.uce_expression_preprocessing = "denoised" if args.endpoint_denoising != "none" else "raw"
    cfg.pca_artifact = cfg.resolve_path(args.pca_artifact)
    cfg.ot_method = "partial"
    cfg.ot_partial_mass = 0.95
    cfg.ot_cost_space = "pca"
    cfg.ot_pair_bank_mode = "interval"
    cfg.ot_pool_size = args.ot_pool_size
    cfg.ot_pairs_per_pool = args.ot_pairs_per_pool
    cfg.batch_size = args.batch_size
    cfg.gene_chunk_size = args.gene_chunk_size
    cfg.epochs = args.epochs
    cfg.score_flow_noise_scale = args.noise_scale
    cfg.uce_sampling = "systematic"
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    selected = pd.read_csv(cfg.out_dir / "selected_genes.csv")
    gene_ids = selected["gene_id"].astype(str).tolist()
    split_path = cfg.resolve_path(args.timepoint_split)
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
        batch_size=cfg.ot_pool_size,
        seed=cfg.seed,
        time_coordinates=split.get("time_coordinates"),
    )
    sampler, validation_sampler = full_sampler.split_validation(0.1, cfg.seed + 17)
    pca = PCADenoiser.load(cfg.pca_artifact)
    expected_hash = hashlib.sha256("\n".join(gene_ids).encode()).hexdigest()
    if pca.components.shape[1] != len(gene_ids) or pca.metadata.get("gene_ids_sha256") != expected_hash:
        raise ValueError("PCA artifact gene panel does not match selected_genes.csv")
    cre_path = cfg.resolve_path(args.cre_token_arrays) if args.cre_token_arrays else cfg.out_dir / "cre_token_arrays.npz"
    cre_inputs = load_cre_npz(cre_path, device)

    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    model = ScoreFlowStreamModel(
        n_genes=len(gene_ids),
        cre_dim=int(cre_inputs["cre_embeddings"].shape[-1]),
        d_model=cfg.d_model,
        n_heads=cfg.n_heads,
        n_layers=cfg.n_layers,
        dropout=cfg.dropout,
        variant=args.variant,
        positional_encoding=cfg.positional_encoding,
        n_context_tokens=cfg.n_context_tokens,
        state_dim=cfg.uce_embedding_dim,
        time_dim=cfg.score_flow_time_dim,
    ).to(device)
    encoder = build_online_uce_encoder(cfg, selected, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)
    stem = f"score_flow_{args.variant}_{args.experiment_label}"
    checkpoint_path = cfg.out_dir / f"model_{stem}.pt"
    metrics_path = cfg.out_dir / f"train_metrics_{stem}.csv"

    def save_checkpoint(current_model, current_optimizer, selection, metrics, gene_scale, flow_scale):
        payload = {
            "model": current_model.state_dict(),
            "optimizer": current_optimizer.state_dict(),
            "config": cfg.to_dict(),
            "model_contract": "online_uce_score_flow_v1",
            "gene_ids": gene_ids,
            "cre_token_arrays": str(cre_path),
            "timepoint_split": str(split_path),
            "selection": selection,
            "gene_scale": gene_scale.detach().cpu(),
            "flow_scale": flow_scale.detach().cpu(),
        }
        temporary = checkpoint_path.with_suffix(".pt.tmp")
        torch.save(payload, temporary)
        temporary.replace(checkpoint_path)
        pd.DataFrame(metrics).to_csv(metrics_path, index=False)

    result = train_score_flow_steps(
        cfg,
        sampler,
        model,
        optimizer,
        cre_inputs,
        encoder,
        pca,
        args.steps_per_epoch,
        validation_sampler=validation_sampler,
        checkpoint_callback=save_checkpoint,
    )
    pd.DataFrame(result.metrics).to_csv(metrics_path, index=False)
    print(f"Wrote {checkpoint_path}")
    print(f"Wrote {metrics_path}")
    print(f"Stopped early: {result.stopped_early}")


if __name__ == "__main__":
    main()
