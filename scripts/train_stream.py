#!/usr/bin/env python
"""Train standard CFM or STREAM model variants."""

from __future__ import annotations

import argparse
import json

import pandas as pd
import torch

from stream_model.config import StreamConfig, apply_config_overrides
from stream_model.data import H5adIntervalSampler
from stream_model.train import artifact_stem, build_model, load_cre_npz, train_steps
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
    device = torch.device(args.device or cfg.device if torch.cuda.is_available() else "cpu")
    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    selected = pd.read_csv(cfg.out_dir / "selected_genes.csv")
    gene_ids = selected["gene_id"].tolist()
    loss_gene_indices = _load_gene_subset_indices(gene_ids, args.loss_gene_subset, cfg)
    with (cfg.out_dir / "timepoint_split.json").open() as handle:
        split = json.load(handle)
    cells = pd.read_csv(cfg.cell_metadata_csv, index_col=0)
    sampler = H5adIntervalSampler.from_adata_dir(
        cfg.adata_dir,
        cells,
        gene_ids,
        [tuple(interval) for interval in split["train_intervals"]],
        batch_size=cfg.batch_size,
        seed=cfg.seed,
        state_embeddings_dir=cfg.uce_embedding_dir if cfg.cell_state == "uce" and cfg.uce_mode == "cached" else None,
        state_dim=cfg.uce_embedding_dim if cfg.cell_state == "uce" else None,
        time_coordinates=split.get("time_coordinates"),
    )

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
    if args.init_checkpoint is not None:
        checkpoint = torch.load(cfg.resolve_path(args.init_checkpoint), map_location=device)
        model.load_state_dict(checkpoint["model"], strict=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)
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
            tags=["stream", cfg.model_variant, cfg.cell_state, cfg.dataset_name, cfg.time_coordinate],
        )
        if loss_gene_indices is not None:
            wandb_run.config["loss_gene_subset"] = args.loss_gene_subset
            wandb_run.config["n_loss_genes"] = len(loss_gene_indices)
    metrics = train_steps(
        cfg,
        sampler,
        model,
        optimizer,
        cre_inputs=cre_inputs,
        steps_per_epoch=args.steps_per_epoch,
        wandb_run=wandb_run,
        loss_gene_indices=loss_gene_indices,
        state_encoder=state_encoder,
    )

    stem = artifact_stem(cfg)
    metrics_path = cfg.out_dir / f"train_metrics_{stem}.csv"
    ckpt_path = cfg.out_dir / f"model_{stem}.pt"
    pd.DataFrame(metrics).to_csv(metrics_path, index=False)
    torch.save(
        {
            "model": model.state_dict(),
            "config": cfg.to_dict(),
            "model_contract": "online_uce_autonomous_v1" if state_encoder is not None else "legacy_v1",
            "gene_ids": gene_ids,
            "cre_token_arrays": str(cre_token_path),
        },
        ckpt_path,
    )
    if wandb_run is not None:
        wandb_run.summary["final_train_loss"] = float(metrics[-1]["loss"]) if metrics else None
        wandb_run.summary["checkpoint_path"] = str(ckpt_path)
        wandb_run.finish()
    print(f"Wrote {metrics_path}")
    print(f"Wrote {ckpt_path}")


if __name__ == "__main__":
    main()
