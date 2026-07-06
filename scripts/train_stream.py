#!/usr/bin/env python
"""Train standard CFM or STREAM model variants."""

from __future__ import annotations

import argparse
import json

import pandas as pd
import torch

from stream_model.config import StreamConfig, apply_config_overrides
from stream_model.data import H5adIntervalSampler
from stream_model.train import build_model, load_cre_npz, train_steps


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/stream_mouse_dev.yaml")
    parser.add_argument("--variant", choices=["standard_cfm", "film", "cross_attention"], default=None)
    parser.add_argument("--hvg-csv", default=None)
    parser.add_argument("--n-hvg", type=int, default=None)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--wandb-run-name", default=None)
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
    )
    if args.variant is not None:
        cfg.model_variant = args.variant
    device = torch.device(args.device or cfg.device if torch.cuda.is_available() else "cpu")
    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    selected = pd.read_csv(cfg.out_dir / "selected_genes.csv")
    gene_ids = selected["gene_id"].tolist()
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
    )

    cre_inputs = None
    cre_dim = None
    if cfg.model_variant != "standard_cfm":
        cre_inputs = load_cre_npz(cfg.out_dir / "cre_token_arrays.npz", device)
        cre_dim = int(cre_inputs["cre_embeddings"].shape[-1])
    model = build_model(cfg, n_genes=len(gene_ids), cre_dim=cre_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)
    wandb_run = None
    if cfg.use_wandb:
        import wandb

        run_name = cfg.wandb_run_name or f"{cfg.model_variant}_{len(gene_ids)}genes_heldout_timepoints"
        wandb_run = wandb.init(
            project=cfg.wandb_project,
            entity=cfg.wandb_entity,
            mode=cfg.wandb_mode,
            name=run_name,
            config=cfg.to_dict(),
            tags=["stream", cfg.model_variant, "mouse_dev"],
        )
    metrics = train_steps(
        cfg,
        sampler,
        model,
        optimizer,
        cre_inputs=cre_inputs,
        steps_per_epoch=args.steps_per_epoch,
        wandb_run=wandb_run,
    )

    metrics_path = cfg.out_dir / f"train_metrics_{cfg.model_variant}.csv"
    ckpt_path = cfg.out_dir / f"model_{cfg.model_variant}.pt"
    pd.DataFrame(metrics).to_csv(metrics_path, index=False)
    torch.save({"model": model.state_dict(), "config": cfg.to_dict()}, ckpt_path)
    if wandb_run is not None:
        wandb_run.summary["final_train_loss"] = float(metrics[-1]["loss"]) if metrics else None
        wandb_run.summary["checkpoint_path"] = str(ckpt_path)
        wandb_run.finish()
    print(f"Wrote {metrics_path}")
    print(f"Wrote {ckpt_path}")


if __name__ == "__main__":
    main()
