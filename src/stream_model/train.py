"""Training and evaluation routines for STREAM models."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .models import StandardCFM, StreamModel, mse_cfm_loss
from .ot import ot_cfm_batch


def load_cre_npz(path: str | Path, device: torch.device) -> dict[str, torch.Tensor]:
    raw = np.load(path, allow_pickle=True)
    return {
        "cre_embeddings": torch.as_tensor(raw["embeddings"], device=device),
        "cre_mask": torch.as_tensor(raw["mask"], device=device),
        "signed_distance": torch.as_tensor(raw["signed_distance"], device=device),
        "is_promoter": torch.as_tensor(raw["is_promoter"], device=device),
    }


def build_model(config, n_genes: int, cre_dim: int | None = None) -> torch.nn.Module:
    if config.model_variant == "standard_cfm":
        return StandardCFM(n_genes=n_genes, hidden_dim=2 * config.d_model, n_layers=3, dropout=config.dropout)
    if cre_dim is None:
        raise ValueError("cre_dim is required for STREAM variants")
    variant = "cross_attention" if config.model_variant == "cross_attention" else "film"
    return StreamModel(
        n_genes=n_genes,
        cre_dim=cre_dim,
        d_model=config.d_model,
        n_heads=config.n_heads,
        n_layers=config.n_layers,
        dropout=config.dropout,
        variant=variant,
        positional_encoding=config.positional_encoding,
        n_context_tokens=config.n_context_tokens,
    )


def train_steps(config, sampler, model, optimizer, cre_inputs=None, steps_per_epoch: int = 100) -> list[dict[str, float]]:
    device = next(model.parameters()).device
    metrics: list[dict[str, float]] = []
    for epoch in range(config.epochs):
        model.train()
        for step in range(steps_per_epoch):
            batch = sampler.sample()
            x0 = torch.as_tensor(batch.x0, device=device)
            x1 = torch.as_tensor(batch.x1, device=device)
            xt, target, _tau = ot_cfm_batch(
                x0,
                x1,
                batch.t0,
                batch.t1,
                epsilon=config.ot_epsilon,
                iterations=config.ot_iterations,
            )
            if cre_inputs is None:
                pred = model(xt)
            else:
                pred = model(xt, **cre_inputs)
            loss = mse_cfm_loss(pred, target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            metrics.append({"epoch": epoch, "step": step, "loss": float(loss.detach().cpu())})
    return metrics
