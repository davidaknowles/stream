"""Training and evaluation routines for STREAM models."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .models import StandardCFM, StreamModel, mse_cfm_loss
from .ot import ot_cfm_batch, ot_cfm_batch_with_state


def load_cre_npz(path: str | Path, device: torch.device) -> dict[str, torch.Tensor]:
    raw = np.load(path, allow_pickle=True)
    return {
        "cre_embeddings": torch.as_tensor(raw["embeddings"], device=device),
        "cre_mask": torch.as_tensor(raw["mask"], device=device),
        "signed_distance": torch.as_tensor(raw["signed_distance"], device=device),
        "is_promoter": torch.as_tensor(raw["is_promoter"], device=device),
    }


def build_model(config, n_genes: int, cre_dim: int | None = None) -> torch.nn.Module:
    state_dim = config.uce_embedding_dim if config.cell_state == "uce" else n_genes
    if config.model_variant == "standard_cfm":
        return StandardCFM(
            n_genes=n_genes,
            hidden_dim=2 * config.d_model,
            n_layers=3,
            dropout=config.dropout,
            state_dim=state_dim,
        )
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
        state_dim=state_dim,
    )


def artifact_stem(config, variant: str | None = None) -> str:
    """Return a model/metric stem that keeps alternate cell states separate."""

    variant = variant or config.model_variant
    stem = variant if config.cell_state == "expression" else f"{variant}_{config.cell_state}"
    return f"{stem}_{config.experiment_label}" if getattr(config, "experiment_label", "") else stem


def predict_stream_chunked(
    model,
    x: torch.Tensor,
    cre_inputs: dict[str, torch.Tensor],
    gene_chunk_size: int,
) -> torch.Tensor:
    """Predict STREAM velocities in gene chunks to control GPU memory."""

    n_genes = int(cre_inputs["cre_embeddings"].shape[0])
    if gene_chunk_size <= 0 or gene_chunk_size >= n_genes:
        return model(x, **cre_inputs)
    chunks = []
    for start in range(0, n_genes, gene_chunk_size):
        end = min(start + gene_chunk_size, n_genes)
        gene_indices = torch.arange(start, end, device=x.device, dtype=torch.long)
        chunks.append(model(x, **cre_inputs, gene_indices=gene_indices))
    return torch.cat(chunks, dim=1)


def stream_chunked_loss(
    model,
    x: torch.Tensor,
    target: torch.Tensor,
    cre_inputs: dict[str, torch.Tensor],
    gene_chunk_size: int,
    loss_gene_indices: list[int] | np.ndarray | torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute STREAM MSE without materializing all genes at once."""

    n_genes = target.shape[1]
    indices = _loss_gene_index_tensor(loss_gene_indices, target.device)
    if indices is None:
        if gene_chunk_size <= 0 or gene_chunk_size >= n_genes:
            return mse_cfm_loss(model(x, **cre_inputs), target)
        indices = torch.arange(n_genes, device=target.device, dtype=torch.long)
    if gene_chunk_size <= 0 or gene_chunk_size >= len(indices):
        pred = model(x, **cre_inputs, gene_indices=indices)
        return mse_cfm_loss(pred, target.index_select(1, indices))
    loss = target.new_tensor(0.0)
    for gene_indices in indices.split(gene_chunk_size):
        pred = model(x, **cre_inputs, gene_indices=gene_indices)
        target_chunk = target.index_select(1, gene_indices)
        loss = loss + mse_cfm_loss(pred, target_chunk) * (len(gene_indices) / len(indices))
    return loss


def backward_stream_chunked_loss(
    model,
    x: torch.Tensor,
    target: torch.Tensor,
    cre_inputs: dict[str, torch.Tensor],
    gene_chunk_size: int,
    loss_gene_indices: list[int] | np.ndarray | torch.Tensor | None = None,
) -> float:
    """Backpropagate STREAM MSE one gene chunk at a time."""

    n_genes = target.shape[1]
    indices = _loss_gene_index_tensor(loss_gene_indices, target.device)
    if indices is None:
        if gene_chunk_size <= 0 or gene_chunk_size >= n_genes:
            loss = mse_cfm_loss(model(x, **cre_inputs), target)
            loss.backward()
            return float(loss.detach().cpu())
        indices = torch.arange(n_genes, device=target.device, dtype=torch.long)
    if gene_chunk_size <= 0 or gene_chunk_size >= len(indices):
        pred = model(x, **cre_inputs, gene_indices=indices)
        loss = mse_cfm_loss(pred, target.index_select(1, indices))
        loss.backward()
        return float(loss.detach().cpu())
    total = 0.0
    for gene_indices in indices.split(gene_chunk_size):
        pred = model(x, **cre_inputs, gene_indices=gene_indices)
        target_chunk = target.index_select(1, gene_indices)
        loss = mse_cfm_loss(pred, target_chunk) * (len(gene_indices) / len(indices))
        loss.backward()
        total += float(loss.detach().cpu())
    return total


def _loss_gene_index_tensor(
    loss_gene_indices: list[int] | np.ndarray | torch.Tensor | None,
    device: torch.device,
) -> torch.Tensor | None:
    if loss_gene_indices is None:
        return None
    indices = torch.as_tensor(loss_gene_indices, device=device, dtype=torch.long).flatten()
    if len(indices) == 0:
        raise ValueError("loss_gene_indices must not be empty")
    return indices


def train_steps(
    config,
    sampler,
    model,
    optimizer,
    cre_inputs=None,
    steps_per_epoch: int = 100,
    wandb_run=None,
    loss_gene_indices: list[int] | np.ndarray | torch.Tensor | None = None,
) -> list[dict[str, float]]:
    device = next(model.parameters()).device
    loss_index_tensor = _loss_gene_index_tensor(loss_gene_indices, device)
    metrics: list[dict[str, float]] = []
    for epoch in range(config.epochs):
        model.train()
        for step in range(steps_per_epoch):
            batch = sampler.sample()
            x0 = torch.as_tensor(batch.x0, device=device)
            x1 = torch.as_tensor(batch.x1, device=device)
            if batch.state0 is None:
                xt, target, _tau = ot_cfm_batch(
                    x0, x1, batch.t0, batch.t1, epsilon=config.ot_epsilon, iterations=config.ot_iterations
                )
                state_t = xt
            else:
                state0 = torch.as_tensor(batch.state0, device=device)
                state1 = torch.as_tensor(batch.state1, device=device)
                xt, target, _tau, state_t = ot_cfm_batch_with_state(
                    x0,
                    x1,
                    state0,
                    state1,
                    batch.t0,
                    batch.t1,
                    epsilon=config.ot_epsilon,
                    iterations=config.ot_iterations,
                )
            optimizer.zero_grad(set_to_none=True)
            if cre_inputs is None:
                pred = model(state_t)
                if loss_index_tensor is not None:
                    pred = pred.index_select(1, loss_index_tensor)
                    target_loss = target.index_select(1, loss_index_tensor)
                else:
                    target_loss = target
                loss = mse_cfm_loss(pred, target_loss)
                loss.backward()
                value = float(loss.detach().cpu())
            else:
                value = backward_stream_chunked_loss(
                    model,
                    state_t,
                    target,
                    cre_inputs,
                    config.gene_chunk_size,
                    loss_gene_indices=loss_index_tensor,
                )
            optimizer.step()
            row = {"epoch": epoch, "step": step, "loss": value}
            metrics.append(row)
            if wandb_run is not None:
                global_step = epoch * steps_per_epoch + step
                wandb_run.log(
                    {
                        "train/loss": value,
                        "train/epoch": epoch,
                        "train/step": step,
                        "model_variant": config.model_variant,
                        "cell_state": config.cell_state,
                    },
                    step=global_step,
                )
    return metrics
