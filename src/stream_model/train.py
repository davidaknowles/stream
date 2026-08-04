"""Training and evaluation routines for STREAM models."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from .denoise import denoise_selected_counts
from .models import StandardCFM, StreamModel, mse_cfm_loss
from .ot import (
    cfm_interpolate,
    coupling_diagnostics,
    coupling_kwargs,
    sample_coupling_pairs,
    transport_plan,
)
from .rollout import mean_shift_metrics, projected_euler_rollout
from .pair_bank import IntervalPairBank


@dataclass(frozen=True)
class FixedValidationBatch:
    """A deterministic CFM target and its unpaired endpoint distributions."""

    day0: str
    day1: str
    t0: float
    t1: float
    state_t: torch.Tensor
    target: torch.Tensor
    x0: torch.Tensor
    x1: torch.Tensor


def uce_input_expression(
    config,
    expression: torch.Tensor,
    pca=None,
    denoised_expression: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return the autonomous expression-derived input to the frozen UCE encoder."""

    preprocessing = getattr(config, "uce_expression_preprocessing", "raw")
    if preprocessing == "raw":
        return expression
    if preprocessing == "pca":
        if pca is None:
            raise ValueError("PCA preprocessing before UCE requires a fitted PCA artifact")
        return pca.reconstruct_tensor(expression)
    if preprocessing == "denoised":
        return expression if denoised_expression is None else denoised_expression
    raise ValueError("uce_expression_preprocessing must be raw, pca, or denoised")


@dataclass(frozen=True)
class TrainingResult:
    train_metrics: list[dict[str, float]]
    validation_metrics: list[dict[str, float]]
    best_validation_loss: float | None
    best_epoch: int | None
    best_global_step: int | None
    stopped_early: bool


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
    if config.cell_state == "expression":
        stem = variant
    elif getattr(config, "uce_mode", "cached") == "online":
        stem = f"{variant}_online_uce"
    else:
        stem = f"{variant}_{config.cell_state}"
    if getattr(config, "dynamics_coordinates", "count") == "gene_scaled":
        stem = f"{stem}_gene_scaled"
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


@torch.no_grad()
def build_fixed_validation_batches(
    config,
    sampler,
    device: torch.device,
    batches_per_interval: int = 1,
    state_encoder=None,
    seed: int | None = None,
    pca=None,
    coordinates=None,
) -> list[FixedValidationBatch]:
    """Materialize fixed, interval-stratified OT-CFM validation examples."""

    if batches_per_interval <= 0:
        raise ValueError("batches_per_interval must be positive")
    base_seed = int(config.seed if seed is None else seed)
    batches: list[FixedValidationBatch] = []
    for interval_index, (day0, day1) in enumerate(sampler.intervals):
        for batch_index in range(batches_per_interval):
            sampled = sampler.sample_interval(day0, day1)
            x0 = torch.as_tensor(sampled.x0, device=device)
            x1 = torch.as_tensor(sampled.x1, device=device)
            batch_seed = base_seed + interval_index * 10_000 + batch_index
            generator = torch.Generator(device=device).manual_seed(batch_seed)
            ot_cost_space = getattr(config, "ot_cost_space", "expression")
            if ot_cost_space == "pca":
                if pca is None:
                    raise ValueError("PCA OT cost requires a fitted PCA artifact")
                cost_x0 = torch.as_tensor(pca.transform_counts(sampled.x0), device=device)
                cost_x1 = torch.as_tensor(pca.transform_counts(sampled.x1), device=device)
                cost_metric = "scaled_euclidean"
            else:
                cost_x0 = cost_x1 = None
                cost_metric = "expression_cosine"
            cost, coupling = transport_plan(
                x0,
                x1,
                cost_x0=cost_x0,
                cost_x1=cost_x1,
                cost_metric=cost_metric,
                **coupling_kwargs(config),
            )
            i0, i1 = sample_coupling_pairs(coupling, int(config.batch_size), generator=generator)
            paired_x0 = x0[i0]
            paired_x1 = x1[i1]
            xt, _raw_target, tau = cfm_interpolate(
                paired_x0, paired_x1, sampled.t0, sampled.t1, generator=generator
            )
            endpoint_denoising = getattr(config, "endpoint_denoising", "none")
            denoised_xt = None
            if endpoint_denoising in {"pca", "knn", "metacell"}:
                if pca is None:
                    raise ValueError("Endpoint denoising requires a fitted PCA artifact")
                denoising_kwargs = {
                    "n_neighbors": int(getattr(config, "denoising_neighbors", 15)),
                    "n_metacells": int(getattr(config, "denoising_metacells", 512)),
                    "device": device,
                }
                target_x0 = torch.as_tensor(
                    denoise_selected_counts(
                        endpoint_denoising,
                        sampled.x0,
                        i0.cpu().numpy(),
                        pca,
                        seed=batch_seed * 2,
                        **denoising_kwargs,
                    ),
                    device=device,
                )
                target_x1 = torch.as_tensor(
                    denoise_selected_counts(
                        endpoint_denoising,
                        sampled.x1,
                        i1.cpu().numpy(),
                        pca,
                        seed=batch_seed * 2 + 1,
                        **denoising_kwargs,
                    ),
                    device=device,
                )
                denoised_xt, target, _target_tau = cfm_interpolate(
                    target_x0, target_x1, sampled.t0, sampled.t1, tau=tau
                )
            else:
                target = _raw_target
            if coordinates is not None:
                target = coordinates.to_model(target)
            if state_encoder is not None:
                state_t = state_encoder.encode(
                    uce_input_expression(config, xt, pca, denoised_xt), seed=batch_seed
                )
            elif sampled.state0 is None:
                state_t = coordinates.to_model(xt) if coordinates is not None else xt
            else:
                state0 = torch.as_tensor(sampled.state0, device=device)
                state1 = torch.as_tensor(sampled.state1, device=device)
                state_t = (1.0 - tau) * state0[i0] + tau * state1[i1]
            batches.append(
                FixedValidationBatch(
                    day0=day0,
                    day1=day1,
                    t0=sampled.t0,
                    t1=sampled.t1,
                    state_t=state_t.detach().cpu(),
                    target=target.detach().cpu(),
                    x0=paired_x0.detach().cpu(),
                    x1=paired_x1.detach().cpu(),
                )
            )
    return batches


def validation_velocity_scale(
    batches: list[FixedValidationBatch],
    loss_gene_indices: list[int] | np.ndarray | torch.Tensor | None = None,
) -> torch.Tensor:
    """Estimate per-gene target RMS with a robust floor for normalized MSE."""

    if not batches:
        raise ValueError("At least one validation batch is required")
    indices = _loss_gene_index_tensor(loss_gene_indices, torch.device("cpu"))
    targets = [batch.target if indices is None else batch.target.index_select(1, indices) for batch in batches]
    second_moment = torch.cat(targets).float().square().mean(dim=0)
    positive = second_moment[second_moment > 0]
    floor = max(float(torch.median(positive)) * 1e-3, 1e-6) if len(positive) else 1e-6
    return second_moment.clamp_min(floor).sqrt()


def _predict_loss_genes(
    model,
    state: torch.Tensor,
    cre_inputs: dict[str, torch.Tensor] | None,
    gene_chunk_size: int,
    loss_gene_indices: torch.Tensor | None,
) -> torch.Tensor:
    if cre_inputs is None:
        prediction = model(state)
        return prediction if loss_gene_indices is None else prediction.index_select(1, loss_gene_indices)
    if loss_gene_indices is None:
        return predict_stream_chunked(model, state, cre_inputs, gene_chunk_size)
    if gene_chunk_size <= 0 or gene_chunk_size >= len(loss_gene_indices):
        return model(state, **cre_inputs, gene_indices=loss_gene_indices)
    return torch.cat(
        [model(state, **cre_inputs, gene_indices=chunk) for chunk in loss_gene_indices.split(gene_chunk_size)], dim=1
    )


@torch.no_grad()
def evaluate_fixed_validation(
    model,
    batches: list[FixedValidationBatch],
    velocity_scale: torch.Tensor,
    cre_inputs: dict[str, torch.Tensor] | None,
    gene_chunk_size: int,
    loss_gene_indices: list[int] | np.ndarray | torch.Tensor | None = None,
) -> dict[str, float]:
    """Evaluate interval-balanced raw and velocity-scale-normalized CFM MSE."""

    device = next(model.parameters()).device
    indices = _loss_gene_index_tensor(loss_gene_indices, device)
    scale = velocity_scale.to(device)
    raw_losses = []
    normalized_losses = []
    model.eval()
    for batch in batches:
        state = batch.state_t.to(device)
        target = batch.target.to(device)
        if indices is not None:
            target = target.index_select(1, indices)
        prediction = _predict_loss_genes(model, state, cre_inputs, gene_chunk_size, indices)
        squared_error = (prediction - target).square()
        raw_losses.append(float(squared_error.mean().cpu()))
        normalized_losses.append(float((squared_error / scale.square()).mean().cpu()))
    return {
        "val_loss_raw": float(np.mean(raw_losses)),
        "val_loss_normalized": float(np.mean(normalized_losses)),
    }


@torch.no_grad()
def evaluate_observed_rollouts(
    config,
    model,
    batches: list[FixedValidationBatch],
    cre_inputs: dict[str, torch.Tensor] | None,
    state_encoder,
    steps: int,
    loss_gene_indices: list[int] | np.ndarray | torch.Tensor | None = None,
    pca=None,
    coordinates=None,
) -> dict[str, float]:
    """Diagnose autonomous endpoint prediction on observed validation intervals."""

    if state_encoder is None and config.cell_state != "expression":
        return {}
    device = next(model.parameters()).device
    indices = None if loss_gene_indices is None else np.asarray(loss_gene_indices, dtype=np.int64)
    interval_metrics = []
    model.eval()
    for interval_index, batch in enumerate(batches):
        x0 = batch.x0.to(device)
        model_x0 = coordinates.to_model(x0) if coordinates is not None else x0

        def velocity_fn(current_model_x, seed):
            current_x = (
                coordinates.to_expression(current_model_x)
                if coordinates is not None
                else current_model_x
            )
            state = (
                state_encoder.encode(uce_input_expression(config, current_x, pca), seed)
                if state_encoder is not None
                else current_model_x
            )
            return _predict_loss_genes(model, state, cre_inputs, config.gene_chunk_size, None)

        predicted_model = projected_euler_rollout(
            model_x0,
            batch.t0,
            batch.t1,
            velocity_fn,
            steps=steps,
            seed=int(config.seed) + interval_index * 10_000,
        )
        predicted = (
            coordinates.to_expression(predicted_model)
            if coordinates is not None
            else predicted_model
        )
        interval_metrics.append(
            mean_shift_metrics(batch.x0.numpy(), batch.x1.numpy(), predicted.cpu().numpy(), indices)
        )
    return {
        f"observed_rollout_{key}": float(np.nanmean([row[key] for row in interval_metrics]))
        for key in interval_metrics[0]
    }


def train_steps(
    config,
    sampler,
    model,
    optimizer,
    cre_inputs=None,
    steps_per_epoch: int = 100,
    wandb_run=None,
    loss_gene_indices: list[int] | np.ndarray | torch.Tensor | None = None,
    state_encoder=None,
    validation_batches: list[FixedValidationBatch] | None = None,
    validation_every_epochs: int = 1,
    early_stopping_patience: int = 12,
    early_stopping_min_delta: float = 0.005,
    validation_ema_alpha: float = 0.3,
    rollout_every_validations: int = 5,
    validation_rollout_steps: int = 4,
    checkpoint_callback: Callable | None = None,
    pca=None,
    coordinates=None,
) -> TrainingResult:
    device = next(model.parameters()).device
    loss_index_tensor = _loss_gene_index_tensor(loss_gene_indices, device)
    metrics: list[dict[str, float]] = []
    validation_metrics: list[dict[str, float]] = []
    velocity_scale = (
        validation_velocity_scale(validation_batches, loss_gene_indices) if validation_batches is not None else None
    )
    best_validation_loss = None
    best_epoch = None
    best_global_step = None
    validation_ema = None
    checks_without_improvement = 0
    validation_checks = 0
    stopped_early = False
    paired_pool = None
    pool_cursor = 0
    pool_refills = 0
    pairs_per_pool = max(int(config.batch_size), int(getattr(config, "ot_pairs_per_pool", 0)))
    pair_bank = (
        IntervalPairBank(config, sampler, device, pca=pca)
        if getattr(config, "ot_pair_bank_mode", "sequential") == "interval"
        else None
    )
    for epoch in range(config.epochs):
        model.train()
        for step in range(steps_per_epoch):
            global_step = epoch * steps_per_epoch + step
            denoised_xt = None
            if pair_bank is not None:
                microbatch = pair_bank.next()
                x0 = microbatch.raw_x0
                x1 = microbatch.raw_x1
                xt, _raw_target, tau = cfm_interpolate(x0, x1, microbatch.t0, microbatch.t1)
                denoised_xt, target, _target_tau = cfm_interpolate(
                    microbatch.target_x0,
                    microbatch.target_x1,
                    microbatch.t0,
                    microbatch.t1,
                    tau=tau,
                )
                batch_day0 = microbatch.day0
                batch_day1 = microbatch.day1
                pool_refill_index = microbatch.refresh
                pool_diagnostics = microbatch.diagnostics
                paired_state0 = microbatch.state0
                paired_state1 = microbatch.state1
            else:
                if paired_pool is None or pool_cursor + config.batch_size > pairs_per_pool:
                    batch = sampler.sample()
                    x0_pool = torch.as_tensor(batch.x0, device=device)
                    x1_pool = torch.as_tensor(batch.x1, device=device)
                    generator = torch.Generator(device=device).manual_seed(int(config.seed) + pool_refills * 100_003)
                    cost, coupling = transport_plan(x0_pool, x1_pool, **coupling_kwargs(config))
                    i0, i1 = sample_coupling_pairs(coupling, pairs_per_pool, generator=generator)
                    paired_pool = {
                        "x0": x0_pool[i0],
                        "x1": x1_pool[i1],
                        "state0": None,
                        "state1": None,
                        "t0": batch.t0,
                        "t1": batch.t1,
                        "day0": batch.day0,
                        "day1": batch.day1,
                        "diagnostics": coupling_diagnostics(cost, coupling),
                    }
                    if batch.state0 is not None:
                        state0_pool = torch.as_tensor(batch.state0, device=device)
                        state1_pool = torch.as_tensor(batch.state1, device=device)
                        paired_pool["state0"] = state0_pool[i0]
                        paired_pool["state1"] = state1_pool[i1]
                    pool_cursor = 0
                    pool_refills += 1
                end = pool_cursor + config.batch_size
                x0 = paired_pool["x0"][pool_cursor:end]
                x1 = paired_pool["x1"][pool_cursor:end]
                xt, target, tau = cfm_interpolate(x0, x1, paired_pool["t0"], paired_pool["t1"])
                batch_day0 = paired_pool["day0"]
                batch_day1 = paired_pool["day1"]
                pool_refill_index = pool_refills - 1
                pool_diagnostics = paired_pool["diagnostics"]
                paired_state0 = (
                    None if paired_pool["state0"] is None else paired_pool["state0"][pool_cursor:end]
                )
                paired_state1 = (
                    None if paired_pool["state1"] is None else paired_pool["state1"][pool_cursor:end]
                )
                pool_cursor = end
            if coordinates is not None:
                target = coordinates.to_model(target)
            if state_encoder is not None:
                state_t = state_encoder.encode(
                    uce_input_expression(config, xt, pca, denoised_xt),
                    seed=int(config.seed) + global_step * config.batch_size,
                )
            elif paired_state0 is None:
                state_t = coordinates.to_model(xt) if coordinates is not None else xt
            else:
                state_t = (1.0 - tau) * paired_state0 + tau * paired_state1
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
            row = {
                "epoch": epoch,
                "step": step,
                "loss": value,
                "day0": batch_day0,
                "day1": batch_day1,
                "ot_pool_refill": pool_refill_index,
                **pool_diagnostics,
            }
            metrics.append(row)
            if wandb_run is not None:
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
        should_validate = validation_batches is not None and (
            (epoch + 1) % validation_every_epochs == 0 or epoch + 1 == config.epochs
        )
        if not should_validate:
            continue
        validation_checks += 1
        current = evaluate_fixed_validation(
            model,
            validation_batches,
            velocity_scale,
            cre_inputs,
            config.gene_chunk_size,
            loss_gene_indices,
        )
        normalized = current["val_loss_normalized"]
        validation_ema = (
            normalized
            if validation_ema is None
            else validation_ema_alpha * normalized + (1.0 - validation_ema_alpha) * validation_ema
        )
        improved = best_validation_loss is None or validation_ema < best_validation_loss * (1.0 - early_stopping_min_delta)
        if improved:
            best_validation_loss = validation_ema
            best_epoch = epoch
            best_global_step = (epoch + 1) * steps_per_epoch
            checks_without_improvement = 0
        else:
            checks_without_improvement += 1
        row = {
            "epoch": epoch,
            "global_step": (epoch + 1) * steps_per_epoch,
            **current,
            "val_loss_normalized_ema": validation_ema,
            "best_val_loss_normalized_ema": best_validation_loss,
            "checks_without_improvement": checks_without_improvement,
            "is_best": int(improved),
        }
        run_rollout = rollout_every_validations > 0 and (
            validation_checks == 1
            or validation_checks % rollout_every_validations == 0
            or checks_without_improvement >= early_stopping_patience
            or epoch + 1 == config.epochs
        )
        if run_rollout:
            row.update(
                evaluate_observed_rollouts(
                    config,
                    model,
                    validation_batches,
                    cre_inputs,
                    state_encoder,
                    validation_rollout_steps,
                    loss_gene_indices,
                    pca,
                    coordinates,
                )
            )
        validation_metrics.append(row)
        if wandb_run is not None:
            wandb_run.log({f"validation/{key}": value for key, value in row.items()}, step=row["global_step"])
        if improved and checkpoint_callback is not None:
            checkpoint_callback(model, optimizer, row, metrics, validation_metrics)
        if checks_without_improvement >= early_stopping_patience:
            stopped_early = True
            break
    return TrainingResult(
        train_metrics=metrics,
        validation_metrics=validation_metrics,
        best_validation_loss=best_validation_loss,
        best_epoch=best_epoch,
        best_global_step=best_global_step,
        stopped_early=stopped_early,
    )
