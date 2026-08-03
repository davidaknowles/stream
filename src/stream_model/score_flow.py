"""Simulation-free stochastic-interpolant targets and score-flow rollout."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

from .pair_bank import IntervalPairBank
from .train import uce_input_expression


def robust_gene_scale(x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
    """Estimate fixed per-gene count scales with a robust nonzero floor."""

    values = torch.cat([x0.float(), x1.float()])
    scale = values.std(dim=0, unbiased=False)
    positive = scale[scale > 0]
    floor = positive.median() * 0.05 if len(positive) else scale.new_tensor(1.0)
    return scale.clamp_min(max(float(floor), 1e-3))


def stochastic_interpolant(
    x0: torch.Tensor,
    x1: torch.Tensor,
    gene_scale: torch.Tensor,
    t0: float,
    t1: float,
    noise_amplitude: float = 0.2,
    tau_min: float = 0.02,
    generator: torch.Generator | None = None,
    tau: torch.Tensor | None = None,
    noise: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return noisy states, endpoint-velocity targets, noise, and path position."""

    if t1 <= t0:
        raise ValueError("t1 must exceed t0")
    if not 0 <= tau_min < 0.5:
        raise ValueError("tau_min must be in [0, 0.5)")
    batch = x0.shape[0]
    if tau is None:
        tau = torch.rand((batch, 1), device=x0.device, generator=generator)
        tau = tau_min + (1.0 - 2.0 * tau_min) * tau
    else:
        tau = tau.reshape(batch, 1).to(device=x0.device, dtype=x0.dtype)
    if noise is None:
        noise = torch.randn(x0.shape, device=x0.device, dtype=x0.dtype, generator=generator)
    scale = gene_scale.to(device=x0.device, dtype=x0.dtype).reshape(1, -1)
    gamma = float(noise_amplitude) * torch.sin(math.pi * tau)
    xt = (1.0 - tau) * x0 + tau * x1 + gamma * scale * noise
    velocity = (x1 - x0) / float(t1 - t0)
    # Keep the Gaussian bridge exact. UCE tokenization naturally ignores nonpositive
    # perturbed entries, while projected rollout remains nonnegative.
    return xt, velocity, noise, tau


def coupled_score_flow_fields(
    prediction: torch.Tensor,
    tau: torch.Tensor,
    gene_scale: torch.Tensor,
    interval_duration: float,
    noise_amplitude: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Construct autonomous velocity, coupled flow, and standardized score."""

    if interval_duration <= 0:
        raise ValueError("interval_duration must be positive")
    tau = tau.reshape(-1, 1).to(prediction)
    scale = gene_scale.to(prediction).reshape(1, -1)
    gamma = (float(noise_amplitude) * torch.sin(math.pi * tau)).clamp_min(1e-3)
    gamma_prime = float(noise_amplitude) * math.pi * torch.cos(math.pi * tau)
    autonomous_velocity = prediction[..., 0]
    conditional_velocity = prediction[..., 1]
    noise = prediction[..., 2]
    coupled_flow = conditional_velocity + gamma_prime * scale * noise / float(interval_duration)
    standardized_score = -noise / gamma
    return autonomous_velocity, coupled_flow, standardized_score


def score_flow_loss(
    prediction: torch.Tensor,
    velocity_target: torch.Tensor,
    noise_target: torch.Tensor,
    flow_scale: torch.Tensor,
    tau: torch.Tensor,
    gene_scale: torch.Tensor,
    interval_duration: float,
    noise_amplitude: float,
    score_weight: float = 1.0,
    autonomous_weight: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Train shared conditional velocity/noise fields and the autonomous control."""

    autonomous_velocity, coupled_flow, _score = coupled_score_flow_fields(
        prediction, tau, gene_scale, interval_duration, noise_amplitude
    )
    scale_values = gene_scale.to(prediction).reshape(1, -1)
    gamma_prime = float(noise_amplitude) * math.pi * torch.cos(math.pi * tau.reshape(-1, 1).to(prediction))
    coupled_flow_target = velocity_target + gamma_prime * scale_values * noise_target / float(interval_duration)
    noise_prediction = prediction[..., 2]
    scale = flow_scale.to(prediction).reshape(1, -1).clamp_min(1e-6)
    coupled_scale = torch.sqrt(
        scale.square() + (gamma_prime * scale_values / float(interval_duration)).square()
    )
    autonomous_loss = ((autonomous_velocity - velocity_target) / scale).square().mean()
    flow_loss = ((coupled_flow - coupled_flow_target) / coupled_scale).square().mean()
    score_loss = (noise_prediction - noise_target).square().mean()
    loss = float(autonomous_weight) * autonomous_loss + flow_loss + float(score_weight) * score_loss
    return loss, {
        "loss": float(loss.detach().cpu()),
        "autonomous_velocity_loss_normalized": float(autonomous_loss.detach().cpu()),
        "coupled_flow_loss_normalized": float(flow_loss.detach().cpu()),
        "score_noise_mse": float(score_loss.detach().cpu()),
    }


def predict_score_flow_chunked(model, state, tau, cre_inputs, gene_chunk_size):
    """Predict both heads in gene chunks to bound activation memory."""

    n_genes = int(cre_inputs["cre_embeddings"].shape[0])
    if gene_chunk_size <= 0 or gene_chunk_size >= n_genes:
        return model(state, tau, **cre_inputs)
    chunks = []
    for start in range(0, n_genes, gene_chunk_size):
        indices = torch.arange(start, min(start + gene_chunk_size, n_genes), device=state.device)
        chunks.append(model(state, tau, **cre_inputs, gene_indices=indices))
    return torch.cat(chunks, dim=1)


def calibrate_scales(pair_bank: IntervalPairBank, n_batches: int = 64) -> tuple[torch.Tensor, torch.Tensor]:
    """Estimate perturbation and velocity scales from training-only paired endpoints."""

    endpoints = []
    velocities = []
    for _ in range(n_batches):
        batch = pair_bank.next()
        endpoints.extend([batch.target_x0.detach().cpu(), batch.target_x1.detach().cpu()])
        velocities.append(((batch.target_x1 - batch.target_x0) / (batch.t1 - batch.t0)).detach().cpu())
    x = torch.cat(endpoints)
    velocity = torch.cat(velocities)
    gene_scale = robust_gene_scale(x[: len(x) // 2], x[len(x) // 2 :])
    flow_scale = velocity.square().mean(dim=0).sqrt()
    positive = flow_scale[flow_scale > 0]
    # Match a 1e-3 second-moment floor: scale is the square root of that moment.
    floor = positive.median() * math.sqrt(1e-3) if len(positive) else flow_scale.new_tensor(1.0)
    return gene_scale, flow_scale.clamp_min(max(float(floor), 1e-4))


@dataclass(frozen=True)
class ScoreFlowTrainingResult:
    metrics: list[dict[str, float]]
    gene_scale: torch.Tensor
    flow_scale: torch.Tensor
    stopped_early: bool


@torch.no_grad()
def evaluate_score_flow_bank(config, bank, model, cre_inputs, state_encoder, pca, gene_scale, flow_scale):
    """Evaluate one deterministic stochastic-interpolant batch per interval."""

    model.eval()
    rows = []
    generator = torch.Generator(device=gene_scale.device).manual_seed(int(config.seed) + 55_901)
    for _ in bank.intervals:
        batch = bank.next()
        xt, velocity_target, noise_target, tau = stochastic_interpolant(
            batch.target_x0,
            batch.target_x1,
            gene_scale,
            batch.t0,
            batch.t1,
            noise_amplitude=config.score_flow_noise_scale,
            tau_min=config.score_flow_tau_min,
            generator=generator,
        )
        state = state_encoder.encode(uce_input_expression(config, xt, pca, xt), seed=int(config.seed) + 55_901)
        prediction = predict_score_flow_chunked(model, state, tau, cre_inputs, config.gene_chunk_size)
        _loss, row = score_flow_loss(
            prediction,
            velocity_target,
            noise_target,
            flow_scale,
            tau,
            gene_scale,
            batch.t1 - batch.t0,
            config.score_flow_noise_scale,
            config.score_flow_score_weight,
            config.score_flow_autonomous_weight,
        )
        rows.append(row)
    return {
        key: float(np.mean([row[key] for row in rows]))
        for key in (
            "loss",
            "autonomous_velocity_loss_normalized",
            "coupled_flow_loss_normalized",
            "score_noise_mse",
        )
    }


def train_score_flow_steps(
    config,
    sampler,
    model,
    optimizer,
    cre_inputs,
    state_encoder,
    pca,
    steps_per_epoch: int,
    validation_sampler=None,
    patience: int = 12,
    min_delta: float = 0.005,
    checkpoint_callback=None,
) -> ScoreFlowTrainingResult:
    """Fit flow and denoising-score heads from an interval-stratified OT bank."""

    device = next(model.parameters()).device
    bank = IntervalPairBank(config, sampler, device, pca=pca)
    validation_bank = (
        None if validation_sampler is None else IntervalPairBank(config, validation_sampler, device, pca=pca)
    )
    gene_scale, flow_scale = calibrate_scales(bank)
    gene_scale = gene_scale.to(device)
    flow_scale = flow_scale.to(device)
    metrics = []
    best_ema = None
    ema = None
    checks_without_improvement = 0
    stopped_early = False
    for epoch in range(config.epochs):
        model.train()
        epoch_rows = []
        for step in range(steps_per_epoch):
            batch = bank.next()
            xt, velocity_target, noise_target, tau = stochastic_interpolant(
                batch.target_x0,
                batch.target_x1,
                gene_scale,
                batch.t0,
                batch.t1,
                noise_amplitude=config.score_flow_noise_scale,
                tau_min=config.score_flow_tau_min,
            )
            state = state_encoder.encode(
                uce_input_expression(config, xt, pca, xt),
                seed=int(config.seed) + epoch * steps_per_epoch + step,
            )
            optimizer.zero_grad(set_to_none=True)
            total = 0.0
            autonomous_total = 0.0
            flow_total = 0.0
            score_total = 0.0
            n_genes = velocity_target.shape[1]
            chunk_size = max(1, int(config.gene_chunk_size))
            for start in range(0, n_genes, chunk_size):
                indices = torch.arange(start, min(start + chunk_size, n_genes), device=device)
                prediction = model(state, tau, **cre_inputs, gene_indices=indices)
                loss, row = score_flow_loss(
                    prediction,
                    velocity_target.index_select(1, indices),
                    noise_target.index_select(1, indices),
                    flow_scale.index_select(0, indices),
                    tau,
                    gene_scale.index_select(0, indices),
                    batch.t1 - batch.t0,
                    config.score_flow_noise_scale,
                    config.score_flow_score_weight,
                    config.score_flow_autonomous_weight,
                )
                weight = len(indices) / n_genes
                (loss * weight).backward()
                total += row["loss"] * weight
                autonomous_total += row["autonomous_velocity_loss_normalized"] * weight
                flow_total += row["coupled_flow_loss_normalized"] * weight
                score_total += row["score_noise_mse"] * weight
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_rows.append((total, autonomous_total, flow_total, score_total))
        mean_loss, mean_autonomous, mean_flow, mean_score = np.mean(epoch_rows, axis=0).tolist()
        validation = (
            {
                "loss": mean_loss,
                "autonomous_velocity_loss_normalized": mean_autonomous,
                "coupled_flow_loss_normalized": mean_flow,
                "score_noise_mse": mean_score,
            }
            if validation_bank is None
            else evaluate_score_flow_bank(
                config, validation_bank, model, cre_inputs, state_encoder, pca, gene_scale, flow_scale
            )
        )
        validation_loss = validation["loss"]
        ema = validation_loss if ema is None else 0.3 * validation_loss + 0.7 * ema
        row = {
            "epoch": epoch,
            "loss": mean_loss,
            "autonomous_velocity_loss_normalized": mean_autonomous,
            "coupled_flow_loss_normalized": mean_flow,
            "score_noise_mse": mean_score,
            "validation_loss": validation_loss,
            "validation_autonomous_velocity_loss_normalized": validation[
                "autonomous_velocity_loss_normalized"
            ],
            "validation_coupled_flow_loss_normalized": validation["coupled_flow_loss_normalized"],
            "validation_score_noise_mse": validation["score_noise_mse"],
            "validation_loss_ema": ema,
        }
        metrics.append(row)
        if best_ema is None or ema < best_ema * (1.0 - min_delta):
            best_ema = ema
            checks_without_improvement = 0
            if checkpoint_callback is not None:
                checkpoint_callback(model, optimizer, row, metrics, gene_scale, flow_scale)
        else:
            checks_without_improvement += 1
        if checks_without_improvement >= patience:
            stopped_early = True
            break
    return ScoreFlowTrainingResult(metrics, gene_scale.detach().cpu(), flow_scale.detach().cpu(), stopped_early)


@torch.no_grad()
def score_flow_rollout(
    x0: torch.Tensor,
    t0: float,
    t1: float,
    predict_fn,
    gene_scale: torch.Tensor,
    steps: int,
    seed: int,
    diffusion: float = 0.0,
    noise_amplitude: float = 0.2,
    dynamics_mode: str = "coupled",
    score_control: str = "learned",
) -> torch.Tensor:
    """Roll out autonomous or analytically coupled score-flow dynamics."""

    if steps <= 0 or t1 <= t0 or diffusion < 0:
        raise ValueError("Require positive horizon/steps and nonnegative diffusion")
    if dynamics_mode not in {"autonomous", "coupled"}:
        raise ValueError("dynamics_mode must be autonomous or coupled")
    if score_control not in {"learned", "zero"}:
        raise ValueError("score_control must be learned or zero")
    if dynamics_mode == "autonomous" and diffusion > 0:
        raise ValueError("The autonomous control is deterministic")
    x = x0.clone()
    dt = float(t1 - t0) / steps
    generator = torch.Generator(device=x.device).manual_seed(seed)
    scale = gene_scale.to(x).reshape(1, -1)
    for step in range(steps):
        tau = x.new_full((len(x), 1), (step + 0.5) / steps)
        prediction = predict_fn(x, tau, seed)
        autonomous_velocity, coupled_flow, standardized_score = coupled_score_flow_fields(
            prediction, tau, scale.flatten(), t1 - t0, noise_amplitude
        )
        drift = autonomous_velocity if dynamics_mode == "autonomous" else coupled_flow
        if diffusion > 0:
            if score_control == "learned":
                drift = drift + diffusion * scale * standardized_score
            noise = torch.randn(x.shape, device=x.device, dtype=x.dtype, generator=generator)
            x = x + dt * drift + math.sqrt(2.0 * diffusion * dt) * scale * noise
        else:
            x = x + dt * drift
        x.clamp_(min=0.0)
    return x
