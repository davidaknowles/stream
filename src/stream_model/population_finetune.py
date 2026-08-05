"""Population-level fine-tuning through explicit score-flow rollouts."""

from __future__ import annotations

import math
from dataclasses import dataclass
from collections.abc import Callable, Sequence

import torch

from .ot import sinkhorn_coupling
from .score_flow import coupled_score_flow_fields


TRAINABLE_SCORE_FLOW_PREFIXES = (
    "time_mlp.",
    "conditional_velocity_head.",
    "noise_head.",
)


def configure_population_finetuning(
    model, include_dynamics: bool = True
) -> list[torch.nn.Parameter]:
    """Freeze the representation and autonomous control, returning rollout parameters."""

    trainable = []
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(
            include_dynamics and name.startswith(TRAINABLE_SCORE_FLOW_PREFIXES)
        )
        if parameter.requires_grad:
            trainable.append(parameter)
    if include_dynamics and not trainable:
        raise ValueError("No score-flow parameters were selected for population fine-tuning")
    return trainable


def snapshot_parameters(parameters: Sequence[torch.nn.Parameter]) -> list[torch.Tensor]:
    """Copy pretrained values used for an anti-forgetting anchor."""

    return [parameter.detach().clone() for parameter in parameters]


def parameter_anchor_loss(
    parameters: Sequence[torch.nn.Parameter], references: Sequence[torch.Tensor]
) -> torch.Tensor:
    if len(parameters) != len(references) or not parameters:
        raise ValueError("Parameters and references must be nonempty and aligned")
    return torch.stack([(parameter - reference).square().mean() for parameter, reference in zip(parameters, references)]).mean()


def _entropic_ot_with_detached_plan(
    x: torch.Tensor,
    y: torch.Tensor,
    epsilon: float,
    iterations: int,
    source_weights: torch.Tensor | None = None,
    target_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    cost = torch.cdist(x.float(), y.float()).square() / x.shape[1]
    source_weights = (
        cost.new_full((len(x),), 1.0 / len(x))
        if source_weights is None
        else source_weights.to(cost) / source_weights.sum()
    )
    target_weights = (
        cost.new_full((len(y),), 1.0 / len(y))
        if target_weights is None
        else target_weights.to(cost) / target_weights.sum()
    )
    with torch.no_grad():
        if source_weights.requires_grad or target_weights.requires_grad:
            log_a = torch.log(source_weights.detach())
            log_b = torch.log(target_weights.detach())
            log_k = -cost.detach() / float(epsilon)
            u = torch.zeros_like(log_a)
            v = torch.zeros_like(log_b)
            for _ in range(iterations):
                u = log_a - torch.logsumexp(log_k + v.unsqueeze(0), dim=1)
                v = log_b - torch.logsumexp(log_k + u.unsqueeze(1), dim=0)
            coupling = torch.exp(log_k + u.unsqueeze(1) + v.unsqueeze(0))
            source_potential = float(epsilon) * (u - log_a)
            target_potential = float(epsilon) * (v - log_b)
        else:
            coupling = sinkhorn_coupling(
                cost.detach(),
                epsilon=epsilon,
                iterations=iterations,
                source_marginal=source_weights,
                target_marginal=target_weights,
            )
            source_potential = target_potential = None
    reference = source_weights.detach()[:, None] * target_weights.detach()[None, :]
    kl = torch.sum(
        coupling
        * (torch.log(coupling.clamp_min(1e-30)) - torch.log(reference))
    )
    objective = torch.sum(coupling * cost) + float(epsilon) * kl
    if source_weights.requires_grad:
        objective = objective + torch.sum(
            (source_weights - source_weights.detach()) * source_potential
        )
    if target_weights.requires_grad:
        objective = objective + torch.sum(
            (target_weights - target_weights.detach()) * target_potential
        )
    return objective


def differentiable_sinkhorn_divergence(
    x: torch.Tensor,
    y: torch.Tensor,
    epsilon: float = 0.05,
    iterations: int = 80,
    x_weights: torch.Tensor | None = None,
    y_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Debiased entropic OT with envelope gradients through endpoint coordinates."""

    if x.ndim != 2 or y.ndim != 2 or x.shape[1] != y.shape[1]:
        raise ValueError("x and y must be matrices with the same feature dimension")
    xy = _entropic_ot_with_detached_plan(
        x, y, epsilon, iterations, x_weights, y_weights
    )
    xx = _entropic_ot_with_detached_plan(
        x, x, epsilon, iterations, x_weights, x_weights
    )
    yy = _entropic_ot_with_detached_plan(
        y, y, epsilon, iterations, y_weights, y_weights
    )
    return xy - 0.5 * xx - 0.5 * yy


@dataclass(frozen=True)
class GrowthRolloutResult:
    state: torch.Tensor
    weights: torch.Tensor
    growth_rate_mean_square: torch.Tensor
    growth_rate_rms: torch.Tensor
    weight_kl: torch.Tensor
    effective_sample_size: torch.Tensor


def differentiable_growth_rollout(
    x0: torch.Tensor,
    t0: float,
    t1: float,
    predict_growth_fn: Callable[
        [torch.Tensor, torch.Tensor, int], tuple[torch.Tensor | None, torch.Tensor]
    ],
    gene_scale: torch.Tensor,
    steps: int,
    seed: int,
    diffusion: float,
    noise_amplitude: float,
    particles: int = 1,
    dynamics_mode: str = "coupled",
    score_control: str = "learned",
    max_growth_rate: float = 4.0,
    brownian_noise: Sequence[torch.Tensor] | None = None,
) -> GrowthRolloutResult:
    """Jointly integrate expression states and normalized relative-growth weights."""

    if dynamics_mode not in {"none", "coupled"}:
        raise ValueError("dynamics_mode must be none or coupled")
    if score_control not in {"learned", "zero"}:
        raise ValueError("score_control must be learned or zero")
    if steps <= 0 or particles <= 0 or t1 <= t0 or diffusion < 0:
        raise ValueError("Require positive steps/particles/horizon and nonnegative diffusion")
    if max_growth_rate <= 0:
        raise ValueError("max_growth_rate must be positive")
    x = x0.repeat_interleave(particles, dim=0)
    log_weights = x.new_full((len(x),), -math.log(len(x)))
    dt = float(t1 - t0) / steps
    scale = gene_scale.to(x).reshape(1, -1)
    generator = torch.Generator(device=x.device).manual_seed(seed)
    squared_rates = []
    if brownian_noise is not None and len(brownian_noise) != steps:
        raise ValueError("brownian_noise must contain one tensor per integration step")
    for step in range(steps):
        tau = x.new_full((len(x), 1), (step + 0.5) / steps)
        prediction, growth_logits = predict_growth_fn(x, tau, seed + step)
        weights = torch.softmax(log_weights, dim=0)
        rates = float(max_growth_rate) * torch.tanh(growth_logits.reshape(-1))
        rates = rates - torch.sum(weights * rates)
        squared_rates.append(torch.sum(weights * rates.square()))
        log_weights = log_weights + dt * rates
        log_weights = log_weights - torch.logsumexp(log_weights, dim=0)
        if dynamics_mode == "none":
            continue
        if prediction is None:
            raise ValueError("A score-flow prediction is required when dynamics are enabled")
        _autonomous, flow, score = coupled_score_flow_fields(
            prediction, tau, gene_scale, t1 - t0, noise_amplitude
        )
        if score_control == "zero":
            score = torch.zeros_like(score)
        drift = flow + float(diffusion) * scale * score
        if diffusion > 0:
            noise = (
                brownian_noise[step].to(x)
                if brownian_noise is not None
                else torch.randn(
                    x.shape, device=x.device, dtype=x.dtype, generator=generator
                )
            )
            x = x + dt * drift + math.sqrt(2.0 * diffusion * dt) * scale * noise
        else:
            x = x + dt * drift
        x = torch.clamp(x, min=0.0)
    weights = torch.softmax(log_weights, dim=0)
    weight_kl = torch.sum(
        weights * (torch.log(weights.clamp_min(1e-30)) + math.log(len(weights)))
    ).clamp_min(0.0)
    growth_rate_mean_square = torch.stack(squared_rates).mean()
    return GrowthRolloutResult(
        state=x,
        weights=weights,
        growth_rate_mean_square=growth_rate_mean_square,
        growth_rate_rms=growth_rate_mean_square.detach().sqrt(),
        weight_kl=weight_kl,
        effective_sample_size=weights.square().sum().reciprocal(),
    )


def differentiable_score_flow_rollout(
    x0: torch.Tensor,
    t0: float,
    t1: float,
    predict_fn: Callable[[torch.Tensor, torch.Tensor, int], torch.Tensor],
    gene_scale: torch.Tensor,
    steps: int,
    seed: int,
    diffusion: float,
    noise_amplitude: float,
    particles: int = 1,
    brownian_noise: Sequence[torch.Tensor] | None = None,
) -> torch.Tensor:
    """Euler-Maruyama rollout retaining gradients through dynamical-coordinate updates.

    ``predict_fn`` may intentionally detach its UCE boundary input. The
    additive state update still carries endpoint gradients across rollout steps.
    """

    if steps <= 0 or particles <= 0 or t1 <= t0 or diffusion < 0:
        raise ValueError("Require positive steps/particles/horizon and nonnegative diffusion")
    x = x0.repeat_interleave(particles, dim=0)
    dt = float(t1 - t0) / steps
    scale = gene_scale.to(x).reshape(1, -1)
    generator = torch.Generator(device=x.device).manual_seed(seed)
    if brownian_noise is not None and len(brownian_noise) != steps:
        raise ValueError("brownian_noise must contain one tensor per integration step")
    for step in range(steps):
        tau = x.new_full((len(x), 1), (step + 0.5) / steps)
        prediction = predict_fn(x, tau, seed + step)
        _autonomous, flow, score = coupled_score_flow_fields(
            prediction, tau, gene_scale, t1 - t0, noise_amplitude
        )
        drift = flow + float(diffusion) * scale * score
        if diffusion > 0:
            noise = (
                brownian_noise[step].to(x)
                if brownian_noise is not None
                else torch.randn(x.shape, device=x.device, dtype=x.dtype, generator=generator)
            )
            if noise.shape != x.shape:
                raise ValueError("Brownian noise shape does not match rollout particles")
            x = x + dt * drift + math.sqrt(2.0 * diffusion * dt) * scale * noise
        else:
            x = x + dt * drift
        x = torch.clamp(x, min=0.0)
    return x


def population_endpoint_loss(
    predicted: torch.Tensor,
    observed: torch.Tensor,
    source: torch.Tensor,
    pca,
    gene_scale: torch.Tensor,
    sinkhorn_epsilon: float = 0.05,
    sinkhorn_iterations: int = 80,
    mean_weight: float = 0.1,
    covariance_weight: float = 0.01,
    predicted_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compare independently sampled endpoint populations without cell pairing."""

    predicted_pca = pca.transform_tensor(predicted)
    observed_pca = pca.transform_tensor(observed)
    sinkhorn = differentiable_sinkhorn_divergence(
        predicted_pca,
        observed_pca,
        epsilon=sinkhorn_epsilon,
        iterations=sinkhorn_iterations,
        x_weights=predicted_weights,
    )
    scale = gene_scale.to(predicted).reshape(-1).clamp_min(1e-3)
    if predicted_weights is None:
        predicted_mean = predicted.mean(0)
        predicted_pca_mean = predicted_pca.mean(0)
    else:
        normalized_weights = predicted_weights / predicted_weights.sum()
        predicted_mean = torch.sum(normalized_weights[:, None] * predicted, dim=0)
        predicted_pca_mean = torch.sum(
            normalized_weights[:, None] * predicted_pca, dim=0
        )
    mean = ((predicted_mean - observed.mean(0)) / scale).square().mean()
    predicted_centered = predicted_pca - predicted_pca_mean
    observed_centered = observed_pca - observed_pca.mean(0)
    if predicted_weights is None:
        predicted_cov = (
            predicted_centered.T @ predicted_centered / max(len(predicted_pca) - 1, 1)
        )
    else:
        predicted_cov = (
            predicted_centered.T
            @ (normalized_weights[:, None] * predicted_centered)
        ) / (1.0 - normalized_weights.square().sum()).clamp_min(1e-6)
    observed_cov = observed_centered.T @ observed_centered / max(len(observed_pca) - 1, 1)
    covariance = (predicted_cov - observed_cov).square().mean()
    total = sinkhorn + float(mean_weight) * mean + float(covariance_weight) * covariance
    source_sinkhorn = differentiable_sinkhorn_divergence(
        pca.transform_tensor(source), observed_pca, epsilon=sinkhorn_epsilon, iterations=sinkhorn_iterations
    )
    return total, {
        "endpoint_loss": total,
        "sinkhorn": sinkhorn,
        "persistence_sinkhorn": source_sinkhorn,
        "sinkhorn_skill": 1.0 - sinkhorn / source_sinkhorn.clamp_min(1e-8),
        "gene_mean_loss": mean,
        "pca_covariance_loss": covariance,
    }
