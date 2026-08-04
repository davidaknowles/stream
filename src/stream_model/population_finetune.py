"""Population-level fine-tuning through explicit score-flow rollouts."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

import torch

from .ot import sinkhorn_coupling
from .score_flow import coupled_score_flow_fields


TRAINABLE_SCORE_FLOW_PREFIXES = (
    "time_mlp.",
    "conditional_velocity_head.",
    "noise_head.",
)


def configure_population_finetuning(model) -> list[torch.nn.Parameter]:
    """Freeze the representation and autonomous control, returning rollout parameters."""

    trainable = []
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith(TRAINABLE_SCORE_FLOW_PREFIXES))
        if parameter.requires_grad:
            trainable.append(parameter)
    if not trainable:
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
) -> torch.Tensor:
    cost = torch.cdist(x.float(), y.float()).square() / x.shape[1]
    with torch.no_grad():
        coupling = sinkhorn_coupling(cost.detach(), epsilon=epsilon, iterations=iterations)
    reference = cost.new_tensor(1.0 / (x.shape[0] * y.shape[0]))
    kl = torch.sum(coupling * (torch.log(coupling.clamp_min(1e-30)) - torch.log(reference)))
    return torch.sum(coupling * cost) + float(epsilon) * kl


def differentiable_sinkhorn_divergence(
    x: torch.Tensor,
    y: torch.Tensor,
    epsilon: float = 0.05,
    iterations: int = 80,
) -> torch.Tensor:
    """Debiased entropic OT with envelope gradients through endpoint coordinates."""

    if x.ndim != 2 or y.ndim != 2 or x.shape[1] != y.shape[1]:
        raise ValueError("x and y must be matrices with the same feature dimension")
    xy = _entropic_ot_with_detached_plan(x, y, epsilon, iterations)
    xx = _entropic_ot_with_detached_plan(x, x, epsilon, iterations)
    yy = _entropic_ot_with_detached_plan(y, y, epsilon, iterations)
    return xy - 0.5 * xx - 0.5 * yy


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
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compare independently sampled endpoint populations without cell pairing."""

    predicted_pca = pca.transform_tensor(predicted)
    observed_pca = pca.transform_tensor(observed)
    sinkhorn = differentiable_sinkhorn_divergence(
        predicted_pca, observed_pca, epsilon=sinkhorn_epsilon, iterations=sinkhorn_iterations
    )
    scale = gene_scale.to(predicted).reshape(-1).clamp_min(1e-3)
    mean = ((predicted.mean(0) - observed.mean(0)) / scale).square().mean()
    predicted_centered = predicted_pca - predicted_pca.mean(0)
    observed_centered = observed_pca - observed_pca.mean(0)
    predicted_cov = predicted_centered.T @ predicted_centered / max(len(predicted_pca) - 1, 1)
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
