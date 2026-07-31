"""Causal expression-space rollout and endpoint distribution metrics."""

from __future__ import annotations

import numpy as np
import torch

from .ot import sinkhorn_coupling


@torch.no_grad()
def projected_euler_rollout(
    x0: torch.Tensor,
    t0: float,
    t1: float,
    velocity_fn,
    steps: int,
    seed: int,
) -> torch.Tensor:
    if steps <= 0:
        raise ValueError("steps must be positive")
    if t1 <= t0:
        raise ValueError("t1 must exceed t0")
    x = x0.clone()
    step_size = float(t1 - t0) / steps
    for _ in range(steps):
        x = torch.clamp(x + step_size * velocity_fn(x, seed), min=0.0)
    return x


def r2_score(target: np.ndarray, prediction: np.ndarray) -> float:
    target = np.asarray(target, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    denominator = np.square(target - target.mean()).sum()
    if denominator == 0:
        return float("nan")
    return float(1.0 - np.square(prediction - target).sum() / denominator)


def mean_shift_metrics(
    x0: np.ndarray,
    x1: np.ndarray,
    predicted_x1: np.ndarray,
    indices: np.ndarray | list[int] | None = None,
) -> dict[str, float]:
    if indices is not None:
        x0 = x0[:, indices]
        x1 = x1[:, indices]
        predicted_x1 = predicted_x1[:, indices]
    observed_shift = x1.mean(axis=0) - x0.mean(axis=0)
    predicted_shift = predicted_x1.mean(axis=0) - x0.mean(axis=0)
    sorted_observed = np.sort(x1, axis=0)
    sorted_predicted = np.sort(predicted_x1, axis=0)
    return {
        "mean_shift_r2": r2_score(observed_shift, predicted_shift),
        "mean_shift_mae": float(np.mean(np.abs(observed_shift - predicted_shift))),
        "mean_gene_wasserstein1": float(np.mean(np.abs(sorted_observed - sorted_predicted))),
    }


def entropic_ot_cost(x: torch.Tensor, y: torch.Tensor, epsilon: float, iterations: int = 200) -> torch.Tensor:
    cost = torch.cdist(x.double(), y.double()).square() / x.shape[1]
    coupling = sinkhorn_coupling(cost, epsilon=epsilon, iterations=iterations).double()
    a = coupling.new_full((x.shape[0],), 1.0 / x.shape[0])
    b = coupling.new_full((y.shape[0],), 1.0 / y.shape[0])
    reference = a[:, None] * b[None, :]
    kl = torch.sum(coupling * (torch.log(coupling.clamp_min(1e-300)) - torch.log(reference)))
    return torch.sum(coupling * cost) + epsilon * kl


def sinkhorn_divergence(
    x: np.ndarray,
    y: np.ndarray,
    epsilon: float,
    iterations: int = 200,
) -> float:
    xt = torch.as_tensor(x, dtype=torch.float64)
    yt = torch.as_tensor(y, dtype=torch.float64)
    xy = entropic_ot_cost(xt, yt, epsilon, iterations)
    xx = entropic_ot_cost(xt, xt, epsilon, iterations)
    yy = entropic_ot_cost(yt, yt, epsilon, iterations)
    return float(torch.clamp(xy - 0.5 * xx - 0.5 * yy, min=0.0))
