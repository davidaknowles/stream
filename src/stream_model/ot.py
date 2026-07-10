"""Minibatch OT and CFM objective utilities."""

from __future__ import annotations

import torch


def pairwise_squared_cost(x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
    x0n = torch.nn.functional.normalize(x0.float(), dim=1)
    x1n = torch.nn.functional.normalize(x1.float(), dim=1)
    return torch.cdist(x0n, x1n, p=2).pow(2)


def sinkhorn_coupling(
    cost: torch.Tensor,
    epsilon: float = 0.05,
    iterations: int = 80,
) -> torch.Tensor:
    """Compute an entropic OT coupling with uniform marginals."""

    if cost.ndim != 2:
        raise ValueError("cost must be a matrix")
    n, m = cost.shape
    log_a = cost.new_full((n,), -torch.log(torch.tensor(float(n), device=cost.device)))
    log_b = cost.new_full((m,), -torch.log(torch.tensor(float(m), device=cost.device)))
    log_k = -cost / epsilon
    u = torch.zeros_like(log_a)
    v = torch.zeros_like(log_b)
    for _ in range(iterations):
        u = log_a - torch.logsumexp(log_k + v.unsqueeze(0), dim=1)
        v = log_b - torch.logsumexp(log_k + u.unsqueeze(1), dim=0)
    return torch.exp(log_k + u.unsqueeze(1) + v.unsqueeze(0))


def sample_coupling_pairs(coupling: torch.Tensor, n_pairs: int, generator: torch.Generator | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    flat = coupling.reshape(-1)
    flat = flat / flat.sum()
    idx = torch.multinomial(flat, n_pairs, replacement=True, generator=generator)
    return idx // coupling.shape[1], idx % coupling.shape[1]


def cfm_interpolate(
    x0: torch.Tensor,
    x1: torch.Tensor,
    t0: float | torch.Tensor,
    t1: float | torch.Tensor,
    tau: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if tau is None:
        tau = torch.rand((x0.shape[0], 1), device=x0.device, dtype=x0.dtype)
    elif tau.ndim == 1:
        tau = tau[:, None]
    dt = torch.as_tensor(t1, device=x0.device, dtype=x0.dtype) - torch.as_tensor(t0, device=x0.device, dtype=x0.dtype)
    if torch.any(dt <= 0):
        raise ValueError("t1 must be greater than t0")
    xt = (1.0 - tau) * x0 + tau * x1
    target = (x1 - x0) / dt
    return xt, target, tau


def ot_cfm_batch(
    x0: torch.Tensor,
    x1: torch.Tensor,
    t0: float,
    t1: float,
    epsilon: float = 0.05,
    iterations: int = 80,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        coupling = sinkhorn_coupling(pairwise_squared_cost(x0, x1), epsilon=epsilon, iterations=iterations)
        i0, i1 = sample_coupling_pairs(coupling, min(x0.shape[0], x1.shape[0]))
    return cfm_interpolate(x0[i0], x1[i1], t0, t1)


def ot_cfm_batch_with_state(
    x0: torch.Tensor,
    x1: torch.Tensor,
    state0: torch.Tensor,
    state1: torch.Tensor,
    t0: float,
    t1: float,
    epsilon: float = 0.05,
    iterations: int = 80,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Couple expression states by OT and interpolate an aligned state representation.

    OT and the CFM velocity target stay in expression space. ``state0`` and
    ``state1`` are row-aligned auxiliary cell representations, such as UCE,
    which are interpolated with the same sampled OT pairs and CFM time.
    """

    if state0.shape[0] != x0.shape[0] or state1.shape[0] != x1.shape[0]:
        raise ValueError("Auxiliary states must align with expression batch rows")
    with torch.no_grad():
        coupling = sinkhorn_coupling(pairwise_squared_cost(x0, x1), epsilon=epsilon, iterations=iterations)
        i0, i1 = sample_coupling_pairs(coupling, min(x0.shape[0], x1.shape[0]))
    xt, target, tau = cfm_interpolate(x0[i0], x1[i1], t0, t1)
    state_t = (1.0 - tau) * state0[i0] + tau * state1[i1]
    return xt, target, tau, state_t
