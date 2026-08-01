"""Minibatch OT and CFM objective utilities."""

from __future__ import annotations

import torch


def coupling_kwargs(config) -> dict[str, float | int | str]:
    """Return a backwards-compatible coupling configuration."""

    return {
        "epsilon": config.ot_epsilon,
        "iterations": config.ot_iterations,
        "method": getattr(config, "ot_method", "balanced"),
        "partial_mass": getattr(config, "ot_partial_mass", 0.95),
        "marginal_relaxation": getattr(config, "ot_marginal_relaxation", 0.1),
    }


def pairwise_squared_cost(x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
    x0n = torch.nn.functional.normalize(x0.float(), dim=1)
    x1n = torch.nn.functional.normalize(x1.float(), dim=1)
    return torch.cdist(x0n, x1n, p=2).pow(2)


def sinkhorn_coupling(
    cost: torch.Tensor,
    epsilon: float = 0.05,
    iterations: int = 80,
    source_marginal: torch.Tensor | None = None,
    target_marginal: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute an entropic OT coupling with uniform marginals."""

    if cost.ndim != 2:
        raise ValueError("cost must be a matrix")
    n, m = cost.shape
    source_marginal = cost.new_full((n,), 1.0 / n) if source_marginal is None else source_marginal
    target_marginal = cost.new_full((m,), 1.0 / m) if target_marginal is None else target_marginal
    if source_marginal.shape != (n,) or target_marginal.shape != (m,):
        raise ValueError("Marginals must match the cost matrix dimensions")
    if torch.any(source_marginal <= 0) or torch.any(target_marginal <= 0):
        raise ValueError("Sinkhorn marginals must be positive")
    if not torch.isclose(source_marginal.sum(), target_marginal.sum(), rtol=1e-5, atol=1e-7):
        raise ValueError("Source and target marginals must have equal total mass")
    log_a = torch.log(source_marginal)
    log_b = torch.log(target_marginal)
    log_k = -cost / epsilon
    u = torch.zeros_like(log_a)
    v = torch.zeros_like(log_b)
    for _ in range(iterations):
        u = log_a - torch.logsumexp(log_k + v.unsqueeze(0), dim=1)
        v = log_b - torch.logsumexp(log_k + u.unsqueeze(1), dim=0)
    return torch.exp(log_k + u.unsqueeze(1) + v.unsqueeze(0))


def partial_sinkhorn_coupling(
    cost: torch.Tensor,
    transported_mass: float = 0.95,
    epsilon: float = 0.05,
    iterations: int = 80,
) -> torch.Tensor:
    """Entropic partial OT via dummy source and target points."""

    if not 0 < transported_mass < 1:
        raise ValueError("transported_mass must be between zero and one")
    n, m = cost.shape
    unmatched_mass = 1.0 - transported_mass
    augmented = cost.new_zeros((n + 1, m + 1))
    augmented[:n, :m] = cost
    augmented[n, m] = cost.max().detach() + 50.0 * epsilon
    source = torch.cat([cost.new_full((n,), 1.0 / n), cost.new_tensor([unmatched_mass])])
    target = torch.cat([cost.new_full((m,), 1.0 / m), cost.new_tensor([unmatched_mass])])
    plan = sinkhorn_coupling(
        augmented,
        epsilon=epsilon,
        iterations=iterations,
        source_marginal=source,
        target_marginal=target,
    )
    return plan[:n, :m]


def unbalanced_sinkhorn_coupling(
    cost: torch.Tensor,
    marginal_relaxation: float = 0.1,
    epsilon: float = 0.05,
    iterations: int = 80,
) -> torch.Tensor:
    """Entropic UOT with KL penalties on both marginal deviations."""

    if marginal_relaxation <= 0:
        raise ValueError("marginal_relaxation must be positive")
    n, m = cost.shape
    log_a = cost.new_full((n,), -torch.log(cost.new_tensor(float(n))))
    log_b = cost.new_full((m,), -torch.log(cost.new_tensor(float(m))))
    log_k = -cost / epsilon
    exponent = marginal_relaxation / (marginal_relaxation + epsilon)
    log_u = torch.zeros_like(log_a)
    log_v = torch.zeros_like(log_b)
    for _ in range(iterations):
        log_u = exponent * (log_a - torch.logsumexp(log_k + log_v.unsqueeze(0), dim=1))
        log_v = exponent * (log_b - torch.logsumexp(log_k + log_u.unsqueeze(1), dim=0))
    return torch.exp(log_k + log_u.unsqueeze(1) + log_v.unsqueeze(0))


def transport_coupling(
    cost: torch.Tensor,
    method: str = "balanced",
    epsilon: float = 0.05,
    iterations: int = 80,
    partial_mass: float = 0.95,
    marginal_relaxation: float = 0.1,
) -> torch.Tensor:
    if method == "balanced":
        return sinkhorn_coupling(cost, epsilon=epsilon, iterations=iterations)
    if method == "partial":
        return partial_sinkhorn_coupling(cost, partial_mass, epsilon, iterations)
    if method == "unbalanced":
        return unbalanced_sinkhorn_coupling(cost, marginal_relaxation, epsilon, iterations)
    raise ValueError("OT method must be balanced, partial, or unbalanced")


def transport_plan(
    x0: torch.Tensor,
    x1: torch.Tensor,
    **coupling_options,
) -> tuple[torch.Tensor, torch.Tensor]:
    cost = pairwise_squared_cost(x0, x1)
    return cost, transport_coupling(cost, **coupling_options)


def coupling_diagnostics(cost: torch.Tensor, coupling: torch.Tensor) -> dict[str, float]:
    """Summarize transported mass, concentration, and marginal imbalance."""

    total = coupling.sum()
    normalized = coupling / total
    positive = normalized[normalized > 0]
    entropy = -(positive * torch.log(positive)).sum()
    row_mass = normalized.sum(dim=1)
    column_mass = normalized.sum(dim=0)
    return {
        "ot_plan_mass": float(total.detach().cpu()),
        "ot_mean_pair_cost": float((normalized * cost).sum().detach().cpu()),
        "ot_effective_edges": float(torch.exp(entropy).detach().cpu()),
        "ot_max_edge_mass": float(normalized.max().detach().cpu()),
        "ot_row_mass_cv": float((row_mass.std() / row_mass.mean()).detach().cpu()),
        "ot_column_mass_cv": float((column_mass.std() / column_mass.mean()).detach().cpu()),
    }


def sample_coupling_pairs(coupling: torch.Tensor, n_pairs: int, generator: torch.Generator | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    flat = coupling.reshape(-1)
    flat = flat / flat.sum()
    idx = torch.multinomial(flat, n_pairs, replacement=True, generator=generator)
    return idx // coupling.shape[1], idx % coupling.shape[1]


def sample_transport_indices(
    x0: torch.Tensor,
    x1: torch.Tensor,
    n_pairs: int,
    generator: torch.Generator | None = None,
    **coupling_options,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Solve OT on a cell pool and sample endpoint indices from its plan."""

    if n_pairs <= 0:
        raise ValueError("n_pairs must be positive")
    _cost, coupling = transport_plan(x0, x1, **coupling_options)
    return sample_coupling_pairs(coupling, n_pairs, generator=generator)


def cfm_interpolate(
    x0: torch.Tensor,
    x1: torch.Tensor,
    t0: float | torch.Tensor,
    t1: float | torch.Tensor,
    tau: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if tau is None:
        tau = torch.rand((x0.shape[0], 1), device=x0.device, dtype=x0.dtype, generator=generator)
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
    generator: torch.Generator | None = None,
    method: str = "balanced",
    partial_mass: float = 0.95,
    marginal_relaxation: float = 0.1,
    n_pairs: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        i0, i1 = sample_transport_indices(
            x0,
            x1,
            min(x0.shape[0], x1.shape[0]) if n_pairs is None else n_pairs,
            generator=generator,
            method=method,
            epsilon=epsilon,
            iterations=iterations,
            partial_mass=partial_mass,
            marginal_relaxation=marginal_relaxation,
        )
    return cfm_interpolate(x0[i0], x1[i1], t0, t1, generator=generator)


def ot_cfm_batch_with_state(
    x0: torch.Tensor,
    x1: torch.Tensor,
    state0: torch.Tensor,
    state1: torch.Tensor,
    t0: float,
    t1: float,
    epsilon: float = 0.05,
    iterations: int = 80,
    generator: torch.Generator | None = None,
    method: str = "balanced",
    partial_mass: float = 0.95,
    marginal_relaxation: float = 0.1,
    n_pairs: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Couple expression states by OT and interpolate an aligned state representation.

    OT and the CFM velocity target stay in expression space. ``state0`` and
    ``state1`` are row-aligned auxiliary cell representations, such as UCE,
    which are interpolated with the same sampled OT pairs and CFM time.
    """

    if state0.shape[0] != x0.shape[0] or state1.shape[0] != x1.shape[0]:
        raise ValueError("Auxiliary states must align with expression batch rows")
    with torch.no_grad():
        i0, i1 = sample_transport_indices(
            x0,
            x1,
            min(x0.shape[0], x1.shape[0]) if n_pairs is None else n_pairs,
            generator=generator,
            method=method,
            epsilon=epsilon,
            iterations=iterations,
            partial_mass=partial_mass,
            marginal_relaxation=marginal_relaxation,
        )
    xt, target, tau = cfm_interpolate(x0[i0], x1[i1], t0, t1, generator=generator)
    state_t = (1.0 - tau) * state0[i0] + tau * state1[i1]
    return xt, target, tau, state_t
