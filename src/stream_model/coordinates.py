"""Gene-scaled dynamical coordinates with expression-space boundaries."""

from __future__ import annotations

from dataclasses import dataclass

import torch


def robust_gene_scale(values: torch.Tensor) -> torch.Tensor:
    """Estimate positive per-gene expression scales with a robust floor."""

    if values.ndim != 2 or len(values) < 2:
        raise ValueError("values must contain at least two cells by genes")
    scale = values.float().std(dim=0, unbiased=False)
    positive = scale[scale > 0]
    floor = positive.median() * 0.05 if len(positive) else scale.new_tensor(1.0)
    return scale.clamp_min(max(float(floor), 1e-3))


def estimate_sampler_gene_scale(sampler, cells_per_day: int = 512) -> torch.Tensor:
    """Estimate gene scales from training cells without materializing all stages."""

    if cells_per_day <= 0:
        raise ValueError("cells_per_day must be positive")
    days = sorted({day for interval in sampler.intervals for day in interval})
    if not days:
        raise ValueError("sampler must contain at least one interval")
    total = 0
    sums = None
    sums_of_squares = None
    original_batch_size = sampler.batch_size
    sampler.batch_size = cells_per_day
    try:
        for day in days:
            sampled = sampler.sample_day(day)
            values = sampled[0] if isinstance(sampled, tuple) else sampled
            tensor = torch.as_tensor(values).double()
            sums = tensor.sum(0) if sums is None else sums + tensor.sum(0)
            batch_squares = tensor.square().sum(0)
            sums_of_squares = (
                batch_squares if sums_of_squares is None else sums_of_squares + batch_squares
            )
            total += len(tensor)
    finally:
        sampler.batch_size = original_batch_size
    variance = (sums_of_squares / total - (sums / total).square()).clamp_min(0.0)
    scale = variance.sqrt().float()
    positive = scale[scale > 0]
    floor = positive.median() * 0.05 if len(positive) else scale.new_tensor(1.0)
    return scale.clamp_min(max(float(floor), 1e-3))


@dataclass(frozen=True)
class GeneScaleCoordinates:
    """Map between count-like expression ``X`` and model coordinates ``Y=X/q``."""

    gene_scale: torch.Tensor
    mode: str = "gene_scaled"

    def __post_init__(self) -> None:
        if self.mode not in {"count", "gene_scaled"}:
            raise ValueError("Coordinate mode must be count or gene_scaled")
        if self.gene_scale.ndim != 1 or torch.any(self.gene_scale <= 0):
            raise ValueError("gene_scale must be a positive vector")

    @property
    def scale(self) -> torch.Tensor:
        if self.mode == "gene_scaled":
            return self.gene_scale
        return torch.ones_like(self.gene_scale)

    def _scale_for(self, values: torch.Tensor) -> torch.Tensor:
        return self.scale.to(device=values.device, dtype=values.dtype).reshape(1, -1)

    def to_model(self, expression: torch.Tensor) -> torch.Tensor:
        """Convert count-like expression or velocity to model coordinates."""

        return expression / self._scale_for(expression)

    def to_expression(self, model_values: torch.Tensor) -> torch.Tensor:
        """Convert model-coordinate expression or velocity to count space."""

        return model_values * self._scale_for(model_values)

    def perturbation_scale(self, reference: torch.Tensor) -> torch.Tensor:
        """Return the Gaussian bridge scale in the active model coordinates."""

        if self.mode == "gene_scaled":
            return torch.ones_like(self.gene_scale, device=reference.device, dtype=reference.dtype)
        return self.gene_scale.to(device=reference.device, dtype=reference.dtype)


def coordinates_from_checkpoint(
    checkpoint: dict, device: torch.device | str = "cpu"
) -> GeneScaleCoordinates:
    """Load a coordinate contract while preserving legacy count-space checkpoints."""

    mode = checkpoint.get(
        "dynamics_coordinates",
        checkpoint.get("config", {}).get("dynamics_coordinates", "count"),
    )
    raw_scale = checkpoint.get("gene_scale")
    if raw_scale is None:
        raise ValueError("A gene_scale is required for an explicit coordinate contract")
    return GeneScaleCoordinates(torch.as_tensor(raw_scale, device=device).float(), mode=mode)
