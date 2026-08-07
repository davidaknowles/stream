"""Fine-grained cell-subtype assignment and population dynamics metrics."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .rollout import r2_score


@dataclass(frozen=True)
class SubtypeCentroidClassifier:
    labels: np.ndarray
    centroids: np.ndarray
    metadata: dict[str, object]

    def __post_init__(self) -> None:
        labels = np.asarray(self.labels).astype(str)
        centroids = np.asarray(self.centroids, dtype=np.float32)
        if centroids.ndim != 2 or centroids.shape[0] != len(labels):
            raise ValueError("Centroids must have one row per subtype label")
        object.__setattr__(self, "labels", labels)
        object.__setattr__(self, "centroids", centroids)

    def predict_coordinates(self, coordinates: np.ndarray, chunk_size: int = 4096) -> np.ndarray:
        coordinates = np.asarray(coordinates, dtype=np.float32)
        if coordinates.ndim != 2 or coordinates.shape[1] != self.centroids.shape[1]:
            raise ValueError("Coordinate dimensions do not match subtype centroids")
        centroid_norm = np.square(self.centroids).sum(axis=1)
        assignments = []
        for start in range(0, len(coordinates), chunk_size):
            chunk = coordinates[start : start + chunk_size]
            distances = (
                np.square(chunk).sum(axis=1, keepdims=True)
                + centroid_norm[None, :]
                - 2.0 * chunk @ self.centroids.T
            )
            assignments.append(np.argmin(distances, axis=1))
        return np.concatenate(assignments) if assignments else np.empty(0, dtype=np.int64)

    def save(self, path: str | Path) -> None:
        np.savez_compressed(
            path,
            labels=self.labels,
            centroids=self.centroids,
            metadata=np.asarray(json.dumps(self.metadata, sort_keys=True)),
        )

    @classmethod
    def load(cls, path: str | Path) -> "SubtypeCentroidClassifier":
        raw = np.load(path, allow_pickle=False)
        return cls(
            labels=raw["labels"].astype(str),
            centroids=np.asarray(raw["centroids"], dtype=np.float32),
            metadata=json.loads(str(raw["metadata"])),
        )


@dataclass
class SubtypeStatistics:
    mass: np.ndarray
    expression_sum: np.ndarray

    @classmethod
    def empty(cls, n_subtypes: int, n_genes: int) -> "SubtypeStatistics":
        return cls(
            mass=np.zeros(n_subtypes, dtype=np.float64),
            expression_sum=np.zeros((n_subtypes, n_genes), dtype=np.float64),
        )

    def update(
        self,
        expression: np.ndarray,
        assignments: np.ndarray,
        weights: np.ndarray | None = None,
    ) -> None:
        expression = np.asarray(expression, dtype=np.float32)
        assignments = np.asarray(assignments, dtype=np.int64)
        if len(expression) != len(assignments):
            raise ValueError("Expression and subtype assignments must align")
        if weights is None:
            weights = np.ones(len(expression), dtype=np.float64)
        else:
            weights = np.asarray(weights, dtype=np.float64)
            if weights.shape != (len(expression),):
                raise ValueError("Weights must have one value per cell")
            # Express normalized particle weights as effective cell counts.
            weights = weights * len(expression) / weights.sum()
        self.mass += np.bincount(assignments, weights=weights, minlength=len(self.mass))
        np.add.at(self.expression_sum, assignments, expression * weights[:, None])


def jensen_shannon_divergence(first: np.ndarray, second: np.ndarray) -> float:
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    first = first / first.sum()
    second = second / second.sum()
    midpoint = 0.5 * (first + second)

    def kl(left: np.ndarray) -> float:
        present = left > 0
        return float(np.sum(left[present] * np.log(left[present] / midpoint[present])))

    return 0.5 * (kl(first) + kl(second))


def subtype_dynamics_metrics(
    source: SubtypeStatistics,
    observed: SubtypeStatistics,
    predicted: SubtypeStatistics,
    *,
    gene_indices: np.ndarray | None = None,
    min_cells: float = 5.0,
) -> dict[str, float | int]:
    """Compare subtype abundance and within-subtype pseudobulk expression shifts."""

    if not (
        source.mass.shape == observed.mass.shape == predicted.mass.shape
        and source.expression_sum.shape
        == observed.expression_sum.shape
        == predicted.expression_sum.shape
    ):
        raise ValueError("Source, observed, and predicted subtype statistics must align")
    source_proportion = source.mass / source.mass.sum()
    observed_proportion = observed.mass / observed.mass.sum()
    predicted_proportion = predicted.mass / predicted.mass.sum()
    persistence_js = jensen_shannon_divergence(source_proportion, observed_proportion)
    predicted_js = jensen_shannon_divergence(predicted_proportion, observed_proportion)

    eligible = (
        (source.mass >= min_cells)
        & (observed.mass >= min_cells)
        & (predicted.mass >= min_cells)
    )
    source_mean = source.expression_sum[eligible] / source.mass[eligible, None]
    observed_mean = observed.expression_sum[eligible] / observed.mass[eligible, None]
    predicted_mean = predicted.expression_sum[eligible] / predicted.mass[eligible, None]
    if gene_indices is not None:
        indices = np.asarray(gene_indices, dtype=np.int64)
        source_mean = source_mean[:, indices]
        observed_mean = observed_mean[:, indices]
        predicted_mean = predicted_mean[:, indices]
    observed_shift = observed_mean - source_mean
    predicted_shift = predicted_mean - source_mean
    subtype_r2 = np.asarray(
        [r2_score(target, estimate) for target, estimate in zip(observed_shift, predicted_shift)],
        dtype=np.float64,
    )
    finite_r2 = subtype_r2[np.isfinite(subtype_r2)]
    return {
        "subtype_js_divergence": predicted_js,
        "subtype_persistence_js_divergence": persistence_js,
        "subtype_js_skill": (
            1.0 - predicted_js / persistence_js if persistence_js > 0 else float("nan")
        ),
        "subtype_total_variation": float(
            0.5 * np.abs(predicted_proportion - observed_proportion).sum()
        ),
        "subtype_proportion_mae": float(
            np.abs(predicted_proportion - observed_proportion).mean()
        ),
        "n_subtypes_total": int(len(source.mass)),
        "n_subtypes_pseudobulk": int(eligible.sum()),
        "subtype_pseudobulk_shift_r2_pooled": (
            r2_score(observed_shift.ravel(), predicted_shift.ravel())
            if eligible.any()
            else float("nan")
        ),
        "subtype_pseudobulk_shift_r2_macro": (
            float(finite_r2.mean()) if len(finite_r2) else float("nan")
        ),
        "subtype_pseudobulk_shift_mae": (
            float(np.abs(observed_shift - predicted_shift).mean())
            if eligible.any()
            else float("nan")
        ),
    }
