"""Train-only PCA transforms for OT costs and gene-space endpoint smoothing."""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


def log_normalize_counts(counts: np.ndarray, target_sum: float = 10_000.0) -> tuple[np.ndarray, np.ndarray]:
    counts = np.asarray(counts, dtype=np.float32)
    library_size = counts.sum(axis=1, dtype=np.float64).astype(np.float32)
    scale = np.divide(target_sum, library_size, out=np.zeros_like(library_size), where=library_size > 0)
    return np.log1p(counts * scale[:, None]), library_size


@dataclass(frozen=True)
class PCADenoiser:
    components: np.ndarray
    mean: np.ndarray
    explained_variance: np.ndarray
    target_sum: float = 10_000.0
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def n_components(self) -> int:
        return int(self.components.shape[0])

    def transform_counts(self, counts: np.ndarray, whiten: bool = True) -> np.ndarray:
        normalized, _library_size = log_normalize_counts(counts, self.target_sum)
        coordinates = (normalized - self.mean) @ self.components.T
        if whiten:
            coordinates = coordinates / np.sqrt(np.maximum(self.explained_variance, 1e-8))
        return np.asarray(coordinates, dtype=np.float32)

    def reconstruct_counts(self, counts: np.ndarray) -> np.ndarray:
        normalized, library_size = log_normalize_counts(counts, self.target_sum)
        coordinates = (normalized - self.mean) @ self.components.T
        reconstructed = coordinates @ self.components + self.mean
        reconstructed = np.clip(reconstructed, 0.0, 20.0)
        count_scale = library_size / self.target_sum
        return np.asarray(np.expm1(reconstructed) * count_scale[:, None], dtype=np.float32)

    def save(self, path: str | Path, metadata: dict[str, object] | None = None) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            components=self.components,
            mean=self.mean,
            explained_variance=self.explained_variance,
            target_sum=np.asarray(self.target_sum),
            metadata=np.asarray(json.dumps(self.metadata if metadata is None else metadata, sort_keys=True)),
        )

    @classmethod
    def load(cls, path: str | Path) -> "PCADenoiser":
        raw = np.load(path, allow_pickle=False)
        return cls(
            components=np.asarray(raw["components"], dtype=np.float32),
            mean=np.asarray(raw["mean"], dtype=np.float32),
            explained_variance=np.asarray(raw["explained_variance"], dtype=np.float32),
            target_sum=float(raw["target_sum"]),
            metadata=json.loads(str(raw["metadata"])) if "metadata" in raw else {},
        )


def fit_pca_denoiser(
    sampler,
    train_days: list[str],
    total_cells: int,
    n_components: int,
    seed: int,
    target_sum: float = 10_000.0,
) -> PCADenoiser:
    """Fit randomized PCA to a stage-balanced sample without advancing sampler RNG."""

    from sklearn.decomposition import PCA

    if total_cells < n_components:
        raise ValueError("PCA fit requires at least as many cells as components")
    per_day = int(np.ceil(total_cells / len(train_days)))
    old_size = sampler.batch_size
    old_rng_state = copy.deepcopy(sampler.rng.bit_generator.state)
    sampler.batch_size = per_day
    try:
        samples = [np.asarray(sampler.sample_day(day), dtype=np.float32) for day in train_days]
    finally:
        sampler.batch_size = old_size
        sampler.rng.bit_generator.state = old_rng_state
    matrix = np.vstack(samples)[:total_cells]
    normalized, _library_size = log_normalize_counts(matrix, target_sum)
    pca = PCA(n_components=n_components, svd_solver="randomized", random_state=seed)
    pca.fit(normalized)
    return PCADenoiser(
        components=np.asarray(pca.components_, dtype=np.float32),
        mean=np.asarray(pca.mean_, dtype=np.float32),
        explained_variance=np.asarray(pca.explained_variance_, dtype=np.float32),
        target_sum=target_sum,
    )
