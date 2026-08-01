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

    def reconstruct_tensor(self, counts):
        """Apply the same reconstruction to a torch count tensor on its current device."""

        import torch

        library_size = counts.sum(dim=1)
        scale = torch.where(
            library_size > 0,
            counts.new_tensor(self.target_sum) / library_size,
            torch.zeros_like(library_size),
        )
        normalized = torch.log1p(counts * scale[:, None])
        components = torch.as_tensor(self.components, device=counts.device, dtype=counts.dtype)
        mean = torch.as_tensor(self.mean, device=counts.device, dtype=counts.dtype)
        coordinates = (normalized - mean) @ components.T
        reconstructed = torch.clamp(coordinates @ components + mean, min=0.0, max=20.0)
        return torch.expm1(reconstructed) * (library_size / self.target_sum)[:, None]

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


def _counts_from_log_expression(
    log_expression: np.ndarray,
    library_size: np.ndarray,
    target_sum: float,
) -> np.ndarray:
    del target_sum
    expression = np.expm1(np.clip(log_expression, 0.0, 20.0))
    profile_sum = expression.sum(axis=1, dtype=np.float64).astype(np.float32)
    count_scale = np.divide(
        np.asarray(library_size, dtype=np.float32),
        profile_sum,
        out=np.zeros_like(profile_sum),
        where=profile_sum > 0,
    )
    return np.asarray(expression * count_scale[:, None], dtype=np.float32)


def knn_smooth_selected_counts(
    counts: np.ndarray,
    coordinates: np.ndarray,
    selected_indices: np.ndarray,
    n_neighbors: int = 15,
    target_sum: float = 10_000.0,
    device: str = "cpu",
    query_chunk_size: int = 64,
) -> np.ndarray:
    """Smooth selected cells over same-stage PCA neighbors while preserving library size."""

    import torch

    counts = np.asarray(counts, dtype=np.float32)
    coordinates = np.asarray(coordinates, dtype=np.float32)
    selected_indices = np.asarray(selected_indices, dtype=np.int64)
    if len(counts) != len(coordinates):
        raise ValueError("counts and coordinates must have the same number of cells")
    invalid_indices = len(selected_indices) and (
        selected_indices.min() < 0 or selected_indices.max() >= len(counts)
    )
    if selected_indices.ndim != 1 or invalid_indices:
        raise ValueError("selected_indices must index rows of counts")
    k = min(int(n_neighbors), len(counts))
    if k <= 0:
        raise ValueError("n_neighbors must be positive")
    unique_indices, inverse = np.unique(selected_indices, return_inverse=True)
    coordinate_tensor = torch.as_tensor(coordinates, device=device)
    neighbors = []
    with torch.no_grad():
        for start in range(0, len(unique_indices), query_chunk_size):
            query_indices = unique_indices[start : start + query_chunk_size]
            queries = coordinate_tensor[torch.as_tensor(query_indices, device=device)]
            distances = torch.cdist(queries, coordinate_tensor)
            neighbors.append(torch.topk(distances, k=k, largest=False).indices.cpu().numpy())
    neighbor_indices = np.vstack(neighbors) if neighbors else np.empty((0, k), dtype=np.int64)
    normalized, library_size = log_normalize_counts(counts, target_sum)
    smoothed_unique = np.empty((len(unique_indices), counts.shape[1]), dtype=np.float32)
    for start in range(0, len(unique_indices), query_chunk_size):
        stop = min(start + query_chunk_size, len(unique_indices))
        smoothed_unique[start:stop] = normalized[neighbor_indices[start:stop]].mean(axis=1)
    selected_library_size = library_size[unique_indices]
    return _counts_from_log_expression(smoothed_unique, selected_library_size, target_sum)[inverse]


def metacell_smooth_selected_counts(
    counts: np.ndarray,
    coordinates: np.ndarray,
    selected_indices: np.ndarray,
    n_metacells: int = 512,
    target_sum: float = 10_000.0,
    seed: int = 0,
) -> np.ndarray:
    """Map selected cells to same-stage PCA metacell centroids, preserving library size."""

    from sklearn.cluster import MiniBatchKMeans

    counts = np.asarray(counts, dtype=np.float32)
    coordinates = np.asarray(coordinates, dtype=np.float32)
    selected_indices = np.asarray(selected_indices, dtype=np.int64)
    if len(counts) != len(coordinates):
        raise ValueError("counts and coordinates must have the same number of cells")
    invalid_indices = len(selected_indices) and (
        selected_indices.min() < 0 or selected_indices.max() >= len(counts)
    )
    if selected_indices.ndim != 1 or invalid_indices:
        raise ValueError("selected_indices must index rows of counts")
    clusters = min(int(n_metacells), len(counts))
    if clusters <= 0:
        raise ValueError("n_metacells must be positive")
    labels = MiniBatchKMeans(
        n_clusters=clusters,
        batch_size=min(2048, len(counts)),
        n_init=1,
        max_iter=100,
        random_state=seed,
    ).fit_predict(coordinates)
    normalized, library_size = log_normalize_counts(counts, target_sum)
    selected_labels = labels[selected_indices]
    unique_labels, inverse = np.unique(selected_labels, return_inverse=True)
    centroids = np.empty((len(unique_labels), counts.shape[1]), dtype=np.float32)
    for index, label in enumerate(unique_labels):
        centroids[index] = normalized[labels == label].mean(axis=0)
    return _counts_from_log_expression(centroids[inverse], library_size[selected_indices], target_sum)


def denoise_selected_counts(
    method: str,
    counts: np.ndarray,
    selected_indices: np.ndarray,
    pca: PCADenoiser,
    *,
    n_neighbors: int = 15,
    n_metacells: int = 512,
    seed: int = 0,
    device: str = "cpu",
) -> np.ndarray:
    """Apply a configured train-fitted endpoint denoiser to selected stage cells."""

    selected_indices = np.asarray(selected_indices, dtype=np.int64)
    if method == "pca":
        return pca.reconstruct_counts(np.asarray(counts)[selected_indices])
    coordinates = pca.transform_counts(counts, whiten=False)
    if method == "knn":
        return knn_smooth_selected_counts(
            counts,
            coordinates,
            selected_indices,
            n_neighbors=n_neighbors,
            target_sum=pca.target_sum,
            device=device,
        )
    if method == "metacell":
        return metacell_smooth_selected_counts(
            counts,
            coordinates,
            selected_indices,
            n_metacells=n_metacells,
            target_sum=pca.target_sum,
            seed=seed,
        )
    raise ValueError("method must be pca, knn, or metacell")


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
