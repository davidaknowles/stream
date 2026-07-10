"""AnnData-backed sampling utilities for STREAM training."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


def parse_day_value(day: str) -> float:
    day_str = str(day)
    prefix = day_str[:1].upper()
    match = re.search(r"([0-9]+(?:\.[0-9]+)?)", str(day))
    if not match:
        raise ValueError(f"Could not parse day value from {day!r}")
    value = float(match.group(1))
    if prefix == "P":
        return 19.0 + value
    return value


def ordered_days(days: list[str] | pd.Series | np.ndarray) -> list[str]:
    return sorted({str(day) for day in days}, key=parse_day_value)


def adjacent_intervals(days: list[str], heldout_days: set[str] | None = None) -> list[tuple[str, str]]:
    heldout_days = heldout_days or set()
    ordered = ordered_days(days)
    return [(a, b) for a, b in zip(ordered[:-1], ordered[1:], strict=True) if a not in heldout_days and b not in heldout_days]


@dataclass(frozen=True)
class IntervalBatch:
    x0: np.ndarray
    x1: np.ndarray
    t0: float
    t1: float
    day0: str
    day1: str
    state0: np.ndarray | None = None
    state1: np.ndarray | None = None


class H5adIntervalSampler:
    """Sample expression batches from backed h5ad files by adjacent day interval."""

    def __init__(
        self,
        manifest: pd.DataFrame,
        gene_indices: np.ndarray,
        intervals: list[tuple[str, str]],
        batch_size: int,
        seed: int = 1337,
        state_embeddings_dir: str | Path | None = None,
        state_dim: int | None = None,
    ):
        self.manifest = manifest
        self.gene_indices = np.asarray(gene_indices)
        self.intervals = intervals
        self.batch_size = batch_size
        self.rng = np.random.default_rng(seed)
        self._adata_cache = {}
        self.state_embeddings_dir = None if state_embeddings_dir is None else Path(state_embeddings_dir)
        self.state_dim = state_dim
        self._state_cache = {}

    @classmethod
    def from_adata_dir(
        cls,
        adata_dir: str | Path,
        cell_metadata: pd.DataFrame,
        gene_ids: list[str],
        intervals: list[tuple[str, str]],
        batch_size: int,
        seed: int = 1337,
        state_embeddings_dir: str | Path | None = None,
        state_dim: int | None = None,
    ) -> "H5adIntervalSampler":
        import anndata as ad

        rows = []
        gene_indices = None
        adata_paths = sorted(Path(adata_dir).glob("*.h5ad"))
        cell_days = cell_metadata.set_index("cell_id")["day"]
        for file_id, path in enumerate(adata_paths):
            a = ad.read_h5ad(path, backed="r")
            var_gene_ids = pd.Index(a.var["gene_id"] if "gene_id" in a.var else a.var_names)
            if gene_indices is None:
                gene_indices = var_gene_ids.get_indexer(gene_ids)
                if np.any(gene_indices < 0):
                    missing = np.asarray(gene_ids)[gene_indices < 0][:10]
                    raise ValueError(f"Selected genes missing from {path}: {missing}")
            obs_cell_id = a.obs["cell_id"] if "cell_id" in a.obs else a.obs_names
            days = pd.Series(obs_cell_id).map(cell_days).to_numpy()
            for row_idx, day in enumerate(days):
                if pd.isna(day):
                    continue
                rows.append({"file_id": file_id, "path": str(path), "row_idx": row_idx, "day": str(day)})
            a.file.close()
        if gene_indices is None:
            raise ValueError(f"No .h5ad files found in {adata_dir}")
        sampler = cls(
            pd.DataFrame(rows),
            gene_indices,
            intervals,
            batch_size,
            seed,
            state_embeddings_dir=state_embeddings_dir,
            state_dim=state_dim,
        )
        if sampler.state_embeddings_dir is not None:
            sampler._validate_state_files(adata_paths)
        return sampler

    def sample(self) -> IntervalBatch:
        day0, day1 = self.intervals[self.rng.integers(0, len(self.intervals))]
        first = self._sample_day(day0)
        second = self._sample_day(day1)
        if self.state_embeddings_dir is None:
            x0, x1 = first, second
            state0 = state1 = None
        else:
            x0, state0 = first
            x1, state1 = second
        return IntervalBatch(
            x0=x0,
            x1=x1,
            t0=parse_day_value(day0),
            t1=parse_day_value(day1),
            day0=day0,
            day1=day1,
            state0=state0,
            state1=state1,
        )

    def _sample_day(self, day: str) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        rows = self.manifest[self.manifest["day"] == day]
        if rows.empty:
            raise ValueError(f"No cells available for day {day}")
        chosen = rows.iloc[self.rng.choice(len(rows), size=self.batch_size, replace=len(rows) < self.batch_size)]
        chunks = []
        state_chunks = []
        for path, group in chosen.groupby("path"):
            a = self._open(path)
            row_idx = np.sort(group["row_idx"].to_numpy())
            x = a.X[row_idx, :][:, self.gene_indices]
            if hasattr(x, "toarray"):
                x = x.toarray()
            chunks.append(np.asarray(x, dtype=np.float32))
            if self.state_embeddings_dir is not None:
                state_chunks.append(np.asarray(self._open_state(path)[row_idx], dtype=np.float32))
        x_out = np.vstack(chunks)
        if self.state_embeddings_dir is None:
            return x_out
        return x_out, np.vstack(state_chunks)

    def _open(self, path: str):
        if path not in self._adata_cache:
            import anndata as ad

            self._adata_cache[path] = ad.read_h5ad(path, backed="r")
        return self._adata_cache[path]

    def _state_path(self, adata_path: str | Path) -> Path:
        if self.state_embeddings_dir is None:
            raise RuntimeError("No auxiliary state embedding directory configured")
        return self.state_embeddings_dir / f"{Path(adata_path).stem}.npy"

    def _validate_state_files(self, adata_paths: list[Path]) -> None:
        for path in adata_paths:
            state_path = self._state_path(path)
            if not state_path.exists():
                raise FileNotFoundError(f"Missing auxiliary state embeddings: {state_path}")
            state = np.load(state_path, mmap_mode="r")
            if state.shape[0] != self.manifest.loc[self.manifest["path"] == str(path), "row_idx"].nunique():
                raise ValueError(f"Embedding row count does not match AnnData rows: {state_path}")
            if self.state_dim is not None and state.shape[1] != self.state_dim:
                raise ValueError(f"Expected state dimension {self.state_dim} in {state_path}, found {state.shape[1]}")

    def _open_state(self, path: str):
        if path not in self._state_cache:
            self._state_cache[path] = np.load(self._state_path(path), mmap_mode="r")
        return self._state_cache[path]


def load_selected_genes(gene_metadata_csv: str | Path, hvg_csv: str | Path, n_hvg: int) -> pd.DataFrame:
    genes = pd.read_csv(gene_metadata_csv, index_col=0)
    hvgs = pd.read_csv(hvg_csv)
    if "variance" in hvgs.columns:
        hvgs = hvgs.sort_values("variance", ascending=False)
    selected = hvgs.merge(genes, left_on="gene", right_on="gene_id", how="inner")
    selected = selected[selected["gene_type"] == "protein_coding"].drop_duplicates("gene_id").head(n_hvg)
    if len(selected) < n_hvg and len(hvgs) > n_hvg:
        raise ValueError(
            f"Only found {len(selected):,} protein-coding HVGs in {hvg_csv}; "
            f"need {n_hvg:,}. Regenerate the HVG table with more genes."
        )
    return selected[["gene_id", "gene_short_name", "gene_type", "chr", "variance"]].reset_index(drop=True)
