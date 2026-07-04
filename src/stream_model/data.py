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


class H5adIntervalSampler:
    """Sample expression batches from backed h5ad files by adjacent day interval."""

    def __init__(
        self,
        manifest: pd.DataFrame,
        gene_indices: np.ndarray,
        intervals: list[tuple[str, str]],
        batch_size: int,
        seed: int = 1337,
    ):
        self.manifest = manifest
        self.gene_indices = np.asarray(gene_indices)
        self.intervals = intervals
        self.batch_size = batch_size
        self.rng = np.random.default_rng(seed)
        self._adata_cache = {}

    @classmethod
    def from_adata_dir(
        cls,
        adata_dir: str | Path,
        cell_metadata: pd.DataFrame,
        gene_ids: list[str],
        intervals: list[tuple[str, str]],
        batch_size: int,
        seed: int = 1337,
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
        return cls(pd.DataFrame(rows), gene_indices, intervals, batch_size, seed)

    def sample(self) -> IntervalBatch:
        day0, day1 = self.intervals[self.rng.integers(0, len(self.intervals))]
        x0 = self._sample_day(day0)
        x1 = self._sample_day(day1)
        return IntervalBatch(x0=x0, x1=x1, t0=parse_day_value(day0), t1=parse_day_value(day1), day0=day0, day1=day1)

    def _sample_day(self, day: str) -> np.ndarray:
        rows = self.manifest[self.manifest["day"] == day]
        if rows.empty:
            raise ValueError(f"No cells available for day {day}")
        chosen = rows.iloc[self.rng.choice(len(rows), size=self.batch_size, replace=len(rows) < self.batch_size)]
        chunks = []
        for path, group in chosen.groupby("path"):
            a = self._open(path)
            row_idx = np.sort(group["row_idx"].to_numpy())
            x = a.X[row_idx, :][:, self.gene_indices]
            if hasattr(x, "toarray"):
                x = x.toarray()
            chunks.append(np.asarray(x, dtype=np.float32))
        return np.vstack(chunks)

    def _open(self, path: str):
        if path not in self._adata_cache:
            import anndata as ad

            self._adata_cache[path] = ad.read_h5ad(path, backed="r")
        return self._adata_cache[path]


def load_selected_genes(gene_metadata_csv: str | Path, hvg_csv: str | Path, n_hvg: int) -> pd.DataFrame:
    genes = pd.read_csv(gene_metadata_csv, index_col=0)
    hvgs = pd.read_csv(hvg_csv)
    selected = hvgs.head(n_hvg).merge(genes, left_on="gene", right_on="gene_id", how="inner")
    selected = selected[selected["gene_type"] == "protein_coding"].drop_duplicates("gene_id")
    return selected[["gene_id", "gene_short_name", "gene_type", "chr", "variance"]].reset_index(drop=True)
