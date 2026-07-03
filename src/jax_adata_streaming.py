from __future__ import annotations

from dataclasses import dataclass
from collections import Counter
import csv
from pathlib import Path
from typing import Callable, Iterable

import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
import umap
from scipy import sparse
from sklearn.decomposition import TruncatedSVD


@dataclass(frozen=True)
class StreamingUmapConfig:
    chunk_cells: int = 5000
    n_hvg: int = 2000
    n_pcs: int = 50
    pca_train_cells: int = 250_000
    umap_train_cells: int = 250_000
    umap_transform_batch: int = 50_000
    umap_plot_cells: int = 500_000
    random_seed: int = 7
    normalize_log1p: str = "auto"
    reuse_streaming_hvg: bool = True
    n_neighbors: int = 15
    umap_min_dist: float = 0.5

    @classmethod
    def from_env(cls) -> "StreamingUmapConfig":
        import os

        return cls(
            chunk_cells=int(os.environ.get("STREAM_CHUNK_CELLS", "5000")),
            n_hvg=int(os.environ.get("N_HVG", "2000")),
            n_pcs=int(os.environ.get("N_PCS", "50")),
            pca_train_cells=int(os.environ.get("PCA_TRAIN_CELLS", "250000")),
            umap_train_cells=int(os.environ.get("UMAP_TRAIN_CELLS", "250000")),
            umap_transform_batch=int(os.environ.get("UMAP_TRANSFORM_BATCH", "50000")),
            umap_plot_cells=int(os.environ.get("UMAP_PLOT_CELLS", "500000")),
            random_seed=int(os.environ.get("RANDOM_SEED", "7")),
            normalize_log1p=os.environ.get("NORMALIZE_LOG1P", "auto"),
            reuse_streaming_hvg=os.environ.get("REUSE_STREAMING_HVG", "1") == "1",
            n_neighbors=int(os.environ.get("N_NEIGHBORS", "15")),
            umap_min_dist=float(os.environ.get("UMAP_MIN_DIST", "0.5")),
        )


def _close_backed(adata) -> None:
    if getattr(adata, "isbacked", False) and getattr(adata, "file", None) is not None:
        adata.file.close()


def _normalize_log1p_chunk(x):
    if sparse.issparse(x):
        x = x.astype(np.float32).tocsr(copy=True)
        counts = np.asarray(x.sum(axis=1)).ravel().astype(np.float32)
        scale = np.divide(1e4, counts, out=np.zeros_like(counts), where=counts > 0)
        x = x.multiply(scale[:, None]).tocsr()
        x.data = np.log1p(x.data)
        return x

    x = np.asarray(x, dtype=np.float32)
    counts = x.sum(axis=1).astype(np.float32)
    scale = np.divide(1e4, counts, out=np.zeros_like(counts), where=counts > 0)
    x = x * scale[:, None]
    return np.log1p(x, out=x)


def _chunk_looks_like_counts(x) -> bool:
    vals = x.data if sparse.issparse(x) else np.asarray(x).ravel()
    vals = vals[: min(100_000, vals.shape[0])]
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return False
    rounded = np.round(vals[: min(10_000, vals.size)])
    return np.nanmax(vals) > 50 and np.allclose(vals[: rounded.size], rounded)


def _maybe_normalize(x, do_normalize: bool):
    if do_normalize:
        return _normalize_log1p_chunk(x)
    if sparse.issparse(x):
        return x.astype(np.float32)
    return np.asarray(x, dtype=np.float32)


def _iter_matrix_chunks(
    paths: Iterable[Path],
    *,
    var_idx=None,
    chunk_size: int,
):
    global_offset = 0
    for path in paths:
        adata = sc.read_h5ad(path, backed="r")
        for start in range(0, adata.n_obs, chunk_size):
            end = min(start + chunk_size, adata.n_obs)
            x = adata.X[start:end, :]
            if var_idx is not None:
                x = x[:, var_idx]
            yield path.name, global_offset + start, global_offset + end, x
        global_offset += adata.n_obs
        _close_backed(adata)


def _make_logger(out_dir: Path) -> Callable[[str], None]:
    progress_log = out_dir / "streaming_umap_progress.log"

    def log_progress(message: str) -> None:
        with progress_log.open("a") as handle:
            handle.write(message + "\n")

    return log_progress


def _sample_plot_indices(total_cells: int, config: StreamingUmapConfig) -> np.ndarray:
    rng = np.random.default_rng(config.random_seed)
    pca_train_n = min(config.pca_train_cells, total_cells)
    umap_train_n = min(config.umap_train_cells, total_cells)
    plot_n = min(config.umap_plot_cells, total_cells)
    rng.choice(total_cells, size=pca_train_n, replace=False)
    rng.choice(total_cells, size=umap_train_n, replace=False)
    return np.sort(rng.choice(total_cells, size=plot_n, replace=False))


def _load_selected_cell_metadata(
    df_cell_path: Path,
    selected_indices: np.ndarray,
    columns: tuple[str, ...] = ("major_trajectory", "celltype_update"),
) -> pd.DataFrame:
    selected = set(int(i) for i in selected_indices)
    rows = []
    with df_cell_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row_idx, row in enumerate(reader):
            if row_idx in selected:
                rows.append({column: row.get(column, "NA") or "NA" for column in columns})
                if len(rows) == len(selected_indices):
                    break
    return pd.DataFrame(rows, columns=columns)


def add_cell_metadata_to_umap_sample(
    umap_df: pd.DataFrame,
    *,
    df_cell_path: Path,
    total_cells: int,
    config: StreamingUmapConfig | None = None,
) -> pd.DataFrame:
    """Add df_cell.csv label columns to a UMAP plotting sample."""

    config = config or StreamingUmapConfig.from_env()
    plot_idx = _sample_plot_indices(total_cells, config)
    labels = _load_selected_cell_metadata(df_cell_path, plot_idx)
    if len(labels) != len(umap_df):
        raise ValueError(
            f"Loaded {len(labels):,} labels for {len(umap_df):,} UMAP sample rows"
        )
    enriched = umap_df.reset_index(drop=True).copy()
    for column in labels.columns:
        enriched[column] = labels[column].to_numpy()
    return enriched


def _add_umap_group_labels(
    plot_df: pd.DataFrame,
    label_col: str,
    *,
    skip_labels: set[str] | None = None,
    max_labels: int | None = None,
    min_cells: int = 500,
    fontsize: int = 7,
) -> None:
    skip_labels = skip_labels or set()
    counts = plot_df[label_col].value_counts()
    labels = [
        label
        for label, n_cells in counts.items()
        if label not in skip_labels and n_cells >= min_cells
    ]
    if max_labels is not None:
        labels = labels[:max_labels]

    for label in labels:
        group = plot_df.loc[plot_df[label_col] == label, ["UMAP1", "UMAP2"]]
        if group.empty:
            continue
        x = float(group["UMAP1"].median())
        y = float(group["UMAP2"].median())
        text = plt.text(
            x,
            y,
            str(label).replace("_", " "),
            ha="center",
            va="center",
            fontsize=fontsize,
            weight="bold",
            color="black",
            clip_on=False,
            bbox={
                "boxstyle": "round,pad=0.18",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.72,
            },
        )
        text.set_path_effects([pe.withStroke(linewidth=2.0, foreground="white")])


def plot_umap_by_cell_type(
    umap_df: pd.DataFrame,
    *,
    out_dir: Path,
    savefig: Callable[[str], None] | None = None,
    top_n_celltypes: int = 30,
) -> None:
    """Write UMAP plots colored by broad and fine cell-type labels."""

    if savefig is None:
        def savefig(name: str) -> None:
            plt.tight_layout()
            plt.savefig(out_dir / name, bbox_inches="tight")
            plt.close()

    if "major_trajectory" in umap_df.columns:
        plt.figure(figsize=(8, 6))
        sns.scatterplot(
            data=umap_df,
            x="UMAP1",
            y="UMAP2",
            hue="major_trajectory",
            s=1,
            linewidth=0,
            alpha=0.35,
        )
        _add_umap_group_labels(
            umap_df,
            "major_trajectory",
            min_cells=1_000,
            fontsize=7,
        )
        plt.legend(markerscale=6, bbox_to_anchor=(1.02, 1), loc="upper left")
        plt.title("Full-data streaming UMAP sample by major trajectory")
        savefig("full_umap_by_major_trajectory.png")

    if "celltype_update" in umap_df.columns:
        top = umap_df["celltype_update"].value_counts().head(top_n_celltypes).index
        plot_df = umap_df.copy()
        plot_df["celltype_update_top"] = np.where(
            plot_df["celltype_update"].isin(top),
            plot_df["celltype_update"],
            "Other",
        )
        plt.figure(figsize=(8, 6))
        sns.scatterplot(
            data=plot_df,
            x="UMAP1",
            y="UMAP2",
            hue="celltype_update_top",
            s=1,
            linewidth=0,
            alpha=0.35,
        )
        _add_umap_group_labels(
            plot_df,
            "celltype_update_top",
            skip_labels={"Other"},
            max_labels=top_n_celltypes,
            min_cells=500,
            fontsize=6,
        )
        plt.legend(markerscale=6, bbox_to_anchor=(1.02, 1), loc="upper left")
        plt.title(f"Full-data streaming UMAP sample by top {top_n_celltypes} cell types")
        savefig("full_umap_by_celltype_update_top30.png")


def run_streaming_umap(
    *,
    h5ad_files: list[Path],
    out_dir: Path,
    adata_summary: pd.DataFrame,
    common_genes: pd.Index,
    obs_all: pd.DataFrame,
    stage_cols: list[str],
    savefig: Callable[[str], None],
    cell_metadata_path: Path | None = None,
    config: StreamingUmapConfig | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Compute a memory-bounded full-data UMAP and return the plotting sample."""

    config = config or StreamingUmapConfig.from_env()
    out_dir.mkdir(parents=True, exist_ok=True)
    log_progress = _make_logger(out_dir)
    rng = np.random.default_rng(config.random_seed)

    total_cells = int(adata_summary["n_obs"].sum())
    n_vars = int(adata_summary["n_vars"].iloc[0])
    log_progress(f"Streaming full-data UMAP for {total_cells:,} cells x {n_vars:,} genes")

    first_chunk = next(
        _iter_matrix_chunks(
            h5ad_files,
            chunk_size=min(config.chunk_cells, 5000),
        )
    )[3]
    do_normalize = (
        _chunk_looks_like_counts(first_chunk)
        if config.normalize_log1p == "auto"
        else config.normalize_log1p == "1"
    )
    log_progress(f"Normalize/log1p chunks: {do_normalize}")
    del first_chunk

    hvg_path = out_dir / "streaming_hvg_genes.csv"
    if hvg_path.exists() and config.reuse_streaming_hvg:
        hvg_genes = pd.read_csv(hvg_path)["gene"].astype(str)
        hvg_idx = common_genes.get_indexer(hvg_genes)
        hvg_idx = hvg_idx[hvg_idx >= 0]
        log_progress(f"Reusing {len(hvg_idx):,} streaming HVGs from {hvg_path}")
    else:
        sum_x = np.zeros(n_vars, dtype=np.float64)
        sum_x2 = np.zeros(n_vars, dtype=np.float64)
        seen = 0
        for _, start, end, x in _iter_matrix_chunks(
            h5ad_files,
            chunk_size=config.chunk_cells,
        ):
            x = _maybe_normalize(x, do_normalize)
            sum_x += np.asarray(x.sum(axis=0)).ravel()
            sum_x2 += (
                np.asarray(x.power(2).sum(axis=0)).ravel()
                if sparse.issparse(x)
                else np.square(x).sum(axis=0)
            )
            seen += end - start
            if seen % 500_000 == 0 or seen == total_cells:
                log_progress(f"HVG pass: {seen:,}/{total_cells:,} cells")
            del x

        mean_x = sum_x / seen
        var_x = np.maximum(sum_x2 / seen - mean_x**2, 0)
        hvg_idx = np.argsort(var_x)[-min(config.n_hvg, n_vars) :]
        hvg_idx.sort()
        pd.DataFrame(
            {"gene": common_genes[hvg_idx].astype(str), "variance": var_x[hvg_idx]}
        ).to_csv(hvg_path, index=False)
        log_progress(f"Selected {len(hvg_idx):,} streaming HVGs")

    n_pcs = min(config.n_pcs, len(hvg_idx) - 1)
    pca_train_n = min(config.pca_train_cells, total_cells)
    pca_train_idx = np.sort(rng.choice(total_cells, size=pca_train_n, replace=False))
    np.save(out_dir / "pca_train_indices.npy", pca_train_idx)

    sample_blocks = []
    cursor = 0
    next_sample_log = 25_000
    for _, start, end, x in _iter_matrix_chunks(
        h5ad_files,
        var_idx=hvg_idx,
        chunk_size=config.chunk_cells,
    ):
        take = pca_train_idx[(pca_train_idx >= start) & (pca_train_idx < end)] - start
        if take.size:
            x = _maybe_normalize(x, do_normalize)
            block = x[take, :] if sparse.issparse(x) else np.asarray(x, dtype=np.float32)[take, :]
            sample_blocks.append(block.tocsr() if sparse.issparse(block) else sparse.csr_matrix(block))
            cursor += take.size
            if cursor >= next_sample_log or cursor == pca_train_n:
                log_progress(f"PCA/SVD sample collection: {cursor:,}/{pca_train_n:,} cells")
                next_sample_log += 25_000
        del x

    sample_matrix = sparse.vstack(sample_blocks, format="csr").astype(np.float32)
    del sample_blocks
    log_progress(
        f"Fitting TruncatedSVD on {sample_matrix.shape[0]:,} sampled cells x "
        f"{sample_matrix.shape[1]:,} HVGs"
    )
    svd = TruncatedSVD(n_components=n_pcs, random_state=config.random_seed)
    svd.fit(sample_matrix)
    del sample_matrix

    pca_path = out_dir / "full_streaming_pca.npy"
    pca_scores = np.lib.format.open_memmap(
        pca_path,
        mode="w+",
        dtype=np.float32,
        shape=(total_cells, n_pcs),
    )
    for _, start, end, x in _iter_matrix_chunks(
        h5ad_files,
        var_idx=hvg_idx,
        chunk_size=config.chunk_cells,
    ):
        x = _maybe_normalize(x, do_normalize)
        pca_scores[start:end, :] = svd.transform(x).astype(np.float32)
        if end % 500_000 == 0 or end == total_cells:
            log_progress(f"SVD transform pass: {end:,}/{total_cells:,} cells")
        del x
    pca_scores.flush()

    pd.DataFrame(
        {
            "pc": np.arange(1, n_pcs + 1),
            "explained_variance_ratio": svd.explained_variance_ratio_,
        }
    ).to_csv(out_dir / "streaming_pca_variance.csv", index=False)

    train_n = min(config.umap_train_cells, total_cells)
    train_idx = np.sort(rng.choice(total_cells, size=train_n, replace=False))
    np.save(out_dir / "umap_train_indices.npy", train_idx)
    reducer = umap.UMAP(
        n_neighbors=config.n_neighbors,
        min_dist=config.umap_min_dist,
        metric="euclidean",
        random_state=config.random_seed,
        low_memory=True,
        verbose=False,
    )
    log_progress(f"Fitting UMAP on {train_n:,} sampled cells, then projecting all cells.")
    reducer.fit(np.asarray(pca_scores[train_idx, :]))

    coords_path = out_dir / "full_umap_streaming.npy"
    coords = np.lib.format.open_memmap(
        coords_path,
        mode="w+",
        dtype=np.float32,
        shape=(total_cells, 2),
    )
    for start in range(0, total_cells, config.umap_transform_batch):
        end = min(start + config.umap_transform_batch, total_cells)
        coords[start:end, :] = reducer.transform(
            np.asarray(pca_scores[start:end, :])
        ).astype(np.float32)
        if end % 500_000 == 0 or end == total_cells:
            log_progress(f"UMAP transform: {end:,}/{total_cells:,} cells")
    coords.flush()

    pd.DataFrame(np.asarray(coords), columns=["UMAP1", "UMAP2"]).to_csv(
        out_dir / "full_umap_streaming_coordinates.csv.gz",
        index=False,
    )

    plt.figure(figsize=(7, 6))
    plt.hexbin(coords[:, 0], coords[:, 1], gridsize=450, bins="log", mincnt=1, cmap="viridis")
    plt.colorbar(label="log10(cell count)")
    plt.xlabel("UMAP1")
    plt.ylabel("UMAP2")
    plt.title("Full-data streaming UMAP density")
    savefig("full_umap_streaming_density.png")

    plot_n = min(config.umap_plot_cells, total_cells)
    plot_idx = np.sort(rng.choice(total_cells, size=plot_n, replace=False))
    umap_df = pd.DataFrame(np.asarray(coords[plot_idx, :]), columns=["UMAP1", "UMAP2"])
    obs_reset = obs_all.reset_index(drop=True)
    umap_df["source_file"] = obs_reset["source_file"].iloc[plot_idx].to_numpy()
    for col in stage_cols[:4]:
        umap_df[col] = obs_reset[col].iloc[plot_idx].astype(str).to_numpy()
    if cell_metadata_path is not None and cell_metadata_path.exists():
        labels = _load_selected_cell_metadata(cell_metadata_path, plot_idx)
        for column in labels.columns:
            umap_df[column] = labels[column].to_numpy()
    umap_df.to_csv(out_dir / "full_umap_streaming_plot_sample.csv.gz", index=False)

    plt.figure(figsize=(7, 6))
    sns.scatterplot(
        data=umap_df,
        x="UMAP1",
        y="UMAP2",
        hue="source_file",
        s=1,
        linewidth=0,
        alpha=0.35,
    )
    plt.legend(markerscale=6, bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.title("Full-data streaming UMAP sample by source file")
    savefig("full_umap_streaming_sample_by_file.png")
    plot_umap_by_cell_type(umap_df, out_dir=out_dir, savefig=savefig)

    summary = {
        "total_cells": total_cells,
        "n_hvg": int(len(hvg_idx)),
        "n_pcs": int(n_pcs),
        "pca_train_cells": int(pca_train_n),
        "umap_train_cells": int(train_n),
        "plot_cells": int(plot_n),
        "normalized_log1p": bool(do_normalize),
    }
    return umap_df, summary


def summarize_cell_metadata(df_cell_path: Path, out_dir: Path) -> dict[str, pd.DataFrame]:
    """Summarize labels and per-embryo staging from df_cell.csv."""

    out_dir.mkdir(parents=True, exist_ok=True)
    label_counts = {"major_trajectory": Counter(), "celltype_update": Counter()}
    embryo_day = Counter()
    embryo_celltype = Counter()
    row_count = 0

    with df_cell_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        columns = reader.fieldnames or []
        for row in reader:
            row_count += 1
            embryo = row.get("embryo_id", "") or "NA"
            day = row.get("day", "") or "NA"
            major = row.get("major_trajectory", "") or "NA"
            celltype = row.get("celltype_update", "") or "NA"
            label_counts["major_trajectory"][major] += 1
            label_counts["celltype_update"][celltype] += 1
            embryo_day[(embryo, day)] += 1
            embryo_celltype[(embryo, celltype)] += 1

    label_rows = [
        {"column": column, "label": label, "n_cells": n_cells}
        for column, counts in label_counts.items()
        for label, n_cells in counts.items()
    ]
    label_summary = pd.DataFrame(label_rows).sort_values(
        ["column", "n_cells"],
        ascending=[True, False],
    )
    label_summary.to_csv(out_dir / "cell_label_counts.csv", index=False)

    embryo_staging = pd.DataFrame(
        [
            {"embryo_id": embryo, "day": day, "n_cells": n_cells}
            for (embryo, day), n_cells in embryo_day.items()
        ]
    ).sort_values(["day", "embryo_id"])
    embryo_staging.to_csv(out_dir / "embryo_staging_summary.csv", index=False)

    embryo_labels = pd.DataFrame(
        [
            {"embryo_id": embryo, "celltype_update": celltype, "n_cells": n_cells}
            for (embryo, celltype), n_cells in embryo_celltype.items()
        ]
    ).sort_values(["embryo_id", "n_cells"], ascending=[True, False])
    embryo_labels.to_csv(out_dir / "embryo_celltype_counts.csv", index=False)

    presence = pd.DataFrame(
        [
            {
                "metadata_source": df_cell_path.name,
                "field": "major_trajectory",
                "present": "major_trajectory" in columns,
                "n_unique": len(label_counts["major_trajectory"]),
            },
            {
                "metadata_source": df_cell_path.name,
                "field": "celltype_update",
                "present": "celltype_update" in columns,
                "n_unique": len(label_counts["celltype_update"]),
            },
            {
                "metadata_source": df_cell_path.name,
                "field": "day",
                "present": "day" in columns,
                "n_unique": len({day for _, day in embryo_day}),
            },
            {
                "metadata_source": df_cell_path.name,
                "field": "embryo_id",
                "present": "embryo_id" in columns,
                "n_unique": len({embryo for embryo, _ in embryo_day}),
            },
        ]
    )
    presence.to_csv(out_dir / "cell_metadata_presence.csv", index=False)

    return {
        "presence": presence,
        "label_summary": label_summary,
        "embryo_staging": embryo_staging,
        "embryo_labels": embryo_labels,
        "row_count": pd.DataFrame([{"n_cells": row_count}]),
    }
