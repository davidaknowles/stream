"""Helpers for plotting STREAM/CFM velocities with scVelo."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from .config import StreamConfig
from .data import parse_day_value


def scvelo_cache_dir(config: StreamConfig, n_cells: int, seed: int) -> Path:
    return config.out_dir / "scvelo_stream" / f"cache_n{n_cells}_seed{seed}"


def load_or_sample_model_adata(
    config: StreamConfig,
    n_cells: int = 5000,
    seed: int = 1337,
    cache_dir: str | Path | None = None,
    force: bool = False,
):
    """Load cached sampled AnnData, or sample and cache it."""

    import anndata as ad

    cache_dir = Path(cache_dir) if cache_dir is not None else scvelo_cache_dir(config, n_cells, seed)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / "base_sample.h5ad"
    if path.exists() and not force:
        return ad.read_h5ad(path)
    adata = sample_model_adata(config, n_cells=n_cells, seed=seed)
    adata = order_days_for_plot(adata)
    adata.write_h5ad(path, compression="gzip")
    return adata


def sample_model_adata(
    config: StreamConfig,
    n_cells: int = 5000,
    seed: int = 1337,
    umap_coordinates_csv: str | Path | None = None,
):
    """Sample cells, selected-gene expression, metadata, and precomputed UMAP coordinates."""

    import anndata as ad

    rng = np.random.default_rng(seed)
    selected = pd.read_csv(config.out_dir / "selected_genes.csv")
    gene_ids = selected["gene_id"].tolist()
    file_rows = []
    offset = 0
    for path in sorted(config.adata_dir.glob("*.h5ad")):
        backed = ad.read_h5ad(path, backed="r")
        n_obs = backed.n_obs
        obs = backed.obs[["cell_id", "day", "embryo_id", "experimental_batch"]].copy()
        obs["source_file"] = path.name
        obs["local_idx"] = np.arange(n_obs)
        obs["global_idx"] = offset + obs["local_idx"]
        file_rows.append(obs)
        offset += n_obs
        backed.file.close()
    obs_all = pd.concat(file_rows, axis=0, ignore_index=True)
    if n_cells < len(obs_all):
        sampled = obs_all.iloc[rng.choice(len(obs_all), size=n_cells, replace=False)].copy()
    else:
        sampled = obs_all.copy()
    sampled = sampled.sort_values("global_idx").reset_index(drop=True)

    x = _read_expression_rows(config.adata_dir, sampled, gene_ids)
    coords_path = Path(umap_coordinates_csv) if umap_coordinates_csv is not None else config.project_root / "outputs/jax_adata_eda/full_umap_streaming_coordinates.csv.gz"
    coords = read_umap_rows(coords_path, sampled["global_idx"].to_numpy())
    obs = sampled.drop(columns=["local_idx"]).copy()
    obs.index = obs["cell_id"].astype(str)
    var = selected.set_index("gene_id")[["gene_short_name", "gene_type", "chr"]].copy()
    adata = ad.AnnData(X=x, obs=obs, var=var)
    adata.obsm["X_umap"] = coords.astype(np.float32)
    return adata


def _read_expression_rows(adata_dir: Path, sampled: pd.DataFrame, gene_ids: list[str]) -> np.ndarray:
    import anndata as ad

    chunks = []
    for source_file, group in sampled.groupby("source_file", sort=False):
        path = adata_dir / source_file
        backed = ad.read_h5ad(path, backed="r")
        var_gene_ids = pd.Index(backed.var["gene_id"] if "gene_id" in backed.var else backed.var_names)
        gene_idx = var_gene_ids.get_indexer(gene_ids)
        if np.any(gene_idx < 0):
            missing = np.asarray(gene_ids)[gene_idx < 0][:10]
            raise ValueError(f"Selected genes missing from {path}: {missing}")
        local = group["local_idx"].to_numpy()
        order = np.argsort(local)
        local_sorted = local[order]
        x = backed.X[local_sorted, :][:, gene_idx]
        if hasattr(x, "toarray"):
            x = x.toarray()
        restored = np.empty_like(np.asarray(x, dtype=np.float32))
        restored[order] = np.asarray(x, dtype=np.float32)
        chunks.append((group.index.to_numpy(), restored))
        backed.file.close()
    out = np.empty((len(sampled), len(gene_ids)), dtype=np.float32)
    for idx, values in chunks:
        out[idx] = values
    return out


def read_umap_rows(path: str | Path, global_indices: np.ndarray, chunksize: int = 1_000_000) -> np.ndarray:
    targets = np.asarray(global_indices, dtype=np.int64)
    order = np.argsort(targets)
    sorted_targets = targets[order]
    coords = np.empty((len(targets), 2), dtype=np.float32)
    cursor = 0
    start = 0
    for chunk in pd.read_csv(path, chunksize=chunksize):
        end = start + len(chunk)
        lo = cursor
        while cursor < len(sorted_targets) and sorted_targets[cursor] < end:
            cursor += 1
        if cursor > lo:
            rel = sorted_targets[lo:cursor] - start
            coords[order[lo:cursor]] = chunk.iloc[rel][["UMAP1", "UMAP2"]].to_numpy(dtype=np.float32)
        start = end
        if cursor == len(sorted_targets):
            break
    if cursor != len(sorted_targets):
        raise ValueError(f"Only found {cursor}/{len(sorted_targets)} requested UMAP rows in {path}")
    return coords


def load_variant_model(config: StreamConfig, variant: str, device: str):
    import torch
    from .train import build_model, load_cre_npz

    cfg = replace(config, model_variant=variant)
    n_genes = len(pd.read_csv(config.out_dir / "selected_genes.csv"))
    cre_inputs = None
    cre_dim = None
    if variant != "standard_cfm":
        cre_inputs = load_cre_npz(config.out_dir / "cre_token_arrays.npz", torch.device(device))
        cre_dim = int(cre_inputs["cre_embeddings"].shape[-1])
    model = build_model(cfg, n_genes=n_genes, cre_dim=cre_dim).to(device)
    checkpoint = torch.load(config.out_dir / f"model_{variant}.pt", map_location=device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, cre_inputs


def predict_variant_velocity(adata, config: StreamConfig, variant: str, device: str = "cpu", batch_size: int = 512) -> np.ndarray:
    import torch

    model, cre_inputs = load_variant_model(config, variant, device)
    x = np.asarray(adata.X, dtype=np.float32)
    outputs = []
    with torch.no_grad():
        for start in range(0, x.shape[0], batch_size):
            xb = torch.as_tensor(x[start : start + batch_size], device=device)
            if cre_inputs is None:
                pred = model(xb)
            else:
                pred = model(xb, **cre_inputs)
            outputs.append(pred.detach().cpu().numpy().astype(np.float32))
    return np.vstack(outputs)


def load_or_predict_variant_velocity(
    adata,
    config: StreamConfig,
    variant: str,
    device: str = "cpu",
    batch_size: int = 512,
    cache_dir: str | Path | None = None,
    force: bool = False,
) -> np.ndarray:
    """Load cached model velocities, or predict and cache them."""

    cache_dir = Path(cache_dir) if cache_dir is not None else scvelo_cache_dir(config, adata.n_obs, int(config.seed))
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"velocity_{variant}.npy"
    if path.exists() and not force:
        return np.load(path).astype(np.float32)
    velocity = predict_variant_velocity(adata, config, variant=variant, device=device, batch_size=batch_size)
    np.save(path, velocity.astype(np.float16))
    return velocity


def add_velocity_layer(adata, velocity: np.ndarray, vkey: str = "velocity"):
    out = adata.copy()
    out.layers["spliced"] = np.asarray(out.X, dtype=np.float32)
    out.layers[vkey] = velocity.astype(np.float32)
    return out


def plot_velocity_stream(
    adata,
    output_path: str | Path,
    vkey: str = "velocity",
    color: str = "embryonic_day",
    color_map: str = "viridis",
    n_neighbors: int = 30,
):
    """Compute scVelo graph/embedding and save a velocity_embedding_stream plot."""

    import matplotlib.pyplot as plt
    import scvelo as scv

    scv.settings.verbosity = 2
    scv.pp.neighbors(adata, n_neighbors=n_neighbors, use_rep="X")
    scv.tl.velocity_graph(adata, vkey=vkey, n_jobs=1)
    scv.tl.velocity_embedding(adata, basis="umap", vkey=vkey)
    scv.pl.velocity_embedding_stream(
        adata,
        basis="umap",
        vkey=vkey,
        color=color,
        color_map=color_map,
        colorbar=True,
        legend_loc="right margin",
        xlabel="",
        ylabel="",
        title="",
        frameon=False,
        show=False,
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def cache_velocity_stream_inputs(
    adata,
    output_path: str | Path,
    vkey: str = "velocity",
    n_neighbors: int = 30,
    force: bool = False,
) -> Path:
    """Cache UMAP-space velocity and stream grid arrays for fast restyling."""

    output_path = Path(output_path)
    if output_path.exists() and not force:
        return output_path

    import scvelo as scv
    from scvelo.plotting.velocity_embedding_stream import compute_velocity_on_grid

    scv.settings.verbosity = 2
    scv.pp.neighbors(adata, n_neighbors=n_neighbors, use_rep="X")
    scv.tl.velocity_graph(adata, vkey=vkey, n_jobs=1)
    scv.tl.velocity_embedding(adata, basis="umap", vkey=vkey)
    x = np.asarray(adata.obsm["X_umap"], dtype=np.float32)
    v = np.asarray(adata.obsm[f"{vkey}_umap"], dtype=np.float32)
    x_grid, v_grid = compute_velocity_on_grid(
        X_emb=x,
        V_emb=v,
        density=1,
        smooth=None,
        min_mass=None,
        n_neighbors=None,
        autoscale=False,
        adjust_for_stream=True,
        cutoff_perc=None,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        X_umap=x.astype(np.float32),
        V_umap=v.astype(np.float32),
        X_grid=x_grid.astype(np.float32),
        V_grid=v_grid.astype(np.float32),
    )
    return output_path


def plot_cached_velocity_stream(
    adata,
    cache_path: str | Path,
    output_path: str | Path,
    color: str = "embryonic_day",
    color_map: str = "viridis",
):
    """Render a velocity stream plot from cached embedding/grid arrays."""

    import matplotlib.pyplot as plt
    import scvelo as scv

    cached = np.load(cache_path)
    scv.pl.velocity_embedding_stream(
        adata,
        basis="umap",
        vkey="velocity",
        color=color,
        color_map=color_map,
        colorbar=True,
        legend_loc="right margin",
        xlabel="",
        ylabel="",
        title="",
        frameon=False,
        show=False,
        X=cached["X_umap"],
        V=cached["V_umap"],
        X_grid=cached["X_grid"],
        V_grid=cached["V_grid"],
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def order_days_for_plot(adata):
    if "day" in adata.obs:
        categories = sorted(adata.obs["day"].astype(str).unique(), key=parse_day_value)
        adata.obs["day"] = pd.Categorical(adata.obs["day"].astype(str), categories=categories, ordered=True)
        adata.obs["embryonic_day"] = adata.obs["day"].astype(str).map(parse_day_value).astype(float)
    return adata
