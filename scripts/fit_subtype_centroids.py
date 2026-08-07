#!/usr/bin/env python
"""Fit fine-grained atlas subtype centroids using observed timepoints only."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

from stream_model.config import StreamConfig
from stream_model.data import canonical_day_label
from stream_model.denoise import PCADenoiser
from stream_model.subtype import SubtypeCentroidClassifier


def balanced_split(
    cells: pd.DataFrame,
    label_column: str,
    train_per_subtype: int,
    validation_per_subtype: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for label, group in cells.groupby(label_column, sort=True):
        order = rng.permutation(len(group))
        n_train = min(train_per_subtype, max(1, len(group) - validation_per_subtype))
        n_validation = min(validation_per_subtype, len(group) - n_train)
        for partition, indices in (
            ("train", order[:n_train]),
            ("validation", order[n_train : n_train + n_validation]),
        ):
            selected = group.iloc[indices][["cell_id"]].copy()
            selected["subtype"] = str(label)
            selected["partition"] = partition
            rows.append(selected)
    return pd.concat(rows, ignore_index=True)


def load_coordinates(
    adata_dir: Path,
    gene_ids: list[str],
    selected_cells: pd.DataFrame,
    pca: PCADenoiser,
    chunk_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    selected = selected_cells.set_index("cell_id")
    coordinate_chunks = []
    label_chunks = []
    partition_chunks = []
    gene_indices = None
    for path in sorted(adata_dir.glob("*.h5ad")):
        atlas = ad.read_h5ad(path, backed="r")
        var_gene_ids = pd.Index(atlas.var["gene_id"] if "gene_id" in atlas.var else atlas.var_names)
        if gene_indices is None:
            gene_indices = var_gene_ids.get_indexer(gene_ids)
            if np.any(gene_indices < 0):
                raise ValueError(f"Selected genes are missing from {path}")
        obs_ids = pd.Index(
            atlas.obs["cell_id"].astype(str) if "cell_id" in atlas.obs else atlas.obs_names.astype(str)
        )
        row_indices = np.flatnonzero(obs_ids.isin(selected.index))
        for start in range(0, len(row_indices), chunk_size):
            rows = row_indices[start : start + chunk_size]
            counts = atlas.X[rows, :][:, gene_indices]
            if hasattr(counts, "toarray"):
                counts = counts.toarray()
            coordinate_chunks.append(pca.transform_counts(np.asarray(counts, dtype=np.float32)))
            info = selected.loc[obs_ids[rows]]
            label_chunks.append(info["subtype"].astype(str).to_numpy())
            partition_chunks.append(info["partition"].astype(str).to_numpy())
        atlas.file.close()
    if not coordinate_chunks:
        raise ValueError("No selected subtype-classifier cells were found in AnnData shards")
    return (
        np.vstack(coordinate_chunks),
        np.concatenate(label_chunks),
        np.concatenate(partition_chunks),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/stream_mouse_dev.yaml")
    parser.add_argument("--out-dir", default="outputs/stream_hvg10000")
    parser.add_argument("--timepoint-split", required=True)
    parser.add_argument("--pca-artifact", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--label-column", default="celltype_update")
    parser.add_argument("--train-per-subtype", type=int, default=512)
    parser.add_argument("--validation-per-subtype", type=int, default=128)
    parser.add_argument("--read-chunk-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    cfg = StreamConfig.from_yaml(args.config)
    out_dir = cfg.resolve_path(args.out_dir)
    selected_genes = pd.read_csv(out_dir / "selected_genes.csv")
    gene_ids = selected_genes["gene_id"].astype(str).tolist()
    pca = PCADenoiser.load(cfg.resolve_path(args.pca_artifact))
    gene_hash = hashlib.sha256("\n".join(gene_ids).encode()).hexdigest()
    if pca.components.shape[1] != len(gene_ids) or pca.metadata.get("gene_ids_sha256") != gene_hash:
        raise ValueError("PCA artifact does not match the selected gene panel")
    with cfg.resolve_path(args.timepoint_split).open() as handle:
        split = json.load(handle)
    heldout = {canonical_day_label(day) for day in split["heldout_days"]}
    cells = pd.read_csv(cfg.cell_metadata_csv, usecols=["cell_id", "day", args.label_column])
    cells["day"] = cells["day"].map(canonical_day_label)
    cells = cells.loc[~cells["day"].isin(heldout)].dropna(subset=[args.label_column])
    chosen = balanced_split(
        cells,
        args.label_column,
        args.train_per_subtype,
        args.validation_per_subtype,
        args.seed,
    )
    coordinates, labels, partitions = load_coordinates(
        cfg.adata_dir,
        gene_ids,
        chosen,
        pca,
        args.read_chunk_size,
    )
    train = partitions == "train"
    validation = partitions == "validation"
    subtype_labels = np.unique(labels[train])
    centroids = np.vstack([coordinates[train & (labels == label)].mean(axis=0) for label in subtype_labels])
    provisional = SubtypeCentroidClassifier(subtype_labels, centroids, {})
    validation_prediction = provisional.predict_coordinates(coordinates[validation])
    validation_truth = pd.Index(subtype_labels).get_indexer(labels[validation])
    valid = validation_truth >= 0
    per_subtype_accuracy = []
    for subtype_index in np.unique(validation_truth[valid]):
        subtype_rows = validation_truth == subtype_index
        per_subtype_accuracy.append(np.mean(validation_prediction[subtype_rows] == subtype_index))
    metadata = {
        "label_column": args.label_column,
        "heldout_days": sorted(heldout),
        "n_subtypes": int(len(subtype_labels)),
        "n_train_cells": int(train.sum()),
        "n_validation_cells": int(valid.sum()),
        "validation_accuracy": float(np.mean(validation_prediction[valid] == validation_truth[valid])),
        "validation_balanced_accuracy": float(np.mean(per_subtype_accuracy)),
        "pca_artifact": str(cfg.resolve_path(args.pca_artifact)),
        "gene_ids_sha256": gene_hash,
        "seed": args.seed,
    }
    classifier = SubtypeCentroidClassifier(subtype_labels, centroids, metadata)
    output = cfg.resolve_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    classifier.save(output)
    print(json.dumps(metadata, indent=2, sort_keys=True))
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
