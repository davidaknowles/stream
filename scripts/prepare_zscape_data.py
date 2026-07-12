#!/usr/bin/env python
"""Download and convert the ZSCAPE reference atlas and ZEPA cCREs for STREAM."""

from __future__ import annotations

import argparse
import gzip
import json
import shutil
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import mmread


ZSCAPE_BASE = "http://trapnell-lab-s3-zscape.s3-website-us-west-2.amazonaws.com"
DEFAULTS = {
    "matrix": f"{ZSCAPE_BASE}/zscape_reference_raw_counts.mtx.gz",
    "cells": f"{ZSCAPE_BASE}/zscape_reference_cell_metadata.csv.gz",
    "genes": f"{ZSCAPE_BASE}/zscape_reference_gene_metadata.csv.gz",
    "ccres": "https://ndownloader.figshare.com/files/45645213",
    "dynamic_ccres": "https://ndownloader.figshare.com/files/45645216",
    "fasta": "https://ftp.ensembl.org/pub/release-113/fasta/danio_rerio/dna/Danio_rerio.GRCz11.dna.primary_assembly.fa.gz",
    "gtf": "https://ftp.ensembl.org/pub/release-113/gtf/danio_rerio/Danio_rerio.GRCz11.113.gtf.gz",
}


def download(url: str, path: Path) -> None:
    if path.exists() and path.stat().st_size > 0:
        print(f"Reusing {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    print(f"Downloading {url} -> {path}", flush=True)
    with urllib.request.urlopen(url) as response, tmp.open("wb") as handle:
        shutil.copyfileobj(response, handle)
    tmp.replace(path)


def normalize_metadata(path: Path, kind: str) -> pd.DataFrame:
    table = pd.read_csv(path)
    if kind == "cells":
        id_column = next((col for col in ("cell_id", "cell", "barcode", "Unnamed: 0") if col in table.columns), None)
        if id_column is None:
            id_column = table.columns[0]
        table = table.rename(columns={id_column: "cell_id"})
        if "timepoint" not in table.columns:
            raise ValueError(f"ZSCAPE metadata lacks timepoint; columns are {table.columns.tolist()}")
        table["day"] = table["timepoint"].astype(str).str.replace(r"\\.0$", "", regex=True)
        return table
    id_column = next((col for col in ("gene_id", "id", "Unnamed: 0") if col in table.columns), None)
    if id_column is None:
        id_column = table.columns[0]
    table = table.rename(columns={id_column: "gene_id"})
    if "gene_short_name" not in table.columns:
        table["gene_short_name"] = table.get("gene_name", table["gene_id"])
    return table


def _column(table: pd.DataFrame, names: tuple[str, ...]) -> str:
    normalized = {str(col).strip().lower().replace("_", "").replace(" ", ""): col for col in table.columns}
    for name in names:
        column = normalized.get(name)
        if column is not None:
            return column
    raise ValueError(f"Could not find one of {names} in ZEPA columns: {table.columns.tolist()}")


def zepa_excel_to_bed(excel_paths: list[Path], output_path: Path) -> int:
    """Convert a ZEPA cCRE worksheet to a normalized BED6-like table."""

    candidates = []
    for excel_path in excel_paths:
        try:
            sheets = pd.read_excel(excel_path, sheet_name=None)
        except ValueError as exc:
            if "Sheet name is an empty list" not in str(exc):
                raise
            print(f"Skipping ZEPA workbook with no readable worksheets: {excel_path}", flush=True)
            continue
        for sheet, table in sheets.items():
            try:
                chrom = _column(table, ("chr", "chrom", "chromosome", "seqnames"))
                start = _column(table, ("start", "chromstart", "startposition"))
                end = _column(table, ("end", "chromend", "endposition"))
            except ValueError:
                continue
            ident = next((col for col in ("cCRE_id", "ccre_id", "id", "peak_id", "peak") if col in table.columns), None)
            state = next((col for col in ("state", "class", "annotation", "type") if col in table.columns), None)
            out = pd.DataFrame({"chrom": table[chrom].astype(str), "start": pd.to_numeric(table[start]), "end": pd.to_numeric(table[end])})
            out = out.dropna().astype({"start": int, "end": int})
            out = out[out["end"] > out["start"]]
            prefix = f"{excel_path.stem}_{sheet}"
            out["ccre_id"] = table.loc[out.index, ident].astype(str).to_numpy() if ident else [f"{prefix}_{i}" for i in out.index]
            out["state"] = table.loc[out.index, state].astype(str).to_numpy() if state else "ZEPA"
            candidates.append(out)
    if not candidates:
        raise ValueError("No coordinate worksheet found in ZEPA cCRE workbook")
    cres = pd.concat(candidates, ignore_index=True).drop_duplicates(["chrom", "start", "end"])
    cres["chrom"] = cres["chrom"].where(cres["chrom"].str.startswith("chr"), "chr" + cres["chrom"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(output_path, "wt") as handle:
        cres.to_csv(handle, sep="\t", header=False, index=False)
    return len(cres)


def write_h5ad_shards(
    matrix_path: Path,
    all_cells: pd.DataFrame,
    cells: pd.DataFrame,
    genes: pd.DataFrame,
    output_dir: Path,
    shard_cells: int,
) -> int:
    import anndata as ad

    print("Reading Matrix Market counts into sparse memory", flush=True)
    matrix = mmread(matrix_path).tocsr()
    if matrix.shape[1] == len(all_cells) and matrix.shape[0] == len(genes):
        matrix = matrix.T.tocsr()
    elif matrix.shape[0] != len(all_cells) or matrix.shape[1] != len(genes):
        raise ValueError(f"Matrix shape {matrix.shape} does not match cells={len(all_cells)} genes={len(genes)}")
    matrix = matrix[cells["matrix_row"].to_numpy()].tocsr()
    output_dir.mkdir(parents=True, exist_ok=True)
    if any(output_dir.glob("*.h5ad")):
        raise FileExistsError(f"Refusing to overwrite existing AnnData shards in {output_dir}; move them to scratch first.")
    var = genes.set_index("gene_id", drop=False).copy()
    for shard, start in enumerate(range(0, len(cells), shard_cells)):
        end = min(start + shard_cells, len(cells))
        obs = cells.iloc[start:end].set_index("cell_id", drop=False).copy()
        ad.AnnData(X=matrix[start:end], obs=obs, var=var).write_h5ad(output_dir / f"zscape_reference_{shard:03d}.h5ad")
        print(f"Wrote cells {start:,}-{end:,}", flush=True)
    return len(cells)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="downloads/zscape")
    parser.add_argument("--shard-cells", type=int, default=100_000)
    parser.add_argument("--control-target", default="ctrl-uninj")
    parser.add_argument("--include-injected-controls", action="store_true")
    parser.add_argument("--skip-matrix", action="store_true")
    args = parser.parse_args()
    root = Path(args.out_dir)
    raw = root / "raw"
    references = root / "references"
    for name in ("matrix", "cells", "genes", "ccres", "dynamic_ccres"):
        suffix = ".xlsx" if "ccres" in name else ".gz"
        download(DEFAULTS[name], raw / f"{name}{suffix}")
    for name in ("fasta", "gtf"):
        download(DEFAULTS[name], references / Path(DEFAULTS[name]).name)

    ccre_bed = root / "zepa_ccres_grcz11.bed.gz"
    n_ccres = zepa_excel_to_bed([raw / "ccres.xlsx", raw / "dynamic_ccres.xlsx"], ccre_bed)
    all_cells = normalize_metadata(raw / "cells.gz", "cells")
    cells = all_cells.copy()
    genes = normalize_metadata(raw / "genes.gz", "genes")
    if "temp" in cells.columns:
        cells = cells[cells["temp"].astype(str).eq("28C")].copy()
    if "gene_target" in cells.columns and not args.include_injected_controls:
        cells = cells[cells["gene_target"].astype(str).eq(args.control_target)].copy()
    if cells.empty:
        raise ValueError("No cells remain after control filtering")
    cells["matrix_row"] = cells.index.to_numpy()
    cells.to_csv(root / "df_cell.csv", index=True)
    genes.to_csv(root / "df_gene.csv", index=True)
    n_cells = 0 if args.skip_matrix else write_h5ad_shards(raw / "matrix.gz", all_cells, cells, genes, root / "adata", args.shard_cells)
    with (root / "preprocessing_manifest.json").open("w") as handle:
        json.dump({"n_cells": n_cells, "n_genes": len(genes), "n_ccres": n_ccres, "control_target": args.control_target, "include_injected_controls": args.include_injected_controls}, handle, indent=2)
    print(f"Prepared ZSCAPE at {root}")


if __name__ == "__main__":
    main()
