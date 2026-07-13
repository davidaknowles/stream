"""Genome annotation and cCRE linking utilities."""

from __future__ import annotations

import gzip
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd


def _open_text(path: str | Path):
    path = Path(path)
    if path.suffix == ".gz":
        return gzip.open(path, "rt")
    return path.open()


def parse_gtf_attributes(raw: str) -> dict[str, str]:
    attrs: dict[str, str] = {}
    for item in raw.strip().rstrip(";").split(";"):
        item = item.strip()
        if not item:
            continue
        if " " not in item:
            continue
        key, value = item.split(" ", 1)
        attrs[key] = value.strip().strip('"')
    return attrs


def normalize_chrom(chrom: str) -> str:
    chrom = str(chrom)
    return chrom if chrom.startswith("chr") else f"chr{chrom}"


def parse_gtf_tss(
    gtf_path: str | Path,
    gene_ids: Iterable[str] | None = None,
    gene_type: str | None = "protein_coding",
) -> pd.DataFrame:
    """Parse gene TSS coordinates from a GTF.

    GTF coordinates are 1-based inclusive. Returned TSS coordinates are 0-based
    single-base positions suitable for distance calculations against BED files.
    """

    keep_ids = set(gene_ids) if gene_ids is not None else None
    rows: list[dict[str, object]] = []
    with _open_text(gtf_path) as handle:
        for line in handle:
            if not line or line.startswith("#"):
                continue
            chrom, _source, feature, start, end, _score, strand, _frame, attrs_raw = line.rstrip("\n").split("\t")
            if feature != "gene":
                continue
            attrs = parse_gtf_attributes(attrs_raw)
            gid = attrs.get("gene_id", "").split(".")[0]
            if keep_ids is not None and gid not in keep_ids:
                continue
            gtype = attrs.get("gene_type") or attrs.get("gene_biotype")
            if gene_type is not None and gtype != gene_type:
                continue
            start_i = int(start)
            end_i = int(end)
            tss0 = start_i - 1 if strand == "+" else end_i - 1
            rows.append(
                {
                    "gene_id": gid,
                    "gene_name": attrs.get("gene_name", gid),
                    "gene_type": gtype,
                    "chrom": normalize_chrom(chrom),
                    "strand": strand,
                    "tss0": tss0,
                }
            )
    return pd.DataFrame(rows).drop_duplicates("gene_id")


def read_ccre_bed(path: str | Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    with _open_text(path) as handle:
        for i, line in enumerate(handle):
            if not line.strip() or line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            chrom, start, end = fields[:3]
            accession = fields[3] if len(fields) > 3 else f"ccre_{i}"
            element_id = fields[4] if len(fields) > 4 else accession
            state = fields[5] if len(fields) > 5 else "unknown"
            start_i = int(start)
            end_i = int(end)
            rows.append(
                {
                    "ccre_id": accession,
                    "element_id": element_id,
                    "state": state,
                    "chrom": normalize_chrom(chrom),
                    "start": start_i,
                    "end": end_i,
                    "midpoint": (start_i + end_i) // 2,
                    "is_synthetic": False,
                }
            )
    return pd.DataFrame(rows)


def link_cres_to_genes(
    tss: pd.DataFrame,
    cres: pd.DataFrame,
    window_bp: int = 100_000,
    promoter_window_bp: int = 1_000,
    synthetic_promoter_bp: int = 512,
) -> pd.DataFrame:
    """Link cCREs to genes and ensure each gene has a promoter token."""

    links: list[dict[str, object]] = []
    cres_by_chrom = {chrom: df.sort_values("midpoint").reset_index(drop=True) for chrom, df in cres.groupby("chrom")}

    for gene in tss.itertuples(index=False):
        chrom_cres = cres_by_chrom.get(gene.chrom)
        gene_links: list[dict[str, object]] = []
        if chrom_cres is not None and not chrom_cres.empty:
            mids = chrom_cres["midpoint"].to_numpy()
            lo = np.searchsorted(mids, int(gene.tss0) - window_bp, side="left")
            hi = np.searchsorted(mids, int(gene.tss0) + window_bp, side="right")
            for cre in chrom_cres.iloc[lo:hi].itertuples(index=False):
                signed = int(cre.midpoint) - int(gene.tss0)
                gene_links.append(
                    {
                        "gene_id": gene.gene_id,
                        "gene_name": gene.gene_name,
                        "chrom": gene.chrom,
                        "tss0": int(gene.tss0),
                        "strand": gene.strand,
                        "ccre_id": cre.ccre_id,
                        "element_id": cre.element_id,
                        "state": cre.state,
                        "start": int(cre.start),
                        "end": int(cre.end),
                        "midpoint": int(cre.midpoint),
                        "signed_distance": signed,
                        "abs_distance": abs(signed),
                        "is_synthetic": bool(cre.is_synthetic),
                        "is_promoter": False,
                    }
                )

        if gene_links:
            promoter_idx = int(np.argmin([row["abs_distance"] for row in gene_links]))
            if gene_links[promoter_idx]["abs_distance"] <= promoter_window_bp:
                gene_links[promoter_idx]["is_promoter"] = True
            else:
                gene_links.insert(0, _synthetic_promoter_link(gene, synthetic_promoter_bp))
        else:
            gene_links.append(_synthetic_promoter_link(gene, synthetic_promoter_bp))

        promoter_links = [row for row in gene_links if row["is_promoter"]]
        if len(promoter_links) != 1:
            raise ValueError(f"Expected one promoter token for {gene.gene_id}, found {len(promoter_links)}")

        gene_links.sort(key=lambda row: (not row["is_promoter"], row["abs_distance"], row["ccre_id"]))
        for rank, row in enumerate(gene_links):
            row["token_rank"] = rank
            links.append(row)

    return pd.DataFrame(links)


def _synthetic_promoter_link(gene, width: int) -> dict[str, object]:
    half = width // 2
    start = max(0, int(gene.tss0) - half)
    end = int(gene.tss0) + half
    return {
        "gene_id": gene.gene_id,
        "gene_name": gene.gene_name,
        "chrom": gene.chrom,
        "tss0": int(gene.tss0),
        "strand": gene.strand,
        "ccre_id": f"synthetic_promoter:{gene.gene_id}",
        "element_id": f"synthetic_promoter:{gene.gene_id}",
        "state": "synthetic_promoter",
        "start": start,
        "end": end,
        "midpoint": int(gene.tss0),
        "signed_distance": 0,
        "abs_distance": 0,
        "is_synthetic": True,
        "is_promoter": True,
    }


def build_token_arrays(
    links: pd.DataFrame,
    embeddings: pd.DataFrame,
    max_tokens: int,
) -> dict[str, np.ndarray]:
    """Pack per-cCRE embeddings into fixed per-gene token tensors."""

    merged = links.merge(embeddings, on="ccre_id", how="left", validate="many_to_one")
    emb_cols = [col for col in merged.columns if col.startswith("emb_")]
    if not emb_cols:
        raise ValueError("Embedding table must contain emb_* columns")
    genes = links[["gene_id", "gene_name"]].drop_duplicates("gene_id").reset_index(drop=True)
    gene_to_idx = {gid: i for i, gid in enumerate(genes["gene_id"])}
    emb_dim = len(emb_cols)
    token_embeddings = np.zeros((len(genes), max_tokens, emb_dim), dtype=np.float32)
    signed_distance = np.zeros((len(genes), max_tokens), dtype=np.float32)
    is_promoter = np.zeros((len(genes), max_tokens), dtype=bool)
    mask = np.zeros((len(genes), max_tokens), dtype=bool)
    for row in merged.sort_values(["gene_id", "token_rank"]).itertuples(index=False):
        rank = int(row.token_rank)
        if rank >= max_tokens:
            continue
        gi = gene_to_idx[row.gene_id]
        token_embeddings[gi, rank] = np.asarray([getattr(row, col) for col in emb_cols], dtype=np.float32)
        signed_distance[gi, rank] = float(row.signed_distance)
        is_promoter[gi, rank] = bool(row.is_promoter)
        mask[gi, rank] = True
    return {
        "gene_id": genes["gene_id"].to_numpy(dtype=str),
        "gene_name": genes["gene_name"].to_numpy(dtype=str),
        "embeddings": token_embeddings,
        "signed_distance": signed_distance,
        "is_promoter": is_promoter,
        "mask": mask,
    }


def build_token_arrays_from_matrix(
    links: pd.DataFrame,
    ccre_ids: np.ndarray,
    embeddings: np.ndarray,
    max_tokens: int,
) -> dict[str, np.ndarray]:
    """Pack token arrays from a dense cCRE matrix without a wide DataFrame."""

    if embeddings.ndim != 2 or len(ccre_ids) != embeddings.shape[0]:
        raise ValueError("CRE ids and embedding matrix must have aligned two-dimensional shapes")
    ccre_to_idx = {str(ccre_id): i for i, ccre_id in enumerate(ccre_ids)}
    genes = links[["gene_id", "gene_name"]].drop_duplicates("gene_id").reset_index(drop=True)
    gene_to_idx = {gid: i for i, gid in enumerate(genes["gene_id"])}
    emb_dim = int(embeddings.shape[1])
    token_embeddings = np.zeros((len(genes), max_tokens, emb_dim), dtype=np.float32)
    signed_distance = np.zeros((len(genes), max_tokens), dtype=np.float32)
    is_promoter = np.zeros((len(genes), max_tokens), dtype=bool)
    mask = np.zeros((len(genes), max_tokens), dtype=bool)
    for row in links.sort_values(["gene_id", "token_rank"]).itertuples(index=False):
        rank = int(row.token_rank)
        if rank >= max_tokens:
            continue
        try:
            emb_index = ccre_to_idx[str(row.ccre_id)]
        except KeyError as exc:
            raise ValueError(f"Missing embedding for cCRE {row.ccre_id}") from exc
        gi = gene_to_idx[row.gene_id]
        token_embeddings[gi, rank] = embeddings[emb_index]
        signed_distance[gi, rank] = float(row.signed_distance)
        is_promoter[gi, rank] = bool(row.is_promoter)
        mask[gi, rank] = True
    return {
        "gene_id": genes["gene_id"].to_numpy(dtype=str),
        "gene_name": genes["gene_name"].to_numpy(dtype=str),
        "embeddings": token_embeddings,
        "signed_distance": signed_distance,
        "is_promoter": is_promoter,
        "mask": mask,
    }
