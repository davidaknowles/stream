"""AlphaGenome-based CRE embedding extraction."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

DNA_TO_INDEX = {"A": 0, "C": 1, "G": 2, "T": 3}


def one_hot_dna(seq: str) -> torch.Tensor:
    arr = np.zeros((len(seq), 4), dtype=np.float32)
    for i, base in enumerate(seq.upper()):
        idx = DNA_TO_INDEX.get(base)
        if idx is not None:
            arr[i, idx] = 1.0
    return torch.from_numpy(arr)


class FastaExtractor:
    """Small wrapper around optional FASTA random-access libraries."""

    def __init__(self, fasta_path: str | Path):
        self.fasta_path = str(Path(fasta_path).expanduser())
        self.backend = None
        self.kind = ""
        try:
            import pysam

            self.backend = pysam.FastaFile(self.fasta_path)
            self.kind = "pysam"
            return
        except Exception:
            pass
        try:
            import pyfaidx

            self.backend = pyfaidx.Fasta(self.fasta_path, as_raw=True, sequence_always_upper=True)
            self.kind = "pyfaidx"
            return
        except Exception:
            pass
        try:
            import pyfastx

            self.backend = pyfastx.Fasta(self.fasta_path)
            self.kind = "pyfastx"
            return
        except Exception as exc:
            raise RuntimeError(
                "Install pyfastx, pyfaidx, or pysam for FASTA extraction. "
                f"Could not open {self.fasta_path}."
            ) from exc

    def fetch(self, chrom: str, start: int, end: int) -> str:
        start = max(0, int(start))
        end = max(start, int(end))
        chroms = _chrom_candidates(chrom)
        if self.kind == "pyfastx":
            # pyfastx uses 1-based inclusive coordinates.
            last_error = None
            for candidate in chroms:
                try:
                    return self.backend.fetch(candidate, (start + 1, end)).upper()
                except Exception as exc:
                    last_error = exc
            raise KeyError(f"None of {chroms} found in {self.fasta_path}") from last_error
        if self.kind == "pyfaidx":
            last_error = None
            for candidate in chroms:
                try:
                    return str(self.backend[candidate][start:end]).upper()
                except Exception as exc:
                    last_error = exc
            raise KeyError(f"None of {chroms} found in {self.fasta_path}") from last_error
        last_error = None
        for candidate in chroms:
            try:
                return self.backend.fetch(candidate, start, end).upper()
            except Exception as exc:
                last_error = exc
        raise KeyError(f"None of {chroms} found in {self.fasta_path}") from last_error


def _chrom_candidates(chrom: str) -> list[str]:
    chrom = str(chrom)
    candidates = [chrom]
    if chrom.startswith("chr"):
        candidates.append(chrom[3:])
    else:
        candidates.append(f"chr{chrom}")
    if chrom in {"chrM", "chrMT", "M", "MT"}:
        candidates.extend(["chrM", "chrMT", "M", "MT"])
    out = []
    for candidate in candidates:
        if candidate not in out:
            out.append(candidate)
    return out


class AlphaGenomeCREEmbedder:
    """Extract pooled AlphaGenome 128bp trunk embeddings for CRE windows."""

    def __init__(
        self,
        checkpoint: str | Path,
        repo: str | Path,
        device: str = "cuda",
        organism_index: int = 1,
        sequence_bp: int = 131_072,
    ):
        checkpoint = Path(checkpoint).expanduser()
        repo = Path(repo).expanduser()
        if not checkpoint.exists():
            raise FileNotFoundError(f"AlphaGenome checkpoint does not exist: {checkpoint}")
        src_dir = repo / "src"
        if src_dir.exists():
            sys.path.insert(0, str(src_dir))
        from alphagenome_pytorch import AlphaGenome

        self.device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
        self.model = AlphaGenome.from_pretrained(str(checkpoint), device=str(self.device))
        self.model.eval()
        self.organism_index = organism_index
        self.sequence_bp = int(sequence_bp)

    @torch.no_grad()
    def embed_sequences(self, seqs: list[str]) -> np.ndarray:
        if not seqs:
            return np.zeros((0, 3072), dtype=np.float32)
        batch = torch.stack([one_hot_dna(_pad_or_trim(seq, self.sequence_bp)) for seq in seqs]).to(self.device)
        emb = self.model.encode(batch, organism_index=self.organism_index, resolutions=(128,))
        x = emb["embeddings_128bp"]
        if x.shape[1] == 3072:
            x = x.transpose(1, 2)
        pooled = x.mean(dim=1)
        return pooled.float().cpu().numpy()


def _pad_or_trim(seq: str, length: int) -> str:
    seq = seq.upper()
    if len(seq) == length:
        return seq
    if len(seq) > length:
        extra = len(seq) - length
        left = extra // 2
        return seq[left : left + length]
    pad = length - len(seq)
    left = pad // 2
    return ("N" * left) + seq + ("N" * (pad - left))


def embed_cre_table(
    cre_table: pd.DataFrame,
    fasta_path: str | Path,
    checkpoint: str | Path,
    repo: str | Path,
    batch_size: int,
    sequence_bp: int,
    device: str,
    organism_index: int = 1,
) -> pd.DataFrame:
    fasta = FastaExtractor(fasta_path)
    embedder = AlphaGenomeCREEmbedder(
        checkpoint=checkpoint,
        repo=repo,
        device=device,
        organism_index=organism_index,
        sequence_bp=sequence_bp,
    )
    rows: list[dict[str, object]] = []
    seqs: list[str] = []
    ids: list[str] = []
    half = sequence_bp // 2
    unique = cre_table.drop_duplicates("ccre_id")
    total = len(unique)
    done = 0
    for cre in unique.itertuples(index=False):
        center = int(cre.midpoint)
        seqs.append(fasta.fetch(cre.chrom, center - half, center + half))
        ids.append(cre.ccre_id)
        if len(seqs) == batch_size:
            rows.extend(_embed_batch(embedder, ids, seqs))
            done += len(ids)
            if done % max(batch_size * 25, 1) == 0 or done == total:
                print(f"AlphaGenome CRE embeddings: {done:,}/{total:,}", flush=True)
            ids, seqs = [], []
    rows.extend(_embed_batch(embedder, ids, seqs))
    done += len(ids)
    if ids:
        print(f"AlphaGenome CRE embeddings: {done:,}/{total:,}", flush=True)
    return pd.DataFrame(rows)


def _embed_batch(embedder: AlphaGenomeCREEmbedder, ids: list[str], seqs: list[str]) -> list[dict[str, object]]:
    if not ids:
        return []
    embeddings = embedder.embed_sequences(seqs)
    rows = []
    for ccre_id, emb in zip(ids, embeddings, strict=True):
        row: dict[str, object] = {"ccre_id": ccre_id}
        row.update({f"emb_{i}": float(value) for i, value in enumerate(emb)})
        rows.append(row)
    return rows
