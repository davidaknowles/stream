"""Streaming UCE tokenization and inference helpers.

The upstream UCE evaluator materializes an ``int64`` dense count matrix. These
helpers retain its token convention while operating directly on sparse rows.
"""

from __future__ import annotations

import importlib.util
import math
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn


UCE_TOKEN_DIM = 5120
UCE_MODEL_DIM = 1280
UCE_N_TOKENS = 145_469
UCE_CHROM_TOKEN_OFFSET = 143_574
UCE_CLS_TOKEN = 3
UCE_CHROM_CLOSE_TOKEN = 2


@dataclass(frozen=True)
class UCEGeneMetadata:
    token_ids: np.ndarray
    chrom_ids: np.ndarray
    starts: np.ndarray

    @property
    def valid(self) -> np.ndarray:
        return self.token_ids >= 0


def load_uce_gene_metadata(
    var: pd.DataFrame,
    protein_embeddings_path: str | Path,
    species_chrom_path: str | Path,
    species_offsets_path: str | Path,
    species: str = "mouse",
    gene_symbol_column: str = "gene_short_name",
) -> UCEGeneMetadata:
    """Map AnnData variables to UCE protein and chromosome token metadata."""

    if gene_symbol_column not in var.columns:
        raise ValueError(f"AnnData var table lacks {gene_symbol_column!r}")
    protein_embeddings = torch.load(protein_embeddings_path, map_location="cpu", weights_only=False)
    gene_to_rank = {str(gene).upper(): rank for rank, gene in enumerate(protein_embeddings)}
    with Path(species_offsets_path).open("rb") as handle:
        offset = pickle.load(handle)[species]

    chrom = pd.read_csv(species_chrom_path)
    chrom["spec_chrom"] = pd.Categorical(chrom["species"].astype(str) + "_" + chrom["chromosome"].astype(str))
    chrom = chrom.loc[chrom["species"].eq(species), ["gene_symbol", "start", "spec_chrom"]].copy()
    chrom["gene_symbol"] = chrom["gene_symbol"].astype(str).str.upper()
    chrom = chrom.drop_duplicates("gene_symbol").set_index("gene_symbol")
    chrom_codes = chrom["spec_chrom"].cat.codes

    symbols = var[gene_symbol_column].astype(str).str.upper()
    token_ids = np.asarray([gene_to_rank.get(symbol, -1) for symbol in symbols], dtype=np.int64)
    token_ids[token_ids >= 0] += int(offset)
    mapped_chrom = symbols.map(chrom_codes)
    starts = pd.to_numeric(symbols.map(chrom["start"]), errors="coerce")
    chrom_ids = mapped_chrom.fillna(-1).astype(np.int64).to_numpy()
    start_values = starts.fillna(-1).astype(np.int64).to_numpy()
    invalid = (token_ids < 0) | (chrom_ids < 0) | (start_values < 0)
    token_ids[invalid] = -1
    return UCEGeneMetadata(token_ids=token_ids, chrom_ids=chrom_ids, starts=start_values)


def sample_uce_sentence(
    row,
    gene_metadata: UCEGeneMetadata,
    rng: np.random.Generator,
    sample_size: int = 1024,
) -> np.ndarray | None:
    """Create one UCE cell sentence from a sparse CSR row."""

    columns = row.indices
    values = row.data
    usable = gene_metadata.valid[columns] & (values > 0)
    columns = columns[usable]
    if len(columns) == 0:
        return None
    weights = np.log1p(values[usable].astype(np.float64, copy=False))
    total = weights.sum()
    if total <= 0:
        return None
    chosen = rng.choice(columns, size=sample_size, replace=True, p=weights / total)
    chroms = gene_metadata.chrom_ids[chosen]
    sentence = np.empty(1 + sample_size + 2 * len(np.unique(chroms)), dtype=np.int64)
    sentence[0] = UCE_CLS_TOKEN
    cursor = 1
    unique_chroms = np.unique(chroms)
    rng.shuffle(unique_chroms)  # Matches the UCE reference chromosome ordering.
    for chrom in unique_chroms:
        sentence[cursor] = UCE_CHROM_TOKEN_OFFSET + int(chrom)
        cursor += 1
        on_chrom = chosen[chroms == chrom]
        ordered = on_chrom[np.argsort(gene_metadata.starts[on_chrom], kind="stable")]
        count = len(ordered)
        sentence[cursor : cursor + count] = gene_metadata.token_ids[ordered]
        cursor += count
        sentence[cursor] = UCE_CHROM_CLOSE_TOKEN
        cursor += 1
    return sentence


def load_uce_model(uce_dir: str | Path, checkpoint: str | Path, device: torch.device) -> torch.nn.Module:
    """Load the official 33-layer UCE checkpoint into the reference model."""

    model_path = Path(uce_dir) / "model.py"
    spec = importlib.util.spec_from_file_location("stream_uce_model", model_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import UCE model from {model_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = module.TransformerModel(
        token_dim=UCE_TOKEN_DIM,
        d_model=UCE_MODEL_DIM,
        nhead=20,
        d_hid=5120,
        nlayers=33,
        dropout=0.05,
        output_dim=UCE_MODEL_DIM,
    )
    model.pe_embedding = nn.Embedding.from_pretrained(torch.zeros(UCE_N_TOKENS, UCE_TOKEN_DIM))
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    return model.eval().to(device)


@torch.inference_mode()
def embed_uce_sentences(model: torch.nn.Module, sentences: np.ndarray, device: torch.device) -> np.ndarray:
    """Embed one exact-length bucket with BF16 Flash Attention and no padding."""

    token_ids = torch.as_tensor(sentences.T, device=device, dtype=torch.long)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        embedded = nn.functional.normalize(model.pe_embedding(token_ids), dim=2)
        encoded = model.encoder(embedded) * math.sqrt(model.d_model)
        encoded = model.pos_encoder(encoded)
        output = model.transformer_encoder(encoded)
        cell_embeddings = nn.functional.normalize(model.decoder(output)[0], dim=1)
    return cell_embeddings.float().cpu().numpy()
