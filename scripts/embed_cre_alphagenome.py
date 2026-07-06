#!/usr/bin/env python
"""Embed linked CRE/promoter windows with AlphaGenome and pack token arrays."""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from stream_model.alphagenome_embed import embed_cre_table
from stream_model.config import StreamConfig, apply_config_overrides
from stream_model.genome import build_token_arrays


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/stream_mouse_dev.yaml")
    parser.add_argument("--hvg-csv", default=None)
    parser.add_argument("--n-hvg", type=int, default=None)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    cfg = StreamConfig.from_yaml(args.config)
    apply_config_overrides(cfg, hvg_csv=args.hvg_csv, n_hvg=args.n_hvg, out_dir=args.out_dir)
    if cfg.alphagenome_checkpoint is None:
        raise SystemExit("Set alphagenome_checkpoint in the config before embedding CREs.")
    links_path = cfg.out_dir / "cre_gene_links.csv"
    links = pd.read_csv(links_path)
    embeddings = embed_cre_table(
        links,
        fasta_path=cfg.fasta,
        checkpoint=cfg.alphagenome_checkpoint,
        repo=cfg.alphagenome_repo,
        batch_size=cfg.alphagenome_batch_size,
        sequence_bp=cfg.alphagenome_sequence_bp,
        device=args.device or cfg.device,
        organism_index=cfg.alphagenome_organism_index,
    )
    embeddings.to_csv(cfg.out_dir / "cre_embeddings.csv.gz", index=False)
    arrays = build_token_arrays(links, embeddings, max_tokens=cfg.max_cres_per_gene)
    np.savez_compressed(cfg.out_dir / "cre_token_arrays.npz", **arrays)
    print(f"Wrote {cfg.out_dir / 'cre_embeddings.csv.gz'}")
    print(f"Wrote {cfg.out_dir / 'cre_token_arrays.npz'}")


if __name__ == "__main__":
    main()
