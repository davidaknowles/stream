#!/usr/bin/env python
"""Embed linked CRE/promoter windows with AlphaGenome and pack token arrays."""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from stream_model.alphagenome_embed import embed_cre_table, embed_tss_context_tokens
from stream_model.config import StreamConfig, apply_config_overrides
from stream_model.genome import build_token_arrays_from_matrix


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/stream_mouse_dev.yaml")
    parser.add_argument("--hvg-csv", default=None)
    parser.add_argument("--n-hvg", type=int, default=None)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--embedding-mode", choices=["independent", "tss_context"], default="independent")
    parser.add_argument("--context-flank-bp", type=int, default=100000)
    args = parser.parse_args()
    cfg = StreamConfig.from_yaml(args.config)
    apply_config_overrides(cfg, hvg_csv=args.hvg_csv, n_hvg=args.n_hvg, out_dir=args.out_dir)
    if cfg.alphagenome_checkpoint is None:
        raise SystemExit("Set alphagenome_checkpoint in the config before embedding CREs.")
    links_path = cfg.out_dir / "cre_gene_links.csv"
    links = pd.read_csv(links_path)
    if args.embedding_mode == "tss_context":
        arrays = embed_tss_context_tokens(
            links,
            fasta_path=cfg.fasta,
            checkpoint=cfg.alphagenome_checkpoint,
            repo=cfg.alphagenome_repo,
            batch_size=cfg.alphagenome_batch_size,
            sequence_bp=cfg.alphagenome_sequence_bp,
            device=args.device or cfg.device,
            organism_index=cfg.alphagenome_organism_index,
            flank_bp=args.context_flank_bp,
            max_tokens=cfg.max_cres_per_gene,
            cache_dir=cfg.out_dir,
        )
        output_path = cfg.out_dir / "cre_token_arrays_tss_context.npz"
    else:
        ccre_ids, embeddings = embed_cre_table(
            links,
            fasta_path=cfg.fasta,
            checkpoint=cfg.alphagenome_checkpoint,
            repo=cfg.alphagenome_repo,
            batch_size=cfg.alphagenome_batch_size,
            sequence_bp=cfg.alphagenome_sequence_bp,
            device=args.device or cfg.device,
            organism_index=cfg.alphagenome_organism_index,
            cache_dir=cfg.out_dir,
        )
        arrays = build_token_arrays_from_matrix(links, ccre_ids, embeddings, max_tokens=cfg.max_cres_per_gene)
        output_path = cfg.out_dir / "cre_token_arrays.npz"
    np.savez_compressed(output_path, **arrays)
    print(f"Wrote resumable CRE matrix under {cfg.out_dir}")
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
