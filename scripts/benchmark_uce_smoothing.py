#!/usr/bin/env python
"""Compare multinomial and systematic online-UCE embedding variability."""

from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd
import torch

from stream_model.config import StreamConfig, apply_config_overrides
from stream_model.uce import OnlineUCEEncoder, load_uce_gene_metadata, load_uce_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/stream_mouse_dev.yaml")
    parser.add_argument("--out-dir", default="outputs/stream_hvg10000")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--cells", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=16)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    cfg = StreamConfig.from_yaml(args.config)
    apply_config_overrides(cfg, out_dir=args.out_dir, n_hvg=10000)
    selected = pd.read_csv(cfg.out_dir / "selected_genes.csv")
    metadata = load_uce_gene_metadata(
        selected,
        cfg.uce_protein_embeddings,
        cfg.uce_species_chrom,
        cfg.uce_species_offsets,
        species=cfg.uce_species,
    )
    payload = np.load(cfg.resolve_path(args.predictions))
    source_key = next(key for key in payload.files if key.endswith("_source"))
    observed_key = source_key.removesuffix("_source") + "_observed"
    expression = 0.5 * (payload[source_key][: args.cells] + payload[observed_key][: args.cells])
    device = torch.device("cuda")
    model = load_uce_model(cfg.uce_dir, cfg.uce_checkpoint, device)
    rows = []
    for sampling in ("multinomial", "systematic"):
        encoder = OnlineUCEEncoder(model, metadata, device, cfg.uce_sample_size, sampling=sampling)
        embeddings = []
        torch.cuda.synchronize()
        start = time.perf_counter()
        for repeat in range(args.repeats):
            embeddings.append(encoder.encode(expression, seed=cfg.seed + repeat).cpu().numpy())
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        values = np.stack(embeddings)
        center = values.mean(axis=0, keepdims=True)
        cosine = np.sum(values * center, axis=2) / (
            np.linalg.norm(values, axis=2) * np.linalg.norm(center, axis=2)
        )
        rows.append(
            {
                "sampling": sampling,
                "cells": len(expression),
                "repeats": args.repeats,
                "seconds": elapsed,
                "cell_embeddings_per_second": len(expression) * args.repeats / elapsed,
                "mean_embedding_variance": float(values.var(axis=0).mean()),
                "mean_cosine_to_repeat_mean": float(cosine.mean()),
            }
        )
    frame = pd.DataFrame(rows)
    output = cfg.resolve_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
