#!/usr/bin/env python
"""Benchmark large mouse OT endpoint pools without running STREAM."""

from __future__ import annotations

import argparse
import json
import time

import pandas as pd
import torch

from stream_model.config import StreamConfig, apply_config_overrides
from stream_model.data import H5adIntervalSampler
from stream_model.ot import coupling_diagnostics, transport_plan


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/stream_mouse_dev.yaml")
    parser.add_argument("--out-dir", default="outputs/stream_hvg10000")
    parser.add_argument("--n-hvg", type=int, default=10_000)
    parser.add_argument("--pool-sizes", default="1024,2048,4096,8192")
    parser.add_argument("--output", default="outputs/stream_hvg10000/ot_pool_benchmark.csv")
    args = parser.parse_args()

    cfg = StreamConfig.from_yaml(args.config)
    apply_config_overrides(cfg, out_dir=args.out_dir, n_hvg=args.n_hvg)
    selected = pd.read_csv(cfg.out_dir / "selected_genes.csv")
    with (cfg.out_dir / "timepoint_split.json").open() as handle:
        split = json.load(handle)
    interval = tuple(split["train_intervals"][len(split["train_intervals"]) // 2])
    cells = pd.read_csv(cfg.cell_metadata_csv, index_col=0)
    sampler = H5adIntervalSampler.from_adata_dir(
        cfg.adata_dir,
        cells,
        selected["gene_id"].astype(str).tolist(),
        [interval],
        batch_size=1,
        seed=cfg.seed + 919,
        time_coordinates=split.get("time_coordinates"),
    )
    device = torch.device("cuda")
    rows = []
    for pool_size in [int(value) for value in args.pool_sizes.split(",")]:
        sampler.batch_size = pool_size
        started = time.perf_counter()
        batch = sampler.sample_interval(*interval)
        load_seconds = time.perf_counter() - started
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        try:
            x0 = torch.as_tensor(batch.x0, device=device)
            x1 = torch.as_tensor(batch.x1, device=device)
            torch.cuda.synchronize()
            solve_started = time.perf_counter()
            cost, coupling = transport_plan(
                x0,
                x1,
                method="partial",
                epsilon=cfg.ot_epsilon,
                iterations=cfg.ot_iterations,
                partial_mass=0.95,
                marginal_relaxation=0.1,
            )
            torch.cuda.synchronize()
            row = {
                "pool_size": pool_size,
                "status": "ok",
                "load_seconds": load_seconds,
                "solve_seconds": time.perf_counter() - solve_started,
                "peak_gpu_gib": torch.cuda.max_memory_allocated() / 2**30,
                **coupling_diagnostics(cost, coupling),
            }
        except torch.OutOfMemoryError:
            row = {
                "pool_size": pool_size,
                "status": "oom",
                "load_seconds": load_seconds,
                "solve_seconds": float("nan"),
                "peak_gpu_gib": torch.cuda.max_memory_allocated() / 2**30,
            }
        rows.append(row)
        print(row, flush=True)
        del batch
        torch.cuda.empty_cache()
    output = cfg.resolve_path(args.output)
    pd.DataFrame(rows).to_csv(output, index=False)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
