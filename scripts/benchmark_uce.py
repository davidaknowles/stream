#!/usr/bin/env python
"""Measure 33-layer UCE inference throughput on realistic cell sentences.

The UCE reference preprocessing materializes a dense count matrix for an
AnnData input. This benchmark isolates the model path used after sparse cell
counts have been converted to UCE token IDs, so it can be used to size a
streaming production run without creating that dense intermediate.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import json
import math
import sys
import time
from pathlib import Path

import torch
from torch import nn


ATLAS_CELLS = 11_441_407
N_TOKENS = 145_469


def load_uce_model(
    uce_dir: Path,
    checkpoint: Path | None,
    device: torch.device,
) -> torch.nn.Module:
    model_path = uce_dir / "model.py"
    spec = importlib.util.spec_from_file_location("uce_model", model_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import UCE model from {model_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    model = module.TransformerModel(
        token_dim=5120,
        d_model=1280,
        nhead=20,
        d_hid=5120,
        nlayers=33,
        dropout=0.05,
        output_dim=1280,
    )
    # The checkpoint includes the pretrained token embeddings. This mirrors
    # the reference evaluator before it optionally replaces them from disk.
    model.pe_embedding = nn.Embedding.from_pretrained(torch.zeros(N_TOKENS, 5120))
    if checkpoint is not None:
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=True)
    return model.eval().to(device)


def measure(
    model: torch.nn.Module,
    batch_size: int,
    sequence_length: int,
    warmup_batches: int,
    timed_batches: int,
    device: torch.device,
    precision: str,
    no_padding_mask: bool,
) -> dict[str, float | int]:
    token_ids = torch.randint(4, N_TOKENS, (sequence_length, batch_size), device=device)
    token_ids[0] = 3  # UCE CLS token.
    mask = torch.ones((batch_size, sequence_length), device=device)

    def forward() -> None:
        autocast = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if precision == "bf16"
            else contextlib.nullcontext()
        )
        with autocast:
            embedded = model.pe_embedding(token_ids)
            embedded = nn.functional.normalize(embedded, dim=2)
            if no_padding_mask:
                encoded = model.encoder(embedded) * math.sqrt(model.d_model)
                encoded = model.pos_encoder(encoded)
                output = model.transformer_encoder(encoded)
                model.decoder(output)
            else:
                model(embedded, mask=mask)

    with torch.inference_mode():
        for _ in range(warmup_batches):
            forward()
        torch.cuda.synchronize(device)
        start = time.perf_counter()
        for _ in range(timed_batches):
            forward()
        torch.cuda.synchronize(device)
    elapsed_seconds = time.perf_counter() - start
    cells_per_second = batch_size * timed_batches / elapsed_seconds
    return {
        "batch_size": batch_size,
        "sequence_length": sequence_length,
        "warmup_batches": warmup_batches,
        "timed_batches": timed_batches,
        "elapsed_seconds": elapsed_seconds,
        "cells_per_second": cells_per_second,
        "seconds_per_cell": 1 / cells_per_second,
        "atlas_hours": ATLAS_CELLS / cells_per_second / 3600,
        "peak_memory_gib": torch.cuda.max_memory_allocated(device) / 2**30,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--uce-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument(
        "--random-init",
        action="store_true",
        help="Benchmark the published architecture before the checkpoint is available.",
    )
    parser.add_argument("--batch-size", type=int, default=25)
    parser.add_argument("--sequence-length", type=int, default=1065)
    parser.add_argument("--precision", choices=["fp32", "bf16"], default="fp32")
    parser.add_argument(
        "--no-padding-mask",
        action="store_true",
        help="Use only for batches bucketed to one exact unpadded token length.",
    )
    parser.add_argument("--warmup-batches", type=int, default=3)
    parser.add_argument("--timed-batches", type=int, default=10)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("UCE throughput benchmarking requires a CUDA GPU")
    if args.random_init == (args.checkpoint is not None):
        raise ValueError("Provide exactly one of --checkpoint or --random-init")
    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats(device)
    model = load_uce_model(args.uce_dir, args.checkpoint, device)
    result = measure(
        model=model,
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        warmup_batches=args.warmup_batches,
        timed_batches=args.timed_batches,
        device=device,
        precision=args.precision,
        no_padding_mask=args.no_padding_mask,
    )
    result["gpu"] = torch.cuda.get_device_name(device)
    result["atlas_cells"] = ATLAS_CELLS
    result["checkpoint_loaded"] = args.checkpoint is not None
    result["precision"] = args.precision
    result["no_padding_mask"] = args.no_padding_mask
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
