"""Evaluation helpers for held-out timepoint comparisons."""

from __future__ import annotations

import json
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .models import mse_cfm_loss
from .ot import ot_cfm_batch, ot_cfm_batch_with_state
from .train import predict_stream_chunked


@torch.no_grad()
def evaluate_intervals(
    config,
    sampler,
    model,
    cre_inputs=None,
    n_batches: int = 20,
    eval_gene_sets: dict[str, list[int] | np.ndarray | None] | None = None,
    batch_cache: str | Path | None = None,
) -> pd.DataFrame:
    device = next(model.parameters()).device
    rows = []
    model.eval()
    eval_gene_sets = eval_gene_sets or {"full": None}
    eval_indices = {
        name: None if indices is None else torch.as_tensor(indices, device=device, dtype=torch.long)
        for name, indices in eval_gene_sets.items()
    }
    cached = _load_or_create_batches(sampler, n_batches, batch_cache)
    for batch_index, batch in enumerate(cached):
        x0 = torch.as_tensor(batch.x0, device=device)
        x1 = torch.as_tensor(batch.x1, device=device)
        generator = torch.Generator(device=device).manual_seed(int(getattr(config, "seed", 1337)) + batch_index)
        if batch.state0 is None:
            xt, target, _tau = ot_cfm_batch(
                x0, x1, batch.t0, batch.t1, epsilon=config.ot_epsilon, iterations=config.ot_iterations, generator=generator
            )
            state_t = xt
        else:
            state0 = torch.as_tensor(batch.state0, device=device)
            state1 = torch.as_tensor(batch.state1, device=device)
            xt, target, _tau, state_t = ot_cfm_batch_with_state(
                x0,
                x1,
                state0,
                state1,
                batch.t0,
                batch.t1,
                epsilon=config.ot_epsilon,
                iterations=config.ot_iterations,
                generator=generator,
            )
        pred = (
            model(state_t)
            if cre_inputs is None
            else predict_stream_chunked(model, state_t, cre_inputs, config.gene_chunk_size)
        )
        for name, indices in eval_indices.items():
            pred_eval = pred if indices is None else pred.index_select(1, indices)
            target_eval = target if indices is None else target.index_select(1, indices)
            err = (pred_eval - target_eval).detach().cpu().numpy()
            displacement_err = (pred_eval - target_eval) * float(batch.t1 - batch.t0)
            rows.append(
                {
                    "batch": batch_index,
                    "day0": batch.day0,
                    "day1": batch.day1,
                    "eval_gene_set": name,
                    "n_eval_genes": int(pred_eval.shape[1]),
                    "cell_state": getattr(config, "cell_state", "expression"),
                    "loss": float(mse_cfm_loss(pred_eval, target_eval).cpu()),
                    "velocity_mae": float(np.mean(np.abs(err))),
                    "displacement_mse": float(torch.mean(displacement_err.square()).cpu()),
                    "displacement_mae": float(torch.mean(displacement_err.abs()).cpu()),
                }
            )
    return pd.DataFrame(rows)


def _load_or_create_batches(sampler, n_batches: int, batch_cache: str | Path | None):
    if batch_cache is None:
        return [sampler.sample() for _ in range(n_batches)]
    path = Path(batch_cache)
    if path.exists():
        raw = np.load(path, allow_pickle=False)
        metadata = json.loads(str(raw["metadata"].item()))
        if metadata["n_batches"] != n_batches:
            raise ValueError(f"Evaluation cache {path} contains {metadata['n_batches']} batches, expected {n_batches}")
        return [
            SimpleNamespace(
                x0=raw[f"x0_{i}"], x1=raw[f"x1_{i}"], state0=raw.get(f"state0_{i}"), state1=raw.get(f"state1_{i}"),
                t0=float(metadata["t0"][i]), t1=float(metadata["t1"][i]), day0=metadata["day0"][i], day1=metadata["day1"][i],
            )
            for i in range(n_batches)
        ]
    batches = [sampler.sample() for _ in range(n_batches)]
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"metadata": np.asarray(json.dumps({"n_batches": n_batches, "t0": [x.t0 for x in batches], "t1": [x.t1 for x in batches], "day0": [x.day0 for x in batches], "day1": [x.day1 for x in batches]}))}
    for i, batch in enumerate(batches):
        payload[f"x0_{i}"] = batch.x0
        payload[f"x1_{i}"] = batch.x1
        if batch.state0 is not None:
            payload[f"state0_{i}"] = batch.state0
            payload[f"state1_{i}"] = batch.state1
    np.savez_compressed(path, **payload)
    return batches
