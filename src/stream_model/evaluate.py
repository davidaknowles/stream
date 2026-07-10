"""Evaluation helpers for held-out timepoint comparisons."""

from __future__ import annotations

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
) -> pd.DataFrame:
    device = next(model.parameters()).device
    rows = []
    model.eval()
    eval_gene_sets = eval_gene_sets or {"full": None}
    eval_indices = {
        name: None if indices is None else torch.as_tensor(indices, device=device, dtype=torch.long)
        for name, indices in eval_gene_sets.items()
    }
    for batch_index in range(n_batches):
        batch = sampler.sample()
        x0 = torch.as_tensor(batch.x0, device=device)
        x1 = torch.as_tensor(batch.x1, device=device)
        if batch.state0 is None:
            xt, target, _tau = ot_cfm_batch(
                x0, x1, batch.t0, batch.t1, epsilon=config.ot_epsilon, iterations=config.ot_iterations
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
                }
            )
    return pd.DataFrame(rows)
