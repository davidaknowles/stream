"""Evaluation helpers for held-out timepoint comparisons."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from .models import mse_cfm_loss
from .ot import ot_cfm_batch


@torch.no_grad()
def evaluate_intervals(config, sampler, model, cre_inputs=None, n_batches: int = 20) -> pd.DataFrame:
    device = next(model.parameters()).device
    rows = []
    model.eval()
    for _ in range(n_batches):
        batch = sampler.sample()
        x0 = torch.as_tensor(batch.x0, device=device)
        x1 = torch.as_tensor(batch.x1, device=device)
        xt, target, _tau = ot_cfm_batch(
            x0,
            x1,
            batch.t0,
            batch.t1,
            epsilon=config.ot_epsilon,
            iterations=config.ot_iterations,
        )
        pred = model(xt) if cre_inputs is None else model(xt, **cre_inputs)
        err = (pred - target).detach().cpu().numpy()
        rows.append(
            {
                "day0": batch.day0,
                "day1": batch.day1,
                "loss": float(mse_cfm_loss(pred, target).cpu()),
                "velocity_mae": float(np.mean(np.abs(err))),
            }
        )
    return pd.DataFrame(rows)
