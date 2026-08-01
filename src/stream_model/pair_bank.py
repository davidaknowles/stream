"""Interval-stratified OT pair banks for CFM training."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .denoise import denoise_selected_counts
from .ot import coupling_diagnostics, coupling_kwargs, sample_coupling_pairs, transport_plan


@dataclass
class PairBankEntry:
    raw_x0: np.ndarray
    raw_x1: np.ndarray
    target_x0: np.ndarray
    target_x1: np.ndarray
    state0: np.ndarray | None
    state1: np.ndarray | None
    t0: float
    t1: float
    day0: str
    day1: str
    diagnostics: dict[str, float]
    refresh: int
    cursor: int = 0


@dataclass(frozen=True)
class PairMicrobatch:
    raw_x0: torch.Tensor
    raw_x1: torch.Tensor
    target_x0: torch.Tensor
    target_x1: torch.Tensor
    state0: torch.Tensor | None
    state1: torch.Tensor | None
    t0: float
    t1: float
    day0: str
    day1: str
    diagnostics: dict[str, float]
    refresh: int


class IntervalPairBank:
    """Build one reusable OT pair bank per interval and interleave intervals."""

    def __init__(self, config, sampler, device: torch.device, pca=None):
        self.config = config
        self.sampler = sampler
        self.device = device
        self.pca = pca
        self.intervals = list(sampler.intervals)
        if not self.intervals:
            raise ValueError("Pair bank requires at least one interval")
        self.entries: dict[tuple[str, str], PairBankEntry] = {}
        self.refresh_counts = {interval: 0 for interval in self.intervals}
        self.rng = np.random.default_rng(int(config.seed) + 701)
        self.schedule: list[tuple[str, str]] = []

    def _next_interval(self) -> tuple[str, str]:
        if not self.schedule:
            order = self.rng.permutation(len(self.intervals))
            self.schedule = [self.intervals[index] for index in order]
        return self.schedule.pop()

    def _build_entry(self, interval: tuple[str, str]) -> PairBankEntry:
        day0, day1 = interval
        batch = self.sampler.sample_interval(day0, day1)
        raw_x0 = np.asarray(batch.x0, dtype=np.float32)
        raw_x1 = np.asarray(batch.x1, dtype=np.float32)
        x0 = torch.as_tensor(raw_x0, device=self.device)
        x1 = torch.as_tensor(raw_x1, device=self.device)
        ot_cost_space = getattr(self.config, "ot_cost_space", "expression")
        if ot_cost_space == "pca":
            if self.pca is None:
                raise ValueError("PCA OT cost requires a fitted PCA artifact")
            cost_x0 = torch.as_tensor(self.pca.transform_counts(raw_x0), device=self.device)
            cost_x1 = torch.as_tensor(self.pca.transform_counts(raw_x1), device=self.device)
            cost_metric = "scaled_euclidean"
        elif ot_cost_space == "expression":
            cost_x0 = cost_x1 = None
            cost_metric = "expression_cosine"
        else:
            raise ValueError("ot_cost_space must be expression or pca")
        cost, coupling = transport_plan(
            x0,
            x1,
            cost_x0=cost_x0,
            cost_x1=cost_x1,
            cost_metric=cost_metric,
            **coupling_kwargs(self.config),
        )
        refresh = self.refresh_counts[interval]
        interval_index = self.intervals.index(interval)
        generator = torch.Generator(device=self.device).manual_seed(
            int(self.config.seed) + interval_index * 100_003 + refresh * 1_000_003
        )
        n_pairs = max(int(self.config.batch_size), int(self.config.ot_pairs_per_pool))
        i0, i1 = sample_coupling_pairs(coupling, n_pairs, generator=generator)
        i0_np = i0.cpu().numpy()
        i1_np = i1.cpu().numpy()
        selected_raw0 = raw_x0[i0_np]
        selected_raw1 = raw_x1[i1_np]
        endpoint_denoising = getattr(self.config, "endpoint_denoising", "none")
        if endpoint_denoising in {"pca", "knn", "metacell"}:
            if self.pca is None:
                raise ValueError("Endpoint denoising requires a fitted PCA artifact")
            denoising_kwargs = {
                "n_neighbors": int(getattr(self.config, "denoising_neighbors", 15)),
                "n_metacells": int(getattr(self.config, "denoising_metacells", 512)),
                "device": self.device,
            }
            target_x0 = denoise_selected_counts(
                endpoint_denoising,
                raw_x0,
                i0_np,
                self.pca,
                seed=int(self.config.seed) + interval_index * 2 + refresh * 10_007,
                **denoising_kwargs,
            )
            target_x1 = denoise_selected_counts(
                endpoint_denoising,
                raw_x1,
                i1_np,
                self.pca,
                seed=int(self.config.seed) + interval_index * 2 + 1 + refresh * 10_007,
                **denoising_kwargs,
            )
        elif endpoint_denoising == "none":
            target_x0 = selected_raw0
            target_x1 = selected_raw1
        else:
            raise ValueError("endpoint_denoising must be none, pca, knn, or metacell")
        state0 = None if batch.state0 is None else np.asarray(batch.state0, dtype=np.float32)[i0_np]
        state1 = None if batch.state1 is None else np.asarray(batch.state1, dtype=np.float32)[i1_np]
        self.refresh_counts[interval] += 1
        return PairBankEntry(
            raw_x0=selected_raw0,
            raw_x1=selected_raw1,
            target_x0=target_x0,
            target_x1=target_x1,
            state0=state0,
            state1=state1,
            t0=batch.t0,
            t1=batch.t1,
            day0=day0,
            day1=day1,
            diagnostics=coupling_diagnostics(cost, coupling),
            refresh=refresh,
        )

    def next(self) -> PairMicrobatch:
        interval = self._next_interval()
        entry = self.entries.get(interval)
        if entry is None or entry.cursor + self.config.batch_size > len(entry.raw_x0):
            entry = self._build_entry(interval)
            self.entries[interval] = entry
        start = entry.cursor
        end = start + int(self.config.batch_size)
        entry.cursor = end

        def tensor(values):
            return torch.as_tensor(values[start:end], device=self.device)

        return PairMicrobatch(
            raw_x0=tensor(entry.raw_x0),
            raw_x1=tensor(entry.raw_x1),
            target_x0=tensor(entry.target_x0),
            target_x1=tensor(entry.target_x1),
            state0=None if entry.state0 is None else tensor(entry.state0),
            state1=None if entry.state1 is None else tensor(entry.state1),
            t0=entry.t0,
            t1=entry.t1,
            day0=entry.day0,
            day1=entry.day1,
            diagnostics=entry.diagnostics,
            refresh=entry.refresh,
        )
