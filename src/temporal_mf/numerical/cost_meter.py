"""Repeated wall-clock measurements of independent solves."""

from __future__ import annotations

import statistics

import torch

from temporal_mf.numerical.burgers import integrate


def time_solver(
    u0: torch.Tensor,
    *,
    nu: float,
    length: float,
    dt: float,
    t_observe: torch.Tensor,
    repeats: int = 3,
) -> dict:
    samples = []
    last = None
    for _ in range(repeats):
        field, rec = integrate(u0, nu=nu, length=length, dt=dt, t_observe=t_observe)
        samples.append(rec.wall_seconds)
        last = rec
    assert last is not None
    return {
        "median_seconds": float(statistics.median(samples)),
        "min_seconds": float(min(samples)),
        "max_seconds": float(max(samples)),
        "samples": samples,
        "status": last.status,
        "n_steps": last.n_steps,
        "max_cfl": last.max_cfl,
        "per_trajectory_median": float(statistics.median(samples)) / max(u0.shape[0], 1),
    }
