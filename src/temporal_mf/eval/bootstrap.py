"""Trajectory-level paired bootstrap. Frames are not the resampling unit."""

from __future__ import annotations

import numpy as np


def paired_bootstrap_mean_diff(
    errors_a: dict[str, float],
    errors_b: dict[str, float],
    *,
    n_boot: int = 1000,
    seed: int = 950,
) -> dict:
    ids = sorted(set(errors_a) & set(errors_b))
    if not ids:
        raise ValueError("no paired trajectory ids")
    a = np.array([errors_a[i] for i in ids], dtype=np.float64)
    b = np.array([errors_b[i] for i in ids], dtype=np.float64)
    rng = np.random.default_rng(seed)
    diffs = []
    n = len(ids)
    for _ in range(n_boot):
        take = rng.integers(0, n, size=n)
        diffs.append(float(np.mean(a[take] - b[take])))
    arr = np.sort(np.array(diffs))
    return {
        "n_trajectories": n,
        "mean_diff_a_minus_b": float(np.mean(a - b)),
        "ci_low": float(np.quantile(arr, 0.025)),
        "ci_high": float(np.quantile(arr, 0.975)),
        "ids": ids,
    }
