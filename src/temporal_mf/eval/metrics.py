"""Field errors with periodic cell-centered quadrature."""

from __future__ import annotations

import numpy as np


def quadrature_errors(pred: np.ndarray, truth: np.ndarray, length: float, eps: float) -> tuple[float, float]:
    """pred/truth shape (space,). Returns absolute and relative L2."""
    if pred.shape != truth.shape:
        raise ValueError("shape mismatch")
    dx = length / pred.shape[0]
    abs_err = float(np.sqrt(np.sum(dx * (pred - truth) ** 2)))
    denom = max(float(np.sqrt(np.sum(dx * truth**2))), eps)
    return abs_err, abs_err / denom


def trajectory_mean(values: list[float]) -> float:
    if not values:
        return float("nan")
    return float(np.mean(values))
