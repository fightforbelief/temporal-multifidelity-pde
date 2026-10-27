"""POD basis and ridge maps fit on training trajectories only."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge


def pod_basis(snapshots: np.ndarray, rank: int) -> tuple[np.ndarray, np.ndarray]:
    """snapshots (n, nx) -> mean (nx,), basis (nx, rank)."""
    mean = snapshots.mean(axis=0)
    centered = snapshots - mean
    _u, s, vt = np.linalg.svd(centered, full_matrices=False)
    r = min(rank, vt.shape[0])
    return mean, vt[:r].T


def project(u: np.ndarray, mean: np.ndarray, basis: np.ndarray) -> np.ndarray:
    return (u - mean) @ basis


def reconstruct(coeff: np.ndarray, mean: np.ndarray, basis: np.ndarray) -> np.ndarray:
    return coeff @ basis.T + mean


class ARRidge:
    """Autoregressive map in POD coordinates, trained on consecutive train frames."""

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.model = Ridge(alpha=alpha)
        self.mean = None
        self.basis = None

    def fit(self, trajectories: list[np.ndarray], rank: int) -> None:
        snaps = np.concatenate(trajectories, axis=0)
        self.mean, self.basis = pod_basis(snaps, rank)
        xs = []
        ys = []
        for traj in trajectories:
            coeff = project(traj, self.mean, self.basis)
            xs.append(coeff[:-1])
            ys.append(coeff[1:])
        self.model.fit(np.concatenate(xs), np.concatenate(ys))

    def step_np(self, u: np.ndarray) -> np.ndarray:
        coeff = project(u, self.mean, self.basis)
        nxt = self.model.predict(coeff)
        return reconstruct(nxt, self.mean, self.basis)


class ResidualRidge:
    """Predict HF next-state residual over a supplied low-fidelity one-step field.

    Training pairs are (state, lf_next, hf_next) from the training split only.
    """

    def __init__(self, alpha: float = 1.0):
        self.model = Ridge(alpha=alpha)
        self.mean = None
        self.basis = None

    def fit(self, states: np.ndarray, lf_next: np.ndarray, hf_next: np.ndarray, rank: int) -> None:
        self.mean, self.basis = pod_basis(states, rank)
        coeff = project(states, self.mean, self.basis)
        residual = hf_next - lf_next
        self.model.fit(coeff, residual)

    def correct(self, state: np.ndarray, lf_next: np.ndarray) -> np.ndarray:
        coeff = project(state, self.mean, self.basis)
        return lf_next + self.model.predict(coeff)
