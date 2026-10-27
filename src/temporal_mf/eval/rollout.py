"""Autoregressive rollout. After the initial state, only predictions are fed back."""

from __future__ import annotations

from collections.abc import Callable

import torch


def rollout(step_fn: Callable[[torch.Tensor], torch.Tensor], u0: torch.Tensor, n_steps: int) -> torch.Tensor:
    """``step_fn`` maps a state (batch, nx) to the next state. Returns (batch, n_steps+1, nx)."""
    frames = [u0]
    state = u0
    for _ in range(n_steps):
        state = step_fn(state)
        frames.append(state)
    return torch.stack(frames, dim=1)


def teacher_forced(step_fn: Callable[[torch.Tensor], torch.Tensor], truth: torch.Tensor) -> torch.Tensor:
    """``truth`` is (batch, time, nx). Predicts each next frame from the true current frame."""
    preds = []
    for n in range(truth.shape[1] - 1):
        preds.append(step_fn(truth[:, n]))
    return torch.stack(preds, dim=1)
