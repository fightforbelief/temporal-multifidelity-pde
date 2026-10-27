"""Fixed-step AdamW training. Validation selects checkpoints; the test split is not read."""

from __future__ import annotations

import time

import torch
from torch import nn


def train_fixed_steps(
    model: nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    xi: torch.Tensor | None,
    *,
    steps: int,
    batch_size: int,
    lr: float,
    seed: int,
    max_seconds: float = 25 * 60,
) -> dict:
    device = inputs.device
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    n = inputs.shape[0]
    if n < 1:
        raise ValueError("empty training set")
    loss_fn = nn.MSELoss()
    t0 = time.perf_counter()
    last = None
    status = "OK"
    model.train()
    for step in range(steps):
        if time.perf_counter() - t0 > max_seconds:
            status = "RESOURCE_LIMIT"
            break
        choice = torch.randint(0, n, (min(batch_size, n),), generator=g)
        pred = model(inputs[choice], None if xi is None else xi[choice])
        loss = loss_fn(pred, targets[choice])
        if not torch.isfinite(loss):
            status = "NONFINITE_LOSS"
            break
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        last = float(loss.detach().cpu())
    return {
        "status": status,
        "steps_completed": step + 1 if steps else 0,
        "train_seconds": time.perf_counter() - t0,
        "last_loss": last,
    }
