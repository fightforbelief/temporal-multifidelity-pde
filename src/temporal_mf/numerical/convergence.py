"""Temporal refinement diagnostics against a finer computed reference."""

from __future__ import annotations

import torch

from temporal_mf.numerical.burgers import integrate, rel_l2


def temporal_convergence(
    u0: torch.Tensor,
    *,
    nu: float,
    length: float,
    dt_coarse: float,
    t_final: float,
    n_levels: int = 3,
) -> list[dict]:
    """Compare successive dyadic refinements. The finest level is the reference."""
    if n_levels < 3:
        raise ValueError("need at least 3 levels")
    levels = []
    fields = []
    for level in range(n_levels):
        dt = dt_coarse / (2**level)
        n_obs = int(round(t_final / dt_coarse))
        t_obs = torch.linspace(0, t_final, n_obs + 1, dtype=u0.dtype, device=u0.device)
        # observation grid is the coarse grid so every level hits it exactly
        # only if dt divides dt_coarse. dt = dt_coarse/2^k does.
        field, rec = integrate(u0, nu=nu, length=length, dt=dt, t_observe=t_obs)
        fields.append(field)
        levels.append({"level": level, "dt": dt, "status": rec.status, "steps": rec.n_steps, "wall_seconds": rec.wall_seconds, "mass_drift": rec.mass_drift})
    ref = fields[-1]
    for i, field in enumerate(fields):
        err = float(rel_l2(field[:, -1], ref[:, -1]).mean())
        levels[i]["rel_error_vs_finest"] = err
    for i in range(n_levels - 2):
        e1 = levels[i]["rel_error_vs_finest"]
        e2 = levels[i + 1]["rel_error_vs_finest"]
        if e2 > 0:
            levels[i]["observed_order_vs_next"] = float(torch.log2(torch.tensor(e1 / e2)))
        else:
            levels[i]["observed_order_vs_next"] = None
    return levels
