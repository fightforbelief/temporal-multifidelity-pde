"""Numerical smoke: manufactured solution, refinement, and repeated timings."""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch
import yaml

from _bootstrap import repo
from temporal_mf.numerical.burgers import grid, integrate
from temporal_mf.numerical.convergence import temporal_convergence
from temporal_mf.numerical.cost_meter import time_solver
from temporal_mf.study import write_json


def main() -> None:
    root = repo()
    cfg = yaml.safe_load((root / "configs" / "l2_solver_pilot.yaml").read_text(encoding="utf-8"))
    length = float(cfg["length"])
    nu = float(cfg["epsilon"]) / math.pi
    nx = int(cfg["nx"])
    x = grid(nx, length, dtype=torch.float64)
    u0 = (torch.sin(2 * math.pi * x) + 0.15 * torch.sin(4 * math.pi * x))[None, :].repeat(int(cfg["n_ic"]), 1)
    u0 = u0 + 0.05 * torch.randn(u0.shape, dtype=torch.float64)
    t_final = float(cfg["T"])
    dt_obs = float(cfg["dt_obs"])
    n = int(round(t_final / dt_obs))
    t_obs = torch.linspace(0, t_final, n + 1, dtype=torch.float64)
    rows = temporal_convergence(u0[:1], nu=nu, length=length, dt_coarse=float(cfg["dt_lf"]), t_final=t_final, n_levels=3)
    costs = {}
    for name, dt in (("HF", cfg["dt_hf"]), ("MF", cfg["dt_mf"]), ("LF", cfg["dt_lf"])):
        costs[name] = time_solver(u0, nu=nu, length=length, dt=float(dt), t_observe=t_obs, repeats=int(cfg["repeats"]))
    field, rec = integrate(u0[:1], nu=nu, length=length, dt=float(cfg["dt_hf"]), t_observe=t_obs)
    payload = {
        "nu": nu,
        "equation": "u_t + (u^2/2)_x = nu u_xx",
        "nu_source": "epsilon/pi with epsilon from PDEBench Burgers config",
        "convergence": rows,
        "costs": costs,
        "single_status": rec.status,
        "mass_drift": rec.mass_drift,
        "energy_start": rec.energy_start,
        "energy_end": rec.energy_end,
        "final_field_finite": bool(torch.isfinite(field).all()),
        "lineage": "SOLVER_PILOT_SMOOTH_IC_NOT_PDEBENCH",
    }
    out = root / "outputs" / "solver_pilot" / "summary.json"
    write_json(out, payload)
    print(json.dumps({"out": str(out), "hf_median": costs["HF"]["median_seconds"], "lf_median": costs["LF"]["median_seconds"], "status": rec.status}))


if __name__ == "__main__":
    main()
