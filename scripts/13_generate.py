"""Re-integrate selected initial conditions at the locked fixed steps.

Inputs are t=0 only. The 4x step is not run: calibration marked it UNSTABLE_CFL.
"""

from __future__ import annotations

import importlib.util
import json
import time
from pathlib import Path

import jax
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("fixed_dt", ROOT / "scripts" / "12_fixed_dt.py")
fixed = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixed)

DT_OBS = 0.01
LEVELS = {"HF": 268, "LF": 134}


def main() -> None:
    payload = np.load(ROOT / "outputs" / "l2" / "selected_ic.npz")
    u0 = jax.numpy.asarray(payload["u0"].astype(np.float32))
    n_save = 201
    out = {}
    timings = {}
    for name, n_sub in LEVELS.items():
        dt = DT_OBS / n_sub
        runner = fixed.integrate_one(u0[0], dt, n_save)
        print(f"compile {name}", flush=True)
        jax.block_until_ready(runner(u0[0]))
        print(f"integrate {name} n={u0.shape[0]}", flush=True)
        started = time.perf_counter()
        pieces = []
        nonfinite = []
        for i in range(u0.shape[0]):
            piece = jax.block_until_ready(runner(u0[i]))
            pieces.append(piece)
            nonfinite.append(bool(piece[4]))
        elapsed = time.perf_counter() - started
        saved = jax.numpy.stack([p[2] for p in pieces])
        final = jax.numpy.stack([p[0] for p in pieces])
        saved = saved.at[:, -1].set(final)
        pred = np.array(saved)
        if any(nonfinite) or not np.isfinite(pred).all():
            raise SystemExit(f"BLOCKED: {name} produced non-finite values")
        out[name] = pred.astype(np.float32)
        timings[name] = {"wall_seconds": elapsed, "per_trajectory_seconds": elapsed / u0.shape[0], "dt": dt, "status": "OK"}
        print(json.dumps({name: timings[name]}), flush=True)
    np.savez_compressed(ROOT / "outputs" / "l2" / "generated_fields.npz", HF=out["HF"], LF=out["LF"])
    (ROOT / "outputs" / "l2" / "generation_timing.json").write_text(json.dumps(timings, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
