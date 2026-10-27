"""Fixed interior steps on calibration trajectories 0-3.

Candidate steps divide the 0.01 observation interval. A candidate that
overflows or becomes non-finite is recorded as UNSTABLE_CFL and is not
replaced by a smaller step.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "external" / "pdebench" / "pdebench" / "data_gen" / "data_gen_NLE"))

import jax
import jax.numpy as jnp
from jax import lax

import utils

EPSILON = 1.0e-2
PI_INV = 1.0 / float(np.pi)
DX = 1.0 / 1024
NU = EPSILON * PI_INV
DT_OBS = 0.01
FIN = 2.0


def limiting(u, ncell):
    du_l = u[1 : ncell + 3] - u[0 : ncell + 2]
    du_r = u[2 : ncell + 4] - u[1 : ncell + 3]
    du_m = (u[2 : ncell + 4] - u[0 : ncell + 2]) * 0.5
    gradu = utils.VLlimiter(du_l, du_r, du_m)
    u_l = jnp.zeros_like(u).at[1 : ncell + 3].set(u[1 : ncell + 3] - 0.5 * gradu)
    u_r = jnp.zeros_like(u).at[1 : ncell + 3].set(u[1 : ncell + 3] + 0.5 * gradu)
    return u_l, u_r


def flux(u):
    nx = u.shape[-1]
    padded = utils.bc(u, DX, Ncell=nx)
    u_l, u_r = limiting(padded, nx)
    f_l = 0.5 * u_l**2
    f_r = 0.5 * u_r**2
    f_up = 0.5 * (
        f_r[1 : nx + 2]
        + f_l[2 : nx + 3]
        - 0.5 * jnp.abs(u_l[2 : nx + 3] + u_r[1 : nx + 2]) * (u_l[2 : nx + 3] - u_r[1 : nx + 2])
    )
    return f_up + (-EPSILON * PI_INV * (padded[2 : nx + 3] - padded[1 : nx + 2]) / DX)


def update(u, u_tmp, dt):
    nx = u.shape[-1]
    f = flux(u_tmp)
    return u - dt / DX * (f[1 : nx + 1] - f[0:nx])


def integrate_one(u0, dt_nominal: float, n_save: int):
    nx = u0.shape[-1]
    dt_nominal = jnp.float32(dt_nominal)

    def step(carry):
        u, t, saved, i_save, nonfinite, max_cfl = carry
        do_save = (t >= i_save * DT_OBS - 1e-7) & (i_save < n_save)
        saved = lax.cond(do_save, lambda s: s.at[i_save].set(u), lambda s: s, saved)
        i_save = jnp.where(do_save, i_save + 1, i_save)
        remain_save = jnp.maximum(i_save * DT_OBS - t, 0.0)
        remain_end = jnp.maximum(FIN - t, 0.0)
        dt = jnp.minimum(dt_nominal, jnp.minimum(remain_save, remain_end))
        speed = jnp.max(jnp.abs(u))
        cfl = dt_nominal * speed / DX
        u_tmp = update(u, u, dt * 0.5)
        u_next = update(u, u_tmp, dt)
        bad = ~jnp.isfinite(u_next).all()
        u = jnp.where(bad, u, u_next)
        t = t + jnp.where(bad, 0.0, dt)
        return u, t, saved, i_save, nonfinite | bad, jnp.maximum(max_cfl, cfl)

    def run(u_single):
        saved0 = jnp.zeros((n_save, nx), dtype=u_single.dtype)
        return lax.while_loop(
            lambda c: (c[1] < FIN - 1e-8) & (~c[4]),
            step,
            (u_single, jnp.float32(0.0), saved0, jnp.int32(0), jnp.bool_(False), jnp.float32(0.0)),
        )

    compiled = jax.jit(run)
    return compiled


def main() -> None:
    payload = np.load(ROOT / "outputs" / "solver_calibration_input.npz")
    archive = payload["u"].astype(np.float32)
    u0 = jnp.asarray(archive[:, 0, :])
    n_save = archive.shape[1]
    # 268 interior steps per observation matches the published diffusion-limited step.
    # 536 is the convergence reference. 134 and 67 are 2x and 4x and are not reduced if unstable.
    candidates = {"HF": 268, "FINE": 536, "MF": 134, "LF": 67}
    results = {}
    fields = {}
    for name, n_sub in candidates.items():
        dt = DT_OBS / n_sub
        print(f"start {name} dt={dt}", flush=True)
        runner = integrate_one(u0[0], dt, n_save)
        t0 = time.perf_counter()
        pieces = [jax.block_until_ready(runner(u0[i])) for i in range(4)]
        # second pass for timing without compile
        t1 = time.perf_counter()
        pieces = [jax.block_until_ready(runner(u0[i])) for i in range(4)]
        elapsed = time.perf_counter() - t1
        u_f = jnp.stack([p[0] for p in pieces])
        saved = jnp.stack([p[2] for p in pieces])
        nonfinite = [bool(p[4]) for p in pieces]
        max_cfl = [float(p[5]) for p in pieces]
        saved = saved.at[:, -1].set(u_f)
        pred = np.array(saved)
        rel = []
        for i in range(4):
            num = np.linalg.norm(pred[i, -1] - archive[i, -1])
            den = max(np.linalg.norm(archive[i, -1]), 1e-8)
            rel.append(float(num / den))
        unstable = any(nonfinite) or any(c > 1.0 for c in max_cfl) or not np.isfinite(pred).all()
        results[name] = {
            "dt": dt,
            "substeps_per_observation": n_sub,
            "timed_seconds_four_trajectories": elapsed,
            "compile_pass_seconds": t1 - t0,
            "per_trajectory_seconds": elapsed / 4.0,
            "nonfinite": nonfinite,
            "max_advective_cfl_using_nominal_dt": max_cfl,
            "final_rel_vs_archive": rel,
            "mean_final_rel_vs_archive": float(np.mean(rel)),
            "status": "UNSTABLE_CFL" if unstable else "OK",
        }
        fields[name] = pred
        print(json.dumps({name: results[name]}), flush=True)
    hf = fields["HF"]
    fine = fields["FINE"]
    mf = fields["MF"]

    def _pair(a, b):
        vals = []
        for i in range(4):
            num = np.linalg.norm(a[i, -1] - b[i, -1])
            den = max(np.linalg.norm(b[i, -1]), 1e-8)
            vals.append(float(num / den))
        return vals

    conv = _pair(hf, fine)
    hf_mf = _pair(hf, mf)
    summary = {
        "nu": NU,
        "equation": "u_t + d_x(u^2/2) = (epsilon/pi) u_xx",
        "epsilon": EPSILON,
        "domain": [0.0, 1.0],
        "nx": 1024,
        "dx": DX,
        "candidates": results,
        "hf_vs_halfstep_final_rel": conv,
        "hf_vs_halfstep_mean_rel": float(np.mean(conv)),
        "hf_vs_mf_final_rel": hf_mf,
        "hf_vs_mf_mean_rel": float(np.mean(hf_mf)),
        "calibration_indices": [0, 1, 2, 3],
    }
    (ROOT / "outputs" / "solver_calibration.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"hf_vs_halfstep_mean_rel": summary["hf_vs_halfstep_mean_rel"]}))


if __name__ == "__main__":
    main()
