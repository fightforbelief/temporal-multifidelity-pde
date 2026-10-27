"""Reproduce PDEBench Burgers on trajectories 0-3 with the official helpers.

The flux update is called through ``utils.bc`` and ``utils.limiting``. The
source file is not copied. This script does not read trajectories other than
the four calibration rows stored in the npz.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
NLE = ROOT / "external" / "pdebench" / "pdebench" / "data_gen" / "data_gen_NLE"
sys.path.insert(0, str(NLE))

import jax
import jax.numpy as jnp
from jax import lax

import utils

jax.config.update("jax_enable_x64", False)


def main() -> None:
    payload = np.load(ROOT / "outputs" / "solver_calibration_input.npz")
    archive = payload["u"].astype(np.float32)
    x_arch = payload["x"].astype(np.float32)
    t_arch = payload["t"].astype(np.float32)
    nx = 1024
    x_l, x_r = 0.0, 1.0
    epsilon = 1.0e-2
    dx = (x_r - x_l) / nx
    xe = jnp.linspace(x_l, x_r, nx + 1)
    xc = xe[:-1] + 0.5 * dx
    x_err = float(np.max(np.abs(np.array(xc) - x_arch)))
    searches = {}
    for k_tot in (4, 8):
        generated = np.array(utils.init_multi(xc, numbers=10000, k_tot=k_tot, init_key=2022))
        searches[str(k_tot)] = _nearest(generated, archive[:, 0, :])
    out = {
        "x_max_abs_err": x_err,
        "dx": dx,
        "x_arch_ends": [float(x_arch[0]), float(x_arch[-1])],
        "t_arch_count": int(t_arch.size),
        "t_arch_ends": [float(t_arch[0]), float(t_arch[-1])],
        "tensor_time": int(archive.shape[1]),
        "init_multi_search": searches,
        "epsilon": epsilon,
        "nu_official": epsilon / float(np.pi),
        "domain": [x_l, x_r],
        "cfl_config": 0.25,
        "jax_backend": str(jax.default_backend()),
        "note": "Archive initial data are integrated below. init_multi is only a provenance check.",
    }
    (ROOT / "outputs" / "solver_calibration_ic.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps({"x_max_abs_err": x_err, "search": searches, "backend": out["jax_backend"]}))
    _integrate(jnp.asarray(archive[:, 0, :]), archive, t_arch, dx, epsilon, xc)


def _nearest(generated: np.ndarray, target: np.ndarray) -> dict:
    # generated (10000, nx), target (4, nx). Report the best match for each archive row.
    gen_norm = np.linalg.norm(generated, axis=1)
    rows = []
    for i in range(target.shape[0]):
        diff = generated - target[i]
        rel = np.linalg.norm(diff, axis=1) / max(np.linalg.norm(target[i]), 1e-8)
        j = int(np.argmin(rel))
        rows.append({"archive_index": i, "best_generated_index": j, "best_rel": float(rel[j]), "same_order_rel": float(rel[i])})
    return {"per_archive_row": rows}


def _rel_rows(pred, ref) -> list[float]:
    num = np.linalg.norm(pred - ref, axis=1)
    den = np.maximum(np.linalg.norm(ref, axis=1), 1e-8)
    return [float(v) for v in num / den]


def _limiting(u, ncell):
    du_l = u[1 : ncell + 3] - u[0 : ncell + 2]
    du_r = u[2 : ncell + 4] - u[1 : ncell + 3]
    du_m = (u[2 : ncell + 4] - u[0 : ncell + 2]) * 0.5
    gradu = utils.VLlimiter(du_l, du_r, du_m)
    u_l = jnp.zeros_like(u).at[1 : ncell + 3].set(u[1 : ncell + 3] - 0.5 * gradu)
    u_r = jnp.zeros_like(u).at[1 : ncell + 3].set(u[1 : ncell + 3] + 0.5 * gradu)
    return u_l, u_r


def _integrate(u0, archive, t_arch, dx, epsilon, xc) -> None:
    del xc
    pi_inv = 1.0 / jnp.pi
    nx = int(u0.shape[-1])
    cfl = jnp.float32(0.25)
    fin_time = jnp.float32(2.0)
    dt_save = jnp.float32(0.01)
    n_save = int(archive.shape[1])

    def flux(u):
        padded = utils.bc(u, dx, Ncell=nx)
        # Checked-out utils.limiting calls `.loc`, which this JAX does not implement.
        # `.at` is the same index-update API the rest of that file already uses.
        u_l, u_r = _limiting(padded, nx)
        f_l = 0.5 * u_l**2
        f_r = 0.5 * u_r**2
        f_up = 0.5 * (
            f_r[1 : nx + 2]
            + f_l[2 : nx + 3]
            - 0.5 * jnp.abs(u_l[2 : nx + 3] + u_r[1 : nx + 2]) * (u_l[2 : nx + 3] - u_r[1 : nx + 2])
        )
        f_up = f_up + (-epsilon * pi_inv * (padded[2 : nx + 3] - padded[1 : nx + 2]) * (1.0 / dx))
        return f_up

    def update(u, u_tmp, dt):
        f = flux(u_tmp)
        return u - dt * (1.0 / dx) * (f[1 : nx + 1] - f[0:nx])

    def step(carry):
        u, t, tsave, i_save, saved, steps, max_speed = carry
        do_save = (t >= tsave - 1e-7) & (i_save < n_save)
        saved = lax.cond(do_save, lambda s: s.at[i_save].set(u), lambda s: s, saved)
        tsave = jnp.where(do_save, tsave + dt_save, tsave)
        i_save = jnp.where(do_save, i_save + 1, i_save)
        speed = jnp.max(jnp.abs(u))
        dt = jnp.minimum(
            jnp.minimum(utils.Courant(u, dx) * cfl, utils.Courant_diff(dx, epsilon * pi_inv) * cfl),
            jnp.minimum(fin_time - t, jnp.maximum(tsave - t, 0.0)),
        )
        u_tmp = update(u, u, dt * 0.5)
        u_next = update(u, u_tmp, dt)
        advance = dt > 1.0e-8
        u = jnp.where(advance, u_next, u)
        t = t + jnp.where(advance, dt, 0.0)
        return u, t, tsave, i_save, saved, steps + 1, jnp.maximum(max_speed, speed)

    def one(u_single):
        saved0 = jnp.zeros((n_save, nx), dtype=u_single.dtype)
        return lax.while_loop(
            lambda carry: carry[1] < fin_time - 1e-8,
            step,
            (
                u_single,
                jnp.float32(0.0),
                jnp.float32(0.0),
                jnp.int32(0),
                saved0,
                jnp.int32(0),
                jnp.float32(0.0),
            ),
        )

    started = time.perf_counter()
    run = jax.jit(one)
    pieces = []
    for i in range(u0.shape[0]):
        print(f"integrate_calibration_index {i}", flush=True)
        pieces.append(jax.block_until_ready(run(u0[i])))
    elapsed = time.perf_counter() - started
    stacked = [jnp.stack([piece[k] for piece in pieces]) for k in range(7)]
    u_f, t_f, _, i_save, saved, steps, max_speed = stacked
    saved = saved.at[:, -1].set(u_f)
    pred = np.array(saved)
    final_rel = _rel_rows(pred[:, -1, :], archive[:, -1, :])
    mid = min(100, pred.shape[1] - 1)
    mid_rel = _rel_rows(pred[:, mid, :], archive[:, mid, :])
    result = {
        "wall_seconds": elapsed,
        "steps": [int(v) for v in np.array(steps)],
        "final_t": [float(v) for v in np.array(t_f)],
        "n_saved_indices": [int(v) for v in np.array(i_save)],
        "max_speed": [float(v) for v in np.array(max_speed)],
        "final_rel_l2": final_rel,
        "final_mean_rel": float(np.mean(final_rel)),
        "t_index_100_rel_l2": mid_rel,
        "t_index_100_archive_time": float(t_arch[mid]) if mid < t_arch.size else None,
        "scheme": "official utils.bc/limiting/Courant, CFL 0.25, viscosity epsilon/pi",
    }
    (ROOT / "outputs" / "solver_calibration_fv.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    np.savez_compressed(
        ROOT / "outputs" / "solver_calibration_fv_fields.npz",
        pred=pred.astype(np.float32),
        ref=archive.astype(np.float32),
    )
    print(json.dumps({"final_mean_rel": result["final_mean_rel"], "seconds": elapsed, "steps": result["steps"]}))


if __name__ == "__main__":
    main()
