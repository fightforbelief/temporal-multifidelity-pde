"""Independent periodic viscous Burgers solver.

The PDE implemented here is

    u_t + (u^2 / 2)_x = nu * u_xx

on a periodic interval of length ``L``. PDEBench's Burgers generator adds a
viscous flux ``-epsilon/pi * du``, so the matching viscosity is
``nu = epsilon / pi`` when ``epsilon`` is the config value named Nu in the
DaRUS filename. This module does not copy that generator. It uses a
Fourier pseudo-spectral IMEX-Euler step: explicit advection with a 2/3
dealiasing rule, implicit diffusion. Temporal fidelity is the fixed step
``dt``. The step is never silently reduced; an unstable step is flagged.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import torch


@dataclass
class SolveRecord:
    status: str
    dt: float
    n_steps: int
    max_cfl: float
    mass_drift: float
    energy_start: float
    energy_end: float
    wall_seconds: float
    nu: float
    nx: int


def wavenumbers(nx: int, length: float, device, dtype) -> torch.Tensor:
    freq = torch.fft.rfftfreq(nx, d=length / nx, device=device, dtype=dtype)
    return (2.0 * math.pi) * freq


def _dealias(spec: torch.Tensor, nx: int) -> torch.Tensor:
    nfreq = spec.shape[-1]
    modes = torch.arange(nfreq, device=spec.device)
    cutoff = max(nx // 3, 1)
    return torch.where(modes <= cutoff, spec, torch.zeros_like(spec))


def mass(u: torch.Tensor, length: float) -> torch.Tensor:
    dx = length / u.shape[-1]
    return dx * u.sum(dim=-1)


def energy(u: torch.Tensor, length: float) -> torch.Tensor:
    dx = length / u.shape[-1]
    return 0.5 * dx * (u * u).sum(dim=-1)


def advective_cfl(u: torch.Tensor, length: float, dt: float) -> torch.Tensor:
    dx = length / u.shape[-1]
    speed = u.abs().amax(dim=-1).clamp_min(1e-12)
    return dt * speed / dx


def imex_step(
    u: torch.Tensor,
    nu: float,
    length: float,
    dt: float,
    forcing: torch.Tensor | None = None,
) -> torch.Tensor:
    """One IMEX-Euler step. ``forcing`` is f(t) such that u_t = -flux_x + nu u_xx + f."""
    nx = u.shape[-1]
    k = wavenumbers(nx, length, u.device, u.dtype)
    u_hat = torch.fft.rfft(u, dim=-1)
    flux = 0.5 * (u * u)
    flux_hat = _dealias(torch.fft.rfft(flux, dim=-1), nx)
    rhs = u_hat - dt * (1j * k) * flux_hat
    if forcing is not None:
        rhs = rhs + dt * torch.fft.rfft(forcing, dim=-1)
    denom = 1.0 + dt * nu * (k * k)
    return torch.fft.irfft(rhs / denom, n=nx, dim=-1)


def integrate(
    u0: torch.Tensor,
    *,
    nu: float,
    length: float,
    dt: float,
    t_observe: torch.Tensor,
    forcing_fn=None,
) -> tuple[torch.Tensor, SolveRecord]:
    """Integrate from t=0. ``t_observe`` must start at 0 and be divisible by ``dt``."""
    if u0.ndim != 2:
        raise ValueError("u0 must have shape (batch, nx)")
    if dt <= 0:
        raise ValueError("dt must be positive")
    t_obs = t_observe.detach().to(device=u0.device, dtype=u0.dtype)
    if t_obs.ndim != 1 or t_obs.numel() < 1:
        raise ValueError("t_observe must be a non-empty 1D tensor")
    if float(t_obs[0]) != 0.0:
        raise ValueError("t_observe must start at 0")
    if not torch.all(t_obs[1:] > t_obs[:-1]):
        raise ValueError("t_observe must be strictly increasing")
    span = t_obs[1:] - t_obs[:-1]
    steps_per = span / dt
    if ((steps_per - steps_per.round()).abs() > 1e-4).any():
        raise ValueError("dt does not divide observation intervals")
    steps_per_int = steps_per.round().to(dtype=torch.int64).tolist()

    t0 = time.perf_counter()
    with torch.no_grad():
        return _integrate_impl(u0, nu, length, dt, t_obs, steps_per_int, forcing_fn, t0)


def _integrate_impl(u0, nu, length, dt, t_obs, steps_per_int, forcing_fn, t0):
    u = u0.detach().clone()
    frames = [u.clone()]
    max_cfl = float(advective_cfl(u, length, dt).max())
    status = "OK"
    n_steps = 0
    t = 0.0
    e0 = energy(u, length)
    for n_sub in steps_per_int:
        for _ in range(int(n_sub)):
            cfl = float(advective_cfl(u, length, dt).max())
            max_cfl = max(max_cfl, cfl)
            if cfl > 1.25:
                status = "UNSTABLE_CFL"
            force = None
            if forcing_fn is not None:
                force = forcing_fn(t, u)
            u = imex_step(u, nu, length, dt, force)
            t += dt
            n_steps += 1
            if not torch.isfinite(u).all():
                status = "NONFINITE"
                u = torch.nan_to_num(u, nan=0.0, posinf=0.0, neginf=0.0)
                break
        frames.append(u.clone())
        if status == "NONFINITE":
            while len(frames) < t_obs.numel():
                frames.append(u.clone())
            break
    elapsed = time.perf_counter() - t0
    out = torch.stack(frames, dim=1)
    m0 = mass(u0, length)
    m1 = mass(out[:, -1], length)
    drift = float((m1 - m0).abs().max()) if torch.isfinite(m1).all() else float("inf")
    e1 = energy(out[:, -1], length)
    rec = SolveRecord(
        status=status,
        dt=float(dt),
        n_steps=n_steps,
        max_cfl=max_cfl,
        mass_drift=drift,
        energy_start=float(e0.mean()),
        energy_end=float(e1.mean()) if torch.isfinite(e1).all() else float("nan"),
        wall_seconds=elapsed,
        nu=float(nu),
        nx=int(u0.shape[-1]),
    )
    return out, rec


def grid(nx: int, length: float, device="cpu", dtype=torch.float64) -> torch.Tensor:
    dx = length / nx
    return (torch.arange(nx, device=device, dtype=dtype) + 0.5) * dx


def manufactured_forcing(t: float, x: torch.Tensor, nu: float) -> torch.Tensor:
    """Forcing for u = exp(-t) sin(2 pi x / L) is not used here.

    The closed form below assumes the periodic coordinate ``x`` already spans
    one full period and the solution u = exp(-t) * sin(2 pi x / L).
    ``x`` is (nx,). Returns a tensor broadcastable to (batch, nx) by the caller.
    """
    # Placeholder kept for API clarity; the test builds forcing explicitly.
    raise NotImplementedError


def rel_l2(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    diff = torch.linalg.vector_norm(a - b, dim=-1)
    denom = torch.linalg.vector_norm(b, dim=-1).clamp_min(eps)
    return diff / denom
