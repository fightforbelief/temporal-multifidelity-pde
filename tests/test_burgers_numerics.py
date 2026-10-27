import math

import torch

from temporal_mf.numerical.burgers import energy, grid, integrate, mass, rel_l2
from temporal_mf.numerical.convergence import temporal_convergence


def _exact_sine(t, x):
    return torch.exp(torch.as_tensor(-t, dtype=x.dtype)) * torch.sin(2 * math.pi * x)


def test_constant_and_zero_are_steady():
    length = 1.0
    nx = 64
    u0 = torch.ones(2, nx, dtype=torch.float64)
    u0[1].zero_()
    t_obs = torch.tensor([0.0, 0.2, 0.4], dtype=torch.float64)
    field, rec = integrate(u0, nu=0.01, length=length, dt=0.01, t_observe=t_obs)
    assert rec.status == "OK"
    assert torch.allclose(field[0, -1], torch.ones(nx, dtype=torch.float64), atol=1e-10)
    assert torch.allclose(field[1, -1], torch.zeros(nx, dtype=torch.float64), atol=1e-10)


def test_periodic_mass_and_energy():
    length = 1.0
    nx = 64
    x = grid(nx, length)
    u0 = (0.4 * torch.sin(2 * math.pi * x) + 0.1 * torch.sin(4 * math.pi * x))[None, :]
    t_obs = torch.linspace(0, 0.2, 5)
    field, rec = integrate(u0, nu=0.02, length=length, dt=5e-4, t_observe=t_obs)
    assert rec.status == "OK"
    assert abs(float(mass(field[0, -1], length) - mass(u0[0], length))) < 1e-8
    assert float(energy(field[0, -1], length)) <= float(energy(u0[0], length)) + 1e-6


def test_manufactured_solution_and_temporal_order():
    length = 1.0
    nx = 96
    nu = 0.02
    x = grid(nx, length)

    def forcing(_t, u):
        s = torch.sin(2 * math.pi * x)
        c = torch.cos(2 * math.pi * x)
        decay = math.exp(-_t)
        ut = -decay * s
        adv = (decay * s) * (decay * 2 * math.pi * c)
        uxx = -((2 * math.pi) ** 2) * decay * s
        f = ut + adv - nu * uxx
        return f[None, :].to(dtype=u.dtype).expand_as(u)

    def run(dt):
        t_obs = torch.tensor([0.0, 0.1], dtype=torch.float64)
        u0 = _exact_sine(0.0, x)[None, :]
        field, rec = integrate(u0, nu=nu, length=length, dt=dt, t_observe=t_obs, forcing_fn=forcing)
        assert rec.status == "OK"
        return float(rel_l2(field[0, -1], _exact_sine(0.1, x)))

    coarse = run(2e-3)
    fine = run(1e-3)
    assert coarse < 5e-2
    assert fine < coarse


def test_temporal_refinement_decreases_error():
    length = 1.0
    x = grid(48, length)
    u0 = torch.sin(2 * math.pi * x)[None, :]
    rows = temporal_convergence(u0, nu=0.05, length=length, dt_coarse=0.02, t_final=0.08, n_levels=3)
    assert rows[0]["rel_error_vs_finest"] > rows[1]["rel_error_vs_finest"]
    assert rows[-1]["rel_error_vs_finest"] == 0.0


def test_unstable_step_is_flagged_not_shrunk():
    length = 1.0
    nx = 32
    u0 = (3 * torch.sin(2 * math.pi * grid(nx, length)))[None, :]
    t_obs = torch.tensor([0.0, 0.2])
    _field, rec = integrate(u0, nu=1e-4, length=length, dt=0.2, t_observe=t_obs)
    assert rec.status in {"UNSTABLE_CFL", "NONFINITE"}
    assert rec.dt == 0.2
