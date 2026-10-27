import math

import torch

from temporal_mf.numerical.burgers import grid, integrate


def test_independent_steps_change_the_field_and_share_ic_and_times():
    length = 1.0
    x = grid(32, length)
    u0 = torch.sin(2 * math.pi * x)[None, :]
    t_obs = torch.tensor([0.0, 0.1, 0.2])
    hf, rec_h = integrate(u0, nu=0.01, length=length, dt=0.002, t_observe=t_obs)
    lf, rec_l = integrate(u0, nu=0.01, length=length, dt=0.01, t_observe=t_obs)
    assert torch.allclose(hf[:, 0], lf[:, 0])
    assert hf.shape[1] == lf.shape[1] == 3
    assert not torch.allclose(hf[:, -1], lf[:, -1])
    assert rec_h.n_steps > rec_l.n_steps
    assert rec_h.status == "OK" and rec_l.status == "OK"
