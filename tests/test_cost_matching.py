from temporal_mf.fidelity.allocate import allocate_counts
from temporal_mf.fidelity.interpolate import fourier_resample
import torch


def test_budget_does_not_overspend():
    row = allocate_counts(10.0, cost_hf=3.0, cost_lf=1.0, hf_cost_fraction=0.5)
    assert row["budget_actual"] <= 10.0
    assert row["n_hf"] == 1
    assert row["n_lf"] == 5
    assert row["unspent"] == 10.0 - row["budget_actual"]


def test_interpolation_identity():
    u = torch.randn(2, 16)
    same = fourier_resample(u, 16)
    assert torch.allclose(same, u, atol=1e-6)
