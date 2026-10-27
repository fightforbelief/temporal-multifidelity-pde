import numpy as np

from temporal_mf.eval.bootstrap import paired_bootstrap_mean_diff
from temporal_mf.eval.metrics import quadrature_errors
from temporal_mf.eval.pareto import nondominated


def test_relative_floor_and_quadrature():
    truth = np.zeros(8)
    pred = np.zeros(8)
    _abs, rel = quadrature_errors(pred, truth, length=1.0, eps=1e-6)
    assert rel == 0.0
    pred = np.ones(8)
    abs_e, rel_e = quadrature_errors(pred, truth, length=1.0, eps=1e-6)
    assert abs_e > 0
    assert rel_e == abs_e / 1e-6


def test_bootstrap_is_paired_on_ids():
    a = {"a": 1.0, "b": 3.0}
    b = {"a": 1.0, "b": 1.0, "c": 9.0}
    out = paired_bootstrap_mean_diff(a, b, n_boot=50, seed=950)
    assert out["n_trajectories"] == 2
    assert out["ids"] == ["a", "b"]
    assert out["mean_diff_a_minus_b"] == 1.0


def test_pareto_marks_dominated_point():
    rows = nondominated(
        [
            {"name": "good", "cost": 1.0, "error": 1.0},
            {"name": "worse", "cost": 2.0, "error": 2.0},
        ]
    )
    by_name = {r["name"]: r["pareto_dominated"] for r in rows}
    assert by_name["good"] is False
    assert by_name["worse"] is True
