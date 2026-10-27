"""Discrete generation-budget allocation. Counts are paid trajectories, not windows."""

from __future__ import annotations


def allocate_counts(
    budget_seconds: float,
    cost_hf: float,
    cost_lf: float,
    hf_cost_fraction: float,
) -> dict:
    if budget_seconds <= 0 or cost_hf <= 0 or cost_lf <= 0:
        raise ValueError("budget and costs must be positive")
    if not 0.0 <= hf_cost_fraction <= 1.0:
        raise ValueError("hf_cost_fraction must be in [0, 1]")
    hf_budget = budget_seconds * hf_cost_fraction
    lf_budget = budget_seconds - hf_budget
    n_hf = int(hf_budget // cost_hf)
    n_lf = int(lf_budget // cost_lf)
    actual = n_hf * cost_hf + n_lf * cost_lf
    return {
        "n_hf": n_hf,
        "n_lf": n_lf,
        "budget_target": float(budget_seconds),
        "budget_actual": float(actual),
        "unspent": float(budget_seconds - actual),
        "hf_cost_fraction_realized": float((n_hf * cost_hf) / actual) if actual > 0 else None,
        "within_10pct": abs(actual - budget_seconds) / budget_seconds <= 0.1,
    }
