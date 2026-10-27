"""Non-dominated cost/error points. Lower is better on both axes."""

from __future__ import annotations


def nondominated(rows: list[dict], cost_key: str = "cost", error_key: str = "error") -> list[dict]:
    keep = []
    for i, row in enumerate(rows):
        dominated = False
        for j, other in enumerate(rows):
            if i == j:
                continue
            cheaper_or_eq = other[cost_key] <= row[cost_key]
            better_or_eq = other[error_key] <= row[error_key]
            strict = other[cost_key] < row[cost_key] or other[error_key] < row[error_key]
            if cheaper_or_eq and better_or_eq and strict:
                dominated = True
                break
        tagged = dict(row)
        tagged["pareto_dominated"] = dominated
        keep.append(tagged)
    return keep
