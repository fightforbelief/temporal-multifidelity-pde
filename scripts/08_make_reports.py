"""Fill report tables from artifacts that actually exist. Missing stages stay BLOCKED or NOT_RUN."""

from __future__ import annotations

import json
from pathlib import Path

from _bootstrap import repo


def _read(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    root = repo()
    reports = root / "reports"
    reports.mkdir(exist_ok=True)
    env = _read(reports / "01_REMOTE_ENVIRONMENT.json")
    solver = _read(root / "outputs" / "solver_pilot" / "summary.json")
    prep = _read(root / "outputs" / "l2" / "prepare_status.json")
    smoke = _read(root / "outputs" / "l2" / "smoke" / "results.json")
    confirmatory = _read(root / "outputs" / "l2" / "confirmatory" / "results.json")
    l1 = _read(root / "outputs" / "l1" / "status.json")
    lines = ["# 09 Results and statistics", ""]
    if confirmatory is None and smoke is None:
        lines.append("No Level-2 fit results were produced.")
    for name, blob in (("smoke", smoke), ("confirmatory", confirmatory)):
        if not blob:
            lines.append(f"- {name}: NOT_RUN")
            continue
        lines.append(f"## {name}")
        lines.append("")
        lines.append("| method | seed | val rollout rel | test rollout rel | C_data s | C_train s | n_hf | n_lf |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
        for row in blob["results"]:
            test = row.get("test")
            test_s = "" if not test else f"{test['mean_rollout_rel']:.4g}"
            lines.append(
                f"| {row['method']} | {row['seed']} | {row['val']['mean_rollout_rel']:.4g} | {test_s} | "
                f"{row.get('C_data_seconds')} | {row.get('C_train_seconds')} | {row.get('n_hf','')} | {row.get('n_lf','')} |"
            )
        lines.append("")
    (reports / "09_RESULTS_AND_STATISTICS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    summary = {
        "environment_recorded": env is not None,
        "solver_pilot": None if solver is None else solver.get("lineage"),
        "prepare": None if prep is None else prep.get("status"),
        "l1": None if l1 is None else l1.get("status"),
        "l2_smoke": smoke is not None,
        "l2_confirmatory": confirmatory is not None,
    }
    (reports / "RUN_STATUS.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
