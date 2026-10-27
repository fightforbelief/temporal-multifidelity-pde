"""Plot only quantities that already exist on disk. Run on hanqing-win."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "reports" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
L1 = ROOT / "external" / "setinek" / "grid" / "outputs_l1"
OUT = ROOT / "outputs" / "l1"
OUT.mkdir(parents=True, exist_ok=True)


def load_results() -> list[dict]:
    rows = []
    if not L1.exists():
        return rows
    for path in sorted(L1.glob("*/result.csv")):
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                row["run_dir"] = path.parent.name
                rows.append(row)
    dest = OUT / "all_results.csv"
    if rows:
        fields = list(rows[0].keys())
        with dest.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    return rows


def plot_convergence() -> None:
    summary_path = ROOT / "outputs" / "solver_pilot" / "summary.json"
    if not summary_path.exists():
        return
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    rows = [r for r in summary["convergence"] if r["dt"] > 0 and r["rel_error_vs_finest"] > 0]
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.loglog([r["dt"] for r in rows], [r["rel_error_vs_finest"] for r in rows], "o-", color="#1f4e79")
    ax.set_xlabel("time step dt")
    ax.set_ylabel("relative L2 vs finest dt in this pilot")
    ax.set_title("Smooth-IC Burgers pilot, not a PDEBench trajectory")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "fidelity_convergence.png", dpi=140)
    plt.close(fig)


def plot_costs() -> None:
    summary_path = ROOT / "outputs" / "solver_pilot" / "summary.json"
    if not summary_path.exists():
        return
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    names = ["HF", "MF", "LF"]
    med = [summary["costs"][n]["per_trajectory_median"] for n in names]
    lo = [summary["costs"][n]["min_seconds"] / 8 for n in names]
    hi = [summary["costs"][n]["max_seconds"] / 8 for n in names]
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.bar(names, med, color=["#1f4e79", "#5b8c5a", "#c47b2b"])
    ax.vlines(range(3), lo, hi, color="black")
    ax.set_ylabel("seconds per trajectory (8 ICs, T=0.4, nx=64)")
    ax.set_title("Measured wall time, smooth IC, three repeats")
    fig.tight_layout()
    fig.savefig(FIG / "lf_hf_cost.png", dpi=140)
    plt.close(fig)


def plot_anchor(rows: list[dict]) -> None:
    if not rows:
        return
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.2))
    for ax, prefix, title in (
        (axes[0], "long_", "1500 steps, channels 32, CPU"),
        (axes[1], "full_", "20000 steps, channels 32, CPU"),
    ):
        subset = [r for r in rows if r["run_dir"].startswith(prefix)]
        if not subset:
            ax.set_title(title + "\n(no completed run)")
            ax.axis("off")
            continue
        for lambd, marker in (("0.0", "o"), ("20.0", "s")):
            chosen = [r for r in subset if r["lambd"] == lambd]
            budgets = sorted({float(r["budget"]) for r in chosen})
            ys = []
            for budget in budgets:
                vals = [float(r["mean_prediction_error"]) for r in chosen if float(r["budget"]) == budget]
                ys.append(sum(vals) / len(vals))
            ax.plot(budgets, ys, marker=marker, label=f"lambda={lambd}")
        ax.set_xlabel("analytical budget (sum of V-cycles K)")
        ax.set_ylabel("mean relative L2")
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    fig.suptitle("Scaled official truncation_1d UNet. Not the paper's published curve.")
    fig.tight_layout()
    fig.savefig(FIG / "anchor_replication.png", dpi=140)
    plt.close(fig)


def main() -> None:
    rows = load_results()
    plot_convergence()
    plot_costs()
    plot_anchor(rows)
    print(f"rows={len(rows)} figures={FIG}")


if __name__ == "__main__":
    main()
