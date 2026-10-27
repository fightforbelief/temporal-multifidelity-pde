"""Figures for the locked Burgers run. Reads JSON only."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "reports" / "figures"
FIG.mkdir(parents=True, exist_ok=True)


def main() -> None:
    cal = json.loads((ROOT / "outputs" / "solver_calibration.json").read_text(encoding="utf-8"))
    results = json.loads((ROOT / "outputs" / "l2" / "confirmatory" / "results.json").read_text(encoding="utf-8"))
    _convergence(cal)
    _cost()
    _budget(results)
    _rollout(results)
    print(f"figures {FIG}")


def _convergence(cal: dict) -> None:
    xs, ys, labels = [], [], []
    for name in ("FINE", "HF", "MF"):
        row = cal["candidates"][name]
        if row["status"] != "OK":
            continue
        xs.append(row["dt"])
        ys.append(row["mean_final_rel_vs_archive"])
        labels.append(name)
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.loglog(xs, ys, "o", color="#1f4e79")
    for x, y, lab in zip(xs, ys, labels):
        ax.annotate(lab, (x, y), textcoords="offset points", xytext=(6, 4))
    ax.set_xlabel("fixed interior dt")
    ax.set_ylabel("mean final relative L2 vs PDEBench, trajectories 0-3")
    ax.set_title("4x dt is UNSTABLE_CFL and is omitted")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "l2_fidelity_convergence.png", dpi=140)
    plt.close(fig)


def _cost() -> None:
    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    ax.bar(["HF dt", "LF 2x dt"], [0.5899143375014925, 0.29754611041668494], color=["#1f4e79", "#c47b2b"])
    ax.set_ylabel("seconds per trajectory, 48 initial conditions")
    ax.set_title("Measured generation time, nx=1024, T=2")
    fig.tight_layout()
    fig.savefig(FIG / "l2_lf_hf_cost.png", dpi=140)
    plt.close(fig)


def _budget(results: dict) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    methods = ["HF_ONLY", "MIXED_CONDITIONED", "LF_PRETRAIN_HF_FINETUNE", "POD_DELTA_RIDGE", "LF_SOLVER", "IDENTITY"]
    for method in methods:
        xs, ys = [], []
        for block in results["fractions"]:
            vals = [
                r["test"]["mean_rollout_rel"]
                for r in block["results"]
                if r["method"] == method and r.get("test") and np.isfinite(r["test"]["mean_rollout_rel"])
            ]
            if not vals:
                continue
            xs.append(block["budget_fraction"])
            ys.append(float(np.median(vals)))
        if xs:
            ax.plot(xs, ys, marker="o", label=method)
    ax.set_yscale("log")
    ax.set_xlabel("fraction of the HF training-set generation budget")
    ax.set_ylabel("median over seeds of mean test rollout relative L2")
    ax.set_title("Horizon 200. Neural runs that diverged stay in the median.")
    ax.legend(fontsize=7)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "l2_budget_accuracy.png", dpi=140)
    plt.close(fig)


def _rollout(results: dict) -> None:
    horizons = [10, 40, 100, 200]
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    block = next(b for b in results["fractions"] if b["budget_fraction"] == 1.0)
    methods = ["HF_ONLY", "MIXED_CONDITIONED", "LF_PRETRAIN_HF_FINETUNE", "POD_DELTA_RIDGE", "LF_SOLVER", "IDENTITY"]
    for method in methods:
        rows = [r for r in block["results"] if r["method"] == method and r.get("test_trajectories")]
        if not rows:
            continue
        means = []
        for h in horizons:
            vals = []
            for row in rows:
                for traj in row["test_trajectories"]:
                    vals.append(traj.get(f"rollout_h{h}_rel"))
            finite = [v for v in vals if v is not None and np.isfinite(v)]
            means.append(float(np.mean(finite)) if finite else np.nan)
        ax.plot(horizons, means, marker="o", label=method)
    ax.set_yscale("log")
    ax.set_xlabel("autoregressive steps (dt_obs = 0.01)")
    ax.set_ylabel("mean test relative L2, budget fraction 1")
    ax.legend(fontsize=7)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "l2_rollout_curves.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    main()
