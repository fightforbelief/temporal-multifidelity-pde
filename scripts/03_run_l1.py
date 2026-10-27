"""Scaled official Setinek 1D truncation smoke, then a capped matrix if the smoke is cheap."""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

from _bootstrap import repo


def _run(python: Path, grid: Path, overrides: list[str], log_path: Path) -> tuple[int, float]:
    env = os.environ.copy()
    env["JAX_PLATFORMS"] = "cpu"
    env["PYTHONPATH"] = str(grid)
    cmd = [str(python), "-m", "experiments.run", *overrides]
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, cwd=grid, env=env, capture_output=True, text=True)
    elapsed = time.perf_counter() - t0
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(proc.stdout + "\n" + proc.stderr, encoding="utf-8")
    return proc.returncode, elapsed


def main() -> None:
    root = repo()
    python = root / ".venv-jax" / "Scripts" / "python.exe"
    grid = root / "external" / "setinek" / "grid"
    out = root / "outputs" / "l1"
    out.mkdir(parents=True, exist_ok=True)
    if not python.exists():
        (out / "status.json").write_text(json.dumps({"status": "BLOCKED", "reason": "jax venv missing"}), encoding="utf-8")
        raise SystemExit(2)
    smoke = [
        "axis=truncation_1d",
        "arch=unet",
        "lambd=0",
        "budget=6",
        "seed=0",
        "num_training_steps=30",
        "num_warmup_steps=4",
        "num_whitening_samples=8",
        "num_test_samples=4",
        "batch_size=4",
        "base_channels=8",
        "hydra.run.dir=outputs_l1/smoke",
    ]
    # Hydra chdir writes inside grid/outputs_l1 because we override run dir relative to cwd.
    code, elapsed = _run(python, grid, smoke, root / "logs" / "l1_smoke.log")
    status = {"smoke_exit": code, "smoke_seconds": elapsed}
    if code != 0:
        status["status"] = "BLOCKED"
        status["log"] = str(root / "logs" / "l1_smoke.log")
        (out / "status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
        print(json.dumps(status))
        raise SystemExit(code)
    runs = []
    if elapsed < 180:
        for budget in (6, 18):
            for lambd in (0, 20):
                for seed in (0, 1):
                    tag = f"b{budget}_l{lambd}_s{seed}"
                    overrides = [
                        "axis=truncation_1d",
                        "arch=unet",
                        f"lambd={lambd}",
                        f"budget={budget}",
                        f"seed={seed}",
                        "num_training_steps=80",
                        "num_warmup_steps=8",
                        "num_whitening_samples=16",
                        "num_test_samples=8",
                        "batch_size=4",
                        "base_channels=16",
                        f"hydra.run.dir=outputs_l1/{tag}",
                    ]
                    code_i, seconds = _run(python, grid, overrides, root / "logs" / f"l1_{tag}.log")
                    runs.append({"tag": tag, "exit": code_i, "seconds": seconds, "budget": budget, "lambd": lambd, "seed": seed})
                    if seconds > 30 * 60:
                        status["status"] = "RESOURCE_LIMIT"
                        break
    else:
        status["status"] = "PARTIAL"
        status["reason"] = "smoke exceeded 3 minutes; matrix not started"
    if "status" not in status:
        status["status"] = "PASS" if all(r["exit"] == 0 for r in runs) else "PARTIAL"
    status["runs"] = runs
    status["scale_note"] = "Scaled-down official truncation_1d UNet. Not the paper's 20,000-step sweep."
    (out / "status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    print(json.dumps({"status": status["status"], "smoke_seconds": elapsed, "n_runs": len(runs)}))


if __name__ == "__main__":
    main()
