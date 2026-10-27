"""Fit the registered matrix. Confirmatory mode refuses to run without a frozen protocol."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import yaml

from _bootstrap import repo
from temporal_mf.study import run_method_matrix, write_json


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    root = repo()
    config_path = Path(sys.argv[1]) if len(sys.argv) > 1 else root / "configs" / "l2_smoke.yaml"
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    status_path = root / "outputs" / "l2" / "prepare_status.json"
    if not status_path.exists():
        raise SystemExit("BLOCKED: prepare_status.json missing")
    status = json.loads(status_path.read_text(encoding="utf-8"))
    if status.get("status") != "PASS" or status.get("lineage") != "PDEBENCH_IC":
        raise SystemExit("BLOCKED: fidelity bundle is not a verified PDEBench-IC prepare")
    if cfg.get("requires_frozen_protocol"):
        frozen = root / "outputs" / "l2" / "FROZEN_PROTOCOL.json"
        if not frozen.exists():
            raise SystemExit("BLOCKED: confirmatory run requires FROZEN_PROTOCOL.json")
        protocol = json.loads(frozen.read_text(encoding="utf-8"))
        if protocol.get("config_sha256") != _sha(config_path):
            raise SystemExit("BLOCKED: frozen protocol does not match this config")
        if protocol.get("test_opened"):
            raise SystemExit("BLOCKED: test split was already opened")
    bundle_path = root / "data" / "generated" / "l2_bundle.npz"
    data = np.load(bundle_path, allow_pickle=True)
    ids = [str(x) for x in data["ids"].tolist()]
    bundle = {
        "HF": data["HF"],
        "LF": data["LF"],
        "ids": ids,
        "labels": status["labels"],
    }
    # Paid cost is the measured generation wall time from prepare, scaled by
    # the fraction of trajectories actually stored. Both fidelities were
    # generated for every trajectory so the paid cost is HF+LF (+MF recorded
    # separately and not double-counted in the strategy budget).
    n = int(status["n"])
    hf_time = float(status["records"]["HF"]["wall_seconds"])
    lf_time = float(status["records"]["LF"]["wall_seconds"])
    cfg = dict(cfg)
    cfg["cost_per_hf"] = hf_time / n
    cfg["cost_per_lf"] = lf_time / n
    # Equal-budget target: the cost of one HF trajectory for every training
    # trajectory. Mixed strategies must buy LF by spending less on HF.
    n_train = sum(1 for label in status["labels"].values() if label == "train")
    base_budget = n_train * cfg["cost_per_hf"]
    fractions = [float(v) for v in cfg.get("budget_fractions") or [1.0]]
    out_dir = root / "outputs" / "l2" / cfg["stage"]
    out_dir.mkdir(parents=True, exist_ok=True)
    combined = []
    for fraction in fractions:
        cfg_run = dict(cfg)
        cfg_run["budget_seconds"] = base_budget * fraction
        cfg_run["budget_fraction"] = fraction
        sub = out_dir / f"fraction_{fraction}"
        sub.mkdir(parents=True, exist_ok=True)
        result = run_method_matrix(cfg_run, bundle, sub, open_test=bool(cfg.get("open_test")))
        result["prepare_nu"] = status["nu"]
        result["dts"] = status["dts"]
        result["budget_seconds_measured"] = cfg_run["budget_seconds"]
        result["budget_fraction"] = fraction
        write_json(sub / "results.json", result)
        combined.append({"budget_fraction": fraction, "results": result["results"], "n_train": result["n_train"], "n_val": result["n_val"], "n_test": result["n_test"]})
    write_json(out_dir / "results.json", {"fractions": combined, "open_test": bool(cfg.get("open_test"))})
    if cfg.get("open_test"):
        protocol = json.loads((root / "outputs" / "l2" / "FROZEN_PROTOCOL.json").read_text(encoding="utf-8"))
        protocol["test_opened"] = True
        protocol["test_read_count"] = 1
        write_json(root / "outputs" / "l2" / "FROZEN_PROTOCOL.json", protocol)
    print(json.dumps({"stage": cfg["stage"], "n_rows": len(result["results"]), "open_test": result["open_test"]}))


if __name__ == "__main__":
    main()
