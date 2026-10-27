"""Write the confirmatory protocol before any test metric is computed."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from _bootstrap import repo


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    root = repo()
    config = root / "configs" / "l2_native_confirmatory.yaml"
    status = json.loads((root / "outputs" / "l2" / "prepare_status.json").read_text(encoding="utf-8"))
    protocol = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_path": "configs/l2_native_confirmatory.yaml",
        "config_sha256": _sha(config),
        "prepare_sha256": _sha(root / "outputs" / "l2" / "prepare_status.json"),
        "bundle_sha256": _sha(root / "data" / "generated" / "l2_bundle.npz"),
        "test_opened": False,
        "test_read_count": 0,
        "nu": status["nu"],
        "dts": status["dts"],
        "labels": status["labels"],
        "horizons": [10, 40, 100, 200],
        "primary_horizon": 200,
        "divergence_rel": 5.0,
        "budget_fractions": [0.25, 0.5, 1.0],
        "seeds": [0, 1, 2],
        "strategies": ["HF_ONLY", "MIXED_CONDITIONED", "LF_PRETRAIN_HF_FINETUNE", "POD_DELTA_RIDGE", "IDENTITY", "LF_SOLVER"],
        "code_version": "NO_COMMIT",
    }
    dest = root / "outputs" / "l2" / "FROZEN_PROTOCOL.json"
    if dest.exists():
        old = json.loads(dest.read_text(encoding="utf-8"))
        if old.get("test_opened"):
            raise SystemExit("BLOCKED: protocol already opened the test split")
    dest.write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    print(json.dumps({"frozen": True, "config_sha256": protocol["config_sha256"]}))


if __name__ == "__main__":
    main()
