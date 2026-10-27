"""Split the generated fields by trajectory id. Does not read held-out futures for tuning."""

from __future__ import annotations

import json

import numpy as np

from _bootstrap import repo
from temporal_mf.study import prepare_split_table, write_json


def main() -> None:
    root = repo()
    selected = json.loads((root / "outputs" / "l2" / "selection.json").read_text(encoding="utf-8"))
    fields = np.load(root / "outputs" / "l2" / "generated_fields.npz")
    timing = json.loads((root / "outputs" / "l2" / "generation_timing.json").read_text(encoding="utf-8"))
    indices = selected["selected_indices"]
    ids = [f"pdebench-nu0.01-{int(i)}" for i in indices]
    ics = np.load(root / "outputs" / "l2" / "selected_ic.npz")["u0"]
    split = prepare_split_table(ids, ics, seed=20261023)
    if split["near_duplicate_violations"]:
        raise SystemExit("BLOCKED: near-duplicate ICs cross splits")
    out = root / "data" / "generated" / "l2_bundle.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    dx = 1.0 / 1024
    t = np.arange(201, dtype=np.float32) * 0.01
    np.savez_compressed(
        out,
        HF=fields["HF"],
        LF=fields["LF"],
        ids=np.array(ids),
        x=np.linspace(0.5 * dx, 1.0 - 0.5 * dx, 1024).astype(np.float32),
        t=t,
    )
    meta = {
        "status": "PASS",
        "lineage": "PDEBENCH_IC",
        "solver": "finite-volume MUSCL predictor-corrector, viscosity epsilon/pi, fixed interior step",
        "nu": 0.01 / float(np.pi),
        "nu_name": "epsilon_over_pi",
        "calibration_indices_excluded": [0, 1, 2, 3],
        "n": len(ids),
        "labels": split["labels"],
        "dts": {"HF": timing["HF"]["dt"], "LF": timing["LF"]["dt"], "rejected_4x": 0.01 / 67, "rejected_status": "UNSTABLE_CFL"},
        "records": timing,
        "source_file": "1D_Burgers_Sols_Nu0.01.hdf5",
        "source_md5": "e6d9a4f62baf9a29121a816b919e2770",
    }
    write_json(root / "outputs" / "l2" / "prepare_status.json", meta)
    print(json.dumps({"status": "PASS", "n": len(ids), "n_train": sum(v == "train" for v in split["labels"].values())}))


if __name__ == "__main__":
    main()
