"""Build a cost-matched fidelity bundle from PDEBench initial conditions.

Exits BLOCKED when the DaRUS file is absent or the checksum does not match.
Calibration trajectories are excluded from train/val/test.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
import yaml

from _bootstrap import repo
from temporal_mf.fidelity.interpolate import fourier_resample
from temporal_mf.io.pdebench_adapter import inspect_hdf5, load_trajectories
from temporal_mf.numerical.burgers import integrate
from temporal_mf.study import observation_times, prepare_split_table, write_json

EXPECTED_BYTES = 8232968312
EXPECTED_MD5 = "e6d9a4f62baf9a29121a816b919e2770"


def _md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_dt(speed: float, dx: float, dt_obs: float) -> dict:
    cap = 0.8 * dx / max(speed, 1e-6)
    hf = 0.001
    while hf * 4 > cap and hf > 1e-4:
        hf /= 2
    # snap so hf divides dt_obs
    steps = max(int(round(dt_obs / hf)), 1)
    hf = dt_obs / steps
    lf_try = hf * 4
    if lf_try <= cap + 1e-12:
        lf = lf_try
        note = "LF is 4x HF and inside the advective CFL cap"
        admissible_4x = True
    else:
        lf = hf * 2
        note = "4x HF step exceeded the advective CFL cap on the sampled initial data; LF accepted at 2x and 4x is inadmissible"
        admissible_4x = False
    return {"HF": hf, "MF": hf * 2, "LF": lf, "note": note, "admissible_4x": admissible_4x, "cfl_cap_dt": cap}


def main() -> None:
    root = repo()
    cfg = yaml.safe_load((root / "configs" / "l2_smoke.yaml").read_text(encoding="utf-8"))
    raw = root / "data" / "raw" / "1D_Burgers_Sols_Nu0.01.hdf5"
    if not raw.exists() or raw.stat().st_size != EXPECTED_BYTES:
        payload = {
            "status": "BLOCKED",
            "reason": "PDEBench file missing or incomplete",
            "path": str(raw),
            "bytes": raw.stat().st_size if raw.exists() else 0,
            "expected_bytes": EXPECTED_BYTES,
        }
        write_json(root / "outputs" / "l2" / "prepare_status.json", payload)
        print(json.dumps(payload))
        raise SystemExit(2)
    md5 = _md5(raw)
    if md5 != EXPECTED_MD5:
        payload = {"status": "BLOCKED", "reason": "md5 mismatch", "md5": md5, "expected": EXPECTED_MD5}
        write_json(root / "outputs" / "l2" / "prepare_status.json", payload)
        print(json.dumps(payload))
        raise SystemExit(2)
    schema = inspect_hdf5(raw)
    (root / "reports").mkdir(exist_ok=True)
    (root / "reports" / "HDF5_SCHEMA.json").write_text(json.dumps(schema, indent=2), encoding="utf-8")
    n = int(cfg["n_trajectories"])
    calib = [0, 1, 2, 3]
    rng = np.random.default_rng(np.random.SeedSequence(20261023))
    pool = np.arange(4, 10000)
    chosen = rng.choice(pool, size=n, replace=False).tolist()
    loaded = load_trajectories(raw, calib + chosen)
    u = loaded["u"]
    x = loaded["x"]
    t = loaded["t"]
    nx = int(cfg["nx"])
    length = float(cfg["length"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    frames = torch.tensor(u, dtype=torch.float32, device=device)
    resampled = fourier_resample(frames, nx).detach().cpu().numpy()
    eps = float(cfg["epsilon"])
    nu_candidates = {"epsilon_over_pi": eps / math.pi, "epsilon": eps}
    t_obs = observation_times(float(cfg["T"]), float(cfg["dt_obs"]), dtype=torch.float32, device=device)
    # Map archive times onto our observation grid by nearest index.
    if t is None:
        raise SystemExit("BLOCKED: archive has no time coordinate")
    archive_index = [int(np.argmin(np.abs(t - float(tv)))) for tv in t_obs.cpu().numpy()]
    ref = resampled[:4, archive_index]
    ic = torch.tensor(resampled[:4, 0], dtype=torch.float32, device=device)
    nu_errors = {}
    for name, nu in nu_candidates.items():
        field, rec = integrate(ic, nu=nu, length=length, dt=0.001, t_observe=t_obs)
        pred = field.detach().cpu().numpy()
        num = np.linalg.norm(pred[:, -1] - ref[:, -1], axis=1)
        den = np.maximum(np.linalg.norm(ref[:, -1], axis=1), 1e-8)
        nu_errors[name] = {"mean_rel": float(np.mean(num / den)), "status": rec.status, "nu": nu}
    chosen_name = min(nu_errors, key=lambda k: nu_errors[k]["mean_rel"])
    nu = nu_errors[chosen_name]["nu"]
    speed = float(np.max(np.abs(resampled[4:, 0])))
    dx = length / nx
    dts = _stable_dt(speed, dx, float(cfg["dt_obs"]))
    study_ic = torch.tensor(resampled[4:, 0], dtype=torch.float32, device=device)
    fields = {}
    records = {}
    for level, dt in (("HF", dts["HF"]), ("MF", dts["MF"]), ("LF", dts["LF"])):
        field, rec = integrate(study_ic, nu=nu, length=length, dt=dt, t_observe=t_obs)
        if rec.status != "OK":
            payload = {"status": "BLOCKED", "reason": f"{level} solver status {rec.status}", "dt": dt}
            write_json(root / "outputs" / "l2" / "prepare_status.json", payload)
            print(json.dumps(payload))
            raise SystemExit(3)
        fields[level] = field.detach().cpu().numpy()
        records[level] = {
            "status": rec.status,
            "dt": rec.dt,
            "n_steps": rec.n_steps,
            "max_cfl": rec.max_cfl,
            "mass_drift": rec.mass_drift,
            "wall_seconds": rec.wall_seconds,
        }
    ids = [f"pdebench-nu0.01-{int(i)}" for i in chosen]
    split = prepare_split_table(ids, resampled[4:, 0], seed=int(cfg["split_seed"]))
    if split["near_duplicate_violations"]:
        payload = {"status": "BLOCKED", "reason": "near-duplicate ICs cross splits", "pairs": split["near_duplicate_violations"]}
        write_json(root / "outputs" / "l2" / "prepare_status.json", payload)
        raise SystemExit(4)
    out = root / "data" / "generated" / "l2_bundle.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        HF=fields["HF"],
        MF=fields["MF"],
        LF=fields["LF"],
        reference=resampled[4:][:, archive_index],
        ids=np.array(ids),
        x=np.linspace(0.5 * dx, length - 0.5 * dx, nx),
        t=t_obs.cpu().numpy(),
    )
    meta = {
        "status": "PASS",
        "lineage": "PDEBENCH_IC",
        "source_file": raw.name,
        "source_md5": md5,
        "source_bytes": EXPECTED_BYTES,
        "doi": "10.18419/darus-2986",
        "version": "8.0",
        "nu_selection": chosen_name,
        "nu": nu,
        "nu_errors_on_calibration_only": nu_errors,
        "calibration_indices_excluded": calib,
        "dts": dts,
        "records": records,
        "n": n,
        "labels": split["labels"],
        "archive_time_index": archive_index,
        "schema_keys": schema["keys"],
        "x_minmax": None if x is None else [float(np.min(x)), float(np.max(x))],
        "t_minmax": [float(np.min(t)), float(np.max(t))],
        "solver": "independent spectral IMEX-Euler, not the PDEBench finite-volume code",
    }
    write_json(root / "outputs" / "l2" / "prepare_status.json", meta)
    print(json.dumps({"status": "PASS", "nu": nu, "nu_name": chosen_name, "dts": {k: dts[k] for k in ("HF", "MF", "LF")}, "n": n}))


if __name__ == "__main__":
    main()
