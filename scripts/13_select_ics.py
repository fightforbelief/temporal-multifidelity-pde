"""Choose study trajectories from t=0 only. Futures of these rows are not read."""

from __future__ import annotations

import json

import h5py
import numpy as np

from _bootstrap import repo
from temporal_mf.data.splits import ic_hash

N = 48
SEED = 20261023


def main() -> None:
    root = repo()
    path = root / "data" / "raw" / "1D_Burgers_Sols_Nu0.01.hdf5"
    with h5py.File(path, "r") as handle:
        u0 = np.array(handle["tensor"][:, 0, :], dtype=np.float32)
    hashes = [ic_hash(u0[i]) for i in range(u0.shape[0])]
    seen = set(hashes[:4])
    rng = np.random.default_rng(np.random.SeedSequence(SEED))
    order = rng.permutation(np.arange(4, u0.shape[0]))
    selected = []
    skipped = []
    for idx in order.tolist():
        digest = hashes[idx]
        if digest in seen:
            skipped.append({"index": int(idx), "reason": "near_duplicate_ic_hash_5dp"})
            continue
        seen.add(digest)
        selected.append(int(idx))
        if len(selected) == N:
            break
    if len(selected) < N:
        raise SystemExit(f"BLOCKED: only {len(selected)} non-duplicate trajectories")
    out = root / "outputs" / "l2"
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "selected_ic.npz", u0=u0[selected], indices=np.array(selected, dtype=np.int64))
    meta = {
        "n": N,
        "seed": SEED,
        "excluded_calibration": [0, 1, 2, 3],
        "selected_indices": selected,
        "n_skipped_near_duplicate": len(skipped),
        "skipped_near_duplicate": skipped[:50],
        "t0_only": True,
    }
    (out / "selection.json").write_text(json.dumps(meta), encoding="utf-8")
    print(json.dumps({"n": N, "skipped": len(skipped), "first": selected[:5]}))


if __name__ == "__main__":
    main()
