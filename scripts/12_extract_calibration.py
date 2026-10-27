"""Save calibration trajectories 0-3 and coordinates. No other trajectories."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from _bootstrap import repo


def main() -> None:
    root = repo()
    path = root / "data" / "raw" / "1D_Burgers_Sols_Nu0.01.hdf5"
    out = root / "outputs" / "solver_calibration_input.npz"
    with h5py.File(path, "r") as handle:
        tensor = handle["tensor"]
        u = np.array(tensor[:4])
        x = np.array(handle["x-coordinate"])
        t = np.array(handle["t-coordinate"])
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, u=u, x=x, t=t)
    print(f"wrote {out} u={u.shape} x={x.shape} t={t.shape}")


if __name__ == "__main__":
    main()
