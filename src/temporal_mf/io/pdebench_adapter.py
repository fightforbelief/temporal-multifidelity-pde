"""Read-only PDEBench HDF5 adapter. Does not assume unread keys."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np


def _find_dataset(handle: h5py.File, candidates: tuple[str, ...]) -> h5py.Dataset | None:
    for key in candidates:
        if key in handle and isinstance(handle[key], h5py.Dataset):
            return handle[key]
    return None


def inspect_hdf5(path: Path) -> dict:
    path = Path(path)
    info = {"path": str(path), "bytes": path.stat().st_size, "keys": [], "datasets": {}}
    with h5py.File(path, "r") as handle:
        info["keys"] = list(handle.keys())
        info["attrs"] = {k: _jsonable(v) for k, v in handle.attrs.items()}

        def visitor(name, obj):
            if isinstance(obj, h5py.Dataset):
                info["datasets"][name] = {
                    "shape": list(obj.shape),
                    "dtype": str(obj.dtype),
                }

        handle.visititems(visitor)
    return info


def _jsonable(value):
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.ndarray):
        if value.size > 8:
            return {"shape": list(value.shape), "dtype": str(value.dtype)}
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def load_trajectories(path: Path, indices: list[int]) -> dict:
    """Load a subset. Returns u (n, time, space), x, t, and the key that was used."""
    with h5py.File(path, "r") as handle:
        data = _find_dataset(handle, ("tensor", "data", "u", "solution"))
        if data is None:
            raise KeyError(f"no trajectory dataset in keys {list(handle.keys())}")
        x = _find_dataset(handle, ("x-coordinate", "x", "xc", "x_coordinate"))
        t = _find_dataset(handle, ("t-coordinate", "t", "tc", "t_coordinate"))
        take = np.array(sorted(indices), dtype=np.int64)
        if take.min() < 0 or take.max() >= data.shape[0]:
            raise IndexError("trajectory index out of range")
        u = np.array(data[take])
        if u.ndim == 4 and u.shape[-1] == 1:
            u = u[..., 0]
        if u.ndim != 3:
            raise ValueError(f"expected (traj, time, space), got {u.shape}")
        x_arr = None if x is None else np.array(x)
        t_arr = None if t is None else np.array(t)
        return {
            "u": u,
            "x": x_arr,
            "t": t_arr,
            "source_indices": take,
            "dataset_key": data.name,
        }
