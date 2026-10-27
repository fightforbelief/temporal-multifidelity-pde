"""Verify the DaRUS Burgers file, rename it, and inspect it without loading all frames."""

from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np

from _bootstrap import repo

EXPECTED_BYTES = 8232968312
EXPECTED_MD5 = "e6d9a4f62baf9a29121a816b919e2770"


def _md5(path: Path) -> tuple[str, float]:
    digest = hashlib.md5()
    started = time.perf_counter()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest(), time.perf_counter() - started


def _jsonable(value):
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.ndarray):
        if value.size > 16:
            return {"shape": list(value.shape), "dtype": str(value.dtype), "omitted": True}
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def main() -> None:
    root = repo()
    raw_dir = root / "data" / "raw"
    partial = raw_dir / "1D_Burgers_Sols_Nu0.01.hdf5.partial"
    final = raw_dir / "1D_Burgers_Sols_Nu0.01.hdf5"
    source = partial if partial.exists() else final
    if not source.exists():
        raise SystemExit("BLOCKED: Burgers file not found")
    size = source.stat().st_size
    md5, md5_seconds = _md5(source)
    stamp = datetime.fromtimestamp(source.stat().st_mtime, timezone.utc).isoformat()
    acceptance = {
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_path": str(source),
        "bytes": size,
        "expected_bytes": EXPECTED_BYTES,
        "md5": md5,
        "expected_md5": EXPECTED_MD5,
        "md5_seconds": md5_seconds,
        "mtime_utc": stamp,
        "size_ok": size == EXPECTED_BYTES,
        "md5_ok": md5 == EXPECTED_MD5,
    }
    if not (acceptance["size_ok"] and acceptance["md5_ok"]):
        acceptance["status"] = "BLOCKED"
        _write(root, acceptance)
        raise SystemExit(2)
    if source == partial:
        if final.exists():
            acceptance["status"] = "BLOCKED"
            acceptance["reason"] = "final name already exists; partial was not overwritten"
            _write(root, acceptance)
            raise SystemExit(2)
        partial.replace(final)
        acceptance["renamed_to"] = str(final)
    acceptance["path"] = str(final)
    schema = _inspect(final)
    acceptance["schema"] = schema
    acceptance["status"] = "PASS" if schema.get("readable") else "BLOCKED"
    _write(root, acceptance)
    print(json.dumps({"status": acceptance["status"], "bytes": size, "md5": md5, "keys": schema.get("keys")}))
    if acceptance["status"] != "PASS":
        raise SystemExit(3)


def _inspect(path: Path) -> dict:
    out = {"readable": False, "keys": [], "datasets": {}, "attrs": {}}
    with h5py.File(path, "r") as handle:
        out["readable"] = True
        out["keys"] = list(handle.keys())
        out["attrs"] = {k: _jsonable(v) for k, v in handle.attrs.items()}

        def visitor(name, obj):
            if isinstance(obj, h5py.Dataset):
                out["datasets"][name] = {
                    "shape": list(obj.shape),
                    "dtype": str(obj.dtype),
                    "chunks": None if obj.chunks is None else list(obj.chunks),
                    "attrs": {k: _jsonable(v) for k, v in obj.attrs.items()},
                }

        handle.visititems(visitor)
        data_name = _pick(out["datasets"], ("tensor", "data", "u", "solution"))
        x_name = _pick(out["datasets"], ("x-coordinate", "x", "xc", "x_coordinate"))
        t_name = _pick(out["datasets"], ("t-coordinate", "t", "tc", "t_coordinate"))
        out["chosen"] = {"data": data_name, "x": x_name, "t": t_name}
        if data_name is None:
            out["error"] = "no trajectory dataset among tensor/data/u/solution"
            return out
        data = handle[data_name]
        if data.ndim != 3:
            out["error"] = f"trajectory dataset ndim {data.ndim}, shape {list(data.shape)}"
            return out
        n, n_t, n_x = (int(data.shape[0]), int(data.shape[1]), int(data.shape[2]))
        # If time is the short axis stored last, record that; do not transpose yet.
        out["shape_as_stored"] = [n, n_t, n_x]
        x = None if x_name is None else np.array(handle[x_name])
        t = None if t_name is None else np.array(handle[t_name])
        out["x"] = _coord_summary(x)
        out["t"] = _coord_summary(t)
        finite = _finite_sample(data)
        out["finite_sample"] = finite
        u0 = np.array(data[:, 0, :]) if n_t >= n_x else None
        if u0 is not None:
            out["t0_all_trajectories"] = {
                "shape": list(u0.shape),
                "finite": bool(np.isfinite(u0).all()),
                "min": float(np.min(u0)),
                "max": float(np.max(u0)),
            }
            # stored for the later duplicate audit; not a model target
            np.save(path.parents[1] / "generated" / "t0_all.npy", u0.astype(np.float32))
    return out


def _pick(datasets: dict, names: tuple[str, ...]) -> str | None:
    for name in names:
        if name in datasets:
            return name
    return None


def _coord_summary(values: np.ndarray | None) -> dict | None:
    if values is None:
        return None
    flat = np.asarray(values).astype(np.float64).ravel()
    diffs = np.diff(flat)
    return {
        "count": int(flat.size),
        "min": float(flat[0]) if flat.size else None,
        "max": float(flat[-1]) if flat.size else None,
        "monotonic": bool(np.all(diffs > 0)) if diffs.size else False,
        "median_spacing": float(np.median(diffs)) if diffs.size else None,
        "spacing_min": float(np.min(diffs)) if diffs.size else None,
        "spacing_max": float(np.max(diffs)) if diffs.size else None,
    }


def _finite_sample(data: h5py.Dataset) -> dict:
    n = int(data.shape[0])
    take = list(range(0, n, max(n // 20, 1)))[:20]
    take = sorted(set(take + [0, 1, 2, 3, n - 1]))
    bad = 0
    checked = 0
    mins = []
    maxs = []
    for index in take:
        slab = np.array(data[index])
        checked += int(slab.size)
        bad += int(np.size(slab) - np.isfinite(slab).sum())
        mins.append(float(np.nanmin(slab)))
        maxs.append(float(np.nanmax(slab)))
    return {
        "trajectory_indices": take,
        "values_checked": checked,
        "nonfinite": bad,
        "sample_min": min(mins),
        "sample_max": max(maxs),
    }


def _write(root: Path, payload: dict) -> None:
    (root / "outputs").mkdir(exist_ok=True)
    (root / "reports").mkdir(exist_ok=True)
    (root / "data" / "generated").mkdir(parents=True, exist_ok=True)
    (root / "outputs" / "data_acceptance.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    lines = [
        "# 13 — 数据验收",
        "",
        f"- 状态：`{payload['status']}`",
        f"- 检查时间（UTC）：{payload['checked_at_utc']}",
        f"- 路径：`{payload.get('path', payload['source_path'])}`",
        f"- 字节：{payload['bytes']}（期望 {payload['expected_bytes']}）",
        f"- MD5：`{payload['md5']}`（期望 `{payload['expected_md5']}`）",
        f"- MD5 耗时：{payload['md5_seconds']:.2f} 秒",
        f"- 文件时间戳（UTC）：{payload['mtime_utc']}",
        "",
    ]
    schema = payload.get("schema")
    if schema:
        lines.append("## HDF5")
        lines.append("")
        lines.append(f"- 可读：{schema.get('readable')}")
        lines.append(f"- 根键：`{schema.get('keys')}`")
        lines.append(f"- 根属性：`{schema.get('attrs')}`")
        lines.append(f"- 选用数据集：`{schema.get('chosen')}`")
        lines.append(f"- 存储形状：`{schema.get('shape_as_stored')}`")
        if schema.get("error"):
            lines.append(f"- 结构问题：{schema['error']}")
        lines.append("")
        lines.append("数据集：")
        lines.append("")
        for name, meta in schema.get("datasets", {}).items():
            lines.append(f"- `{name}` shape={meta['shape']} dtype={meta['dtype']} chunks={meta['chunks']} attrs={meta['attrs']}")
        lines.append("")
        lines.append(f"- 空间坐标：`{schema.get('x')}`")
        lines.append(f"- 时间坐标：`{schema.get('t')}`")
        lines.append(f"- 抽样有限性：`{schema.get('finite_sample')}`")
        lines.append(f"- 全部 t=0：`{schema.get('t0_all_trajectories')}`")
        lines.append("")
        lines.append("这一步没有计算测试轨迹的预测误差，也没有用 t>0 的场选择粘性或模型。")
    (root / "reports" / "13_DATA_ACCEPTANCE.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
