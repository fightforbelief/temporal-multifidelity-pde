"""Remote host, path, and dependency checks. No secrets."""

from __future__ import annotations

import json
import platform
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def repo_root_from_here() -> Path:
    return Path(__file__).resolve().parents[2]


def assert_windows_repo(repo: Path) -> None:
    if platform.system() != "Windows":
        raise SystemExit(
            "REMOTE_BLOCKED: experiment entry points must run on Windows "
            f"(got {platform.system()})."
        )
    resolved = repo.resolve()
    parts = [p.lower() for p in resolved.parts]
    if "p1-temporal-multifidelity-pde" not in parts:
        raise SystemExit(f"REFUSING write outside dedicated repo: {resolved}")


def package_versions() -> dict:
    out = {
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "platform": platform.platform(),
        "hostname": socket.gethostname(),
    }
    for name in ("numpy", "scipy", "h5py", "torch", "yaml", "matplotlib", "pandas", "sklearn"):
        mod = "yaml" if name == "yaml" else name
        try:
            module = __import__(mod)
            out[name] = getattr(module, "__version__", "unknown")
        except Exception as exc:  # noqa: BLE001
            out[name] = f"MISSING:{type(exc).__name__}"
    try:
        import torch

        out["torch_cuda"] = torch.version.cuda
        out["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            out["gpu_name"] = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            out["gpu_total_memory_bytes"] = int(props.total_memory)
            x = torch.randn(8, 8, device="cuda", requires_grad=True)
            y = (x @ x).sum()
            y.backward()
            out["gpu_backward_ok"] = True
            out["gpu_backward_grad_mean"] = float(x.grad.detach().abs().mean().cpu())
        else:
            out["gpu_backward_ok"] = False
    except Exception as exc:  # noqa: BLE001
        out["cuda_error"] = f"{type(exc).__name__}: {exc}"
    return out


def disk_free_bytes(path: Path) -> int | None:
    try:
        import shutil

        return int(shutil.disk_usage(path).free)
    except Exception:
        return None


def git_sha(path: Path) -> str:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=path,
            check=False,
            capture_output=True,
            text=True,
        )
        if proc.returncode == 0:
            return proc.stdout.strip()
    except Exception:
        pass
    return "NO_COMMIT"


def write_environment_report(repo: Path, output: Path) -> dict:
    info = package_versions()
    info["utc"] = utc_now()
    info["repo"] = str(repo.resolve())
    info["disk_free_bytes"] = disk_free_bytes(repo)
    info["repo_sha"] = git_sha(repo)
    upstream = {}
    for name in ("setinek", "pdebench"):
        ext = repo / "external" / name
        upstream[name] = git_sha(ext) if ext.exists() else "ABSENT"
    info["upstream_sha"] = upstream
    output.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Remote environment",
        "",
        f"- Recorded UTC: {info['utc']}",
        f"- Hostname: `{info['hostname']}`",
        f"- Platform: `{info['platform']}`",
        f"- Python: `{info['executable']}` ({info['python']})",
        f"- Repo: `{info['repo']}`",
        f"- Repo git SHA: `{info['repo_sha']}`",
        f"- Free disk bytes: {info['disk_free_bytes']}",
        f"- numpy {info.get('numpy')} scipy {info.get('scipy')} h5py {info.get('h5py')}",
        f"- torch {info.get('torch')} cuda {info.get('torch_cuda')} available {info.get('cuda_available')}",
        f"- GPU: {info.get('gpu_name', 'none')} backward_ok={info.get('gpu_backward_ok')}",
        f"- Upstream SHAs: `{json.dumps(upstream)}`",
        "",
        "System Python was not modified. This interpreter must be the project `.venv`.",
        "",
    ]
    output.write_text("\n".join(lines), encoding="utf-8")
    (output.with_suffix(".json")).write_text(json.dumps(info, indent=2), encoding="utf-8")
    return info
