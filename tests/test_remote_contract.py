import os
import platform
from pathlib import Path

from temporal_mf.env_audit import repo_root_from_here


def test_platform_is_windows_for_this_study():
    assert platform.system() == "Windows"


def test_interpreter_is_project_venv():
    exe = Path(os.sys.executable).resolve()
    assert ".venv" in exe.parts
    assert "p1-temporal-multifidelity-pde" in [p.lower() for p in exe.parts] or "p1-temporal-multifidelity-pde" in str(exe).lower()


def test_repo_root_resolves():
    root = repo_root_from_here()
    assert (root / "src" / "temporal_mf").is_dir()


def test_cuda_backward():
    import torch

    assert torch.cuda.is_available()
    x = torch.randn(4, 4, device="cuda", requires_grad=True)
    y = (x @ x).sum()
    y.backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
