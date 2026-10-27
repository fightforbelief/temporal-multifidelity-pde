"""Shared script bootstrap. Experiments refuse to run off Windows."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from temporal_mf.env_audit import assert_windows_repo  # noqa: E402


def repo() -> Path:
    assert_windows_repo(ROOT)
    return ROOT
