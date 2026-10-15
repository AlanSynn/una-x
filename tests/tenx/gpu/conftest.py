"""Fixtures for the G10 GPU fold island (campaign una_10x).

Same self-contained style as the other tenx islands.  The numba cache
is forced to the single campaign owner directory BEFORE any module
import triggers compilation, exactly as the native island does.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
for p in (str(REPO / "src"),):
    if p not in sys.path:
        sys.path.insert(0, p)

os.environ.setdefault(
    "NUMBA_CACHE_DIR", str(REPO / ".numba_cache_owner"))
os.makedirs(os.environ["NUMBA_CACHE_DIR"], exist_ok=True)


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "tenx_gpu: G10 GPU fold island (una_10x)")
