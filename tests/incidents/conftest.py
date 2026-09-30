"""Incident reproduction suite (FAILURES task, dossier 03).

Every attempt runs through incident_runner.run_attempt: a spawned child
process with an independent parent-side watchdog, per-attempt heartbeat,
faulthandler stack dump, bounded grace before SIGTERM of OWNED pids only,
reap, process-tree simultaneous-RSS sampling (reusing the HARNESS sampler)
and an attempt envelope (versions, start method, dataset hash/CRS, census,
parameters, stage entered, exit status).

Attempts write only to ARTIFACTS_OUT/incidents (never committed evidence);
the FAILURES receipt curates retained copies.  The pinned Madina tree is
read-only: causal interventions are applied to in-memory module copies,
never to the tree on disk.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
LARGE_E2E = REPO / "tests" / "large_e2e"
for _p in (str(REPO), str(HERE), str(LARGE_E2E)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from _legacy_paths import ARTIFACTS_OUT  # portable-path fix (HARNESS 2026-09-30)

MADINA_SRC = REPO / ".refs" / "madina" / "src"
MADINA_PYTHON = REPO / ".refs" / "venv_madina_legacy" / "bin" / "python"
ATTEMPTS_ROOT = Path(str(ARTIFACTS_OUT)) / "incidents"


def madina_bridge_available() -> bool:
    if not MADINA_SRC.is_dir():
        return False
    try:
        import subprocess
        probe = (
            "import sys; sys.path.insert(0, r'" + str(MADINA_SRC) + "');\n"
            "from madina.zonal.zonal import Zonal\n"
        )
        env = dict(os.environ, NUMBA_CACHE_DIR=str(ATTEMPTS_ROOT / "nbc_probe"))
        r = subprocess.run([sys.executable, "-c", probe], capture_output=True,
                           timeout=120, env=env)
        return r.returncode == 0
    except Exception:
        return False


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "incidents: bounded reproduction attempts against pinned Madina "
        "reports (spawned children, per-attempt watchdog)")


@pytest.fixture(scope="session")
def attempts_root():
    d = ATTEMPTS_ROOT / "run"
    d.mkdir(parents=True, exist_ok=True)
    return d


@pytest.fixture(scope="session")
def madina_python():
    """Interpreter of the dependency-bridged madina_legacy profile: the
    pinned Madina imports pydeck at package level, so attempts run under
    the bridged reference venv, not the harness environment."""
    if not MADINA_PYTHON.is_file():
        pytest.skip("madina_legacy reference venv absent "
                    "(.refs/venv_madina_legacy)")
    return str(MADINA_PYTHON)


@pytest.fixture(scope="session")
def madina_src():
    if not MADINA_SRC.is_dir():
        pytest.skip("pinned madina clone absent (.refs/madina)")
    return str(MADINA_SRC)
