"""SCIENCE suite: bounded bug-registry confirmation probes and the
corrected_v1 reference tests.

Every probe runs in a supervised subprocess with a hard time limit and
an address-space cap (NUMERICS.md: keep dangerous legacy probes in a
supervised subprocess).  Capsules land under ARTIFACTS_OUT/science
(never committed evidence directly); the task evidence dir retains the
curated copies.
"""
from __future__ import annotations

import json
import os
import resource
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
LARGE_E2E = REPO / "tests" / "large_e2e"
for _p in (str(REPO), str(HERE), str(LARGE_E2E), str(REPO / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from _legacy_paths import ARTIFACTS_OUT  # portable-path fix (HARNESS 2026-09-30)

MADINA_PYTHON = REPO / ".refs" / "venv_madina_legacy" / "bin" / "python"
PROBE_SCRIPT = HERE / "_probes.py"

BASELINE_PROBES = [
    "frac_weight_trunc",
    "same_edge_od",
    "coincident_seeds",
    "nonfinite_validation",
]
MADINA_PROBES = [
    "prep_zero_coord",
]

# per-probe wall-time bounds (s): the negative/zero-cost probes involve
# compiled first-runs; generous but hard.
TIMEOUTS = {
    "frac_weight_trunc": 240,
    "same_edge_od": 240,
    "coincident_seeds": 240,
    "nonfinite_validation": 300,
    "prep_zero_coord": 240,
}

MEMCAP_BYTES = 2 * 1024 ** 3     # 2 GiB address space per probe child


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "science: bounded bug-registry probes and corrected_v1 reference tests")


def _memcap():
    def limit():
        resource.setrlimit(resource.RLIMIT_AS, (MEMCAP_BYTES, MEMCAP_BYTES))
    return limit


def run_probe(probe: str, interpreter: str) -> dict:
    out_dir = Path(str(ARTIFACTS_OUT)) / "science" / "probes"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{probe}.json"
    if out.exists():
        out.unlink()
    env = dict(os.environ, NUMBA_DISABLE_JIT="0",
               NUMBA_CACHE_DIR=str(out_dir / "_nbc"))
    cmd = [interpreter, str(PROBE_SCRIPT), "--probe", probe, "--out", str(out)]
    proc = subprocess.run(
        cmd, capture_output=True, text=True,
        timeout=TIMEOUTS[probe], env=env, cwd=str(REPO),
        preexec_fn=_memcap())
    if not out.exists():
        return {"probe": probe, "ok": False,
                "error": f"no capsule (rc={proc.returncode})",
                "stderr": proc.stderr[-2000:]}
    return json.loads(out.read_text())


@pytest.fixture(scope="session")
def baseline_runner():
    return sys.executable


@pytest.fixture(scope="session")
def madina_runner():
    if not MADINA_PYTHON.is_file():
        pytest.skip("madina_legacy reference venv absent")
    return str(MADINA_PYTHON)
