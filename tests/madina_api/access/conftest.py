"""MADINA_ACCESS compat-suite fixtures.

Both parity arms run under the SAME interpreter — the dependency-bridged
reference venv (venv_madina_legacy), the only environment where the
pinned upstream ``madina`` is importable — so digest differences are
attributable to the code under test only.  Digests are bitwise-strict
(IEEE-754 bit patterns, WKB hashes, dtypes, orderings).
"""
from __future__ import annotations

import functools
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
LARGE_E2E = REPO / "tests" / "large_e2e"
for _p in (str(REPO), str(HERE), str(LARGE_E2E), str(REPO / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from _legacy_paths import ARTIFACTS_OUT  # portable-path fix (HARNESS 2026-09-30)

MADINA_PYTHON = REPO / ".refs" / "venv_madina_legacy" / "bin" / "python"
MADINA_REF_SRC = REPO / ".refs" / "madina_ref" / "src"
SCENARIO_SCRIPT = HERE / "_access_scenario.py"
REFLECT_SCRIPT = HERE / "_access_reflection.py"

# scenario names live in _access_scenario.SCENARIO_NAMES (unique module
# name; `from conftest import ...` collides across non-package test dirs)


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "madina_api: facade-vs-pinned-reference parity of the "
        "compat.madina namespace")


@pytest.fixture(scope="session")
def madina_python():
    if not MADINA_PYTHON.is_file():
        pytest.skip("madina_legacy reference venv absent "
                    "(.refs/venv_madina_legacy)")
    if not MADINA_REF_SRC.is_dir():
        pytest.skip("bridged pinned reference tree absent (.refs/madina_ref)")
    return str(MADINA_PYTHON)


def run_arm(madina_python, scenario, arm, *, sabotage=None, timeout=900) -> dict:
    # resolve(): the arm subprocess runs with cwd=out_dir, so a RELATIVE
    # ARTIFACTS_OUT would resolve against that cwd and nest the digest
    # under a spurious out_dir/campaigns/... tree (observed 2026-10-01
    # in the paths suite); absolute paths masked this until the
    # retained evidence run.
    out_dir = (Path(str(ARTIFACTS_OUT)) / "madina_access" / arm).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{scenario}{'__' + sabotage if sabotage else ''}.json"
    if out.exists():
        out.unlink()
    cmd = [madina_python, str(SCENARIO_SCRIPT),
           "--arm", arm, "--scenario", scenario,
           "--out", str(out), "--seed", "20260930"]
    if sabotage:
        cmd += ["--sabotage", sabotage]
    env = dict(os.environ, PYTHONHASHSEED="0")
    proc = subprocess.run(cmd, capture_output=True, text=True,
                          timeout=timeout, env=env, cwd=str(out_dir))
    if not out.exists():
        raise AssertionError(
            f"{arm}/{scenario} produced no digest "
            f"(rc={proc.returncode})\nstdout:\n{proc.stdout[-2000:]}\n"
            f"stderr:\n{proc.stderr[-2000:]}")
    digest = json.loads(out.read_text())
    if "scenario_error" in digest:
        raise AssertionError(
            f"{arm}/{scenario} failed inside the scenario:\n"
            f"{digest['scenario_error'][-2000:]}")
    return digest


def run_reflection(madina_python, arm) -> dict:
    out_dir = (Path(str(ARTIFACTS_OUT)) / "madina_access" / "reflection").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{arm}.json"
    if out.exists():
        out.unlink()
    cmd = [madina_python, str(REFLECT_SCRIPT),
           "--arm", arm, "--out", str(out)]
    env = dict(os.environ, PYTHONHASHSEED="0")
    proc = subprocess.run(cmd, capture_output=True, text=True,
                          timeout=300, env=env, cwd=str(REPO))
    if not out.exists():
        raise AssertionError(
            f"reflection/{arm} produced no digest (rc={proc.returncode})\n"
            f"stdout:\n{proc.stdout[-2000:]}\nstderr:\n{proc.stderr[-2000:]}")
    digest = json.loads(out.read_text())
    if "probe_error" in digest:
        raise AssertionError(
            f"reflection/{arm} failed:\n{digest['probe_error'][-2000:]}")
    return digest


@pytest.fixture(scope="session")
def arm_runner(madina_python):
    return functools.partial(run_arm, madina_python)


@pytest.fixture(scope="session")
def reflection_runner(madina_python):
    return functools.partial(run_reflection, madina_python)
