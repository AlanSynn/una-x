"""Platform geometry / compat-facade parity suite (TOPOLOGY).

Both parity arms run under the SAME interpreter — the dependency-bridged
reference venv (venv_madina_legacy) — with identically seeded RNGs, so
digest differences are attributable to the code under test only.
Digests are bitwise-strict (IEEE-754 bit patterns, WKB hashes, dtypes,
categories, orderings).
"""
from __future__ import annotations

import functools
import json
import os
import subprocess
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

MADINA_PYTHON = REPO / ".refs" / "venv_madina_legacy" / "bin" / "python"
MADINA_REF_SRC = REPO / ".refs" / "madina_ref" / "src"
SCENARIO_SCRIPT = HERE / "_scenario.py"

SCENARIOS = [
    "grid_basic",
    "redundant_keep",
    "redundant_discard",
    "redundant_split",
    "tolerance_snap",
    "custom_weight",
    "prep_geometry",
    "layer_order",
    "clear_reinsert",
    "coincident_od",
    "gdf_input",
    "crs_variant",
    "repeat_network",
    "multilinestring_pure",
]

ERROR_SCENARIOS = [
    "load_layer_bad_name",
    "load_layer_bad_source",
    "csn_unknown_layer",
    "csn_bad_weight_attr",
    "csn_negative_tolerance",
    "csn_bad_treatment",
    "turn_threshold_range",
    "turn_penalty_negative",
    "insert_bad_label",
    "insert_unknown_layer",
    "create_graph_nonbool",
    "layers_bad_key",
    "prep_mixed_z",
    "csn_multilinestring",
]


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "platform_geometry: bitwise parity of the compat.madina facade "
        "against the bridged pinned reference")


@pytest.fixture(scope="session")
def madina_python():
    if not MADINA_PYTHON.is_file():
        pytest.skip("madina_legacy reference venv absent "
                    "(.refs/venv_madina_legacy)")
    if not MADINA_REF_SRC.is_dir():
        pytest.skip("bridged pinned reference tree absent (.refs/madina_ref)")
    return str(MADINA_PYTHON)


def run_arm(madina_python, scenario, arm, *, sabotage=None,
            timeout=300) -> dict:
    out_dir = Path(str(ARTIFACTS_OUT)) / "platform_geometry" / arm
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{scenario}{'__' + sabotage if sabotage else ''}.json"
    if out.exists():
        out.unlink()
    cmd = [madina_python, str(SCENARIO_SCRIPT),
           "--arm", arm, "--scenario", scenario,
           "--out", str(out), "--seed", "20260930"]
    if sabotage:
        cmd += ["--sabotage", sabotage]
    env = dict(os.environ, PYTHONHASHSEED="0",
               NUMBA_CACHE_DIR=str(out_dir / "_nbc"))
    proc = subprocess.run(cmd, capture_output=True, text=True,
                          timeout=timeout, env=env, cwd=str(REPO))
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


@pytest.fixture(scope="session")
def arm_runner(madina_python):
    return functools.partial(run_arm, madina_python)
