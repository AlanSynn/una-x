"""L1: immutable baseline source versus candidate source, whole-engine.

Runs _arm_runner.py against the blob-verified baseline source tree and this
candidate tree, ONE CASE PER SUBPROCESS (a long-lived process accumulating
LLVM/Numba compile memory was SIGKILLed by macOS under memory pressure; the
per-case split bounds each process lifetime). For every case:
  1. baseline arm A, 2. baseline arm B -> assert bit-identical (determinism
gate first, per VALIDATION_POLICY), 3. candidate arm -> assert bit-identical
to baseline A.

Set UNA_BASELINE_SRC to the baseline source root (…/src). Without it this
module skips (the L0 suite still runs standalone).
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

TESTS_DIR = Path(__file__).resolve().parent
CANDIDATE_SRC = Path(
    os.environ.get("UNA_CANDIDATE_SRC",
                   str(TESTS_DIR.parents[1] / "src"))).resolve()
BASELINE_SRC = os.environ.get("UNA_BASELINE_SRC")

CASES = ["random_elev", "same_symmetric", "no_z_elev_on", "micro_parallel",
         "extreme_finite_weights", "int32_realdata"]

pytestmark = pytest.mark.skipif(
    BASELINE_SRC is None, reason="UNA_BASELINE_SRC not set; L1 needs the "
    "immutable baseline source tree")


def _load(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def _assert_identical(a: dict, b: dict, label: str):
    assert set(a) == set(b), f"{label}: key mismatch"
    for k in sorted(a):
        va, vb = a[k], b[k]
        assert va.dtype == vb.dtype, f"{label}: {k} dtype {va.dtype} != {vb.dtype}"
        assert va.shape == vb.shape, f"{label}: {k} shape {va.shape} != {vb.shape}"
        assert va.tobytes() == vb.tobytes(), f"{label}: {k} bytes differ"


def _run_arm(src_root: Path, out_path: Path, case: str) -> dict:
    reuse_dir = os.environ.get("L1_REUSE_DIR")
    if reuse_dir:
        cached = Path(reuse_dir) / out_path.name
        if cached.exists():
            print(f"[l1] reusing {cached}", flush=True)
            return _load(cached)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(src_root)
    env.setdefault("NUMBA_CACHE_DIR", str(TESTS_DIR / ".numba_cache"))
    proc = subprocess.run(
        [sys.executable, str(TESTS_DIR / "_arm_runner.py"),
         "--out", str(out_path), "--case", case],
        capture_output=True, text=True, env=env, timeout=1800,
    )
    assert proc.returncode == 0, (
        f"arm runner failed for src={src_root} case={case}\n"
        f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}")
    return _load(out_path)


@pytest.fixture(scope="module")
def all_arms(tmp_path_factory):
    """arms[case] = (baseline_a_dict, baseline_b_dict, candidate_dict)."""
    tmp = tmp_path_factory.mktemp("arms")
    arms = {}
    for case in CASES:
        a = _run_arm(Path(BASELINE_SRC), tmp / f"{case}.base_a.npz", case)
        b = _run_arm(Path(BASELINE_SRC), tmp / f"{case}.base_b.npz", case)
        _assert_identical(a, b, f"{case}: baseline-vs-baseline")
        c = _run_arm(CANDIDATE_SRC, tmp / f"{case}.cand.npz", case)
        _assert_identical(a, c, f"{case}: baseline-vs-candidate")
        arms[case] = (a, b, c)
        print(f"[l1] {case}: deterministic + bit-identical", flush=True)
    return arms


@pytest.mark.parametrize("case", CASES)
def test_baseline_is_deterministic(case, all_arms):
    _assert_identical(all_arms[case][0], all_arms[case][1],
                      f"{case}: baseline-vs-baseline")


@pytest.mark.parametrize("case", CASES)
def test_candidate_bit_identical_to_baseline(case, all_arms):
    _assert_identical(all_arms[case][0], all_arms[case][2],
                      f"{case}: baseline-vs-candidate")
