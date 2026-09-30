"""T13/T14 routing engagement (bounded children, NUMBA_NRT_STATS=1).

T14: on an admitted input the candidate driver's per-call NRT
allocation count is bounded and constant (no per-pop growth) and far
below B0's on the same input — the A1 target made observable.
T13: on a value-refusing input (negative cost, same typing) the
candidate's count equals B0's exactly — the fallback compiled path.
"""
from __future__ import annotations

import json
import os
import subprocess
import tempfile

from _legacy_paths import LEGACY_WS  # portable-path fix (HARNESS 2026-09-30)
CANDIDATE_ROOT = os.environ.get(
    "UNA_A1_CANDIDATE_ROOT",
    str(LEGACY_WS) + "/wt-large-e2e/src")
B0_ROOT = str(LEGACY_WS) + "/wt-b0/src"
CAMPAIGN_PYTHON = (str(LEGACY_WS) + "/venvs/campaign/"
                   "bin/python")
CHILD = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "nrt_child.py")
CHILD_TIMEOUT_S = 1200


def _run_arm(root):
    env = dict(os.environ)
    env["NUMBA_NUM_THREADS"] = env.get("UNA_A1_NUM_THREADS", "2")
    with tempfile.TemporaryDirectory(prefix="a1_nrt_") as workdir:
        # fresh per-arm cache: cache entries written by the suite's
        # alias-loaded candidate pickle the alias module name, which a
        # real-name import cannot resolve (test-infra collision)
        env["NUMBA_CACHE_DIR"] = os.path.join(workdir, "cache")
        out_json = os.path.join(workdir, "out.json")
        proc = subprocess.run(
            [CAMPAIGN_PYTHON, CHILD, root, out_json],
            cwd="/tmp", env=env, capture_output=True, text=True,
            timeout=CHILD_TIMEOUT_S)
        assert proc.returncode == 0, proc.stderr[-3000:]
        with open(out_json) as fh:
            return json.load(fh)


def test_t14_admitted_route_bounded_and_t13_refusal_matches_b0():
    cand = _run_arm(CANDIDATE_ROOT)
    b0 = _run_arm(B0_ROOT)

    # both arms: counts constant across repeated calls
    for arm in (cand, b0):
        assert len(set(arm["admitted_allocs"])) == 1, arm
        assert len(set(arm["refused_allocs"])) == 1, arm

    # T13: refusal -> the exact B0 allocation signature (fallback path)
    assert cand["refused_allocs"] == b0["refused_allocs"], (
        cand["refused_allocs"], b0["refused_allocs"])

    # T14: admitted -> bounded, far below B0's per-pop rate
    cand_adm = cand["admitted_allocs"][0]
    b0_adm = b0["admitted_allocs"][0]
    assert cand_adm < b0_adm // 2, (cand_adm, b0_adm)
