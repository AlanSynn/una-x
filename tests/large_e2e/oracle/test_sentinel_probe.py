"""A2 sentinel probe (dossier-required helper) evaluated against REAL
compiled scopes across the sentinel domain: admission, the crafted
b<=R rounding hazard, maxfloat, inf (bounded child), and D==0."""
from __future__ import annotations

import json
import os
import subprocess

import numpy as np
import pytest

import fixtures
from support import b0
from test_l0_traces import _scope_inputs_for_engine_case

CAMPAIGN_PYTHON = ("/Users/alansynn/orca/workspaces/una-x/venvs/campaign/"
                   "bin/python")
ORACLE_DIR = os.path.dirname(os.path.abspath(__file__))
B0_ROOT = "/Users/alansynn/orca/workspaces/una-x/wt-b0/src"


def _scope_for(case_name):
    b = b0()
    case = [c for c in fixtures.engine_cases()
            if c.name == case_name][0]
    csr, term = _scope_inputs_for_engine_case(b, case)
    scope = np.asarray(b.compact_vector_node_view_scope(
        term["o_terminal_idxs"][0], term["o_terminal_weights"][0],
        csr["pointer"], csr["vector"], csr["weights"], csr["flags"],
        case.cutoff, term["d_count"])[0])
    v_count = int(csr["pointer"].shape[0] - 1)
    return scope, v_count, term["d_count"], case.cutoff


def test_probe_admits_on_normal_cutoff():
    """Ordinary finite cutoff: stored b == cutoff+1, finite and > R, so
    the sentinel admits local pruning (the safe common case)."""
    scope, v_count, d_count, cutoff = _scope_for("l0_tiny_random_dup_loop")
    probe = trace_access_sentinel(scope, v_count, d_count, cutoff)
    assert probe["d_count"] == d_count > 0
    assert probe["b"] == cutoff + 1.0
    assert probe["b_finite"] is True
    assert probe["b_gt_R"] is True
    assert probe["admits_local_pruning"] is True
    assert probe["b_bits"] == np.float64(cutoff + 1.0).tobytes().hex()


def test_probe_refuses_on_two_pow_53_rounding_hazard():
    """Crafted b<=R WITHOUT huge weights beyond 2**53: at cutoff=2**53
    the typed initialization b=1+cutoff rounds back to cutoff, so the
    actual stored bytes FAIL the b>R test — the probe must refuse."""
    assert float(np.float64(1.0) + np.float64(2.0 ** 53)) == 2.0 ** 53
    scope, v_count, d_count, cutoff = _scope_for("flt_sentinel_two_pow_53")
    probe = trace_access_sentinel(scope, v_count, d_count, cutoff)
    assert probe["b"] == cutoff           # rounded sentinel, not cutoff+1
    assert probe["b_finite"] is True
    assert probe["b_gt_R"] is False       # b <= R: crafted hazard domain
    assert probe["admits_local_pruning"] is False


def test_probe_refuses_on_maxfloat_cutoff():
    scope, v_count, d_count, cutoff = _scope_for("flt_maxfloat_cutoff")
    probe = trace_access_sentinel(scope, v_count, d_count, cutoff)
    assert probe["b"] == np.finfo(np.float64).max
    assert probe["b_finite"] is True
    assert probe["b_gt_R"] is False
    assert probe["admits_local_pruning"] is False


def test_probe_documents_d_zero_scope():
    """D==0: no destination tail exists; the probe reports the unchanged
    empty-destination behavior instead of reading out of bounds."""
    scope, v_count, d_count, cutoff = _scope_for("l0_tiny_random_dup_loop")
    probe = trace_access_sentinel(scope, v_count, 0, cutoff)
    assert probe["d_count"] == 0
    assert probe["b"] is None and probe["b_finite"] is None
    assert probe["b_gt_R"] is None
    assert probe["admits_local_pruning"] is False
    assert "D==0" in probe["reason"]


def test_probe_refuses_infinite_sentinel_from_bounded_child():
    """cutoff=+inf runs ONLY in a bounded child (possible unbounded label
    growth). The child's stored sentinel bytes are decoded here and fed
    through the same probe condition: b=inf is not finite -> refuse."""
    env = dict(os.environ)
    env["PYTHONPATH"] = B0_ROOT
    env["NUMBA_CACHE_DIR"] = os.environ["NUMBA_CACHE_DIR"]
    env.pop("L1_REUSE_DIR", None)
    proc = subprocess.run(
        [CAMPAIGN_PYTHON, ORACLE_DIR + "/child_hazard.py",
         "--mode", "inf_cutoff", "--timeout-s", "10"],
        cwd="/tmp", env=env, capture_output=True, text=True, timeout=60)
    lines = [ln for ln in proc.stdout.strip().splitlines()
             if ln.startswith("{")]
    payload = json.loads(lines[-1])
    assert payload["outcome"] == "completed", payload
    scope = np.frombuffer(bytes.fromhex(payload["scope_bits"]),
                          dtype=np.float64)
    v_count = 3  # child graph has 3 network nodes
    b = float(scope[v_count])
    assert np.isinf(b)
    assert not (np.isfinite(b) and b > float(np.inf))
    # the probe helper itself, on the child's stored bytes:
    probe = trace_access_sentinel(scope, v_count, 1, float(np.inf))
    assert probe["admits_local_pruning"] is False


def test_probe_condition_matches_compiled_filter():
    """The probe's refusal on flt_sentinel_two_pow_53 must agree with the
    compiled retained filter: the rounded sentinel destination IS still
    retained (d_adjust == R <= R) — that asymmetry is exactly the hazard
    the probe documents."""
    from comparator import assert_array_bytes_equal  # noqa: F401
    b = b0()
    case = [c for c in fixtures.engine_cases()
            if c.name == "flt_sentinel_two_pow_53"][0]
    csr, term = _scope_inputs_for_engine_case(b, case)
    scope = np.asarray(b.compact_vector_node_view_scope(
        term["o_terminal_idxs"][0], term["o_terminal_weights"][0],
        csr["pointer"], csr["vector"], csr["weights"], csr["flags"],
        case.cutoff, term["d_count"])[0])
    probe = trace_access_sentinel(
        scope, int(csr["pointer"].shape[0] - 1), term["d_count"], case.cutoff)
    adjust = np.asarray(b.adjust_destination_distances(
        scope, term["d_terminal_idxs"], term["d_terminal_weights"],
        term["d_count"]))
    retained = np.where(adjust <= case.cutoff)[0]
    # Probe refuses local pruning, yet the compiled filter keeps the
    # rounded-to-R destination — both facts recorded, hazard documented.
    assert probe["admits_local_pruning"] is False
    assert 0 in retained


def trace_access_sentinel(scope, v_count, d_count, cutoff):
    import trace_access
    return trace_access.sentinel_probe(scope, v_count, d_count, cutoff)
