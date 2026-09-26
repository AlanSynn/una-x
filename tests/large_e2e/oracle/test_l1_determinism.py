"""L1 determinism gate (VALIDATION.md): run actual compiled B0 TWICE —
in-process, in fresh subprocesses, and across numba worker counts — and
require bit-identical outputs. Also pins the frozen compiler profile
and the hazard/refusal domains (bounded child probes only)."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import fixtures
import stub_topology
from comparator import OracleMismatch, assert_array_bytes_equal
from support import b0


CAMPAIGN_PYTHON = ("/Users/alansynn/orca/workspaces/una-x/venvs/campaign/"
                   "bin/python")
ORACLE_DIR = Path(__file__).resolve().parent
B0_ROOT = "/Users/alansynn/orca/workspaces/una-x/wt-b0/src"


# ----------------------------------------------------------------------
# Frozen compiler profile
# ----------------------------------------------------------------------

def test_compiler_profile_frozen_facts():
    b = b0()
    opts = b._accumulate_od_flow.targetoptions
    assert opts.get("fastmath") is True
    assert "nogil" not in opts or opts.get("nogil") is not True, (
        "verified live fact broken: _accumulate_od_flow has nogil")
    for kern in ("compact_vector_node_view_scope", "adjust_destination_distances",
                 "reach_gravity_knn_access"):
        ko = getattr(b, kern).targetoptions
        assert ko.get("nogil") is True and ko.get("fastmath") is True
        assert ko.get("parallel") is False
    for kern in ("od_compact_vector_node_view_scope", "integrated_scope_access"):
        assert getattr(b, kern).targetoptions.get("parallel") is True


def test_profile_environment_matches_golden_record():
    """The environment the golden was built in must be THIS environment
    (frozen dependency profile), otherwise golden bytes are void."""
    import llvmlite
    import numba
    import scipy
    from support import golden_meta
    prof = golden_meta()["profile"]
    assert prof["numba_version"] == numba.__version__
    assert prof["llvmlite_version"] == llvmlite.__version__
    assert prof["numpy_version"] == np.__version__
    assert prof["scipy_version"] == scipy.__version__
    assert prof["platform"]["machine"] == "arm64"
    assert prof["env"]["L1_REUSE_DIR"] is None
    assert golden_meta()["import_root"] == B0_ROOT
    from support import golden_hashes
    assert golden_hashes()["b0_commit"] == (
        "361928e4ba38f34622cafe065b0025244db61368")
    assert golden_hashes()["import_root"] == B0_ROOT
    assert prof["verified_live_facts"]["_accumulate_od_flow_nogil"] is False
    assert prof["numba_threading_layer"] == "workqueue"


# ----------------------------------------------------------------------
# In-process double runs
# ----------------------------------------------------------------------

def test_in_process_double_run_bit_identical():
    b = b0()
    case = fixtures.engine_cases()[10]  # flt_signed_zero — signed-zero exact
    term = case.terminal_arrays()
    topo = stub_topology.StubTopology(case)
    ge = b.Accessibility(topo).graph_engine
    args = (term["o_terminal_idxs"][0], term["o_terminal_weights"][0],
            ge.adjacency_pointer, ge.adjacency_vector,
            ge.adjacency_vector_weights, ge.adjacynct_vector_network_node,
            case.cutoff, term["d_count"])
    out1 = b.compact_vector_node_view_scope(*args)
    out2 = b.compact_vector_node_view_scope(*args)
    assert_array_bytes_equal(np.asarray(out1[0]), np.asarray(out2[0]),
                             "scope/run2")

    adj1 = b.adjust_destination_distances(out1[0], term["d_terminal_idxs"],
                                          term["d_terminal_weights"],
                                          term["d_count"])
    adj2 = b.adjust_destination_distances(out2[0], term["d_terminal_idxs"],
                                          term["d_terminal_weights"],
                                          term["d_count"])
    assert_array_bytes_equal(adj1, adj2, "adjust/run2")
    # signed zero preserved byte-exactly across runs
    assert adj1.tobytes() == adj2.tobytes()

    m1 = b.reach_gravity_knn_access(adj1, term["d_weights"], case.cutoff,
                                    0.001, 0.0, 500.0, 0.0,
                                    np.array([1.0, 1.0, 0.5]), 0.0,
                                    "logistic", 500.0, 1.0, 1.0)
    m2 = b.reach_gravity_knn_access(adj2, term["d_weights"], case.cutoff,
                                    0.001, 0.0, 500.0, 0.0,
                                    np.array([1.0, 1.0, 0.5]), 0.0,
                                    "logistic", 500.0, 1.0, 1.0)
    for a, e, label in zip(m1, m2, ("reach", "gexp", "glog", "knn")):
        assert np.float64(a).tobytes() == np.float64(e).tobytes(), label


def test_flow_engine_repeat_run_bit_identical():
    b = b0()
    eng, settings, _ = __import__("support").flow_engine_case(b, None)
    eng.num_threads = 2
    eng.Centrality(settings)
    first = (eng.edge_flow_AB.copy(), eng.edge_flow_BA.copy(),
             eng.node_flow.copy())
    eng.Centrality(settings)
    assert eng.edge_flow_AB.tobytes() == first[0].tobytes()
    assert eng.edge_flow_BA.tobytes() == first[1].tobytes()
    assert eng.node_flow.tobytes() == first[2].tobytes()


def test_parallel_kernel_identical_across_numba_thread_counts():
    """L1 'multiple independent workers': the parallel=True kernels must
    produce bit-identical outputs with different worker-pool sizes."""
    import runner_arm

    def run_with(env_over, out_npz):
        env = dict(os.environ)
        env["PYTHONPATH"] = B0_ROOT
        env["UNA_ORACLE_ARM_A"] = B0_ROOT
        env["NUMBA_CACHE_DIR"] = os.environ["NUMBA_CACHE_DIR"]
        env.pop("L1_REUSE_DIR", None)
        env.update(env_over)
        proc = subprocess.run(
            [CAMPAIGN_PYTHON, str(ORACLE_DIR / "runner_arm.py"),
             "--out-npz", str(out_npz), "--out-json", str(out_npz) + ".json",
             "--runs", "1"],
            cwd="/tmp", env=env, capture_output=True, text=True, timeout=1800)
        assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-2000:]

    tmp = Path("/tmp/h03_threads")
    tmp.mkdir(exist_ok=True)
    run_with({"NUMBA_NUM_THREADS": "1"}, tmp / "t1.npz")
    run_with({"NUMBA_NUM_THREADS": "4"}, tmp / "t4.npz")

    za, zb = np.load(tmp / "t1.npz"), np.load(tmp / "t4.npz")
    assert set(za.files) == set(zb.files)
    for key in za.files:
        a, b = np.ascontiguousarray(za[key]), np.ascontiguousarray(zb[key])
        assert a.dtype == b.dtype and a.shape == b.shape, key
        assert a.tobytes() == b.tobytes(), (
            f"parallel kernel output differs across worker counts: {key}")


# ----------------------------------------------------------------------
# Fresh-process B0-vs-B0 through the two-root comparator (both = B0)
# ----------------------------------------------------------------------

def test_subprocess_b0_vs_b0_bit_identical(tmp_path):
    sys.path.insert(0, "/Users/alansynn/orca/workspaces/una-x/wt-large-e2e/"
                       "benchmarks/large_e2e/oracle")
    import compare_arms
    report_path = tmp_path / "report.json"
    argv = sys.argv
    sys.argv = ["compare_arms.py", "--workdir", str(tmp_path / "work"),
                "--arm-a", B0_ROOT, "--arm-b", B0_ROOT,
                "--report", str(report_path), "--label", "h03_b0_vs_b0"]
    try:
        code = compare_arms.main()
    finally:
        sys.argv = argv
    assert code == 0, f"compare_arms failed, report={report_path}"
    report = json.loads(report_path.read_text())
    assert report["verdict"] == "BIT_IDENTICAL"
    assert report["arm_a_meta"]["import_root"] == B0_ROOT
    assert report["arm_b_meta"]["import_root"] == B0_ROOT
    assert report["arm_a_meta"]["self_determinism"]["all_identical"]
    assert report["arm_b_meta"]["self_determinism"]["all_identical"]
    assert report["arm_a_meta"]["module_files"]["Engines.Accessibility"].startswith(B0_ROOT)
    assert report["comparison"]["n_keys"] > 900
    assert report["comparison"]["mismatched_keys"] == []


# ----------------------------------------------------------------------
# Hazard / refusal domains — bounded child probes ONLY
# ----------------------------------------------------------------------

def _run_hazard(mode, timeout_s):
    env = dict(os.environ)
    env["PYTHONPATH"] = B0_ROOT
    env["NUMBA_CACHE_DIR"] = os.environ["NUMBA_CACHE_DIR"]
    proc = subprocess.run(
        [CAMPAIGN_PYTHON, str(ORACLE_DIR / "child_hazard.py"),
         "--mode", mode, "--timeout-s", "10"],
        cwd="/tmp", env=env, capture_output=True, text=True,
        timeout=timeout_s)
    lines = [ln for ln in proc.stdout.strip().splitlines() if ln.startswith("{")]
    payload = json.loads(lines[-1]) if lines else {
        "mode": mode, "outcome": "no_output", "returncode": proc.returncode}
    payload["returncode"] = proc.returncode
    return payload


def test_hazard_negative_weight_nontermination_is_bounded():
    """Negative cycle: B0's kernel has no negative-cycle guard; the
    expected outcome is nontermination captured by the bounded child —
    hazard characterization, NOT a numerical golden."""
    payload = _run_hazard("negative_cycle", timeout_s=60)
    assert payload["outcome"] == "timeout_kill", (
        f"re-characterize hazard: B0 completed a negative-cycle probe: {payload}")
    assert payload["returncode"] == 9


@pytest.mark.parametrize("mode", ["nan_edge", "nan_origin", "inf_cutoff"])
def test_hazard_probe_domains_recorded(mode):
    """Record actual B0 behavior for NaN / inf domains from bounded
    children. Either outcome (completed with bytes, or bounded kill) is
    admissible evidence; the test asserts only that the probe was
    safely bounded and produced a record."""
    payload = _run_hazard(mode, timeout_s=60)
    assert payload["outcome"] in ("completed", "timeout_kill", "error"), payload
    if payload["outcome"] == "completed":
        assert "scope_bits" in payload and "scope_dtype" in payload
