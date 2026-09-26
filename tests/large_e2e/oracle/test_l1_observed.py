"""L1-observed (H04-N4): kernel-boundary replay of the typed arrays
captured from the REAL B0 O2 observed pass (accessibility + flow with
the auto p95 gravity cap; H04 reviewer's fixture and settings).

The captured pass was verified byte-identical to the reviewer's
observed run: all six exported accessibility/flow Results files match
the reviewer's recorded sha256s, and the auto-resolved gravity cap
matches exactly (7540.0175). These tests pin the next link of that
chain: LIVE compiled B0 must reproduce every captured kernel-boundary
output byte-for-byte from the captured inputs.

Candidate-vs-B0 use (A1R..F3R): run observed_replay.replay_* with a
candidate namespace (b0_import.load_arm(UNA_ORACLE_ARM_B=...)) and
compare outputs to the SAME stored arrays with
assert_array_bytes_equal. No tolerance, no exception.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

import observed_replay as obsrep
from comparator import OracleMismatch, assert_array_bytes_equal
from mutants import mutant_gradient_where_reversed
from support import b0, golden_hashes
from scipy.sparse.csgraph import dijkstra as scipy_dijkstra


@pytest.fixture(scope="module")
def observed():
    return obsrep.load_observed()


def test_observed_fixture_files_intact(observed):
    """npz file bytes and every stored array must match the sidecar."""
    sidecar, npz = observed
    h = hashlib.sha256(obsrep.NPZ_PATH.read_bytes()).hexdigest()
    assert h == sidecar["npz_sha256"]
    assert set(npz.files) == set(sidecar["arrays"])
    for key, meta in sidecar["arrays"].items():
        a = np.asarray(npz[key])
        assert str(a.dtype) == meta["dtype_name"], key
        assert list(a.shape) == meta["shape"], key
        assert hashlib.sha256(
            a.dtype.str.encode() + str(a.shape).encode() + a.tobytes()
        ).hexdigest() == meta["sha256"], key


def test_observed_provenance_and_env(observed):
    """The fixture must come from the pinned B0 tree at the pinned
    commit, from the H04 reviewer's O2 fixture, in THIS environment."""
    sidecar, _ = observed
    prov = sidecar["provenance"]
    assert prov["b0_root"] == "/Users/alansynn/orca/workspaces/una-x/wt-b0/src"
    assert prov["b0_commit"] == golden_hashes()["b0_commit"]
    assert prov["o2_fixture_sha256"] == (
        "afaf4bf4651a1c9d76a8328ab8e78d99bb35699b3fd378c297c54661dea09492")
    b = b0()
    for mod_file in b.module_files.values():
        assert mod_file.startswith(prov["b0_root"]), mod_file
    # captured in the same pinned toolchain this suite runs in
    import numba
    import numpy
    import scipy
    assert sidecar["env"]["numba"] == numba.__version__
    assert sidecar["env"]["numpy"] == numpy.__version__
    assert sidecar["env"]["scipy"] == scipy.__version__
    # the auto gravity cap the pass resolved (matches the H04 reviewer)
    assert prov["resolved_gravity_cap"]["value"] == 7540.0175
    assert prov["resolved_gravity_cap"]["spec"] == "p95"


def test_observed_dtypes_pin_endpoint_typing(observed):
    """The observed typing facts N4 requires, including the int32
    endpoint inputs: flow CSR indptr/indices/edge_id are int32,
    direction int8; the accessibility CSR endpoints are int64; outputs
    typing as observed."""
    sidecar, _ = observed
    arrs = sidecar["arrays"]
    g = sidecar["calls"]["precompute_dest_gradients"][0]["args"]
    assert arrs[g["_csr_indptr"]]["dtype_name"] == "int32"
    assert arrs[g["_csr_indices"]]["dtype_name"] == "int32"
    assert arrs[g["_csr_edge_id"]]["dtype_name"] == "int32"
    assert arrs[g["_csr_direction"]]["dtype_name"] == "int8"
    assert arrs[g["_csr_weights"]]["dtype_name"] == "float64"
    assert arrs[g["_csr_rev_indptr"]]["dtype_name"] == "int32"
    assert arrs[g["_csr_rev_indices"]]["dtype_name"] == "int32"
    assert arrs[g["_csr_rev_data"]]["dtype_name"] == "float64"
    a0 = sidecar["calls"]["integrated_scope_access"][0]["args"]
    assert arrs[a0["adjacency_pointer"]]["dtype_name"] == "int64"
    assert arrs[a0["adjacency_vector"]]["dtype_name"] == "int64"
    assert arrs[a0["adjacency_vector_weights"]]["dtype_name"] == "float64"
    assert arrs[a0["adjacynct_vector_network_node"]]["dtype_name"] == "bool"
    assert arrs[a0["o_terminal_idxs"]]["dtype_name"] == "int64"
    assert arrs[a0["d_terminal_idxs"]]["dtype_name"] == "int64"
    out0 = sidecar["calls"]["integrated_scope_access"][0]["outputs"]
    assert arrs[out0["reach"]]["dtype_name"] == "int64"
    assert arrs[out0["gravity_exponential"]]["dtype_name"] == "float64"


def test_observed_call_ledger_shape(observed):
    """The capture must reflect the real pass: 2 accessibility kernel
    calls (run + cap derivation), 16 origins, 137 OD kernel calls with
    32 selected zero-buffer replays, 1 gradient assembly, and the base
    Accessibility kernel family NOT called (O2 dispatches to the
    elevation family)."""
    sidecar, _ = observed
    calls = sidecar["calls"]
    assert len(calls["integrated_scope_access"]) == 2
    assert {r["module"] for r in calls["integrated_scope_access"]} == \
        {"AccessibilityWElevation"}
    assert len(calls["compute_trip_volumes"]) == 16
    assert len(calls["accumulate_od_flow"]) == 137
    assert sum(1 for r in calls["accumulate_od_flow"] if r["selected"]) == 32
    assert len(calls["precompute_dest_gradients"]) == 1
    # every replayed call's delivered is bit-equal live vs zero-buffer
    assert all(r["delivered_live"] == r["delivered_zero"]
               for r in calls["accumulate_od_flow"] if r["selected"])


def test_integrated_scope_access_replay_bytes(observed):
    """Live elevation-family kernel must reproduce the captured
    accessibility outputs (both observed calls share deduped inputs;
    record 1 == record 2 by dedup, and both outputs must match)."""
    b = b0()
    sidecar, npz = observed
    for i, rec in enumerate(sidecar["calls"]["integrated_scope_access"]):
        outs, _ = obsrep.replay_integrated_scope_access(b, npz, rec)
        for name, key in rec["outputs"].items():
            assert_array_bytes_equal(
                outs[["reach", "gravity_exponential", "gravity_logistic",
                      "knn_access"].index(name)],
                np.asarray(npz[key]),
                f"observed/isa[{i}]/{name}")


def test_gradient_boundary_replay_bytes(observed):
    b = b0()  # loaded for import-root identity; assembly is scipy-side
    sidecar, npz = observed
    rec = sidecar["calls"]["precompute_dest_gradients"][0]
    # chunk=512 bounds the dense scipy transient (~340 MB) in-suite; the
    # captured outputs were assembled at the engine's chunk rule (1754),
    # so this also proves assembled-byte chunk-invariance.
    assembled = obsrep.replay_gradient(npz, rec, chunk=512)
    for name, key in rec["outputs"].items():
        assert_array_bytes_equal(np.asarray(assembled[name]),
                                 np.asarray(npz[key]),
                                 f"observed/gradient/{name}")


def test_od_flow_kernel_replay_bytes(observed):
    """Every selected observed OD call: fresh zero buffers in, byte-equal
    per-call delta out, bit-equal delivered."""
    b = b0()
    sidecar, npz = observed
    for rec in obsrep.selected_od_records(sidecar):
        out_AB, out_BA, out_node, delivered = obsrep.replay_od_flow(
            b, sidecar, npz, rec)
        assert delivered == rec["delivered_zero"]
        for name, arr in (("out_AB", out_AB), ("out_BA", out_BA),
                          ("out_node_flow", out_node)):
            assert_array_bytes_equal(arr, np.asarray(npz[rec["outputs"][name]]),
                                     f"observed/od{rec['seq']}/{name}")


def test_trip_volume_replay_bytes(observed):
    b = b0()
    sidecar, npz = observed
    for i, rec in enumerate(sidecar["calls"]["compute_trip_volumes"]):
        out = obsrep.replay_trip_volumes(b, npz, rec)
        assert_array_bytes_equal(out, np.asarray(npz[rec["output"]]),
                                 f"observed/ctv[{i}]")


def test_engine_outputs_anchor_present(observed):
    """The observed engine-level outputs captured after each public call
    are present with the observed scale (V=49159 net nodes, E=69957
    edges) and are non-degenerate."""
    sidecar, npz = observed
    reach = np.asarray(npz[sidecar["engine_outputs"]["accessibility"]["reach"]])
    assert reach.shape == (16,) and reach.dtype == np.int64
    AB = np.asarray(npz[sidecar["engine_outputs"]["flow"]["edge_flow_AB"]])
    BA = np.asarray(npz[sidecar["engine_outputs"]["flow"]["edge_flow_BA"]])
    assert AB.shape == (69957,) and BA.shape == (69957,)
    assert np.count_nonzero(AB) > 0 and np.count_nonzero(BA) > 0


def test_negative_observed_gradient_reversed_where_caught(observed):
    """Negative check on the observed fixture: the reversed-where
    gradient mutant must be caught against the OBSERVED outputs."""
    sidecar, npz = observed
    rec = sidecar["calls"]["precompute_dest_gradients"][0]
    indptr = np.asarray(npz[rec["args"]["_csr_rev_indptr"]])
    indices = np.asarray(npz[rec["args"]["_csr_rev_indices"]])
    data = np.asarray(npz[rec["args"]["_csr_rev_data"]])
    n_total = indptr.shape[0] - 1
    from scipy.sparse import csr_matrix
    csr_rev = csr_matrix((data, indices, indptr), shape=(n_total, n_total))
    dest_nodes = np.asarray(npz[rec["args"]["dest_nodes"]])
    s = rec["scalars"]
    bad = mutant_gradient_where_reversed(
        scipy_dijkstra, csr_rev, dest_nodes, float(s["limit"]),
        int(s["n_dest"]), 512)
    caught = []
    for i, name in enumerate(("indptr", "nodes", "dist", "pred")):
        try:
            assert_array_bytes_equal(np.asarray(bad[name]),
                                     np.asarray(npz[rec["outputs"][name]]),
                                     f"negative/observed_gradient/{name}")
        except OracleMismatch:
            caught.append(name)
    assert set(caught) == {"nodes", "dist", "pred"}, caught


def test_negative_observed_od_budget_offset_caught(observed):
    """Negative check on the observed fixture: a driver that drops the
    detour ratio/buffer and budgets the OD at the full gradient limit
    (an F2-class budget-computation bug) must move bytes of the
    observed per-call outputs, and the byte comparator must catch it.
    Proves the stored deltas are discriminative at the observed scale
    (not vacuous). Small absolute offsets are NOT used: the budget
    enters the reach mask by inequality over continuously-valued
    shortest distances, so a +1.0 offset can leave every envelope
    unchanged (verified empirically on these records).

    The scan is deterministic: selected records in capture-seq order;
    the first record whose mutant output differs is pinned."""
    b = b0()
    sidecar, npz = observed
    limit = float(sidecar["calls"]["precompute_dest_gradients"][0]
                  ["scalars"]["limit"])
    caught = None
    for rec in obsrep.selected_od_records(sidecar):
        ref_AB, ref_BA, ref_node, ref_del = obsrep.replay_od_flow(
            b, sidecar, npz, rec)
        mut_AB, mut_BA, mut_node, mut_del = obsrep.replay_od_flow(
            b, sidecar, npz, rec, budget_override=limit)
        differing = [name for name, mut, ref in
                     (("out_AB", mut_AB, ref_AB), ("out_BA", mut_BA, ref_BA),
                      ("out_node_flow", mut_node, ref_node))
                     if np.asarray(mut).tobytes() != np.asarray(ref).tobytes()]
        if differing:
            caught = (rec["seq"], differing[0], mut_AB, ref_AB)
            break
    assert caught is not None, (
        "budget-at-gradient-limit mutant moved no observed output byte "
        "on any selected OD call")
    seq, name, mut, ref = caught
    with pytest.raises(OracleMismatch):
        assert_array_bytes_equal(mut, ref, f"negative/od_budget/od{seq}/{name}")
