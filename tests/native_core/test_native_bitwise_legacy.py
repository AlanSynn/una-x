"""una_legacy bitwise battery: native core vs compiled engines vs oracle.

The native core claims bit-identical execution of the characterized
two-phase scope search (kernels/SPEC.md section 3) on the admitted
domain.  These tests are that claim, machine-checked against BOTH pinned
references on the same battery the CPU_ALGORITHMS suite pins the oracle
with: the compiled engine kernels (A3/A1), the engine driver, and the
pure-Python oracle schedule.

The NaN-on-both-seeds case stays excluded here exactly as everywhere
else (the characterized NONTERMINATION; SPEC.md section 3).
"""
import math

import numpy as np
import pytest

from urban_network_analysis.backends.native import route_scope_access
from urban_network_analysis.backends.native._loader import (
    NativeUnavailableError,
    load_core,
)
from urban_network_analysis.kernels import scope_access as SA

from conftest import CASE_PARAMS, bits, build_case, case_kwargs

try:
    CORE = load_core()
except NativeUnavailableError:  # pragma: no cover - unbuilt host
    CORE = None

needs_core = pytest.mark.skipif(CORE is None,
                                reason="native core not built on this host")


def scratch(max_degree):
    return (np.empty(max_degree, dtype=np.int64),
            np.empty(max_degree, dtype=np.float64))


# ---------------------------------------------------------------------------
# Labels: native vs engine kernels, both routes, whole battery
# ---------------------------------------------------------------------------

@needs_core
@pytest.mark.parametrize("domain", ["a3", "a1"])
@pytest.mark.parametrize("name", CASE_PARAMS)
def test_search_labels_bitwise_vs_engine(name, domain, engines):
    """Native labels == engine A3/A1 kernel labels, byte-for-byte."""
    c = build_case(3 + CASE_PARAMS.index(name), **case_kwargs(name))
    admitted, max_degree = SA.scope_route_admits(
        c["ptr"], c["vec"], c["wts"], c["flag"], c["o"], c["ow"],
        c["cutoff"])
    if not admitted:
        pytest.skip("guard refuses this dtype domain (fallback case)")
    d_count = c["dw"].shape[0]
    off, sw = scratch(max_degree)
    if domain == "a3":
        engine_labels = engines.LAS._a3_scope_search_tailless(
            c["o"][0], c["ow"][0], c["ptr"], c["vec"], c["wts"], c["flag"],
            c["cutoff"], off, sw)[0]
    else:
        engine_labels = engines.LAS._a1_scope_search(
            c["o"][0], c["ow"][0], c["ptr"], c["vec"], c["wts"], c["flag"],
            c["cutoff"], d_count, off, sw)[0]
    native_labels = CORE.search_labels_legacy(
        c["o"][0], c["ow"][0], c["ptr"], c["vec"], c["wts"], c["flag"],
        c["cutoff"], domain, d_count)
    assert bits(native_labels[:engine_labels.shape[0]]) == \
        bits(engine_labels)


@needs_core
@pytest.mark.parametrize("name", CASE_PARAMS)
def test_search_labels_bitwise_vs_oracle(name):
    """Native labels == oracle schedule labels, byte-for-byte, on both
    label domains (the oracle is the portable schedule definition)."""
    c = build_case(3 + CASE_PARAMS.index(name), **case_kwargs(name))
    admitted, _ = SA.scope_route_admits(
        c["ptr"], c["vec"], c["wts"], c["flag"], c["o"], c["ow"],
        c["cutoff"])
    if not admitted:
        pytest.skip("guard refuses this dtype domain (fallback case)")
    d_count = c["dw"].shape[0]
    for domain, tail in (("a3", 0), ("a1", d_count)):
        oracle = SA.scope_search_schedule(
            c["o"][0], c["ow"][0], c["ptr"], c["vec"], c["wts"], c["flag"],
            c["cutoff"], d_count=tail, domain=domain)
        native = CORE.search_labels_legacy(
            c["o"][0], c["ow"][0], c["ptr"], c["vec"], c["wts"], c["flag"],
            c["cutoff"], domain, tail)
        assert bits(native) == bits(oracle), domain


# ---------------------------------------------------------------------------
# Structural semantics on tiny graphs (native vs oracle; the engine side
# of every row here is pinned in tests/cpu_kernels/test_schedule_pinning)
# ---------------------------------------------------------------------------

def tiny_graph(arcs_by_node, weights_by_node, flags_by_node):
    pointer = [0]
    vector = []
    weights = []
    flag = []
    for node in range(len(arcs_by_node)):
        for k, neighbor in enumerate(arcs_by_node[node]):
            vector.append(neighbor)
            weights.append(float(weights_by_node[node][k]))
            flag.append(bool(flags_by_node[node][k]))
        pointer.append(len(vector))
    return (np.asarray(pointer, dtype=np.int64),
            np.asarray(vector, dtype=np.int64),
            np.asarray(weights, dtype=np.float64),
            np.asarray(flag, dtype=np.bool_))


def native_run_pair(ptr, vec, wts, flag, o, ow, cutoff):
    """(native labels, oracle labels) on the a3 domain, for the tiny
    structural graphs."""
    oracle = SA.scope_search_schedule(
        o, ow, ptr, vec, wts, flag, cutoff, d_count=0, domain="a3")
    native = CORE.search_labels_legacy(
        o, ow, ptr, vec, wts, flag, cutoff, "a3", 0)
    return native, oracle


@needs_core
def test_duplicate_arc_row_order_semantics():
    """Phase two assigns in stored row order: the SECOND parallel arc
    wins even when larger (the row-order-overwrite quirk)."""
    ptr, vec, wts, flag = tiny_graph([[1, 1], []], [[2.0, 1.0], []],
                                     [[True, True], []])
    o = np.array([0, 1], dtype=np.int64)
    ow = np.array([0.0, 9e9], dtype=np.float64)
    native, oracle = native_run_pair(ptr, vec, wts, flag, o, ow, 10.0)
    assert bits(native) == bits(oracle)
    assert float(oracle[1]) == 1.0

    ptr2, vec2, wts2, flag2 = tiny_graph([[1, 1], []], [[1.0, 2.0], []],
                                         [[True, True], []])
    native2, oracle2 = native_run_pair(ptr2, vec2, wts2, flag2, o, ow, 10.0)
    assert bits(native2) == bits(oracle2)
    assert float(oracle2[1]) == 2.0  # second, LARGER arc overwrites


@needs_core
def test_self_loop_semantics():
    ptr, vec, wts, flag = tiny_graph([[0, 1], [], []],
                                     [[5.0, 1.0], [], []],
                                     [[True, True], [], []])
    o = np.array([0, 2], dtype=np.int64)
    ow = np.array([0.0, 9e9], dtype=np.float64)
    native, oracle = native_run_pair(ptr, vec, wts, flag, o, ow, 10.0)
    assert bits(native) == bits(oracle)
    assert float(oracle[0]) == 0.0
    assert float(oracle[1]) == 1.0   # leaf gate: assigned, never pushed
    assert float(oracle[2]) == 9e9   # unconditional end-seed write


@needs_core
def test_leaf_push_gate_and_flag_false_semantics():
    ptr, vec, wts, flag = tiny_graph([[1], [2], [], []],
                                     [[1.0], [1.0], [], []],
                                     [[True], [True], [], []])
    o = np.array([0, 3], dtype=np.int64)
    ow = np.array([0.0, 9e9], dtype=np.float64)
    native, oracle = native_run_pair(ptr, vec, wts, flag, o, ow, 10.0)
    assert bits(native) == bits(oracle)
    assert float(oracle[1]) == 1.0
    assert float(oracle[2]) == 11.0  # sentinel: leaf 1 never pushed

    ptr2, vec2, wts2, flag2 = tiny_graph([[1, 2], [], []],
                                         [[1.0, 5.0], [], []],
                                         [[False, True], [], []])
    o2 = np.array([0, 2], dtype=np.int64)  # 3-node graph: end terminal 2
    ow2 = np.array([0.0, 9e9], dtype=np.float64)
    native2, oracle2 = native_run_pair(ptr2, vec2, wts2, flag2, o2, ow2,
                                       10.0)
    assert bits(native2) == bits(oracle2)
    assert float(oracle2[1]) == 1.0  # assigned via unflagged arc (no push)
    assert float(oracle2[2]) == 5.0


@needs_core
def test_seed_gates_strict_and_nan():
    """Seed push gates: strict ``< cutoff`` under the FAST law — NaN and
    negative seeds pass, seeds at cutoff and oversized seeds do not;
    NaN labels compare byte-exactly through bits()."""
    ptr, vec, wts, flag = tiny_graph([[2], [], []], [[7.0], [], []],
                                     [[True], [], []])
    o = np.array([0, 1], dtype=np.int64)
    rows = [
        ("end_at_cutoff", [0.0, 10.0], 10.0, 7.0),
        ("end_below_cutoff", [0.0, 9.999], 9.999, 7.0),
        ("nan_end", [0.0, float("nan")], float("nan"), 7.0),
        ("neg_end", [0.0, -4.0], -4.0, 7.0),
        ("big_end", [0.0, 25.0], 25.0, 7.0),
    ]
    for label, ow_vals, expect_end, expect_arc in rows:
        ow = np.array(ow_vals, dtype=np.float64)
        native, oracle = native_run_pair(ptr, vec, wts, flag, o, ow, 10.0)
        assert bits(native) == bits(oracle), label
        got_end = float(oracle[1])
        if math.isnan(expect_end):
            assert math.isnan(got_end), label
        else:
            assert got_end == expect_end, label
        assert float(oracle[2]) == expect_arc, label


@needs_core
def test_sentinel_bits_match():
    """The native sentinel is ``1.0 + cutoff`` (one binary64 RN add),
    bit-for-bit for adversarial cutoffs on the non-seed slots."""
    for cutoff in (0.0, -0.0, 5e-324, 1.7976931348623157e308, 25.0):
        ptr = np.array([0, 0, 0, 0, 0], dtype=np.int64)
        vec = np.empty(0, dtype=np.int64)
        wts = np.empty(0, dtype=np.float64)
        flag = np.empty(0, dtype=np.bool_)
        o = np.array([0, 3], dtype=np.int64)
        ow = np.array([9e9, 9e9], dtype=np.float64)
        native, oracle = native_run_pair(ptr, vec, wts, flag, o, ow, cutoff)
        assert bits(native) == bits(oracle), repr(cutoff)
        want = np.ones(ptr.shape[0] - 1, dtype=np.float64) + cutoff
        assert bits(oracle[[1, 2]]) == bits(want[[1, 2]]), repr(cutoff)


# ---------------------------------------------------------------------------
# Whole driver: routes, four arrays, engagement honesty
# ---------------------------------------------------------------------------

FOLD = dict(gravity_beta=0.02, gravity_plateau=5.0,
            gravity_logistic_midpoint=8.0, gravity_growth_rate=0.3,
            knn_decay="exponential",
            knn_weights=np.array([0.9, 0.5, 0.25, 0.1]))


def fold_args(c):
    return (c["o"], c["ow"], c["ptr"], c["vec"], c["wts"], c["flag"],
            c["d"], c["dw"], c["dwt"],
            FOLD["gravity_beta"], FOLD["gravity_plateau"],
            FOLD["gravity_logistic_midpoint"], FOLD["gravity_growth_rate"],
            FOLD["knn_decay"], FOLD["knn_weights"], c["cutoff"])


@needs_core
@pytest.mark.parametrize("name", CASE_PARAMS)
def test_driver_bitwise_route_and_engagement(name, engines):
    """Native-engaged driver == engine driver: route tag, four output
    arrays byte-for-byte; engagement honestly reported."""
    if name == "weights_f32":
        pytest.skip("guard-refused domain: compact fallback (delegated)")
    c = build_case(3 + CASE_PARAMS.index(name), **case_kwargs(name))
    engine = engines.integrated(*fold_args(c))

    run = route_scope_access("una_legacy", *fold_args(c))

    assert run.native_engaged is True
    assert run.fallback_reason is None
    n_count = c["ptr"].shape[0] - 1
    admitted, _ = SA.scope_route_admits(
        c["ptr"], c["vec"], c["wts"], c["flag"], c["o"], c["ow"],
        c["cutoff"])
    expected_route = (SA.ROUTE_A3
                      if SA.tail_admits(c["d"], c["dw"].shape[0], n_count)
                      else SA.ROUTE_A1) if admitted else SA.ROUTE_COMPACT
    assert run.route == expected_route
    assert run.result.route == expected_route
    assert bits(run.result.reach) == bits(engine[0])
    assert bits(run.result.gravity_exponential) == bits(engine[1])
    assert bits(run.result.gravity_logistic) == bits(engine[2])
    assert bits(run.result.knn_access) == bits(engine[3])
    assert run.result.reach.dtype == c["o"].dtype
    # Engagement honesty: core counters present and positive on real
    # work; the fingerprint binds the result to the actual binary.
    assert run.engagement["heap_pops"] > 0
    assert run.engagement["rows_scanned"] == run.engagement["heap_pops"] - 1
    assert run.engagement["first_incomplete_origin"] == 2**63 - 1
    assert "g++" in run.engagement["binary_fingerprint"]
    assert run.outputs.status == "complete"
    assert [a.name for a in run.outputs.arrays] == [
        "reach", "gravity_exponential", "gravity_logistic", "knn_access"]


@needs_core
@pytest.mark.parametrize("seed", [21, 22])
def test_driver_fallback_delegation_bitwise(seed):
    """Guard-refusing domains route compact — on the pinned oracle, with
    the refusal recorded and no native engagement (never a partial
    native re-implementation of the compact route)."""
    c = build_case(seed, weights_f32=(seed == 21))
    if seed == 22:  # malformed pointer tail forces the fallback
        ptr = c["ptr"].copy()
        ptr[-1] = ptr[-1] - 1
        c = dict(c, ptr=ptr)
    oracle = SA.integrated_scope_access_oracle(*fold_args(c))
    run = route_scope_access("una_legacy", *fold_args(c))
    assert run.native_engaged is False
    assert run.fallback_reason is not None
    assert "route-compact" in run.fallback_reason
    assert run.route == SA.ROUTE_COMPACT
    assert bits(run.result.reach) == bits(oracle.reach)
    assert bits(run.result.gravity_exponential) == \
        bits(oracle.gravity_exponential)
    assert bits(run.result.gravity_logistic) == bits(oracle.gravity_logistic)
    assert bits(run.result.knn_access) == bits(oracle.knn_access)
    assert run.engagement == {}


@needs_core
def test_driver_noncontiguous_delegates():
    """An admitted-but-strided array never reaches the pointer-reading
    core: the policy delegates and the oracle result stands."""
    c = build_case(5)
    wts_nc = np.repeat(c["wts"], 2)[::2]  # same values, 16-byte stride
    assert wts_nc.dtype == np.float64 and wts_nc.shape == c["wts"].shape
    assert (wts_nc == c["wts"]).all()
    assert not wts_nc.flags["C_CONTIGUOUS"]
    c_nc = dict(c, wts=wts_nc)
    args = fold_args(c_nc)
    oracle = SA.integrated_scope_access_oracle(*args)
    run = route_scope_access("una_legacy", *args)
    assert run.native_engaged is False
    assert "layout" in run.fallback_reason
    assert bits(run.result.reach) == bits(oracle.reach)


@needs_core
def test_driver_a1_negative_terminal_delegates():
    """A1 route with a destination terminal outside [0, V + d): Python
    negative-index semantics belong to the oracle, not the core."""
    c = build_case(5)
    d = c["d"].copy()
    d[0, 0] = -1  # tail_admits refuses; A1 route; read-domain refusal
    c = dict(c, d=d)
    args = fold_args(c)
    oracle = SA.integrated_scope_access_oracle(*args)
    run = route_scope_access("una_legacy", *args)
    n_count = c["ptr"].shape[0] - 1
    assert SA.tail_admits(c["d"], c["dw"].shape[0], n_count) is False
    assert run.route == SA.ROUTE_A1
    assert run.native_engaged is False
    assert "a1-read-domain" in run.fallback_reason
    assert bits(run.result.reach) == bits(oracle.reach)


@needs_core
def test_driver_native_absent_fallback(monkeypatch):
    """With the loader disabled the legacy route IS the pinned oracle —
    same objects' bits, honest reason, no engagement claims."""
    from urban_network_analysis.backends.native import _loader
    c = build_case(5)
    args = fold_args(c)
    oracle = SA.integrated_scope_access_oracle(*args)
    monkeypatch.setenv("UNA_NATIVE_CORE_DISABLE", "1")
    _loader.reset_loader_cache()
    try:
        run = route_scope_access("una_legacy", *args)
    finally:
        _loader.reset_loader_cache()
    assert run.native_engaged is False
    assert run.fallback_reason.startswith("native-unavailable:")
    assert bits(run.result.reach) == bits(oracle.reach)
    assert bits(run.result.knn_access) == bits(oracle.knn_access)
