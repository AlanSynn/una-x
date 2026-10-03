"""Schedule pinning: oracle vs compiled engines, bitwise (CPU_ALGORITHMS).

The oracle in ``urban_network_analysis.kernels.scope_access`` claims
bit-identical execution of the characterized una_legacy scope search.
These tests are that claim, machine-checked per platform: the fast
comparison law, the heap order law, search labels on the case battery,
the driver ladder (routes + four output arrays), admission mirrors,
and the characterized structural semantics (seed gates, duplicate
arcs, self-loops, leaf/flag push gates, sentinel bits).
"""
import math
import struct
import subprocess
import sys

import numpy as np
import pytest
from numba import njit

from urban_network_analysis.Engines import _large_access_scratch as LAS
from urban_network_analysis.Engines.AccessibilityWElevation import (
    compact_vector_node_view_scope,
    integrated_scope_access,
)
from urban_network_analysis.kernels import scope_access as SA

from conftest import CASE_PARAMS, REPO_SRC, bits, build_case, case_kwargs


def wb(x):
    """Byte key for one float (NaN-safe equality)."""
    return struct.pack(">d", float(x))


def item_key(item):
    return (wb(item[0]), int(item[1]))


# ---------------------------------------------------------------------------
# Fast comparison law (scope_access module docstring)
# ---------------------------------------------------------------------------

@njit(cache=False, fastmath=True)
def _engine_lt_le(a, b):
    return (a < b, a <= b)


PROBE_POINTS = [
    ("finite_lt_nan", 1.0, float("nan")),
    ("nan_lt_finite", float("nan"), 1.0),
    ("nan_lt_nan", float("nan"), float("nan")),
    ("descending_finites", 25.0, 1.0),
    ("ascending_finites", 1.0, 25.0),
    ("mz_lt_pz", -0.0, 0.0),
    ("pz_lt_mz", 0.0, -0.0),
    ("inf_lt_nan", float("inf"), float("nan")),
    ("ninf_lt_nan", float("-inf"), float("nan")),
    ("equal_finites", 7.0, 7.0),
]


@pytest.mark.parametrize("name,a,b",
                         PROBE_POINTS, ids=[p[0] for p in PROBE_POINTS])
def test_fast_comparison_law_matches_engine(name, a, b):
    """The oracle's fast_lt/fast_le reproduce the engine's compiled
    ``<``/``<=`` on every comparison form the search schedule uses."""
    eng_lt, eng_le = _engine_lt_le(np.float64(a), np.float64(b))
    assert bool(eng_lt) == SA.fast_lt(a, b)
    assert bool(eng_le) == SA.fast_le(a, b)


def test_nan_comparison_law_semantics():
    """The laws' distinctive rows, documentation-by-test: NaN
    candidates pass eligibility; NaN labels are improvable; NaN entries
    sort as less in the heap; ordinary and signed-zero order preserved."""
    nan = float("nan")
    assert SA.fast_le(nan, 25.0) is True    # NaN <= cutoff -> eligible
    assert SA.fast_lt(4.0, nan) is True     # finite < NaN label -> improves
    assert SA.fast_lt(nan, nan) is True     # NaN < NaN (heap law)
    assert SA.fast_lt(25.0, 1.0) is False   # ordinary order preserved
    assert SA.fast_lt(-0.0, 0.0) is False   # signed zeros: not less
    assert SA.fast_le(0.0, -0.0) is True


# ---------------------------------------------------------------------------
# Heap order law
# ---------------------------------------------------------------------------

from heapq import heappop as _engine_heappop
from heapq import heappush as _engine_heappush


@njit(cache=False, fastmath=True)
def _engine_heap_sequence(wn):
    queue = [(wn[0, 0], wn[0, 1])]
    for i in range(1, wn.shape[0]):
        _engine_heappush(queue, (wn[i, 0], wn[i, 1]))
    out = []
    while len(queue):
        out.append(_engine_heappop(queue))
    return out


HEAP_SEQUENCES = {
    "nan_among_finites": [(2.0, 5), (float("nan"), 1), (2.0, 3), (1.0, 8)],
    "nan_first_push": [(1.5, 7), (float("nan"), 3), (0.5, 9)],
    "tie_heavy": [(2.0, 5), (2.0, 3), (2.0, 9), (1.0, 8), (1.0, 2)],
}


@pytest.mark.parametrize("name", list(HEAP_SEQUENCES), ids=list(HEAP_SEQUENCES))
def test_heap_order_law_matches_engine(name):
    """The oracle heap reproduces the compiled heapq's push/pop order
    byte-exactly, including NaN-weight entries (characterized tuple
    law; see SA._tuple_lt docstring)."""
    seq = HEAP_SEQUENCES[name]
    wn = np.array([[float(w), float(n)] for w, n in seq], dtype=np.float64)
    engine_items = _engine_heap_sequence(wn)
    heap = [(float(seq[0][0]), int(seq[0][1]))]
    for w, n in seq[1:]:
        SA._heap_push(heap, (float(w), int(n)))
    oracle_items = [SA._heap_pop(heap) for _ in range(len(seq))]
    engine_keys = [(wb(w), int(n)) for w, n in engine_items]
    oracle_keys = [(wb(w), int(n)) for w, n in oracle_items]
    assert engine_keys == oracle_keys


# ---------------------------------------------------------------------------
# Search schedule vs compiled kernels (bitwise labels)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("domain", ["a3", "a1"])
def test_search_labels_bitwise(case, domain):
    """Oracle labels == engine A3/A1 kernel labels, byte-for-byte."""
    admitted, max_degree = LAS._a1_scope_admits(
        case["ptr"], case["vec"], case["wts"], case["flag"],
        case["o"], case["ow"], case["cutoff"])
    if not admitted:
        pytest.skip("guard refuses this dtype domain (fallback case)")
    d_count = case["dw"].shape[0]
    scratch_off = np.empty(max_degree, dtype=np.int64)
    scratch_w = np.empty(max_degree, dtype=np.float64)
    if domain == "a3":
        engine_labels = LAS._a3_scope_search_tailless(
            case["o"][0], case["ow"][0], case["ptr"], case["vec"],
            case["wts"], case["flag"], case["cutoff"],
            scratch_off, scratch_w)[0]
    else:
        engine_labels = LAS._a1_scope_search(
            case["o"][0], case["ow"][0], case["ptr"], case["vec"],
            case["wts"], case["flag"], case["cutoff"], d_count,
            scratch_off, scratch_w)[0]
    oracle_labels = SA.scope_search_schedule(
        case["o"][0], case["ow"][0], case["ptr"], case["vec"],
        case["wts"], case["flag"], case["cutoff"],
        d_count=d_count, domain=domain)
    assert bits(engine_labels) == bits(oracle_labels)


def test_a1_a3_prefix_agreement(case):
    """On admitted inputs the A1 tail is never read: A1 and A3 labels
    agree over the network-node domain (A3I proof, pinned live)."""
    admitted, max_degree = LAS._a1_scope_admits(
        case["ptr"], case["vec"], case["wts"], case["flag"],
        case["o"], case["ow"], case["cutoff"])
    if not admitted:
        pytest.skip("fallback-forced case")
    d_count = case["dw"].shape[0]
    scratch_off = np.empty(max_degree, dtype=np.int64)
    scratch_w = np.empty(max_degree, dtype=np.float64)
    a3 = LAS._a3_scope_search_tailless(
        case["o"][0], case["ow"][0], case["ptr"], case["vec"],
        case["wts"], case["flag"], case["cutoff"],
        scratch_off, scratch_w)[0]
    a1 = LAS._a1_scope_search(
        case["o"][0], case["ow"][0], case["ptr"], case["vec"],
        case["wts"], case["flag"], case["cutoff"], d_count,
        scratch_off, scratch_w)[0]
    v = case["ptr"].shape[0] - 1
    assert bits(a1[:v]) == bits(a3)


def test_admission_mirror_matches_engine_guard(case):
    """The oracle admission decision equals the engine guard's
    pure-Python dispatch body on the same inputs."""
    engine = LAS._a1_scope_admits(
        case["ptr"], case["vec"], case["wts"], case["flag"],
        case["o"], case["ow"], case["cutoff"])
    oracle = SA.scope_route_admits(
        case["ptr"], case["vec"], case["wts"], case["flag"],
        case["o"], case["ow"], case["cutoff"])
    assert (bool(engine[0]), int(engine[1])) == (bool(oracle[0]), int(oracle[1]))


@pytest.mark.parametrize("name", CASE_PARAMS)
def test_admission_mirror_adversarial(name):
    """The mirror refuses exactly where the engine guard refuses, on
    hand-built malformed inputs (not just the standard battery)."""
    base = build_case(5)
    good = (base["ptr"], base["vec"], base["wts"], base["flag"],
            base["o"], base["ow"], base["cutoff"])
    engine = LAS._a1_scope_admits(*good)
    oracle = SA.scope_route_admits(*good)
    assert (bool(engine[0]), int(engine[1])) == \
        (bool(oracle[0]), int(oracle[1]))

    bad_cases = {
        "short_weights": (base["ptr"], base["vec"], base["wts"][:-1],
                          base["flag"], base["o"], base["ow"],
                          base["cutoff"]),
        "bad_ptr0": (base["ptr"] + 1, base["vec"], base["wts"], base["flag"],
                     base["o"], base["ow"], base["cutoff"]),
        "ptr_end_mismatch": (
            np.concatenate([base["ptr"][:-1], [base["ptr"][-1] + 1]]),
            base["vec"], base["wts"], base["flag"], base["o"], base["ow"],
            base["cutoff"]),
        "oob_terminal": (base["ptr"], base["vec"], base["wts"], base["flag"],
                         np.array([[0, 10 ** 6]], dtype=np.int64),
                         base["ow"], base["cutoff"]),
        "neg_cutoff": (base["ptr"], base["vec"], base["wts"], base["flag"],
                       base["o"], base["ow"], -1.0),
        "nan_cutoff": (base["ptr"], base["vec"], base["wts"], base["flag"],
                       base["o"], base["ow"], float("nan")),
        "inf_cutoff": (base["ptr"], base["vec"], base["wts"], base["flag"],
                       base["o"], base["ow"], float("inf")),
        "nan_cost": (
            base["ptr"], base["vec"],
            np.concatenate([base["wts"][:-1], [np.nan]]),
            base["flag"], base["o"], base["ow"], base["cutoff"]),
        "neg_cost": (
            base["ptr"], base["vec"],
            np.concatenate([base["wts"][:-1], [-0.5]]),
            base["flag"], base["o"], base["ow"], base["cutoff"]),
        "bool_cutoff": (base["ptr"], base["vec"], base["wts"], base["flag"],
                        base["o"], base["ow"], True),
    }
    for label, args in bad_cases.items():
        engine = LAS._a1_scope_admits(*args)
        oracle = SA.scope_route_admits(*args)
        assert (bool(engine[0]), int(engine[1])) == \
            (bool(oracle[0]), int(oracle[1])), label


# ---------------------------------------------------------------------------
# Driver ladder: routes and bitwise outputs
# ---------------------------------------------------------------------------

FOLD_KW = dict(gravity_beta=0.02, gravity_plateau=5.0,
               gravity_logistic_midpoint=8.0, gravity_growth_rate=0.3,
               knn_decay="exponential",
               knn_weights=np.array([0.9, 0.5, 0.25, 0.1]))


@pytest.mark.parametrize("name", CASE_PARAMS)
def test_driver_bitwise_and_route(name):
    """Oracle driver == engine driver: four output arrays byte-for-byte
    and the same ladder route (A3 when the tail admits, else A1)."""
    if name == "weights_f32":
        pytest.skip("fallback-forced: covered by test_driver_fallback_bitwise")
    kwargs = case_kwargs(name)
    c = build_case(3 + CASE_PARAMS.index(name), **kwargs)
    engine = integrated_scope_access(
        c["o"], c["ow"], c["ptr"], c["vec"], c["wts"], c["flag"],
        c["d"], c["dw"], c["dwt"],
        FOLD_KW["gravity_beta"], FOLD_KW["gravity_plateau"],
        FOLD_KW["gravity_logistic_midpoint"], FOLD_KW["gravity_growth_rate"],
        FOLD_KW["knn_decay"], FOLD_KW["knn_weights"], c["cutoff"])
    result = SA.integrated_scope_access_oracle(
        c["o"], c["ow"], c["ptr"], c["vec"], c["wts"], c["flag"],
        c["d"], c["dw"], c["dwt"],
        FOLD_KW["gravity_beta"], FOLD_KW["gravity_plateau"],
        FOLD_KW["gravity_logistic_midpoint"], FOLD_KW["gravity_growth_rate"],
        FOLD_KW["knn_decay"], FOLD_KW["knn_weights"], c["cutoff"])
    d_count = c["dw"].shape[0]
    n_count = c["ptr"].shape[0] - 1
    admitted, _ = LAS._a1_scope_admits(
        c["ptr"], c["vec"], c["wts"], c["flag"], c["o"], c["ow"],
        c["cutoff"])
    expected_route = (SA.ROUTE_A3 if SA.tail_admits(c["d"], d_count, n_count)
                      else SA.ROUTE_A1) if admitted else SA.ROUTE_COMPACT
    assert result.route == expected_route
    assert bits(result.reach) == bits(engine[0])
    assert bits(result.gravity_exponential) == bits(engine[1])
    assert bits(result.gravity_logistic) == bits(engine[2])
    assert bits(result.knn_access) == bits(engine[3])
    assert result.reach.dtype == c["o"].dtype


@pytest.mark.parametrize("seed", [21, 22])
def test_driver_fallback_bitwise(seed):
    """Guard-refusing value domains route to the compact route; the
    oracle's compact route matches the engine's fallback kernel
    bit-for-byte (schedule equivalence across code shapes)."""
    c = build_case(seed, weights_f32=(seed == 21))
    if seed == 22:  # force fallback through a malformed pointer tail
        ptr = c["ptr"].copy()
        ptr[-1] = ptr[-1] - 1  # pointer[V] != E
        c = dict(c, ptr=ptr)
    engine = integrated_scope_access(
        c["o"], c["ow"], c["ptr"], c["vec"], c["wts"], c["flag"],
        c["d"], c["dw"], c["dwt"],
        FOLD_KW["gravity_beta"], FOLD_KW["gravity_plateau"],
        FOLD_KW["gravity_logistic_midpoint"], FOLD_KW["gravity_growth_rate"],
        FOLD_KW["knn_decay"], FOLD_KW["knn_weights"], c["cutoff"])
    result = SA.integrated_scope_access_oracle(
        c["o"], c["ow"], c["ptr"], c["vec"], c["wts"], c["flag"],
        c["d"], c["dw"], c["dwt"],
        FOLD_KW["gravity_beta"], FOLD_KW["gravity_plateau"],
        FOLD_KW["gravity_logistic_midpoint"], FOLD_KW["gravity_growth_rate"],
        FOLD_KW["knn_decay"], FOLD_KW["knn_weights"], c["cutoff"])
    assert result.route == SA.ROUTE_COMPACT
    assert bits(result.reach) == bits(engine[0])
    assert bits(result.gravity_exponential) == bits(engine[1])
    assert bits(result.gravity_logistic) == bits(engine[2])
    assert bits(result.knn_access) == bits(engine[3])


def test_driver_float32_terminals_characterized():
    """CHARACTERIZED ENGINE LIMIT: float32 origin-terminal weights fall
    outside the guard's dtype domain; the engine's fallback kernel then
    fails to compile the mixed-dtype seed heap (float32 label pops vs
    float64 edge candidates) and numba raises a TypingError.  Recorded
    as the engine's boundary behavior — the oracle refuses such inputs
    with ValueError instead of mirroring a compilation failure."""
    c = build_case(23, terminals_f32=True)
    with pytest.raises(Exception):
        integrated_scope_access(
            c["o"], c["ow"], c["ptr"], c["vec"], c["wts"], c["flag"],
            c["d"], c["dw"], c["dwt"],
            FOLD_KW["gravity_beta"], FOLD_KW["gravity_plateau"],
            FOLD_KW["gravity_logistic_midpoint"],
            FOLD_KW["gravity_growth_rate"], FOLD_KW["knn_decay"],
            FOLD_KW["knn_weights"], c["cutoff"])
    with pytest.raises(ValueError):
        SA.integrated_scope_access_oracle(
            c["o"], c["ow"], c["ptr"], c["vec"], c["wts"], c["flag"],
            c["d"], c["dw"], c["dwt"],
            FOLD_KW["gravity_beta"], FOLD_KW["gravity_plateau"],
            FOLD_KW["gravity_logistic_midpoint"],
            FOLD_KW["gravity_growth_rate"], FOLD_KW["knn_decay"],
            FOLD_KW["knn_weights"], c["cutoff"])


# ---------------------------------------------------------------------------
# Characterized structural semantics (SPEC.md section-3 rules, pinned
# against the engine's fallback kernel — the same schedule, compiled)
# ---------------------------------------------------------------------------

def tiny_graph(arcs_by_node, weights_by_node, flags_by_node):
    """Deterministic CSR from per-node arc lists."""
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


def run_pair(ptr, vec, wts, flag, o, ow, cutoff):
    engine = compact_vector_node_view_scope(
        o, ow, ptr, vec, wts, flag, cutoff, 0)[0]
    oracle = SA.scope_search_schedule(
        o, ow, ptr, vec, wts, flag, cutoff, d_count=0, domain="a3")
    return engine, oracle


def test_duplicate_arc_row_order_semantics():
    """Two parallel arcs to the same node are both snapshot-eligible;
    phase two assigns in row order, so the SECOND arc's candidate wins
    even when larger — the row-order-overwrite quirk, pinned."""
    ptr, vec, wts, flag = tiny_graph([[1, 1], []], [[2.0, 1.0], []],
                                     [[True, True], []])
    o = np.array([0, 1], dtype=np.int64)
    ow = np.array([0.0, 9e9], dtype=np.float64)
    engine, oracle = run_pair(ptr, vec, wts, flag, o, ow, 10.0)
    assert bits(engine) == bits(oracle)
    assert float(oracle[1]) == 1.0  # second (smaller) arc wins

    ptr2, vec2, wts2, flag2 = tiny_graph([[1, 1], []], [[1.0, 2.0], []],
                                         [[True, True], []])
    engine2, oracle2 = run_pair(ptr2, vec2, wts2, flag2, o, ow, 10.0)
    assert bits(engine2) == bits(oracle2)
    assert float(oracle2[1]) == 2.0  # second, LARGER arc overwrites: quirk


def test_self_loop_semantics():
    """A self-candidate can rewrite the node's own label only by strict
    fast improvement, then pushes only under flag AND degree > 1."""
    ptr, vec, wts, flag = tiny_graph([[0, 1], [], []],
                                     [[5.0, 1.0], [], []],
                                     [[True, True], [], []])
    o = np.array([0, 2], dtype=np.int64)
    ow = np.array([0.0, 9e9], dtype=np.float64)
    engine, oracle = run_pair(ptr, vec, wts, flag, o, ow, 10.0)
    assert bits(engine) == bits(oracle)
    # Seed 0.0 at node 0.  The self-arc candidate is 5.0 + 0.0 = 5.0,
    # which IS eligible (5.0 <= 10) but is NOT < the 0.0 seed label, so
    # the self-loop never rewrites the seed (engine-verified: 0.0).
    # The arc to leaf node 1 assigns 1.0; degree 1 blocks the push.
    assert float(oracle[0]) == 0.0
    assert float(oracle[1]) == 1.0
    assert float(oracle[2]) == 9e9  # the unconditional end-seed write


def test_leaf_push_gate_semantics():
    """A flagged degree-1 neighbor is assigned but never pushed, so its
    own neighborhood stays at sentinel — observable two hops away
    (engine-verified graph: seeds at 0 and 3, chain 0->1->2)."""
    ptr, vec, wts, flag = tiny_graph([[1], [2], [], []],
                                     [[1.0], [1.0], [], []],
                                     [[True], [True], [], []])
    o = np.array([0, 3], dtype=np.int64)
    ow = np.array([0.0, 9e9], dtype=np.float64)
    engine, oracle = run_pair(ptr, vec, wts, flag, o, ow, 10.0)
    assert bits(engine) == bits(oracle)
    assert float(oracle[1]) == 1.0           # assigned via flagged arc
    assert float(oracle[2]) == 11.0          # sentinel: 1 never pushed
    assert float(oracle[3]) == 9e9           # end-seed write stands


def test_flag_false_push_gate_semantics():
    """An unflagged arc never pushes its (even improved) neighbor."""
    ptr, vec, wts, flag = tiny_graph([[1, 2], [], []],
                                     [[1.0, 5.0], [], []],
                                     [[False, True], [], []])
    o = np.array([0, 2], dtype=np.int64)
    ow = np.array([0.0, 9e9], dtype=np.float64)
    engine, oracle = run_pair(ptr, vec, wts, flag, o, ow, 10.0)
    assert bits(engine) == bits(oracle)
    assert float(oracle[1]) == 1.0  # assigned via unflagged arc
    assert float(oracle[2]) == 5.0  # assigned via flagged arc


def test_seed_gates_strict_and_nan():
    """Seed push gates are strict ``< cutoff`` under the FAST law: a
    seed exactly at cutoff is not pushed; NaN and negative seeds ARE
    (characterized); oversized seeds are not.  Node 1 is isolated so
    the end-seed label is directly observable."""
    ptr, vec, wts, flag = tiny_graph([[2], [], []], [[7.0], [], []],
                                     [[True], [], []])
    o = np.array([0, 1], dtype=np.int64)
    rows = [
        ("end_at_cutoff", [0.0, 10.0], 10.0, 7.0),
        ("end_below_cutoff", [0.0, 9.999], 9.999, 7.0),
        ("nan_end", [0.0, float("nan")], float("nan"), 7.0),
        ("neg_end", [0.0, -4.0], -4.0, 7.0),
        ("big_end", [0.0, 25.0], 25.0, 7.0),  # seed writes above sentinel
    ]
    for label, ow_vals, expect_end, expect_arc in rows:
        ow = np.array(ow_vals, dtype=np.float64)
        engine, oracle = run_pair(ptr, vec, wts, flag, o, ow, 10.0)
        assert bits(engine) == bits(oracle), label
        got_end = float(oracle[1])
        if math.isnan(expect_end):
            assert math.isnan(got_end), label
        else:
            assert got_end == expect_end, label
        assert float(oracle[2]) == expect_arc, label


def test_nan_seed_pair_nontermination_bounded():
    """CHARACTERIZED NONTERMINATION (not a defect fix): with NaN on BOTH
    origin seeds the schedule has no monotone potential left — a NaN
    candidate improves ANY finite label (not (finite <= NaN)) and any
    finite candidate improves a NaN label (not (NaN <= finite)) — so
    labels oscillate and the heap pumps.  Reproduced on the COMPILED
    engine (evidence cpuk-20261003T0129Z/engine_nan_seed_nontermination,
    exit 124 after 280 s on the cached-compile kernel); here the same
    battery case is pinned on the oracle under a bounded pop cap."""
    c = build_case(5, nan_seed_start=True, nan_seed=True)
    real_pop = SA._heap_pop
    count = {"n": 0}

    def capped_pop(heap):
        if count["n"] >= 100_000:
            raise RuntimeError(
                "nontermination: label oscillation under NaN seeds "
                "(characterized; SPEC.md section 3)")
        count["n"] += 1
        return real_pop(heap)

    SA._heap_pop = capped_pop
    try:
        with pytest.raises(RuntimeError, match="nontermination"):
            SA.scope_search_schedule(
                c["o"][0], c["ow"][0], c["ptr"], c["vec"], c["wts"],
                c["flag"], c["cutoff"], d_count=0, domain="a3")
    finally:
        SA._heap_pop = real_pop
    assert count["n"] >= 100_000


def test_sentinel_bits_match_engine():
    """The sentinel is ``np.ones(domain, dtype) + cutoff`` bit-for-bit
    for adversarial cutoffs (subnormal, huge, negative zero), on the
    non-seed slots (seeds 9e9 sit above every cutoff here, so slots 0
    and 3 are overwritten by the unconditional seed writes — the
    characterized sentinel applies to the remaining nodes)."""
    for cutoff in (0.0, -0.0, 5e-324, 1.7976931348623157e308, 25.0):
        ptr = np.array([0, 0, 0, 0, 0], dtype=np.int64)
        o = np.array([0, 3], dtype=np.int64)
        ow = np.array([9e9, 9e9], dtype=np.float64)
        engine, oracle = run_pair(ptr, np.empty(0, dtype=np.int64),
                                  np.empty(0, dtype=np.float64),
                                  np.empty(0, dtype=np.bool_),
                                  o, ow, cutoff)
        assert bits(engine) == bits(oracle), repr(cutoff)
        want = np.ones(ptr.shape[0] - 1, dtype=np.float64) + cutoff
        assert bits(oracle[[1, 2]]) == bits(want[[1, 2]]), repr(cutoff)


def test_interpreted_mode_admission_mirror():
    """NUMBA_DISABLE_JIT=1 subprocess: the engine guard's interpreted
    body and the oracle mirror make the same admission decisions."""
    code = (
        "import os; os.environ['NUMBA_DISABLE_JIT']='1'; "
        "import sys; sys.path.insert(0, {src!r}); sys.path.insert(0, {tst!r}); "
        "from conftest import build_case; "
        "c = build_case(5); "
        "from urban_network_analysis.Engines import _large_access_scratch as L; "
        "from urban_network_analysis.kernels import scope_access as S; "
        "e = L._a1_scope_admits(c['ptr'], c['vec'], c['wts'], c['flag'], "
        "c['o'], c['ow'], c['cutoff']); "
        "o = S.scope_route_admits(c['ptr'], c['vec'], c['wts'], c['flag'], "
        "c['o'], c['ow'], c['cutoff']); "
        "assert (bool(e[0]), int(e[1])) == (bool(o[0]), int(o[1])); "
        "print('MIRROR OK')\n"
    ).format(src=str(REPO_SRC),
             tst=str(REPO_SRC.parent / "tests" / "cpu_kernels"))
    env = dict(os.environ, NUMBA_DISABLE_JIT="1", PYTHONHASHSEED="0")
    proc = subprocess.run([sys.executable, "-c", code], env=env,
                          capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert "MIRROR OK" in proc.stdout


import os  # noqa: E402  (used only by the subprocess test above)
