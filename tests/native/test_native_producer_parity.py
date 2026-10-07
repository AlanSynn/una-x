"""N00 parity: the native producer record is bit-exact vs numba.

Every test runs the numba ``build_trace`` and the native twin over
IDENTICAL inputs and demands indistinguishable TraceRecords: identity
metadata, packed geometry bytes, content digest, refusal law, and all
counters (tile_temp_bytes excepted by declared design — each route
measures its own scratch).  Bitwise parity per una_legacy/
madina_legacy/corrected_v1: no tolerance anywhere, float bits compared
via tobytes.
"""
import numpy as np
import pytest

from conftest import (CASE_PARAMS, assert_parity, bits, build_case,
                      case_kwargs, record_diffs, ring_case, run_pair)


# --------------------------------------------------------------------------
# The adversarial battery (single source, same ids as the S11 suite):
# equal duplicates, ties, self loops, zero cutoff, same-terminal pairs.
# Refusing battery members (nan/neg seeds are NOT in CASE_PARAMS) are
# covered by the refusal-parity test below.
# --------------------------------------------------------------------------
@pytest.mark.parametrize("name", CASE_PARAMS, ids=CASE_PARAMS)
def test_battery_parity(native_pair, name):
    case = build_case(3 + CASE_PARAMS.index(name), **case_kwargs(name))
    for tile_origins in (64, 3, 1):
        ref, nat = run_pair(native_pair, case, tile_origins=tile_origins)
        assert_parity(ref, nat)


def test_battery_parity_dominated_ties(native_pair):
    """All-equal edge weights: every relaxation distance ties, so the
    heap's total order and the min() tie law are exercised together."""
    case = ring_case(seed=7, equal_weights=True)
    for tile_origins in (64, 7):
        ref, nat = run_pair(native_pair, case, tile_origins=tile_origins)
        assert_parity(ref, nat)


def test_equal_duplicate_edges(native_pair):
    """Duplicate arcs carrying EQUAL weights: the py_min tie law (b < a,
    keep a) must match Python's min() exactly on the label update."""
    case = build_case(11, dup=True, self_loop=True)
    vec = case["vec"]
    wts = case["wts"]
    # Duplicate every arc with its identical weight.
    case["vec"] = np.concatenate([vec, vec])
    case["wts"] = np.concatenate([wts, wts])
    case["flag"] = np.concatenate([case["flag"], case["flag"]])
    ptr = np.zeros(case["ptr"].shape[0], dtype=np.int64)
    ptr[1:] = np.cumsum(np.bincount(
        case["vec"], minlength=case["ptr"].shape[0] - 1))
    case["ptr"] = ptr
    ref, nat = run_pair(native_pair, case)
    assert_parity(ref, nat)


def test_negative_zero_weights(native_pair):
    """-0.0 vs +0.0 must produce identical BITS in both routes (the
    distance add 1.0 + -0.0 == 1.0 + 0.0, and the comparator treats
    them equal) — bit identity, not numeric identity claims."""
    case = build_case(5)
    case["wts"] = np.where(case["wts"] < 3.0, -0.0, case["wts"])
    case["flag"] = np.ones(case["flag"].shape[0], dtype=bool)
    ref, nat = run_pair(native_pair, case)
    assert_parity(ref, nat)
    if ref.complete:
        assert bits(ref.distance_bits) == bits(nat.distance_bits)


def test_empty_destinations(native_pair):
    """d_count = 0: reverse incidence is empty, no candidates anywhere,
    zero-length outputs — both routes, both tiling regimes."""
    case = build_case(13)
    case["d"] = np.zeros((0, 2), dtype=np.int64)
    case["dw"] = np.zeros((0, 2), dtype=np.float64)
    case["dwt"] = np.zeros(0, dtype=np.float64)
    for tile_origins in (64, 1):
        ref, nat = run_pair(native_pair, case, tile_origins=tile_origins)
        assert_parity(ref, nat)
        assert ref.counters.candidates_total == 0


def test_single_node_no_edges(native_pair):
    """N=1, E=0, one origin at the node, one destination at the node:
    the degenerate CSR boundary."""
    case = dict(
        o=np.array([[0, 0]], dtype=np.int64),
        ow=np.array([[0.5, 0.5]]),
        ptr=np.array([0, 0], dtype=np.int64),
        vec=np.zeros(0, dtype=np.int64),
        wts=np.zeros(0, dtype=np.float64),
        flag=np.zeros(0, dtype=bool),
        d=np.array([[0, 0]], dtype=np.int64),
        dw=np.array([[1.0, 1.0]]),
        dwt=np.array([0.5]),
        cutoff=2.0)
    ref, nat = run_pair(native_pair, case)
    assert_parity(ref, nat)


def test_destination_equals_origin(native_pair):
    """A destination pair that includes the origin's own nodes:
    distance 0.0 candidates (exactly-representable, sign-of-zero
    sensitive)."""
    case = ring_case(seed=17)
    case["d"] = np.array([[0, 1], [11, 11], [39, 0]], dtype=np.int64)
    ref, nat = run_pair(native_pair, case)
    assert_parity(ref, nat)


def test_zero_and_tiny_cutoff(native_pair):
    """cutoff 0.0 (only exactly-0.0 paths) and a tiny cutoff (nothing
    reachable): both boundary regimes must match."""
    for cutoff in (0.0, 1.0000001):
        case = ring_case(seed=19, cutoff=cutoff)
        ref, nat = run_pair(native_pair, case)
        assert_parity(ref, nat)


def test_flag_column_false_edges(native_pair):
    """Network flag False removes arcs from the search (the a3 tail
    gate): parity must hold with a mixed flag column."""
    case = ring_case(seed=29)
    rng = np.random.default_rng(4)
    case["flag"] = rng.random(case["vec"].shape[0]) < 0.7
    ref, nat = run_pair(native_pair, case)
    assert_parity(ref, nat)


def test_int64_boundary_indices(native_pair):
    """Origins/destinations at the LAST node ids (v-1): index arithmetic
    at the CSR boundary, pointer tails, reverse-incidence tails."""
    case = ring_case(seed=31)
    v = case["ptr"].shape[0] - 1
    case["o"] = np.array([[v - 1, 0], [v - 2, v - 1]], dtype=np.int64)
    case["ow"] = np.full((2, 2), 0.2)
    case["d"] = np.array([[v - 1, v - 1], [0, v - 1]], dtype=np.int64)
    ref, nat = run_pair(native_pair, case)
    assert_parity(ref, nat)


def test_unreachable_isolated_nodes(native_pair):
    """Nodes with zero degree (unreachable): label defaults to the
    bound b on read, never captured as candidates."""
    case = ring_case(seed=37)
    v = case["ptr"].shape[0] - 1
    ptr = case["ptr"].copy()
    # Cut all arcs into/out of node v-5 -> isolated.
    isolated = v - 5
    keep = case["vec"] != isolated
    case["vec"] = case["vec"][keep]
    case["wts"] = case["wts"][keep]
    case["flag"] = case["flag"][keep]
    src_of = np.repeat(np.arange(v), np.diff(ptr))
    src_of = src_of[keep]
    ptr = np.zeros(v + 1, dtype=np.int64)
    ptr[1:] = np.cumsum(np.bincount(src_of, minlength=v))
    case["ptr"] = ptr
    case["d"] = np.array([[isolated, 0], [1, isolated]], dtype=np.int64)
    ref, nat = run_pair(native_pair, case)
    assert_parity(ref, nat)


# --------------------------------------------------------------------------
# Tiling / epoch regimes (the state machine's clearing laws).
# --------------------------------------------------------------------------
def test_tile_origins_regimes(native_pair):
    """tile_origins in {1, 3, 64, 10^4} over a 40-origin case: partial
    final tiles, single-origin tiles, and one giant tile."""
    case = ring_case(seed=23)
    for tile_origins in (1, 3, 64, 10_000):
        ref, nat = run_pair(native_pair, case, tile_origins=tile_origins)
        assert_parity(ref, nat)


def test_epoch_rollover_regimes(native_pair):
    """epoch_rollover in {1, 3}: stamps/marks cleared BEFORE rollover
    every 1 or 3 origins — the checked-counter path."""
    case = ring_case(seed=23)
    for rollover in (1, 3):
        ref, nat = run_pair(native_pair, case, epoch_rollover=rollover)
        assert_parity(ref, nat)


def test_rollover_with_tiling(native_pair):
    """Rollover AND tiling together (the compound clearing path)."""
    case = ring_case(seed=41)
    ref, nat = run_pair(native_pair, case, epoch_rollover=2,
                        tile_origins=3)
    assert_parity(ref, nat)


# --------------------------------------------------------------------------
# Budget law: INCOMPLETE records must be indistinguishable too.
# --------------------------------------------------------------------------
def test_budget_exceeded_first_tile(native_pair):
    """Budget below the first tile's cost: identical route, refusal
    text, origin boundary, and counters (minus tile_temp_bytes)."""
    case = ring_case(seed=23)
    ref, nat = run_pair(native_pair, case, trace_budget_bytes=1000)
    assert_parity(ref, nat)
    assert not ref.complete
    assert ref.route == "trace_budget_exceeded"
    assert "trace budget exceeded at origin" in ref.refusal


def test_budget_exceeded_mid_stream(native_pair):
    """Per-origin tiles with a budget sized to stop between tiles: the
    triggering tile counts in BOTH routes (build_trace's exact
    boundary), so budget_hit names the same origin."""
    case = ring_case(seed=23)
    # ~57 candidates/origin -> 16*57*20 ~= 18k triggers around origin 20.
    ref, nat = run_pair(native_pair, case, tile_origins=1,
                        trace_budget_bytes=18000)
    assert_parity(ref, nat)
    assert not ref.complete
    assert ref.counters.tiles == nat.counters.tiles
    assert 0 < ref.counters.tiles < 40


def test_budget_exactly_at_boundary(native_pair):
    """A budget exactly equal to the full trace cost completes; one
    byte less stops.  The integer identity (no slack, no rounding) is
    the law under test."""
    case = ring_case(seed=23)
    ref_full, nat_full = run_pair(native_pair, case)
    meta = ref_full.counters.metadata_bytes
    ipb = ref_full.counters.indptr_bytes
    cand = ref_full.counters.candidates_total
    exact = ipb + 16 * cand + meta
    ref, nat = run_pair(native_pair, case, trace_budget_bytes=exact)
    assert_parity(ref, nat)
    assert ref.complete
    ref, nat = run_pair(native_pair, case, trace_budget_bytes=exact - 1)
    assert_parity(ref, nat)
    assert not ref.complete


# --------------------------------------------------------------------------
# Refusal law: the S11 gate refuses BEFORE the native core runs.
# --------------------------------------------------------------------------
def test_admission_refusal_parity(native_pair):
    """Negative and NaN seed weights refuse; the native twin returns
    the SAME compact_fallback INCOMPLETE record (same refusal string,
    zeroed counters)."""
    for name, kw in [("neg", dict(neg_seed=True)),
                     ("nan_end", dict(nan_seed=True))]:
        case = build_case(3, **kw)
        ref, nat = run_pair(native_pair, case)
        assert_parity(ref, nat)
        assert not ref.complete
        assert ref.route == "compact_fallback"
        assert ref.refusal == nat.refusal
        assert ref.counters.candidates_total == 0


def test_refusal_engagement_counts_refusal_not_dispatch(native_pair):
    """A refused call never dispatches the native core: the refusal
    counter moves, the dispatch counter does not."""
    import trace_build_native as tbn
    before = tbn.engagement()
    case = build_case(3, neg_seed=True)
    ref, nat = run_pair(native_pair, case)
    after = tbn.engagement()
    assert not ref.complete
    assert after["admission_refusals"] == before["admission_refusals"] + 1
    assert after["dispatches"] == before["dispatches"]
