"""Analytic / independent-enumeration tests for the corrected_v1 search.

Two scientific checks per NUMERICS.md: exact binary64 expectations from
the declared reference rules, and independent mathematical oracles
(exhaustive simple-path enumeration with Fractions) that validate the
intended math with a different failure mode than the implementation.
"""
from __future__ import annotations

import struct
from fractions import Fraction

import numpy as np
import pytest

pytestmark = pytest.mark.science

from urban_network_analysis.reference import errors
from urban_network_analysis.reference.search import (
    corrected_od_distances,
    corrected_scope_labels,
)


def bits(x: float) -> int:
    return struct.unpack("<Q", struct.pack("<d", float(x)))[0]


def csr(edges, node_count):
    """Deterministic sorted CSR from (u, v, w) triples."""
    edges = sorted(edges)
    pointer = np.zeros(node_count + 1, dtype=np.int64)
    for u, v, _ in edges:
        pointer[u + 1] += 1
        pointer[v + 1] += 1
    pointer = np.cumsum(pointer)
    vector = np.zeros(int(pointer[-1]), dtype=np.int64)
    weights = np.zeros(int(pointer[-1]), dtype=np.float64)
    fill = pointer[:-1].copy()
    for u, v, w in edges:
        vector[fill[u]] = v
        weights[fill[u]] = w
        fill[u] += 1
        vector[fill[v]] = u
        weights[fill[v]] = w
        fill[v] += 1
    return pointer, vector, weights


def enumerate_simple_path_costs(adj, src, dst):
    """Independent oracle: exhaustive simple-path costs, as Fractions.

    ``adj``: {node: [(neighbor, Fraction cost), ...]}.  Returns the set
    of exact path costs src -> dst over all simple paths (src == dst
    yields the single empty path, cost 0).
    """
    out = []

    def dfs(node, visited, cost):
        if node == dst:
            out.append(cost)
            return
        for nbr, c in adj.get(node, ()):
            if nbr not in visited:
                dfs(nbr, visited | {nbr}, cost + c)

    dfs(src, {src}, Fraction(0))
    return out


def frac_adj(edges):
    """Exact rational view of the same binary64 costs (Fraction(float)
    is exact -- no rounding, no limit_denominator), so the enumeration
    oracle computes true minima over the very values the reference
    receives."""
    adj = {}
    for u, v, w in edges:
        f = Fraction(w)
        adj.setdefault(u, []).append((v, f))
        adj.setdefault(v, []).append((u, f))
    return adj


# --------------------------------------------------------------------------
# scope labels vs independent enumeration (rational oracle)
# --------------------------------------------------------------------------

ANALYTIC_GRAPHS = [
    # (name, edges, node_count, origin splits (node pair, weights))
    ("path_dyadic", [(0, 1, 1.5), (1, 2, 2.25), (2, 3, 3.75)], 4, (0, 1, 0.5, 1.0)),
    ("ring", [(0, 1, 1.0), (1, 2, 1.0), (2, 3, 1.0), (3, 0, 1.0)], 4, (0, 1, 0.25, 0.75)),
    ("parallel_same_neighbor", [(0, 1, 2.0), (0, 1, 1.25)], 2, (0, 0, 0.0, 0.0)),
    ("self_loop", [(0, 0, 0.5), (0, 1, 2.0)], 2, (0, 1, 0.1, 1.9)),
    ("two_routes_tie", [(0, 1, 1.0), (1, 2, 1.0), (0, 3, 1.0), (3, 2, 1.0)], 4, (0, 0, 0.0, 0.0)),
    ("zero_cost_cycle", [(0, 1, 0.0), (1, 0, 0.0)], 2, (0, 1, 0.0, 0.0)),
    ("disconnected", [(0, 1, 1.0)], 4, (0, 1, 0.5, 0.5)),
    ("subnormal_costs", [(0, 1, 5e-324), (1, 2, 5e-324)], 3, (0, 0, 0.0, 0.0)),
]


@pytest.mark.parametrize("case", ANALYTIC_GRAPHS, ids=[g[0] for g in ANALYTIC_GRAPHS])
def test_labels_match_exhaustive_enumeration(case):
    """Corrected labels equal the exact enumeration minimum on every
    node (Fractions validate the intended math with an independent
    failure mode; dyadic costs make the binary64 comparison exact)."""
    name, edges, node_count, (ou, ov, wu, wv) = case
    pointer, vector, weights = csr(edges, node_count)
    label = corrected_scope_labels(
        np.array([[ou, ov]], dtype=np.int64),
        np.array([[wu, wv]], dtype=np.float64),
        pointer, vector, weights, 1e9)

    adj = frac_adj(edges)
    for node in range(node_count):
        via_u = min(
            (Fraction(wu) + c for c in enumerate_simple_path_costs(adj, ou, node)),
            default=None)
        via_v = min(
            (Fraction(wv) + c for c in enumerate_simple_path_costs(adj, ov, node)),
            default=None)
        if via_u is None and via_v is None:
            # unreachable in the enumeration oracle: label must be +inf
            assert bits(label[node]) == bits(float("inf")), (
                f"{name}: node {node}: {label[node]!r} should be +inf")
            continue
        candidates = [c for c in (via_u, via_v) if c is not None]
        expected = min(candidates)
        # exact rational comparison against the float64 label
        assert Fraction(label[node]) == expected, (
            f"{name}: node {node}: label {label[node]!r} != {expected}")


def test_labels_bitwise_deterministic_under_ties():
    """Tied heap priorities ((1.0, 1) vs (1.0, 3)) resolve
    deterministically: two runs are bitwise identical."""
    edges = [(0, 1, 1.0), (1, 2, 1.0), (0, 3, 1.0), (3, 2, 1.0)]
    pointer, vector, weights = csr(edges, 4)
    args = (np.array([[0, 0]], dtype=np.int64),
            np.array([[0.0, 0.0]], dtype=np.float64),
            pointer, vector, weights, 10.0)
    l1 = corrected_scope_labels(*args)
    l2 = corrected_scope_labels(*args)
    assert l1.tobytes() == l2.tobytes()
    # the tie survives to the output: two equal-cost routes to node 2
    assert l1[2] == 2.0


# --------------------------------------------------------------------------
# declared seed / boundary / domain rules
# --------------------------------------------------------------------------

def test_coincident_seeds_combine_by_minimum():
    """Corrected rule: split weights (0, 5) at one node -> label 0.0
    (legacy overwrite keeps 5.0; that legacy bit is pinned in
    test_bug_triples)."""
    pointer, vector, weights = csr([(0, 1, 10.0)], 2)
    label = corrected_scope_labels(
        np.array([[0, 0]], dtype=np.int64),
        np.array([[0.0, 5.0]], dtype=np.float64),
        pointer, vector, weights, 100.0)
    assert bits(label[0]) == bits(0.0)


def test_seed_signed_zero_tie_keeps_start_split():
    """Declared combine order: on an exact +/-0.0 tie the start split
    (the incumbent) survives, so seed bits are order-defined."""
    pointer, vector, weights = csr([(0, 1, 10.0)], 2)
    label = corrected_scope_labels(
        np.array([[0, 0]], dtype=np.int64),
        np.array([[0.0, -0.0]], dtype=np.float64),
        pointer, vector, weights, 100.0)
    assert bits(label[0]) == bits(0.0)          # +0.0 (start incumbent)
    label2 = corrected_scope_labels(
        np.array([[0, 0]], dtype=np.int64),
        np.array([[-0.0, 0.0]], dtype=np.float64),
        pointer, vector, weights, 100.0)
    assert bits(label2[0]) == bits(-0.0)        # -0.0 (start incumbent)


def test_cutoff_membership_closed_and_nextafter():
    """Membership is label <= cutoff (closed): at exactly the cutoff a
    destination is in; one ulp beyond is out."""
    edges = [(0, 1, 7.5), (1, 2, 2.5)]
    pointer, vector, weights = csr(edges, 3)
    label = corrected_scope_labels(
        np.array([[0, 0]], dtype=np.int64),
        np.array([[0.0, 0.0]], dtype=np.float64),
        pointer, vector, weights, 10.0)
    assert label[2] == 10.0                     # == cutoff: admitted
    label_tight = corrected_scope_labels(
        np.array([[0, 0]], dtype=np.int64),
        np.array([[0.0, 0.0]], dtype=np.float64),
        pointer, vector, weights,
        np.nextafter(np.float64(10.0), 0))
    assert np.isinf(label_tight[2])             # one ulp below: not admitted


def test_unreachable_nodes_stay_positive_inf():
    pointer, vector, weights = csr([(0, 1, 1.0)], 4)
    label = corrected_scope_labels(
        np.array([[0, 1]], dtype=np.int64),
        np.array([[0.5, 0.5]], dtype=np.float64),
        pointer, vector, weights, 100.0)
    assert bits(label[2]) == bits(float("inf"))
    assert bits(label[3]) == bits(float("inf"))


def test_max_finite_cost_no_overflow_write():
    """max-finite arcs: one such arc is admitted (closed membership,
    big <= big), but a second chained arc overflows the candidate to
    +inf, fails <= cutoff, and is never written (declared init is +inf,
    so there is no ones(...)+cutoff sentinel to overflow)."""
    big = float(np.finfo(np.float64).max)
    pointer, vector, weights = csr([(0, 1, big), (1, 2, big)], 3)
    with np.errstate(over="ignore"):
        label = corrected_scope_labels(
            np.array([[0, 0]], dtype=np.int64),
            np.array([[0.0, 0.0]], dtype=np.float64),
            pointer, vector, weights, big)
    assert bits(label[1]) == bits(big)          # first hop admitted
    assert bits(label[2]) == bits(float("inf"))  # overflow candidate rejected


def test_validation_rejects_invalid_domains():
    pointer, vector, weights = csr([(0, 1, 1.0)], 2)
    seeds = (np.array([[0, 1]], dtype=np.int64),
             np.array([[0.0, 0.0]], dtype=np.float64))
    with pytest.raises(errors.NonFiniteCostError):
        corrected_scope_labels(*seeds, pointer, vector,
                               np.array([1.0, float("nan")]), 10.0)
    with pytest.raises(errors.NonFiniteCostError):
        corrected_scope_labels(*seeds, pointer, vector,
                               np.array([1.0, float("inf")]), 10.0)
    with pytest.raises(errors.NegativeCostError):
        corrected_scope_labels(*seeds, pointer, vector,
                               np.array([1.0, -5.0]), 10.0)
    with pytest.raises(errors.NonFiniteCostError):
        corrected_scope_labels(*seeds, pointer, vector, weights, float("nan"))
    with pytest.raises(errors.NegativeCostError):
        corrected_scope_labels(*seeds, pointer, vector, weights, -1.0)
    with pytest.raises(errors.NonFiniteCostError):
        corrected_scope_labels(
            np.array([[0, 1]], dtype=np.int64),
            np.array([[0.0, float("nan")]], dtype=np.float64),
            pointer, vector, weights, 10.0)
    with pytest.raises(errors.OutOfRangeTerminalError):
        corrected_scope_labels(
            np.array([[0, 9]], dtype=np.int64),
            np.array([[0.0, 0.0]], dtype=np.float64),
            pointer, vector, weights, 10.0)
    with pytest.raises(errors.MalformedGraphError):
        corrected_scope_labels(*seeds, np.array([1, 0], dtype=np.int64),
                               vector, weights, 10.0)   # pointer[0] != 0


# --------------------------------------------------------------------------
# corrected OD assembly (BUG-SAME-EDGE-OD rule)
# --------------------------------------------------------------------------

def test_same_edge_direct_partial_admitted():
    """Origin 10 m / destination 90 m on one 100 m edge -> 80 (the
    along-edge arc), not the legacy endpoint-route 100."""
    pointer, vector, weights = csr([(0, 1, 100.0)], 2)
    od = corrected_od_distances(
        np.array([[0, 1]], dtype=np.int64), np.array([[10.0, 90.0]]), ["e0"],
        pointer, vector, weights, 1000.0,
        ["e0"], np.array([[0, 1]], dtype=np.int64),
        np.array([[0, 1]], dtype=np.int64), np.array([[90.0, 10.0]]))
    assert bits(od[0, 0]) == bits(80.0)


def test_same_edge_direct_looses_to_shorter_endpoint_route():
    """min() over all three candidates: a shorter endpoint route wins."""
    pointer, vector, weights = csr([(0, 1, 100.0)], 2)
    od = corrected_od_distances(
        np.array([[0, 1]], dtype=np.int64), np.array([[10.0, 90.0]]), ["e0"],
        pointer, vector, weights, 1000.0,
        ["e0"], np.array([[0, 1]], dtype=np.int64),
        np.array([[0, 1]], dtype=np.int64), np.array([[20.0, 80.0]]))
    assert bits(od[0, 0]) == bits(10.0)     # direct |10-20| beats 30 via u


def test_distinct_edges_use_endpoint_routes_only():
    """No shared host edge -> no direct candidate; the destination sits
    on a parallel dummy edge id so direct never applies."""
    pointer, vector, weights = csr([(0, 1, 100.0)], 2)
    od = corrected_od_distances(
        np.array([[0, 1]], dtype=np.int64), np.array([[10.0, 90.0]]), ["eo"],
        pointer, vector, weights, 1000.0,
        ["ed"], np.array([[0, 1]], dtype=np.int64),
        np.array([[0, 1]], dtype=np.int64), np.array([[90.0, 10.0]]))
    # endpoint routes only: min(10+90, 90+10) = 100
    assert bits(od[0, 0]) == bits(100.0)


def test_od_unreachable_maps_to_inf():
    pointer, vector, weights = csr([(0, 1, 100.0)], 3)
    od = corrected_od_distances(
        np.array([[0, 1]], dtype=np.int64), np.array([[10.0, 90.0]]), ["eo"],
        pointer, vector, weights, 1000.0,
        ["ed"], np.array([[2, 2]], dtype=np.int64),
        np.array([[2, 2]], dtype=np.int64), np.array([[0.0, 0.0]]))
    assert bits(od[0, 0]) == bits(float("inf"))
