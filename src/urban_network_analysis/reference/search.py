"""Corrected_v1 scope search and OD assembly (SCIENCE reference).

Declarations implemented here (each independently derived, reviewed in
the task evidence, and distinct from legacy pinned behavior):

Corrected scope labels
    ``label[v]`` = shortest-walk distance from the origin POSITION to
    network node v, where the origin sits on its host edge with split
    distances (w_start, w_end) to the two endpoints.

    Seeds combine by MINIMUM, not overwrite (BUG-COINCIDENT-SEEDS
    corrected rule): both split points sit on the same host edge, and
    the distance to a shared endpoint is the shorter arc around that
    edge.  Combine order is declared: the start split seeds first, the
    end split replaces only on strict improvement.  On an exact tie
    (including the -0.0/+0.0 case) the incumbent start value survives,
    so signed-zero bits are a function of the declared order.

    Every reached network node propagates (no degree>1 push gate): the
    gate is a sound optimization for leaf nodes, but the corrected
    reference prefers the simpler rule whose termination proof covers
    every node uniformly; per-origin reference runs are not the
    performance lane.

    Heap priorities are tuples (distance, node_id): deterministic tie
    order, no insertion counter needed.  Relaxation is strict
    improvement with zero tolerance (NUMERICS: no tolerance changes
    radius membership); radius membership is label <= cutoff (closed).

    Stale-entry skips (pop with distance > label[node]) are admitted
    with the standard proof: on nonnegative validated costs, a pop with
    distance == label[node] re-relaxes the same adjacency and writes
    only strictly smaller labels; the first processing already wrote
    those exact candidates, so a second processing is provably a no-op.
    A pop with distance > label[node] carries a superseded value.

    Termination (declared proof): every heap entry is created by a
    strict label decrease, and a decreasing sequence of candidate
    values corresponds to prefix sums along simple paths (any repeated
    node adds nonnegative cost, so such prefixes never strictly
    improve); a finite graph has finitely many simple paths, so the
    heap drains.  Zero-cost cycles produce EQUAL candidates, which
    strict improvement rejects -- no livelock.  NaN/negative costs
    cannot enter: validate_csr refuses them first (BUG-NONFINITE-
    NOVALIDATION corrected rule).

Corrected OD assembly
    For destination j on host edge (u_j, v_j) with splits (s_u, s_v):

        od[i, j] = min(label[u_j] + s_u, label[v_j] + s_v, direct_arc)

    where direct_arc = |x_o - x_j| applies only when origin and
    destination share the same host edge (BUG-SAME-EDGE-OD corrected
    rule): x denotes distance from the edge's start node along the
    host edge (x_o = the origin's start split), so |x_o - x_j| is the
    along-edge arc from the model's own split geometry -- not a
    Euclidean substitution.  Endpoint routes are the corrected-label
    routes.  The minimum evaluates candidates in the declared order
    endpoint-start, endpoint-end, direct, keeping the incumbent on
    exact ties.

    Unreachable destinations (row value exceeds the cutoff or no
    finite route) map to +inf, by declaration; legacy maps them to
    its sentinel-plus-split values instead (retained legacy bits).
"""
from __future__ import annotations

import heapq

import numpy as np

from .errors import CorrectedValidationError
from .graph import validate_cutoff, validate_csr, validate_terminals

__all__ = [
    "corrected_scope_labels",
    "corrected_od_distances",
]


def corrected_scope_labels(
    o_terminal_idxs,
    o_terminal_weights,
    adjacency_pointer,
    adjacency_vector,
    adjacency_weights,
    cutoff,
):
    """Corrected single-origin scope labels (see module docstring).

    Returns a float64 label vector of length node_count; unreachable
    nodes keep +inf.
    """
    node_count, pointer, vector, weights = validate_csr(
        adjacency_pointer, adjacency_vector, adjacency_weights)
    cutoff = validate_cutoff(cutoff)
    idxs, wts = validate_terminals(
        o_terminal_idxs, o_terminal_weights, node_count, what="origin")
    if idxs.shape[0] != 1:
        raise CorrectedValidationError(
            "corrected_scope_labels handles exactly one origin row")
    start = int(idxs[0, 0])
    end = int(idxs[0, 1])
    w_start = float(wts[0, 0])
    w_end = float(wts[0, 1])

    label = np.full(node_count, np.inf, dtype=np.float64)

    # Declared seed combine: start seeds, end replaces on strict
    # improvement only (BUG-COINCIDENT-SEEDS corrected minimum rule).
    label[start] = w_start
    if w_end < label[end]:
        label[end] = w_end

    heap = []
    for seed in (start, end):
        if label[seed] <= cutoff:      # closed radius membership
            heapq.heappush(heap, (label[seed], seed))

    while heap:
        dist, node = heapq.heappop(heap)
        if dist > label[node]:
            # Stale entry: superseded by a strict improvement
            # (admission proof in the module docstring).
            continue
        # Ascending adjacency-offset scan (declared order).
        for j in range(int(pointer[node]), int(pointer[node + 1])):
            candidate = dist + float(weights[j])   # one binary64 RN add
            neighbor = int(vector[j])
            if candidate <= cutoff and candidate < label[neighbor]:
                label[neighbor] = candidate
                heapq.heappush(heap, (candidate, neighbor))
    return label


def _od_row_impl(label, o_edge, o_splits, d_edge_ids, d_edge_nodes, d_splits):
    """One origin's corrected destination distances (see module docstring).

    ``o_edge``: the origin's host-edge id (any hashable; equality is
    identity).  ``d_edge_ids``: (d_count,) host-edge ids.
    ``d_edge_nodes``: (d_count, 2) endpoint indices.  ``d_splits``:
    (d_count, 2) via (start endpoint, end endpoint).

    Declared precondition for the same-edge direct arc: for a given
    host edge id, the origin's and the destination's split columns
    measure from the SAME canonical start node (the legacy Network
    state derives both from the edge's start/end columns, so this
    holds there).  The direct candidate |o_splits[0] - d_splits[j,0]|
    is only meaningful under that shared reference.
    """
    d_count = d_splits.shape[0]
    row = np.empty(d_count, dtype=np.float64)
    o_start_split = float(o_splits[0])
    for j in range(d_count):
        u = int(d_edge_nodes[j, 0])
        v = int(d_edge_nodes[j, 1])
        # Declared candidate order: endpoint-start, endpoint-end, direct.
        best = label[u] + float(d_splits[j, 0])
        alt = label[v] + float(d_splits[j, 1])
        if alt < best:
            best = alt
        if d_edge_ids[j] == o_edge:
            # Along-edge arc between the two positions (x from the
            # edge's start node); exact subtract + abs.
            direct = abs(o_start_split - float(d_splits[j, 0]))
            if direct < best:
                best = direct
        row[j] = best
    return row


def corrected_od_distances(
    o_terminal_idxs,
    o_terminal_weights,
    o_edge_ids,
    adjacency_pointer,
    adjacency_vector,
    adjacency_weights,
    cutoff,
    d_edge_ids,
    d_edge_nodes,
    d_terminal_idxs,
    d_terminal_weights,
):
    """Corrected (o_count, d_count) OD distance matrix.

    Origins run in increasing order (per-origin outputs are independent;
    the canonical partition applies when results are AGGREGATED, see
    partition.py).  See the module docstring for the assembly rule.
    """
    node_count, pointer, vector, weights = validate_csr(
        adjacency_pointer, adjacency_vector, adjacency_weights)
    cutoff = validate_cutoff(cutoff)
    o_idxs, o_wts = validate_terminals(
        o_terminal_idxs, o_terminal_weights, node_count, what="origin")
    d_idxs, d_wts = validate_terminals(
        d_terminal_idxs, d_terminal_weights, node_count, what="destination")

    o_count = o_idxs.shape[0]
    d_count = d_idxs.shape[0]
    o_edge_ids = list(o_edge_ids)
    d_edge_ids = list(d_edge_ids)
    if len(o_edge_ids) != o_count:
        raise CorrectedValidationError(
            f"o_edge_ids length {len(o_edge_ids)} != origin count {o_count}")
    if len(d_edge_ids) != d_count:
        raise CorrectedValidationError(
            f"d_edge_ids length {len(d_edge_ids)} != destination count "
            f"{d_count}")
    d_edge_nodes = np.asarray(d_edge_nodes)
    if d_edge_nodes.shape != (d_count, 2) or d_edge_nodes.dtype != np.int64:
        raise CorrectedValidationError(
            "d_edge_nodes must be (d_count, 2) int64 endpoint indices")

    out = np.empty((o_count, d_count), dtype=np.float64)
    for i in range(o_count):
        label = corrected_scope_labels(
            o_idxs[i:i + 1], o_wts[i:i + 1], pointer, vector, weights, cutoff)
        out[i, :] = _od_row_impl(
            label, o_edge_ids[i], o_wts[i], d_edge_ids, d_edge_nodes, d_wts)
    return out
