"""Private, order-preserving CSR construction for the accessibility engines.

No edge-weight arithmetic is performed here. Unsupported representations return
None so the caller executes its original construction, including its errors.
"""
from __future__ import annotations

import numpy as np


def _try_build_ordered_csr(node_count, start_nodes, end_nodes, ab_weights, ba_weights):
    """Return owned CSR arrays, or None when the exact fast path is not admitted.

    A row contains forward incidences in input-edge order, followed by reverse
    incidences in input-edge order. Parallel edges and both incidences of a
    self-loop are retained. Native int64 endpoints and float64 weights are the
    normal representations produced by the existing accessibility builders;
    int32 endpoints are additionally admitted via an exact value-preserving
    upcast, because the original builder emits int64 neighbors from any
    signed-endpoint input (its list-to-array conversion widens), so the
    observable arrays are identical either way.
    """
    # The histogram output dtype is intp; keep the admitted platform explicit.
    if np.dtype(np.intp).itemsize != 8:
        return None
    if type(node_count) is not int or not 0 <= node_count < np.iinfo(np.intp).max:
        return None
    arrays = (start_nodes, end_nodes, ab_weights, ba_weights)
    if any(type(a) is not np.ndarray or a.ndim != 1 for a in arrays):
        return None
    if ab_weights.dtype != np.dtype(np.float64) or ba_weights.dtype != np.dtype(np.float64):
        return None
    # Signed endpoint arrays only, up to the width the original builder
    # widens to int64 on output. int32 is upcast exactly (no value change);
    # any other dtype keeps the original builder in charge.
    if start_nodes.dtype == np.dtype(np.int64) and end_nodes.dtype == np.dtype(np.int64):
        start_i64 = start_nodes
        end_i64 = end_nodes
    elif (
        start_nodes.dtype == np.dtype(np.int32)
        and end_nodes.dtype == np.dtype(np.int32)
        and node_count <= np.iinfo(np.int32).max
    ):
        start_i64 = start_nodes.astype(np.int64)
        end_i64 = end_nodes.astype(np.int64)
    else:
        return None
    edge_count = start_i64.size
    if any(a.size != edge_count for a in (end_i64, ab_weights, ba_weights)):
        return None
    # Bound integer prefix sums and the doubled incidence-array dimensions.
    if edge_count > min(np.iinfo(np.intp).max, np.iinfo(np.int64).max) // 2:
        return None
    if edge_count and (
        start_i64.min() < 0 or end_i64.min() < 0
        or start_i64.max() >= node_count or end_i64.max() >= node_count
    ):
        return None

    try:
        # Stable sorting of [all forward incidences, all reverse incidences]
        # reproduces BOTH order rules of the original per-node masked scans.
        sources = np.concatenate((start_i64, end_i64))
        counts = np.bincount(sources, minlength=node_count)
        pointer = np.empty(node_count + 1, dtype=np.int64)
        pointer[0] = 0
        np.cumsum(counts, dtype=np.int64, out=pointer[1:])
        del counts
        order = np.argsort(sources, kind="stable")
        del sources

        unsorted_neighbors = np.concatenate((end_i64, start_i64))
        neighbors = unsorted_neighbors[order]
        del unsorted_neighbors
        unsorted_weights = np.concatenate((ab_weights, ba_weights))
        weights = unsorted_weights[order]
        del unsorted_weights, order
        is_network = np.ones(2 * edge_count, dtype=np.bool_)
        return pointer, neighbors, weights, is_network
    except MemoryError:
        # Release this frame's scratch before the caller enters the old path.
        # Do not catch unrelated failures or silently change the input domain.
        return None
