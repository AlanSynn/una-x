"""A3-specific fixtures (task A3I; proof.md sections 3.1 and 7).

The discriminating axis for A3 is the DESTINATION TERMINAL domain:
`_a3_tail_admits` fires exactly when `d_terminal_idxs` has shape
(d_count, 2) with every value in [0, node_count). The builder produces
one admitted typed graph whose destination terminals are then put
through variants:

* valid            — guard fires; tail-free route runs.
* boundary_values  — guard fires with terminals 0 and node_count-1.
* oob_value        — one value == node_count: A3 refuses, A1 route
                     reads the untouched tail sentinel (byte-identical
                     to B0 by construction).
* negative_value   — one value == -1: A3 refuses; the A1 fallback's
                     negative-index semantics are preserved (identity
                     to B0 is what is pinned, not sanity).
* extra_row        — d_terminal_idxs carries d_count+1 rows: shape
                     refusal; adjust reads only the first d_count rows
                     on both arms.
* zero_destinations— d_count == 0: guard fires vacuously; adjust loops
                     zero times.
* shared_terminals — duplicate origin rows and duplicate destination
                     rows (per-origin ownership under identical
                     workloads).
* a1_refusal       — pointer[0] != 0 (not ordered CSR): the A1 guard
                     refuses, so the original kernel loop runs; A3 is
                     unreachable.

All arrays are int64/float64/bool — the admitted typing.
"""
from __future__ import annotations

import numpy as np


def _graph_arrays(seed, v):
    rng = np.random.default_rng(seed)
    pointer = [0]
    nbrs, weights, flags = [], [], []
    for node in range(v):
        targets = {(node + 1) % v, (node + 7) % v, (node + 29) % v}
        deg = int(rng.integers(1, 4))
        for t in sorted(targets)[:deg]:
            nbrs.append(t)
            weights.append(round(float(rng.uniform(0.5, 9.0)), 6))
            flags.append(bool(t % 5 != 3))
        pointer.append(len(nbrs))
    return (
        np.asarray(pointer, dtype=np.int64),
        np.asarray(nbrs, dtype=np.int64),
        np.asarray(weights, dtype=np.float64),
        np.asarray(flags, dtype=np.bool_),
        rng,
    )


def _base(seed=20260925, v=120, o_count=5, d_count=6):
    """Deterministic connected random graph in admitted typing with
    in-domain destination terminals (the guard firing case)."""
    pointer, nbrs, weights, flags, rng = _graph_arrays(seed, v)
    oti = np.stack(
        [(np.arange(o_count) * 11) % v, (np.arange(o_count) * 11 + 1) % v],
        axis=1).astype(np.int64)
    otw = np.round(rng.uniform(0.0, 2.0, (o_count, 2)), 6)
    dti = np.stack(
        [(np.arange(d_count) * 13 + 3) % v,
         (np.arange(d_count) * 13 + 5) % v], axis=1).astype(np.int64)
    dtw = np.round(rng.uniform(0.0, 2.0, (d_count, 2)), 6)
    return dict(
        adjacency_pointer=pointer,
        adjacency_vector=nbrs,
        adjacency_vector_weights=weights,
        adjacynct_vector_network_node=flags,
        o_terminal_idxs=oti,
        o_terminal_weights=otw,
        d_count=d_count,
        d_terminal_idxs=dti,
        d_terminal_weights=dtw,
        d_weights=np.round(rng.uniform(0.5, 3.0, d_count), 6),
        gravity_beta=0.05,
        metric_plateau=0.0,
        metric_midpoint=5.0,
        knn_decay="logistic",
        knn_weights=np.array([1.0, 0.6, 0.3]),
        cutoff=12.5,
        node_count=v,
    )


def valid(seed=20260925, v=120):
    return _base(seed=seed, v=v)


def boundary_values(seed=20260925, v=120):
    arrays = _base(seed=seed, v=v)
    d = arrays["d_count"]
    arrays["d_terminal_idxs"] = np.stack(
        [np.zeros(d, dtype=np.int64),
         np.full(d, v - 1, dtype=np.int64)], axis=1)
    return arrays


def oob_value(seed=20260925, v=120):
    arrays = _base(seed=seed, v=v)
    arrays["d_terminal_idxs"] = arrays["d_terminal_idxs"].copy()
    arrays["d_terminal_idxs"][0, 0] = v  # == node_count: out of domain
    return arrays


def negative_value(seed=20260925, v=120):
    arrays = _base(seed=seed, v=v)
    arrays["d_terminal_idxs"] = arrays["d_terminal_idxs"].copy()
    arrays["d_terminal_idxs"][1, 1] = -1
    return arrays


def extra_row(seed=20260925, v=120):
    arrays = _base(seed=seed, v=v)
    arrays["d_terminal_idxs"] = np.vstack(
        [arrays["d_terminal_idxs"], [[0, 1]]]).astype(np.int64)
    return arrays


def zero_destinations(seed=20260925, v=120):
    arrays = _base(seed=seed, v=v)
    arrays["d_count"] = 0
    arrays["d_terminal_idxs"] = np.zeros((0, 2), dtype=np.int64)
    arrays["d_terminal_weights"] = np.zeros((0, 2), dtype=np.float64)
    arrays["d_weights"] = np.zeros(0, dtype=np.float64)
    return arrays


def shared_terminals(seed=20260925, v=120):
    arrays = _base(seed=seed, v=v)
    arrays["o_terminal_idxs"] = arrays["o_terminal_idxs"].copy()
    arrays["o_terminal_idxs"][1] = arrays["o_terminal_idxs"][0]
    arrays["d_terminal_idxs"] = arrays["d_terminal_idxs"].copy()
    arrays["d_terminal_idxs"][1] = arrays["d_terminal_idxs"][0]
    return arrays


def a1_refusal(seed=20260925, v=120):
    arrays = _base(seed=seed, v=v)
    pointer = arrays["adjacency_pointer"].copy()
    pointer[0] = 1  # not ordered CSR: node 0's incidences start at 1
    arrays["adjacency_pointer"] = pointer
    return arrays


VARIANTS = (
    "valid", "boundary_values", "oob_value", "negative_value",
    "extra_row", "zero_destinations", "shared_terminals", "a1_refusal",
)


def kernel_case(arrays, o_pos):
    """Single-origin kernel-call arrays (B0 kernel signature) plus the
    expected d_count; o_pos indexes the origin rows."""
    return dict(
        o_terminal_idxs=arrays["o_terminal_idxs"][o_pos],
        o_terminal_weights=arrays["o_terminal_weights"][o_pos],
        adjacency_pointer=arrays["adjacency_pointer"],
        adjacency_vector=arrays["adjacency_vector"],
        adjacency_vector_weights=arrays["adjacency_vector_weights"],
        adjacynct_vector_network_node=arrays["adjacynct_vector_network_node"],
        cutoff=arrays["cutoff"],
        d_count=arrays["d_count"],
    )
