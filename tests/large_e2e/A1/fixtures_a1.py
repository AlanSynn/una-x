"""A1-specific boundary fixtures (task A1I; proof.md section 9).

Hand-built kernel-signature arrays for the pins the H03 corpus does
not already carry:

* downstream_dup     — T5 discriminator: a duplicate-incidence row one
                       hop downstream of the origin, whose destination
                       is degree-0 so no stale re-scan can repair the
                       staged later-larger overwrite (the k_dup_* origin
                       rows converge both semantics; H03 documents this
                       in test_negative_mutations.py).
* max_degree_row     — T10: the widest row is strictly wider than every
                       other row, holds duplicate destinations, and the
                       scratch buffer is sized to max_degree.
* true_overflow      — T4: finite weights whose incidence add overflows
                       to +inf inside a scanned row in BOTH arms (the
                       guard admits: every input value is finite).

All arrays are int64/float64/bool — the admitted typing.
"""
from __future__ import annotations

import numpy as np

MAX_FLOAT = float(np.finfo(np.float64).max)


def downstream_dup():
    """Origin row -> node 1; node 1's row has two incidences to
    degree-0 node 2 (cheap first, expensive second). Snapshot judging
    keeps the LAST eligible value (1+5=6); an immediate-update scan
    keeps the cheap value (1+3=4) because the second duplicate then
    fails `c < labels[2]`; node 2 is never pushed, so the distinction
    survives to the final labels."""
    return dict(
        adjacency_pointer=np.array([0, 1, 3, 3], dtype=np.int64),
        adjacency_vector=np.array([1, 2, 2], dtype=np.int64),
        adjacency_vector_weights=np.array([1.0, 3.0, 5.0], dtype=np.float64),
        adjacynct_vector_network_node=np.ones(3, dtype=np.bool_),
        o_terminal_idxs=np.array([[0, 0]], dtype=np.int64),
        o_terminal_weights=np.array([[0.0, 0.0]], dtype=np.float64),
        cutoff=10.0,
        d_count=1,
    )


def max_degree_row():
    """Widest row (degree 6) strictly exceeds every other row (degree
    1); it contains duplicate destinations whose last-eligible staged
    overwrite sticks (degree-1 neighbors are never pushed). Output is
    d_count + V wide, so the trailing slot stays the destination
    sentinel: expected [0, 1, 3, 3.5, 2, 2, 1+cutoff]."""
    return dict(
        adjacency_pointer=np.array([0, 1, 7, 8, 9, 10, 11], dtype=np.int64),
        adjacency_vector=np.array(
            [1, 2, 3, 4, 5, 2, 3, 1, 1, 1, 1], dtype=np.int64),
        adjacency_vector_weights=np.array(
            [1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.5,
             1.0, 1.0, 1.0, 1.0], dtype=np.float64),
        adjacynct_vector_network_node=np.ones(11, dtype=np.bool_),
        o_terminal_idxs=np.array([[0, 0]], dtype=np.int64),
        o_terminal_weights=np.array([[0.0, 0.0]], dtype=np.float64),
        cutoff=10.0,
        d_count=1,
    )


def true_overflow():
    """Finite 1.5e308 weights; the popped 1.5e308 label plus a 1.5e308
    incidence weight overflows to +inf inside the scanned row of node 1
    in BOTH arms (ineligible there). Every input value is finite and
    the cutoff is finite, so the guard admits. Final staged labels:
    [0, 1.0, 0]."""
    return dict(
        adjacency_pointer=np.array([0, 1, 3, 4], dtype=np.int64),
        adjacency_vector=np.array([1, 2, 0, 1], dtype=np.int64),
        adjacency_vector_weights=np.array(
            [1.5e308, 1.5e308, 1.0, 1.0], dtype=np.float64),
        adjacynct_vector_network_node=np.ones(4, dtype=np.bool_),
        o_terminal_idxs=np.array([[0, 2]], dtype=np.int64),
        o_terminal_weights=np.array([[0.0, 0.0]], dtype=np.float64),
        cutoff=MAX_FLOAT,
        d_count=1,
    )


def scratch_cap_refusal():
    """Guard condition 12 with real arrays: one row of degree
    16,777,217 exceeds the scratch cap for every thread count. The
    arrays are ~400 MiB, allocated once in the test process and freed
    with the case (the guard refuses BEFORE any scratch exists, and the
    fallback kernel scans one ineligible row)."""
    row_len = 16_777_217  # > CAP_ELEMS for every threads <= 2**24
    vector = np.zeros(row_len, dtype=np.int64)
    vector[0] = 1  # endpoints valid; capacity is the refusing condition
    return dict(
        adjacency_pointer=np.array([0, row_len, row_len], dtype=np.int64),
        adjacency_vector=vector,
        adjacency_vector_weights=np.zeros(row_len, dtype=np.float64),
        adjacynct_vector_network_node=np.zeros(row_len, dtype=np.bool_),
        o_terminal_idxs=np.array([[0, 0]], dtype=np.int64),
        o_terminal_weights=np.array([[0.0, 0.0]], dtype=np.float64),
        cutoff=10.0,
        d_count=1,
    )


def admitted_base(seed=20260925, v=120, extra_origins=True):
    """A deterministic connected random graph in admitted typing, for
    routing (T13/T14), reuse-independence (T11) and permutation tests."""
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
    o_count = 5 if extra_origins else 1
    oti = np.stack(
        [(np.arange(o_count) * 11) % v, (np.arange(o_count) * 11 + 1) % v],
        axis=1).astype(np.int64)
    otw = np.round(rng.uniform(0.0, 2.0, (o_count, 2)), 6)
    d_count = 4
    dti = np.stack(
        [(np.arange(d_count) * 13 + 3) % v,
         (np.arange(d_count) * 13 + 5) % v], axis=1).astype(np.int64)
    dtw = np.round(rng.uniform(0.0, 2.0, (d_count, 2)), 6)
    return dict(
        adjacency_pointer=np.asarray(pointer, dtype=np.int64),
        adjacency_vector=np.asarray(nbrs, dtype=np.int64),
        adjacency_vector_weights=np.asarray(weights, dtype=np.float64),
        adjacynct_vector_network_node=np.asarray(flags, dtype=np.bool_),
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
    )
