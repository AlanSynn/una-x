"""I00 composition fixtures.

A-side (Accessibility families): deterministic typed graphs whose
DESTINATION TERMINAL domain drives the A1×A3 guard matrix, mirroring
the proven A3 builder shapes. The variant set covers every reachable
dispatch cell of the composed tree:

* both_admit, boundary_values, zero_destinations, shared_terminals
      — A1 admits AND A3 admits: the composed A3 tail-free route fires.
* a3_refuse_oob, a3_refuse_negative, a3_refuse_shape
      — A1 admits, A3 refuses: the A1 scratch route (the fallback) runs.
* a1_refusal
      — A1 refuses: the original kernel loop runs (A3 unreachable by
        dispatch structure — nested inside the admitted branch).

All arrays are int64/float64/bool — the admitted typing. Guard-cell
expectations live in GUARD_EXPECTATIONS and are probed with njit
wrappers (never pure-Python calls: the @overload guards resolve only
under nopython).

F-side (AggregateFlow): the shared oracle spec (crossover-refusing at
its stub size) plus a self-contained long-path spec whose real
crossover ADMITS (the F2 local-overlap route engages under F1
striping), mirroring the F2 suite's engagement design.

F3-side: a minimal deterministic gradient stub (superset read-set,
feeds both arms) with a capture logger.
"""
from __future__ import annotations

import types

import numpy as np
from scipy.sparse import csr_matrix

import fixtures as oracle_fixtures


# ======================================================================
# A-side: guard-matrix graphs
# ======================================================================

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


def _base(seed=20260928, v=120, o_count=5, d_count=6):
    """Deterministic connected random graph in admitted typing with
    in-domain destination terminals (both guards firing)."""
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


def both_admit(seed=20260928, v=120):
    return _base(seed=seed, v=v)


def boundary_values(seed=20260928, v=120):
    arrays = _base(seed=seed, v=v)
    d = arrays["d_count"]
    arrays["d_terminal_idxs"] = np.stack(
        [np.zeros(d, dtype=np.int64),
         np.full(d, v - 1, dtype=np.int64)], axis=1)
    return arrays


def zero_destinations(seed=20260928, v=120):
    arrays = _base(seed=seed, v=v)
    arrays["d_count"] = 0
    arrays["d_terminal_idxs"] = np.zeros((0, 2), dtype=np.int64)
    arrays["d_terminal_weights"] = np.zeros((0, 2), dtype=np.float64)
    arrays["d_weights"] = np.zeros(0, dtype=np.float64)
    return arrays


def shared_terminals(seed=20260928, v=120):
    arrays = _base(seed=seed, v=v)
    arrays["o_terminal_idxs"] = arrays["o_terminal_idxs"].copy()
    arrays["o_terminal_idxs"][1] = arrays["o_terminal_idxs"][0]
    arrays["d_terminal_idxs"] = arrays["d_terminal_idxs"].copy()
    arrays["d_terminal_idxs"][1] = arrays["d_terminal_idxs"][0]
    return arrays


def a3_refuse_oob(seed=20260928, v=120):
    arrays = _base(seed=seed, v=v)
    arrays["d_terminal_idxs"] = arrays["d_terminal_idxs"].copy()
    arrays["d_terminal_idxs"][0, 0] = v  # == node_count: out of domain
    return arrays


def a3_refuse_negative(seed=20260928, v=120):
    arrays = _base(seed=seed, v=v)
    arrays["d_terminal_idxs"] = arrays["d_terminal_idxs"].copy()
    arrays["d_terminal_idxs"][1, 1] = -1
    return arrays


def a3_refuse_shape(seed=20260928, v=120):
    arrays = _base(seed=seed, v=v)
    arrays["d_terminal_idxs"] = np.vstack(
        [arrays["d_terminal_idxs"], [[0, 1]]]).astype(np.int64)
    return arrays


def a1_refusal(seed=20260928, v=120):
    arrays = _base(seed=seed, v=v)
    pointer = arrays["adjacency_pointer"].copy()
    pointer[0] = 1  # not ordered CSR: node 0's incidences start at 1
    arrays["adjacency_pointer"] = pointer
    return arrays


VARIANTS = (
    "both_admit", "boundary_values", "zero_destinations",
    "shared_terminals", "a3_refuse_oob", "a3_refuse_negative",
    "a3_refuse_shape", "a1_refusal",
)

# Expected guard cells per variant: (a1_admitted, a3_admitted-or-None).
# a3 is None where the composed dispatch never evaluates it (A1 refused
# first — the A3 guard is nested inside the A1-admitted branch).
GUARD_EXPECTATIONS = {
    "both_admit": (True, True),
    "boundary_values": (True, True),
    "zero_destinations": (True, True),
    "shared_terminals": (True, True),
    "a3_refuse_oob": (True, False),
    "a3_refuse_negative": (True, False),
    "a3_refuse_shape": (True, False),
    "a1_refusal": (False, None),
}

SEEDS = (20260928, 20260929, 20260930)


# ======================================================================
# F-side: flow specs
# ======================================================================

def flow_spec_oracle():
    """The shared oracle flow spec (tiny; the production dense-overlap
    crossover refuses every OD at this size — the refusing side)."""
    return oracle_fixtures.flow_network_spec()


def flow_spec_long():
    """Long sparse path spec whose real crossover ADMITS at search
    radius 4.0 (n_total=127, per-destination gradient slices of ~5-11
    nodes): the F2 local-overlap route engages under F1 striping.
    Self-contained mirror of the F2 suite's engagement spec."""
    n = 120
    edges = [(i, i + 1, 1.0) for i in range(n - 1)]
    origins = [
        (18, 18, 19, 0.1, 0.9, 3.0),
        (38, 38, 39, 0.5, 0.5, 1.0),
        (58, 58, 59, 0.3, 0.7, 2.0),
        (98, 98, 99, 0.5, 0.5, 1.0),
    ]
    destinations = [
        (20, 20, 21, 0.4, 0.6, 2.0),
        (40, 40, 41, 0.5, 0.5, 1.5),
        (80, 80, 81, 0.5, 0.5, 1.0),
    ]
    return dict(node_count=n, edges=edges, origins=origins,
                destinations=destinations)


# ======================================================================
# F3-side: gradient stubs
# ======================================================================

class CaptureLogger:
    """Records every logger.log call as (args, kwargs)."""

    def __init__(self):
        self.calls = []

    def log(self, *args, **kwargs):
        self.calls.append((args, kwargs))

    def v2_lines(self):
        return [a[1] for a, k in self.calls
                if k.get("v") == 2 and len(a) >= 2]


def make_gradient_stub(arcs, n_net, n_dest, limit=np.inf, n_extra=0):
    """Build a stub engine exposing exactly the attributes
    _precompute_dest_gradients reads — a superset of the B0 read set,
    so the SAME builder feeds both arms.

    arcs: list of (u, v, weight) on the network+virtual node space.
    Node numbering: network [0, n_net), destination virtuals
    [n_net, n_net + n_dest), extras after that.
    """
    n_total = n_net + n_dest + n_extra
    order = sorted(range(len(arcs)), key=lambda i: (arcs[i][0], arcs[i][1]))
    indptr = np.zeros(n_total + 1, dtype=np.int64)
    indices = np.empty(len(arcs), dtype=np.int64)
    weights = np.empty(len(arcs), dtype=np.float64)
    for pos, i in enumerate(order):
        u, v, w = arcs[i]
        indices[pos] = v
        weights[pos] = w
        indptr[u + 1] += 1
    indptr = np.cumsum(indptr)
    fwd = csr_matrix((weights, indices, indptr), shape=(n_total, n_total))
    rev = fwd.T.tocsr()

    stub = types.SimpleNamespace()
    stub._n_network_nodes = n_net
    stub._n_destinations = n_dest
    stub._n_origins = n_extra
    stub._csr_indptr = np.asarray(fwd.indptr)
    stub._csr_indices = np.asarray(fwd.indices)
    stub._csr_weights = fwd.data
    stub._csr_fwd = fwd
    stub._csr_rev = rev
    stub._gradient_limit = lambda ns: limit
    stub.logger = CaptureLogger()
    return stub


def grad_arcs_chain():
    """Deterministic chain network (8 nodes) with two destination
    virtuals (8, 9) attached mid-chain plus one extra (origin) virtual
    (10): wide per-destination slices so forced-small caps produce real
    chunk schedules and forced-tiny caps produce NO_FIT."""
    arcs = [(i, i + 1, 1.0) for i in range(7)]          # chain 0..7
    arcs += [(7, 0, 1.0)]                                # close the ring
    arcs += [(2, 8, 0.5), (5, 8, 0.75)]                  # dest 8 fan-in
    arcs += [(3, 9, 0.5), (6, 9, 1.25)]                  # dest 9 fan-in
    arcs += [(4, 10, 0.5)]                               # origin virtual
    return arcs
