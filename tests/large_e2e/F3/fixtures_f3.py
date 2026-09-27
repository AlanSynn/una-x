"""F3 fixtures: stub gradient engines covering the dossier's fixture
list (spec tests_plan lines 31-33), the caller-measured selector-input
arithmetic used to force schedules through the frozen cap knob, and
the _scipy_dijkstra call-sequence interposer (restart-vs-hybrid
discriminator, proof section 7 / extension 4).

Fixture classes (dossier line 32): zero destinations; one destination;
disconnected/+inf; tied shortest paths (pred compared, not dist);
duplicated source nodes (NOT built: the function derives dest_nodes
from np.arange — duplicates impossible by construction, recorded as a
note); custom/obstacle-style costs (arbitrary nonnegative weights incl.
zero-weight arcs); sorted input/parallel arcs; empty finite rows
(limit=0); uneven final tail.

Cap forcing is exact integer arithmetic MIRRORING the implementation's
caller-side measurement (fixtures_f3.selector_inputs).  If the
implementation changes its fixed_live composition, the forced-c tests
fail loudly — that is the pin, not a fragility.
"""
from __future__ import annotations

import types

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra as scipy_dijkstra


# ======================================================================
# Stub engine
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


def make_stub(arcs, n_net, n_dest, limit=np.inf, n_extra=0):
    """Build a stub engine exposing exactly the attributes
    _precompute_dest_gradients reads (candidate) — a superset of the
    B0 read set, so the SAME builder feeds both arms.

    arcs: list of (u, v, weight) on the network+virtual node space.
    Node numbering: network [0, n_net), destination virtuals
    [n_net, n_net + n_dest), extras after that (origin virtuals etc.).
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


# ======================================================================
# Caller-measured selector inputs (mirror of the implementation's
# measurement; the forcing pin described in the module docstring)
# ======================================================================

def selector_inputs(stub):
    """The integers the implementation passes to select_chunk, derived
    from the SAME live arrays (proof section 3 inputs list)."""
    fwd, rev = stub._csr_fwd, stub._csr_rev
    n_total = stub._csr_indptr.shape[0] - 1
    dest_nodes = np.arange(stub._n_network_nodes,
                           stub._n_network_nodes + stub._n_destinations,
                           dtype=np.int64)
    counts = np.zeros(stub._n_destinations, dtype=np.int64)
    fixed_live = (fwd.data.nbytes + fwd.indices.nbytes + fwd.indptr.nbytes
                  + rev.data.nbytes + rev.indices.nbytes + rev.indptr.nbytes
                  + dest_nodes.nbytes + counts.nbytes)
    return {
        "v_prime": int(n_total),
        "dist_itemsize": np.dtype(np.float64).itemsize,
        "pred_itemsize": np.dtype(np.int32).itemsize,
        "mask_itemsize": np.dtype(np.bool_).itemsize,
        "fixed_live_base": int(fixed_live),
        "row_temporaries": 8 * int(n_total),
        "source_index_cost": int(dest_nodes.dtype.itemsize),
    }


def per_source(inp):
    return (inp["v_prime"] * (inp["dist_itemsize"] + inp["pred_itemsize"]
                              + inp["mask_itemsize"])
            + inp["row_temporaries"] + inp["source_index_cost"])


def cap_for_first_slice_c(stub, lfws, c):
    """DEFAULT_CAP_BYTES value whose first budgeted selection is exactly
    c (budget_k == c * per_source)."""
    inp = selector_inputs(stub)
    return (inp["fixed_live_base"] + lfws.MARGIN_BYTES
            + c * per_source(inp))


def cap_entry_no_fit(stub, lfws):
    """Cap value that makes even c=1 not fit at m_k=0 (entry NO_FIT)."""
    inp = selector_inputs(stub)
    return inp["fixed_live_base"] + lfws.MARGIN_BYTES + per_source(inp) - 1


# ======================================================================
# _scipy_dijkstra call-sequence interposer
# ======================================================================

class CallRecorder:
    """Wraps the candidate module-global _scipy_dijkstra; records the
    (s, e) slice bounds of every call and, optionally, weakref
    finalizers on each call's returned arrays.

    With track_finalizers=True, at the ENTRY of call k (k >= 1) the
    wrapper checks that BOTH arrays (dist, preds) of EVERY prior call
    have already been finalized — i.e. the previous slice's dense pair
    was released before the next Dijkstra call allocated (proof
    section 5's del-before-next-call obligation, verified at CPython
    refcount granularity)."""

    def __init__(self, track_finalizers=False):
        self.bounds = []
        self.index_lens = []
        self.track_finalizers = track_finalizers
        self.counters = []       # per call: [fires] (expect 2 = dist+preds)
        self.entry_checks = []   # per call k >= 1: all prior counters == 2
        self._real = None

    @staticmethod
    def _bump(counter):
        counter[0] += 1

    def install(self, flow_module):
        import weakref
        recorder = self
        self._real = flow_module._scipy_dijkstra

        def wrapper(csr, directed=True, indices=None, limit=np.inf,
                    return_predecessors=False):
            recorder.bounds.append((int(indices[0]), int(indices[-1]) + 1))
            recorder.index_lens.append(int(indices.shape[0]))
            if recorder.track_finalizers:
                ok = all(c[0] == 2 for c in recorder.counters)
                recorder.entry_checks.append(ok)
            dist, preds = recorder._real(
                csr, directed=directed, indices=indices, limit=limit,
                return_predecessors=return_predecessors)
            if recorder.track_finalizers:
                counter = [0]
                weakref.finalize(dist, CallRecorder._bump, counter)
                weakref.finalize(preds, CallRecorder._bump, counter)
                recorder.counters.append(counter)
            return dist, preds

        flow_module._scipy_dijkstra = wrapper
        return self

    def uninstall(self, flow_module):
        flow_module._scipy_dijkstra = self._real

    def position_bounds(self, n_net):
        """Recorded bounds converted from node ids to slice positions:
        indices are dest_nodes[s:e] = arange(n_net, ...) slices, so
        s = first_id - n_net and e = s + len(indices)."""
        return [(first - n_net, first - n_net + n)
                for (first, _), n in zip(self.bounds, self.index_lens)]


# ======================================================================
# Reference implementation — the ORIGINAL element-count formula loop,
# byte-for-byte the B0 :908-961 body (pre-F3), used as the in-process
# reference arm alongside the real B0 module arm.
# ======================================================================

def reference_gradient(stub, limit):
    n_net = stub._n_network_nodes
    n_dest = stub._n_destinations
    n_total = stub._csr_indptr.shape[0] - 1
    dest_nodes = np.arange(n_net, n_net + n_dest, dtype=np.int64)
    chunk = max(1, int(1e8 // max(n_total, 1)))
    idx_parts, dist_parts, pred_parts = [], [], []
    counts = np.zeros(n_dest, dtype=np.int64)
    for s in range(0, n_dest, chunk):
        e = min(s + chunk, n_dest)
        dist, preds = scipy_dijkstra(
            stub._csr_rev, directed=True,
            indices=dest_nodes[s:e], limit=limit,
            return_predecessors=True,
        )
        if dist.ndim == 1:
            dist, preds = dist[None, :], preds[None, :]
        finite = np.isfinite(dist)
        for k in range(e - s):
            cols = np.where(finite[k])[0]
            counts[s + k] = cols.shape[0]
            idx_parts.append(cols.astype(np.int64))
            dist_parts.append(dist[k, cols].astype(np.float64))
            pred_parts.append(preds[k, cols].astype(np.int32))
    indptr = np.zeros(n_dest + 1, dtype=np.int64)
    np.cumsum(counts, out=indptr[1:])
    nodes = (np.concatenate(idx_parts) if idx_parts
             else np.zeros(0, np.int64))
    dist = (np.concatenate(dist_parts) if dist_parts
            else np.zeros(0, np.float64))
    pred = (np.concatenate(pred_parts) if pred_parts
            else np.zeros(0, np.int32))
    return indptr, nodes, dist, pred


# ======================================================================
# Fixture graph builders (dossier fixture list)
# ======================================================================

def _dest_arc(n_net, d, targets):
    """Arcs from destination virtual d to network nodes (backward
    search: dijkstra runs on the REVERSED CSR from the virtuals)."""
    return [(t, n_net + d, 1.0) for t in targets]


def base_case():
    """Connected base: 6 network nodes, 5 destinations, varied costs."""
    arcs = [(0, 1, 1.0), (1, 2, 2.0), (0, 2, 5.0), (2, 3, 1.0),
            (3, 4, 1.0), (4, 5, 1.0), (1, 3, 7.0), (5, 2, 2.5)]
    for d in range(5):
        arcs += _dest_arc(6, d, [d + 1, (d + 3) % 6])
    return dict(name="base", arcs=arcs, n_net=6, n_dest=5, limit=np.inf)


def zero_dest_case():
    arcs = [(0, 1, 1.0), (1, 2, 2.0)]
    return dict(name="zero_dest", arcs=arcs, n_net=3, n_dest=0,
                limit=np.inf)


def one_dest_case():
    arcs = [(0, 1, 1.0), (1, 2, 2.0), (2, 0, 1.5)]
    arcs += _dest_arc(3, 0, [0, 1, 2])
    return dict(name="one_dest", arcs=arcs, n_net=3, n_dest=1,
                limit=np.inf)


def disconnected_case():
    """Destination 1 hangs off an isolated network node (+inf rows);
    destination 0 is reachable."""
    arcs = [(0, 1, 1.0), (1, 2, 1.0)]
    arcs += _dest_arc(3, 0, [0, 1, 2])
    arcs += _dest_arc(3, 1, [4])          # node 4 has no in-arcs
    return dict(name="disconnected", arcs=arcs, n_net=5, n_dest=2,
                limit=np.inf)


def tied_case():
    """Genuine predecessor tie: node 3's two equal-cost predecessors
    (nodes 1 and 2) — pred arrays compared bit-for-bit, never dist
    alone."""
    arcs = [(0, 1, 1.0), (0, 2, 1.0),   # d(1) = d(2) symmetric
            (1, 3, 1.0), (2, 3, 1.0),   # equal-cost preds of 3
            (3, 4, 2.0)]
    arcs += _dest_arc(5, 0, [3, 4])
    arcs += _dest_arc(5, 1, [4])
    return dict(name="tied", arcs=arcs, n_net=5, n_dest=2, limit=np.inf)


def parallel_arcs_case():
    """Sorted rows with PARALLEL arcs (duplicate (u,v) with different
    weights) and a zero-weight arc — nonneg custom costs."""
    arcs = [(0, 1, 2.0), (0, 1, 5.0), (0, 2, 0.0), (1, 2, 1.0),
            (2, 3, 3.0), (0, 3, 9.0)]
    arcs += _dest_arc(4, 0, [1, 2, 3])
    arcs += _dest_arc(4, 1, [3])
    return dict(name="parallel_arcs", arcs=arcs, n_net=4, n_dest=2,
                limit=np.inf)


def custom_costs_case():
    """Arbitrary nonnegative (non-metric) costs — custom-cost class."""
    arcs = [(0, 1, 0.37), (1, 2, 11.2), (0, 2, 5.05), (2, 3, 0.01),
            (3, 1, 7.7), (1, 4, 2.25), (4, 5, 3.5), (5, 3, 1.1)]
    arcs += _dest_arc(6, 0, [0, 2, 4])
    arcs += _dest_arc(6, 1, [1, 3, 5])
    arcs += _dest_arc(6, 2, [2])
    return dict(name="custom_costs", arcs=arcs, n_net=6, n_dest=3,
                limit=np.inf)


def empty_rows_case():
    """limit=0.0: every finite mask empty — all parts empty arrays."""
    arcs = [(0, 1, 1.0), (1, 2, 1.0)]
    arcs += _dest_arc(3, 0, [0, 1, 2])
    arcs += _dest_arc(3, 1, [1])
    return dict(name="empty_rows", arcs=arcs, n_net=3, n_dest=2,
                limit=0.0)


def partial_empty_case():
    """One destination within limit, one beyond it (+inf row) — mixed
    empty/non-empty finite rows."""
    arcs = [(0, 1, 1.0), (1, 2, 50.0)]
    arcs += _dest_arc(3, 0, [0])       # within limit 10
    arcs += _dest_arc(3, 1, [2])       # d = 50 > limit -> empty row
    return dict(name="partial_empty", arcs=arcs, n_net=3, n_dest=2,
                limit=10.0)


def uneven_tail_case():
    """7 destinations — any slice width leaves an uneven final tail."""
    arcs = [(i, (i + 1) % 8, 1.0 + 0.1 * i) for i in range(8)]
    for d in range(7):
        arcs += _dest_arc(8, d, [d, (d + 2) % 8, (d + 5) % 8])
    return dict(name="uneven_tail", arcs=arcs, n_net=8, n_dest=7,
                limit=np.inf)


def all_cases():
    return [base_case(), zero_dest_case(), one_dest_case(),
            disconnected_case(), tied_case(), parallel_arcs_case(),
            custom_costs_case(), empty_rows_case(), partial_empty_case(),
            uneven_tail_case()]


def case_stub(case):
    return make_stub(case["arcs"], case["n_net"], case["n_dest"],
                     limit=case["limit"])
