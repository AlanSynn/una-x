"""F2 fixtures: micro kernel cases (baseline-vs-local differential) and
engine builders for the driver-level batteries.

Single import arm: this worktree's src (the F2I implementation). The
differential is the ROUTE toggle INSIDE the one module — the unchanged
baseline kernel `_accumulate_od_flow` is the reference, the new
`_accumulate_od_flow_local` the candidate; engine-level differentials
flip `_use_local_route`.

Production-faithful emulation of the driver contract:
* dd_buf / pd_buf are +inf / -9999 except on the destination's finite
  gradient slice columns (`cols`), which come from a LIMIT-bounded
  backward Dijkstra off the destination virtual node (engine
  `_precompute_dest_gradients` semantics) and are strictly ascending
  (the dossier step-1 precondition the engine verifies per run);
* out_node_flow has shape n_net when node flow is on, shape 0 when off;
* origin connectors are out-only, destination connectors
  bidirectional; connector arcs carry the snap edge id.
"""
from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra as scipy_dijkstra

import fixtures
import stub_topology
from urban_network_analysis.Engines import AggregateFlow as AF
from urban_network_analysis.Settings import Settings


# ======================================================================
# Engine-level builders (driver batteries)
# ======================================================================

def big_flow_spec():
    """Long sparse path spec for DEFAULT-route engagement tests.

    The shared oracle stub is so small that the production dense-overlap
    crossover (`slice <= n_total // 8`) refuses every OD — correct
    behavior, but it leaves the default routing decision unexercised at
    engine level. This spec gives n_total=127 with per-destination
    gradient slices of ~5-11 nodes, so at search_radius 4.0 the REAL
    `_use_local_route` admits real ODs (probed: fast_calls=2, nonzero
    flow) without bypassing any engine decision. Never modified by
    tests — built here, oracle module untouched.
    """
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


def build_engine(num_threads, extra_overrides=None, spec=None):
    """Fresh AggregateFlow engine + Settings on the oracle flow spec."""
    spec = spec if spec is not None else fixtures.flow_network_spec()
    overrides = fixtures.flow_settings_overrides(node_flow=True)
    if extra_overrides:
        overrides.update(extra_overrides)
    topo = stub_topology.StubFlowTopology(spec)
    settings = stub_topology.make_settings(Settings, accessibility=False,
                                           **overrides)
    eng = AF.AggregateFlow(topo)
    eng.num_threads = int(num_threads)
    return eng, settings


def engine_output_arrays(eng):
    return {
        "edge_flow_AB": eng.edge_flow_AB,
        "edge_flow_BA": eng.edge_flow_BA,
        "edge_flow": eng.edge_flow,
        "node_flow": eng.node_flow,
    }


def assert_engine_bytes(a, b, label):
    for k in a:
        va, vb = a[k], b[k]
        if va is None or vb is None:
            assert (va is None) == (vb is None), f"{label}/{k}: None mismatch"
            continue
        assert va.tobytes() == vb.tobytes(), \
            f"{label}/{k}: engine outputs diverge (F2 route not exact)"


# ======================================================================
# Micro kernel cases
# ======================================================================

def _build_csr(n_nodes, arcs):
    """arcs: (u, v, weight, edge_id, dir_bit) -> engine-dtyped CSR."""
    order = sorted(range(len(arcs)), key=lambda i: (arcs[i][0], arcs[i][1]))
    indptr = np.zeros(n_nodes + 1, dtype=np.int32)
    indices = np.empty(len(arcs), dtype=np.int32)
    weights = np.empty(len(arcs), dtype=np.float64)
    edge_id = np.empty(len(arcs), dtype=np.int32)
    dir_of = np.empty(len(arcs), dtype=np.int8)
    for pos, i in enumerate(order):
        u, v, w, eid, d = arcs[i]
        indices[pos] = v
        weights[pos] = w
        edge_id[pos] = eid
        dir_of[pos] = d
        indptr[u + 1] += 1
    indptr = np.cumsum(indptr).astype(np.int32)
    return indptr, indices, weights, edge_id, dir_of


def _trees(indptr, indices, weights, n_nodes, o_vn, d_vn, limit=None):
    """Forward tree from the origin virtual, LIMIT-bounded backward tree
    from the destination virtual (engine gradient semantics)."""
    fwd = csr_matrix((weights, indices, indptr.astype(np.int64)),
                     shape=(n_nodes, n_nodes))
    rev = fwd.T.tocsr()
    d_o, pred_o = scipy_dijkstra(fwd, directed=True, indices=o_vn,
                                 return_predecessors=True)
    if limit is None:
        d_d, pred_d = scipy_dijkstra(rev, directed=True, indices=d_vn,
                                     return_predecessors=True)
    else:
        d_d, pred_d = scipy_dijkstra(rev, directed=True, indices=d_vn,
                                     limit=limit, return_predecessors=True)
    return (np.asarray(d_o, dtype=np.float64),
            np.asarray(pred_o, dtype=np.int32),
            np.asarray(d_d, dtype=np.float64),
            np.asarray(pred_d, dtype=np.int32))


class MicroCase:
    """One typed kernel-call argument set plus F2 scratch management."""

    def __init__(self, name, indptr, indices, weights, edge_id, dir_of,
                 d_o, dd_buf, pred_o, pd_buf, cols, o_vn, d_vn, o_edge,
                 d_edge, d_shortest, budget, decay, trip_volume, n_net,
                 n_edges, node_flow):
        self.name = name
        self.indptr = indptr
        self.indices = indices
        self.weights = weights
        self.edge_id = edge_id
        self.dir_of = dir_of
        self.d_o = d_o
        self.dd_buf = dd_buf
        self.pred_o = pred_o
        self.pd_buf = pd_buf
        self.cols = cols
        self.o_vn = int(o_vn)
        self.d_vn = int(d_vn)
        self.o_edge = int(o_edge)
        self.d_edge = int(d_edge)
        self.d_shortest = float(d_shortest)
        self.budget = float(budget)
        self.decay = decay          # (curve_id, beta, midpoint)
        self.trip_volume = float(trip_volume)
        self.n_net = int(n_net)
        self.n_edges = int(n_edges)
        self.node_flow = bool(node_flow)
        self.n_total = int(indptr.shape[0] - 1)

    # -- scratch -------------------------------------------------------
    def alloc_scratch(self):
        n = self.n_total
        sc = {
            "reach": np.zeros(n, dtype=np.bool_),
            "cont_o": np.zeros(n, dtype=np.bool_),
            "cont_d": np.zeros(n, dtype=np.bool_),
            "acc_o": np.zeros(n, dtype=np.float64),
            "acc_d": np.zeros(n, dtype=np.float64),
            "reach_nodes": np.empty(n, dtype=np.int64),
            "t_reach": np.empty(n, dtype=np.int64),
            "t_cont_o": np.empty(n, dtype=np.int64),
            "t_cont_d": np.empty(n, dtype=np.int64),
            "t_acc_o": np.empty(n + self.n_edges, dtype=np.int64),
            "t_acc_d": np.empty(n + self.n_edges, dtype=np.int64),
        }
        return sc

    @staticmethod
    def pristine_scratch(sc):
        return {k: v.copy() for k, v in sc.items()
                if k in ("reach", "cont_o", "cont_d", "acc_o", "acc_d")}

    @staticmethod
    def assert_pristine(sc, label):
        for k in ("reach", "cont_o", "cont_d", "acc_o", "acc_d"):
            ref = np.zeros(sc[k].shape[0],
                           dtype=sc[k].dtype)
            assert sc[k].tobytes() == ref.tobytes(), \
                f"{label}: scratch array {k} not pristine after reset"

    # -- kernel calls ----------------------------------------------------
    def _args(self):
        return (self.indptr, self.indices, self.weights, self.edge_id,
                self.dir_of, self.d_o, self.dd_buf, self.pred_o,
                self.pd_buf, self.o_vn, self.d_vn, self.o_edge,
                self.d_edge, self.d_shortest, self.budget,
                int(self.decay[0]), float(self.decay[1]),
                float(self.decay[2]), self.trip_volume, self.n_net)

    def call_orig(self):
        out_node = (np.zeros(self.n_net, dtype=np.float64)
                    if self.node_flow else np.zeros(0, dtype=np.float64))
        out_ab = np.zeros(self.n_edges, dtype=np.float64)
        out_ba = np.zeros(self.n_edges, dtype=np.float64)
        delivered = AF._accumulate_od_flow(*self._args(),
                                           out_ab, out_ba, out_node)
        return out_ab, out_ba, out_node, delivered

    def call_local(self, sc=None, caps=None):
        """Run the local kernel. `caps` optionally overrides the touched
        list arrays with tiny ones (overflow leg). Returns the three
        output buffers, delivered, the of flag and the scratch dict."""
        sc = self.alloc_scratch() if sc is None else sc
        if caps is not None:
            for k, n in caps.items():
                sc[k] = np.empty(n, dtype=np.int64)
        out_node = (np.zeros(self.n_net, dtype=np.float64)
                    if self.node_flow else np.zeros(0, dtype=np.float64))
        out_ab = np.zeros(self.n_edges, dtype=np.float64)
        out_ba = np.zeros(self.n_edges, dtype=np.float64)
        delivered, of = AF._accumulate_od_flow_local(
            *self._args(), out_ab, out_ba, out_node, self.cols,
            sc["reach"], sc["cont_o"], sc["cont_d"], sc["acc_o"],
            sc["acc_d"], sc["reach_nodes"],
            sc["t_reach"], sc["t_cont_o"], sc["t_cont_d"],
            sc["t_acc_o"], sc["t_acc_d"])
        return out_ab, out_ba, out_node, delivered, int(of), sc

    def expected_reach(self):
        """Ascending ground-truth reach set from the cols + predicate."""
        dov = self.d_o[self.cols]
        ddv = self.dd_buf[self.cols]
        keep = (dov < np.inf) & (ddv < np.inf) & (dov + ddv <= self.budget)
        return self.cols[keep]


def _path_arcs(nodes, eid_of):
    arcs = []
    for i in range(len(nodes) - 1):
        u, v = nodes[i], nodes[i + 1]
        arcs.append((u, v, 1.0, eid_of[i], 0))
        arcs.append((v, u, 1.0, eid_of[i], 1))
    return arcs


def _dest_connectors(arcs, d_vn, attach, eid):
    for node, dir_bit in attach:
        arcs.append((d_vn, node, 0.2 + 0.1 * dir_bit, eid, 0))
        arcs.append((node, d_vn, 0.2 + 0.1 * dir_bit, eid, 1))
    return arcs


def _origin_connectors(arcs, o_vn, attach, eid):
    for node, dir_bit in attach:
        arcs.append((o_vn, node, 0.1 + 0.05 * dir_bit, eid, 0))
    return arcs


def _micro(name, n_net, arcs, o_vn, d_vn, o_edge, d_edge, budget=None,
           decay=(1, 0.3, 100.0), trip_volume=1.0, node_flow=True,
           grad_limit=None, tight=False, cols_override=None):
    """Build one MicroCase with production scatter semantics: the
    backward tree (optionally limit-bounded) defines the slice; only
    slice columns are scattered into dd_buf/pd_buf. `cols_override`
    pins an explicit slice (q_sum<=0 adversarial construction)."""
    n_nodes = int(max(max(u, v) for u, v, *_ in arcs) + 1)
    n_nodes = max(n_nodes, int(o_vn) + 1, int(d_vn) + 1)
    indptr, indices, weights, edge_id, dir_of = _build_csr(n_nodes, arcs)
    d_o, pred_o, d_d, pred_d = _trees(indptr, indices, weights, n_nodes,
                                      o_vn, d_vn, limit=grad_limit)
    d_shortest = float(d_o[d_vn])
    if budget is None:
        if np.isfinite(d_shortest):
            budget = d_shortest * 1.5 + 0.2
        else:
            budget = 2.5
    if tight and np.isfinite(d_shortest):
        budget = d_shortest
    if cols_override is None:
        cols = np.flatnonzero(np.isfinite(d_d)).astype(np.int64)
    else:
        cols = np.asarray(sorted(int(v) for v in cols_override),
                          dtype=np.int64)
    dd_buf = np.full(n_nodes, np.inf, dtype=np.float64)
    pd_buf = np.full(n_nodes, -9999, dtype=np.int32)
    dd_buf[cols] = d_d[cols]
    pd_buf[cols] = pred_d[cols]
    n_edges = int(max(eid for *_, eid, _ in arcs)) + 1
    return MicroCase(name, indptr, indices, weights, edge_id, dir_of,
                     d_o, dd_buf, pred_o, pd_buf, cols, o_vn, d_vn,
                     o_edge, d_edge, d_shortest, budget, decay,
                     trip_volume, n_net, n_edges, node_flow)


def micro_cases():
    """The F2 micro battery: the dossier's adversarial classes, each run
    through BOTH kernels with a bitwise differential + pristine check."""
    cases = []

    def base_arcs(o_attach, d_attach, o_eid, d_eid, o_vn, d_vn):
        arcs = _path_arcs([0, 1, 2, 3], [0, 1, 2])
        arcs = _origin_connectors(arcs, o_vn, o_attach, o_eid)
        arcs = _dest_connectors(arcs, d_vn, d_attach, d_eid)
        return arcs

    # Full pipeline on a 4-node path, node flow on and off.
    arcs = base_arcs([(0, 0)], [(2, 0), (3, 1)], 5, 6, 4, 5)
    cases.append(_micro("m_base", 4, arcs, 4, 5, 5, 6))
    cases.append(_micro("m_base_nonode", 4, arcs, 4, 5, 5, 6,
                        node_flow=False))

    # Early exit: unreachable destination virtual (no live connectors).
    arcs = _path_arcs([0, 1, 2, 3], [0, 1, 2])
    arcs = _origin_connectors(arcs, 4, [(0, 0)], 5)
    cases.append(_micro("m_unreach_dest", 4, arcs, 4, 6, 5, 6))

    # q_sum==0: the slice holds EXACTLY the two virtual nodes, so the
    # joint envelope holds there only (both virtuals reached -> no
    # early exit) and every via-arc check fails on reach[x] — pass 1
    # admits nothing and q_sum is 0.
    arcs = base_arcs([(0, 0)], [(2, 0), (3, 1)], 5, 6, 4, 5)
    cases.append(_micro("m_qsum_zero", 4, arcs, 4, 5, 5, 6,
                        tight=True, cols_override=[4, 5]))

    # U-turn / dead-end: destination on the leaf of a spur.
    arcs = _path_arcs([0, 1, 2, 3], [0, 1, 2])
    arcs += [(3, 4, 1.0, 3, 0), (4, 3, 1.0, 3, 1)]      # spur to leaf 4
    arcs = _origin_connectors(arcs, 5, [(0, 0)], 4)
    arcs = _dest_connectors(arcs, 6, [(4, 0)], 5)
    cases.append(_micro("m_uturn_leaf", 5, arcs, 5, 6, 4, 5))

    # Same host edge for origin and destination (snap-edge exclusion).
    arcs = _path_arcs([0, 1, 2, 3], [0, 1, 2])
    arcs = _origin_connectors(arcs, 4, [(0, 0), (1, 1)], 0)
    arcs = _dest_connectors(arcs, 5, [(0, 0), (1, 1)], 0)
    cases.append(_micro("m_same_edge", 4, arcs, 4, 5, 0, 0))

    # Contamination: destination snaps mid-path; origin legs crossing
    # the destination snap edge must be marked, and vice versa.
    arcs = _path_arcs([0, 1, 2, 3], [0, 1, 2])
    arcs = _origin_connectors(arcs, 4, [(0, 0)], 0)
    arcs = _dest_connectors(arcs, 5, [(1, 0), (2, 1)], 1)
    cases.append(_micro("m_contam", 4, arcs, 4, 5, 0, 1))

    # Decay curves over the base network.
    for curve_id, tag in ((0, "equal"), (1, "exponential"),
                          (2, "logistic")):
        arcs = base_arcs([(0, 0)], [(2, 0), (3, 1)], 5, 6, 4, 5)
        cases.append(_micro(f"m_decay_{tag}", 4, arcs, 4, 5, 5, 6,
                            decay=(curve_id, 0.3, 50.0)))

    # Degenerate trip volume and tight budget.
    arcs = base_arcs([(0, 0)], [(2, 0), (3, 1)], 5, 6, 4, 5)
    cases.append(_micro("m_zero_trip", 4, arcs, 4, 5, 5, 6,
                        trip_volume=0.0))
    cases.append(_micro("m_budget_tight", 4, arcs, 4, 5, 5, 6,
                        tight=True))

    # Genuinely sparse slice (F2 regime): limit-bounded backward tree.
    arcs = base_arcs([(0, 0)], [(2, 0), (3, 1)], 5, 6, 4, 5)
    cases.append(_micro("m_sparse_slice", 4, arcs, 4, 5, 5, 6,
                        grad_limit=2.6))

    # Weight ties: uniform weights everywhere (argsort tie behavior is
    # identical code in both kernels; the differential pins it).
    arcs = _path_arcs([0, 1, 2, 3], [0, 1, 2])
    arcs = _origin_connectors(arcs, 4, [(0, 0)], 5)
    arcs = _dest_connectors(arcs, 5, [(3, 0)], 6)
    cases.append(_micro("m_ties", 4, arcs, 4, 5, 5, 6,
                        decay=(2, 0.0, 100.0)))

    return cases


def random_micro_case(seed):
    """Seeded random digraph + engine-style connectors + a random OD.
    Generates pv-outside-reach, disjoint envelopes and duplicate reset
    targets naturally."""
    rng = np.random.default_rng(seed)
    n_net = int(rng.integers(4, 12))
    arcs = []
    eid = 0
    for u in range(n_net):
        for v in range(n_net):
            if u != v and rng.random() < 0.35:
                w = float(rng.uniform(0.2, 2.0))
                arcs.append((u, v, w, eid, 0))
                if rng.random() < 0.6:      # sometimes bidirectional
                    arcs.append((v, u, w, eid, 1))
                eid += 1
    if not arcs:
        arcs.append((0, 1, 1.0, 0, 0))
        arcs.append((1, 0, 1.0, 0, 1))
        eid = 1
    o_vn = n_net
    d_vn = n_net + 1
    o_attach = [(int(rng.integers(0, n_net)), 0)]
    d_attach = [(int(rng.integers(0, n_net)), 0)]
    o_eid, d_eid = eid, eid + 1
    arcs = _origin_connectors(arcs, o_vn, o_attach, o_eid)
    arcs = _dest_connectors(arcs, d_vn, d_attach, d_eid)
    ratio = float(rng.uniform(1.0, 1.6))
    return _fuzz_micro(f"r_case_{seed}", n_net, arcs, o_vn, d_vn, o_eid,
                       d_eid, ratio)


def _fuzz_micro(name, n_net, arcs, o_vn, d_vn, o_eid, d_eid, ratio):
    """Fuzz variant: budget = d_shortest * ratio (engine 'ratio' mode)."""
    case = _micro(name, n_net, arcs, o_vn, d_vn, o_eid, d_eid)
    if np.isfinite(case.d_shortest):
        case.budget = case.d_shortest * ratio
    return case
