"""F1 fixtures: engine cases, micro kernel cases, and the serial
striped-replica of the flow origin driver.

Three fixture layers, all two-arm (B0 via oracle support.b0(), candidate
via cand_load — same arguments, byte-compared outputs):

* ENGINE_CASES  — AggregateFlow Centrality runs on the oracle's
  adversarial flow_network_spec, extended here with elevation,
  obstacle and observer topology variants (F1FlowTopology).
* micro cases   — hand-built tiny CSR graphs + engine-style dense
  scratch buffers, driven straight through each arm's COMPILED
  _accumulate_od_flow (T1/T13/M3 battery).
* striped_replica — serial Python replica of _process_origins_aggregate
  (mirroring the oracle trace_origin_loop approach) that calls the
  arm's COMPILED kernel per OD, keeps per-stripe partials before the
  reduction, and folds them in a knob-chosen order (T3; M1/M2 knobs).

Mutant knobs exist ONLY for the negative tests (M1 fold order, M2
contiguous partition); production knobs are the defaults.
"""
from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra as scipy_dijkstra

import fixtures
import stub_topology


# ======================================================================
# Topology variants (obstacles / elevation / observers)
# ======================================================================

class _StubObservers:
    """Minimal observer_points container read by the engine's
    _accumulate_observer_flows (edge-snapped)."""

    def __init__(self, edge_ids):
        self.geometry = np.zeros(len(edge_ids), dtype=object)
        self.snap_to = "edge"
        self.nearest_edge_id = np.asarray(edge_ids, dtype=np.int64)


class F1FlowTopology(stub_topology.StubFlowTopology):
    """StubFlowTopology plus the optional engine surfaces the oracle
    stub leaves switched off: net.z (elevation), obstacle arc
    penalties, observer points. Everything else is byte-identical to
    the oracle stub's behavior."""

    def __init__(self, spec, origin_weights=None, z=None,
                 obstacle_penalties=None, observer_edge_ids=None):
        super().__init__(spec, origin_weights=origin_weights)
        if z is not None:
            self.network.z = np.asarray(z, dtype=np.float64)
        self._obstacle_penalties = None
        if obstacle_penalties is not None:
            p_ab, p_ba = obstacle_penalties
            self._obstacle_penalties = (
                np.asarray(p_ab, dtype=np.float64),
                np.asarray(p_ba, dtype=np.float64),
            )
            self.obstacles = object()  # non-None sentinel, engine reads only this
        if observer_edge_ids is not None:
            self.observer_points = _StubObservers(observer_edge_ids)

    def get_obstacle_arc_penalties(self):
        return self._obstacle_penalties


N_EDGES = len(fixtures.flow_network_spec()["edges"])

# Per-node z for the elevation case (deterministic, non-monotone).
ELEV_Z = [0.0, 2.0, 0.5, 3.0, 1.0, 4.0, 0.25, 0.0, 1.5]
# Per-edge obstacle penalties (AB, BA) for the obstacle case.
OBS_P_AB = np.array([0.5, 0.0, 1.0, 0.0, 0.25, 0.0, 0.0, 0.75, 0.0, 0.0, 0.0, 0.5])
OBS_P_BA = np.array([0.0, 0.5, 0.0, 1.0, 0.0, 0.25, 0.75, 0.0, 0.0, 0.0, 0.0, 0.0])


# ======================================================================
# Engine cases (T2/T5/T8/T9/T12 batteries)
# ======================================================================

ENGINE_CASES = {
    "base":            dict(),
    "no_node_flow":    dict(node_flow=False),
    "decay_equal":     dict(overrides={"flow_path_detour_penalty": "equal"}),
    "decay_logistic":  dict(overrides={"flow_path_detour_penalty": "logistic"}),
    "elevation":       dict(z=ELEV_Z,
                            overrides={"elevation": True,
                                       "elevation_penalty": 4.0}),
    "obstacles":       dict(obstacle_penalties=(OBS_P_AB, OBS_P_BA)),
    "observers":       dict(observer_edge_ids=[0, 5, 11]),
    "tight_budget":    dict(overrides={"flow_detour_ratio": 1.0,
                                       "flow_detour_buffer": 0.0}),
    "no_o_weights":    dict(overrides={"flow_origin_weights": False}),
}


def build_engine(ns, case_name, num_threads, spec=None, extra_overrides=None):
    """Fresh engine + settings for one case at one K profile.

    Each (case, K) pair is a SEPARATE B0-vs-candidate reference pair;
    K is never compared across (proof 10: K-matched pairs only)."""
    case = ENGINE_CASES[case_name]
    spec = spec if spec is not None else fixtures.flow_network_spec()
    overrides = dict(fixtures.flow_settings_overrides(
        node_flow=(case.get("node_flow", True))))
    overrides.update(case.get("overrides", {}))
    if extra_overrides:
        overrides.update(extra_overrides)
    topo = F1FlowTopology(
        spec,
        z=case.get("z"),
        obstacle_penalties=case.get("obstacle_penalties"),
        observer_edge_ids=case.get("observer_edge_ids"),
    )
    settings = stub_topology.make_settings(ns.Settings, accessibility=False,
                                           **overrides)
    eng = ns.AggregateFlow(topo)
    eng.num_threads = int(num_threads)
    return eng, settings


def engine_output_arrays(eng):
    """All engine result arrays as a name->ndarray dict (None where the
    engine leaves a surface disabled)."""
    return {
        "edge_flow_AB": eng.edge_flow_AB,
        "edge_flow_BA": eng.edge_flow_BA,
        "edge_flow": eng.edge_flow,
        "node_flow": eng.node_flow,
        "observer_flow_AB": getattr(eng, "observer_flow_AB", None),
        "observer_flow_BA": getattr(eng, "observer_flow_BA", None),
        "observer_flow_total": getattr(eng, "observer_flow_total", None),
    }


# ======================================================================
# Micro kernel cases (T1/T13 battery, M3 mutant target)
# ======================================================================

def _build_csr(n_nodes, arcs):
    """arcs: (u, v, weight, edge_id, dir_bit). Returns engine-dtyped CSR
    arrays (int32 indptr/indices, f8 weights, i4 edge ids, i1 dirs)."""
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


def _trees(indptr, indices, weights, n_nodes, o_vn, d_vn):
    fwd = csr_matrix((weights, indices, indptr.astype(np.int64)),
                     shape=(n_nodes, n_nodes))
    rev = fwd.T.tocsr()
    d_o, pred_o = scipy_dijkstra(fwd, directed=True, indices=o_vn,
                                 return_predecessors=True)
    d_d, pred_d = scipy_dijkstra(rev, directed=True, indices=d_vn,
                                 return_predecessors=True)
    return (np.asarray(d_o, dtype=np.float64),
            np.asarray(pred_o, dtype=np.int32),
            np.asarray(d_d, dtype=np.float64),
            np.asarray(pred_d, dtype=np.int32))


class MicroCase:
    """One typed kernel-call argument set (engine-style buffers)."""

    def __init__(self, name, indptr, indices, weights, edge_id, dir_of,
                 d_o, dd_buf, pred_o, pd_buf, o_vn, d_vn, o_edge, d_edge,
                 d_shortest, budget, decay, trip_volume, n_net, n_edges,
                 node_flow):
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

    def out_buffers(self):
        n = self.indptr.shape[0] - 1
        out_ab = np.zeros(self.n_edges, dtype=np.float64)
        out_ba = np.zeros(self.n_edges, dtype=np.float64)
        out_node = (np.zeros(n, dtype=np.float64) if self.node_flow
                    else np.zeros(0, dtype=np.float64))
        return out_ab, out_ba, out_node

    def call(self, kernel):
        out_ab, out_ba, out_node = self.out_buffers()
        delivered = kernel(
            self.indptr, self.indices, self.weights, self.edge_id,
            self.dir_of, self.d_o, self.dd_buf, self.pred_o, self.pd_buf,
            self.o_vn, self.d_vn, self.o_edge, self.d_edge,
            self.d_shortest, self.budget,
            int(self.decay[0]), float(self.decay[1]), float(self.decay[2]),
            self.trip_volume, self.n_net,
            out_ab, out_ba, out_node,
        )
        return out_ab, out_ba, out_node, delivered


def _micro(name, n_net, arcs, o_vn, d_vn, o_edge, d_edge, budget=None,
           decay=(1, 0.3, 100.0), trip_volume=1.0, node_flow=True,
           scatter_cols=None, tight=False):
    """Build one MicroCase. `scatter_cols=None` scatters the FULL
    destination tree (dense semantics); a column list emulates the
    production sparse gradient (dd/pd live only on those columns)."""
    n_nodes = int(max(max(u, v) for u, v, *_ in arcs)
                   + 1)
    n_nodes = max(n_nodes, int(o_vn) + 1, int(d_vn) + 1)
    indptr, indices, weights, edge_id, dir_of = _build_csr(n_nodes, arcs)
    d_o, pred_o, d_d, pred_d = _trees(indptr, indices, weights, n_nodes,
                                      o_vn, d_vn)
    d_shortest = float(d_o[d_vn])
    # budget must stay FINITE: the production driver never calls the
    # kernel with a non-finite d_shortest/budget (it skips those ODs),
    # and under fastmath the kernel's inf comparisons are free to fold.
    if budget is None:
        if np.isfinite(d_shortest):
            budget = d_shortest * 1.5 + 0.2
        else:
            budget = 2.5
    if tight and np.isfinite(d_shortest):
        budget = d_shortest
    dd_buf = np.full(n_nodes, np.inf, dtype=np.float64)
    pd_buf = np.full(n_nodes, -9999, dtype=np.int32)
    if scatter_cols is None:
        live = np.isfinite(d_d)
        dd_buf[live] = d_d[live]
        pd_buf[live] = pred_d[live]
    else:
        cols = np.asarray(sorted(scatter_cols), dtype=np.int64)
        dd_buf[cols] = d_d[cols]
        pd_buf[cols] = pred_d[cols]
    n_edges = int(max(eid for *_, eid, _ in arcs)) + 1
    return MicroCase(name, indptr, indices, weights, edge_id, dir_of,
                     d_o, dd_buf, pred_o, pd_buf, o_vn, d_vn, o_edge,
                     d_edge, d_shortest, budget, decay, trip_volume,
                     n_net, n_edges, node_flow)


def _path_arcs(nodes, eid_of):
    """Bidirectional path arcs through `nodes`; eid_of[i] hosts
    (nodes[i] -> nodes[i+1])."""
    arcs = []
    for i in range(len(nodes) - 1):
        u, v = nodes[i], nodes[i + 1]
        eid = eid_of[i]
        arcs.append((u, v, 1.0, eid, 0))
        arcs.append((v, u, 1.0, eid, 1))
    return arcs


def _dest_connectors(arcs, d_vn, attach, eid):
    """Bidirectional destination connectors (engine semantics: both
    directions carry the snap edge id; dir bits 0/1)."""
    for node, dir_bit in attach:
        arcs.append((d_vn, node, 0.2 + 0.1 * dir_bit, eid, 0))
        arcs.append((node, d_vn, 0.2 + 0.1 * dir_bit, eid, 1))
    return arcs


def _origin_connectors(arcs, o_vn, attach, eid):
    """Unidirectional origin connectors (engine semantics: V_o -> net
    only)."""
    for node, dir_bit in attach:
        arcs.append((o_vn, node, 0.1 + 0.05 * dir_bit, eid, 0))
    return arcs


def micro_cases():
    """The T1 battery. Classes (proof 10): empty reach both kinds,
    q_sum==0, contamination marking, u-turn exclusion, snap-edge
    exclusion, node flow on/off zero-shape buffer, decay curves,
    zero trip volume, tight budget."""
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

    # Unreachable destination virtual node (no live connectors).
    arcs = _path_arcs([0, 1, 2, 3], [0, 1, 2])
    arcs = _origin_connectors(arcs, 4, [(0, 0)], 5)
    cases.append(_micro("m_unreach_dest", 4, arcs, 4, 6, 5, 6))

    # q_sum==0: joint envelope holds at the virtual nodes only, every
    # candidate via-arc falls outside it (sparse-gradient semantics,
    # budget == d_shortest).
    arcs = base_arcs([(0, 0)], [(2, 0), (3, 1)], 5, 6, 4, 5)
    cases.append(_micro("m_qsum_zero", 4, arcs, 4, 5, 5, 6,
                        scatter_cols=[4, 5], tight=True))

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

    # Degenerate trip volume (scale == 0 path) and tight budget.
    arcs = base_arcs([(0, 0)], [(2, 0), (3, 1)], 5, 6, 4, 5)
    cases.append(_micro("m_zero_trip", 4, arcs, 4, 5, 5, 6,
                        trip_volume=0.0))
    cases.append(_micro("m_budget_tight", 4, arcs, 4, 5, 5, 6,
                        tight=True))
    return cases


# ======================================================================
# Serial striped replica of the origin driver (T3; M1/M2 knobs)
# ======================================================================

def striped_replica(ns, engine, settings, K, partition="strided",
                    fold="ascending", kernel=None):
    """Serial replica of _process_origins_aggregate for a fixed K,
    calling the arm's COMPILED kernel per OD.

    Mirrors AggregateFlow.py:966-1160 (the driver this campaign pins):
    stripe membership range(slot, n_origins, K); per-stripe private
    AB/BA/node buffers plus dd_buf/pd_buf scatter/reset; slot-order
    reduction. The knobs exist ONLY for the M1/M2 negative tests:
      partition='contiguous' — block decomposition [b*(n//K):...] instead
        of the production strided range (M2).
      fold='descending'      — reduction folds slots K-1..0 (M1).
    Returns per-stripe partials, the folded finals, and bookkeeping."""
    if kernel is None:
        kernel = ns._accumulate_od_flow

    g_indptr, g_nodes, g_dist, g_pred = engine._precompute_dest_gradients(
        engine._prepare_params(settings))
    csr_indptr = engine._csr_indptr
    csr_indices = engine._csr_indices
    csr_weights = engine._csr_weights
    csr_edge_id = engine._csr_edge_id
    csr_direction = engine._csr_direction
    csr_fwd = engine._csr_fwd
    n_total_nodes = csr_indptr.shape[0] - 1
    n_net = engine._n_network_nodes
    n_dest = engine._n_destinations
    origins = engine.topology.origins
    dest = engine.topology.destinations
    n_origins = int(len(origins.node_weight))

    params = engine._prepare_params(settings)
    if params["path_penalty"] == "equal":
        decay_curve_id = ns._DECAY_EQUAL
    elif params["path_penalty"] == "exponential":
        decay_curve_id = ns._DECAY_EXPONENTIAL
    else:
        decay_curve_id = ns._DECAY_LOGISTIC
    decay_beta = float(params["route_beta"])
    decay_midpoint = (float(params["route_midpoint"])
                      if params["route_midpoint"] > 0 else 200.0)

    radius = float(params["search_radius"])
    gravity_beta = float(params["beta"])
    use_nearest = bool(params["closest_dest"])
    decay_on = bool(params["decay"])
    decay_curve_dcy = params["decay_curve"]
    use_o_weights = bool(params["use_o_weights"])
    use_d_weights = bool(params["use_d_weights"])
    dest_weights = np.asarray(dest.node_weight, dtype=np.float64)
    dest_node_ids = np.arange(n_net, n_net + n_dest, dtype=np.int64)
    dest_edge_ids = np.asarray(dest.nearest_edge_id, dtype=np.int64)
    mode = params["mode"]
    ratio = float(params["ratio"])
    buffer_ = float(params["buffer"])
    grad_limit = float(engine._gradient_limit(params))

    n_edges = engine.edge_flow_AB.shape[0]
    local_AB = [np.zeros(n_edges, dtype=np.float64) for _ in range(K)]
    local_BA = [np.zeros(n_edges, dtype=np.float64) for _ in range(K)]
    if engine.node_flow is not None:
        local_node = [np.zeros(n_net, dtype=np.float64) for _ in range(K)]
    else:
        _empty_node = np.zeros(0, dtype=np.float64)
        local_node = [_empty_node] * K

    stripe_members = []
    if partition == "strided":
        members = [list(range(slot, n_origins, K)) for slot in range(K)]
    elif partition == "contiguous":
        block = -(-n_origins // K)
        members = [list(range(slot * block,
                             min((slot + 1) * block, n_origins)))
                   for slot in range(K)]
    else:
        raise ValueError(f"unknown partition {partition!r}")
    stripe_members.extend(members)

    n_kernel_calls = 0
    for slot in range(K):
        buf_AB = local_AB[slot]
        buf_BA = local_BA[slot]
        buf_node = local_node[slot]
        dd_buf = np.full(n_total_nodes, np.inf, dtype=np.float64)
        pd_buf = np.full(n_total_nodes, -9999, dtype=np.int32)

        for o_pos in members[slot]:
            o_weight = float(origins.node_weight[o_pos])
            if use_o_weights and o_weight == 0.0:
                continue
            if not use_o_weights:
                o_weight = 1.0

            origin_virtual = int(engine._first_origin_node + o_pos)
            o_edge_id = int(origins.nearest_edge_id[o_pos])

            d_o, pred_o = scipy_dijkstra(
                csr_fwd, directed=True, indices=origin_virtual,
                limit=grad_limit, return_predecessors=True)
            d_o = np.asarray(d_o, dtype=np.float64)
            pred_o = np.asarray(pred_o, dtype=np.int32)

            d_shortest_arr = d_o[dest_node_ids]
            trip_vols = np.asarray(ns._compute_trip_volumes(
                o_weight, dest_weights, d_shortest_arr,
                radius, gravity_beta, decay_on, decay_curve_dcy,
                use_nearest, use_d_weights,
                params["decay_method"], float(params["gravity_cap"]),
            ), dtype=np.float64)

            for d_idx in range(n_dest):
                d_shortest = float(d_shortest_arr[d_idx])
                trip_vol = float(trip_vols[d_idx])
                if trip_vol <= 0.0 or not np.isfinite(d_shortest):
                    continue
                if d_shortest > radius:
                    continue
                budget = float(ns._cutoff_for_shortest(
                    d_shortest, mode, ratio, buffer_))

                s0, s1 = g_indptr[d_idx], g_indptr[d_idx + 1]
                cols = g_nodes[s0:s1]
                dd_buf[cols] = g_dist[s0:s1]
                pd_buf[cols] = g_pred[s0:s1]

                kernel(
                    csr_indptr, csr_indices, csr_weights,
                    csr_edge_id, csr_direction,
                    d_o, dd_buf, pred_o, pd_buf,
                    origin_virtual, int(dest_node_ids[d_idx]),
                    o_edge_id, int(dest_edge_ids[d_idx]),
                    d_shortest, budget,
                    decay_curve_id, decay_beta, decay_midpoint,
                    trip_vol, n_net,
                    buf_AB, buf_BA, buf_node,
                )
                n_kernel_calls += 1

                dd_buf[cols] = np.inf
                pd_buf[cols] = -9999

    order = (range(K - 1, -1, -1) if fold == "descending"
             else range(K))
    final_AB = np.zeros(n_edges, dtype=np.float64)
    final_BA = np.zeros(n_edges, dtype=np.float64)
    final_node = (np.zeros(n_net, dtype=np.float64)
                  if engine.node_flow is not None
                  else np.zeros(0, dtype=np.float64))
    fold_sequence = []
    for slot in order:
        final_AB += local_AB[slot]
        final_BA += local_BA[slot]
        if engine.node_flow is not None:
            final_node += local_node[slot]
        fold_sequence.append(int(slot))

    return {
        "local_AB": local_AB,
        "local_BA": local_BA,
        "local_node": local_node,
        "final_AB": final_AB,
        "final_BA": final_BA,
        "final_node": final_node,
        "stripe_members": stripe_members,
        "fold_sequence": fold_sequence,
        "n_kernel_calls": n_kernel_calls,
    }
