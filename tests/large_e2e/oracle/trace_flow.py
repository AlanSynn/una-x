"""Explanatory Python traces of the AggregateFlow kernel and driver.

trace_accumulate_od_flow replicates B0's _accumulate_od_flow step by
step with a per-OD journal (reach list, both argsort orders, both
contamination masks, q_sum, scale, acc_o/acc_d, out deltas, delivered).
The journal's out_AB/out_BA/out_node_flow are THIS OD's DELTA: the
exact += writes the kernel performed, replayed in the same order into
zero buffers (byte-identical to a fresh-buffer compiled run; the live
driver buffers are additionally recorded as out_*_cumulative). The
replica still accumulates into the caller's live buffers so the
per-stripe fold reproduces the driver's float association exactly.
Scalar exp/arc-lookup semantics are taken from the COMPILED B0 _decay
and _find_arc callables passed in as `decay_fn` / `find_arc_fn` so the
journal uses the frozen profile's exact floating behavior (numba np.exp
is not assumed bit-equal to numpy's).

trace_origin_loop replicates _process_origins_aggregate serially for a
fixed K: static stripe membership range(slot, n_origins, K), per-stripe
dd_buf/pd_buf scatter/reset, per-stripe partials, and the final
reduction folding slots 0..K-1 in order. numpy argsort is used for the
tree-pass orders; the empirical tie behavior is separately pinned by
the golden tests (argsort-tie fixtures) — any divergence there is a
finding, not something to paper over.

Mutant knobs exist ONLY for the negative-mutation tests.
"""
from __future__ import annotations

import numpy as np


def trace_accumulate_od_flow(decay_fn, find_arc_fn,
                             indptr, indices, weights, edge_id_of_arc,
                             dir_of_arc, d_o, d_d, pred_o, pred_d,
                             origin_virtual_node, dest_virtual_node,
                             o_edge_id, d_edge_id, d_shortest, budget,
                             decay_curve_id, decay_beta, decay_midpoint,
                             trip_volume, n_net,
                             out_AB=None, out_BA=None, out_node_flow=None,
                             journal=True, stale_scratch=False,
                             arc_evals=None):
    """Faithful replica of the via-arc loading kernel.

    decay_fn(curve_id, beta, midpoint, excess) and
    find_arc_fn(indptr, indices, u, v) MUST be the compiled B0 callables.

    out_* default to fresh zero buffers (delta semantics). Pass live
    driver buffers to accumulate into them.

    stale_scratch=True is the F2 negative mutant marker hook — the
    scratch-reset behavior lives in the driver, not here; the driver
    replica in trace_origin_loop uses it to skip the dd_buf/pd_buf
    reset between ODs.

    Returns (out_AB, out_BA, out_node_flow, delivered, journal).
    """
    n_nodes = d_o.shape[0]
    if out_AB is None or out_BA is None or out_node_flow is None:
        raise ValueError(
            "out_AB/out_BA/out_node_flow are required; allocate fresh zero "
            "buffers for delta semantics or pass live driver buffers")
    journal_data = {}
    # Every += the kernel performs on the caller's live buffers is
    # recorded here so the journal can report THIS OD's delta. The delta
    # is reconstructed by replaying the exact same writes in the exact
    # same order into zero buffers — identical bytes to what a fresh
    # compiled run with fresh buffers produces (same q/f values, same
    # association order); no float subtraction is ever used.
    writes = []

    reach = np.zeros(n_nodes, dtype=np.bool_)
    n_r = 0
    reach_decisions = []
    for v in range(n_nodes):
        dov = d_o[v]
        ddv = d_d[v]
        accept = bool(dov < np.inf and ddv < np.inf and dov + ddv <= budget)
        reach[v] = accept
        if accept:
            n_r += 1
        reach_decisions.append(accept)

    if not reach[dest_virtual_node] or not reach[origin_virtual_node]:
        journal_data.update({
            "early_return": "no_od_reach",
            "reach": reach.copy(),
            "n_r": n_r,
            "writes": writes,
            "out_AB": np.zeros_like(out_AB),
            "out_BA": np.zeros_like(out_BA),
            "out_node_flow": np.zeros_like(out_node_flow),
            "delivered": 0.0,
        })
        return out_AB, out_BA, out_node_flow, 0.0, journal_data

    reach_nodes = np.empty(n_r, dtype=np.int64)
    p = 0
    for v in range(n_nodes):
        if reach[v]:
            reach_nodes[p] = v
            p += 1

    order_o = reach_nodes[np.argsort(d_o[reach_nodes])]   # ascending d_o
    order_d = reach_nodes[np.argsort(d_d[reach_nodes])]   # ascending d_d

    cont_o = np.zeros(n_nodes, dtype=np.bool_)
    cont_d = np.zeros(n_nodes, dtype=np.bool_)
    for i in range(n_r):
        v = order_o[i]
        pv = pred_o[v]
        if pv < 0:
            continue
        if cont_o[pv]:
            cont_o[v] = True
        elif v < n_net and pv < n_net:
            ai = find_arc_fn(indptr, indices, pv, v)
            if ai >= 0 and edge_id_of_arc[ai] == d_edge_id:
                cont_o[v] = True
    for i in range(n_r):
        v = order_d[i]
        pv = pred_d[v]
        if pv < 0:
            continue
        if cont_d[pv]:
            cont_d[v] = True
        elif v < n_net and pv < n_net:
            ai = find_arc_fn(indptr, indices, v, pv)
            if ai >= 0 and edge_id_of_arc[ai] == o_edge_id:
                cont_d[v] = True

    q_sum = 0.0
    pass1 = []
    for i in range(n_r):
        u = reach_nodes[i]
        if cont_o[u]:
            continue
        for ai in range(indptr[u], indptr[u + 1]):
            x = indices[ai]
            if not reach[x] or cont_d[x]:
                continue
            arc_w = weights[ai]
            if d_o[u] + arc_w + d_d[x] > budget:
                continue
            eid = edge_id_of_arc[ai]
            if u < n_net and x < n_net and (eid == o_edge_id or eid == d_edge_id):
                continue
            if pred_d[x] == u or pred_o[u] == x:
                continue
            excess = d_o[u] + arc_w + d_d[x] - d_shortest
            if excess < 0.0:
                excess = 0.0
            q = float(decay_fn(decay_curve_id, decay_beta, decay_midpoint, excess))
            q_sum += q
            if journal:
                pass1.append([int(u), int(ai), int(x), float(excess), q])

    if q_sum <= 0.0:
        journal_data.update({
            "early_return": "q_sum_zero",
            "reach": reach.copy(), "n_r": n_r,
            "reach_nodes": reach_nodes.copy(),
            "order_o": order_o.copy(), "order_d": order_d.copy(),
            "cont_o": cont_o.copy(), "cont_d": cont_d.copy(),
            "q_sum": q_sum,
            "writes": writes,
            "out_AB": np.zeros_like(out_AB),
            "out_BA": np.zeros_like(out_BA),
            "out_node_flow": np.zeros_like(out_node_flow),
            "delivered": 0.0,
        })
        return out_AB, out_BA, out_node_flow, 0.0, journal_data

    scale = trip_volume / q_sum

    acc_o = np.zeros(n_nodes, dtype=np.float64)
    acc_d = np.zeros(n_nodes, dtype=np.float64)
    seeds = []
    for i in range(n_r):
        u = reach_nodes[i]
        if cont_o[u]:
            continue
        for ai in range(indptr[u], indptr[u + 1]):
            x = indices[ai]
            if not reach[x] or cont_d[x]:
                continue
            arc_w = weights[ai]
            if d_o[u] + arc_w + d_d[x] > budget:
                continue
            eid = edge_id_of_arc[ai]
            if u < n_net and x < n_net and (eid == o_edge_id or eid == d_edge_id):
                continue
            if pred_d[x] == u or pred_o[u] == x:
                continue
            excess = d_o[u] + arc_w + d_d[x] - d_shortest
            if excess < 0.0:
                excess = 0.0
            q = scale * float(decay_fn(decay_curve_id, decay_beta,
                                       decay_midpoint, excess))
            if eid >= 0:
                if dir_of_arc[ai] == 0:
                    out_AB[eid] += q
                    if journal:
                        writes.append(("AB", int(eid), q))
                else:
                    out_BA[eid] += q
                    if journal:
                        writes.append(("BA", int(eid), q))
            acc_o[u] += q
            acc_d[x] += q
            if journal:
                seeds.append([int(u), int(ai), int(x), q])

    for i in range(n_r - 1, -1, -1):
        v = order_o[i]
        f = acc_o[v]
        if f <= 0.0:
            continue
        pv = pred_o[v]
        if pv < 0:
            continue
        ai = find_arc_fn(indptr, indices, pv, v)
        if ai >= 0:
            eid = edge_id_of_arc[ai]
            if eid >= 0:
                if dir_of_arc[ai] == 0:
                    out_AB[eid] += f
                    if journal:
                        writes.append(("AB", int(eid), f))
                else:
                    out_BA[eid] += f
                    if journal:
                        writes.append(("BA", int(eid), f))
        acc_o[pv] += f

    for i in range(n_r - 1, -1, -1):
        v = order_d[i]
        f = acc_d[v]
        if f <= 0.0:
            continue
        pv = pred_d[v]
        if pv < 0:
            continue
        ai = find_arc_fn(indptr, indices, v, pv)
        if ai >= 0:
            eid = edge_id_of_arc[ai]
            if eid >= 0:
                if dir_of_arc[ai] == 0:
                    out_AB[eid] += f
                    if journal:
                        writes.append(("AB", int(eid), f))
                else:
                    out_BA[eid] += f
                    if journal:
                        writes.append(("BA", int(eid), f))
        acc_d[pv] += f

    if out_node_flow.shape[0] > 0:
        for i in range(n_r):
            v = reach_nodes[i]
            if v < n_net:
                val = acc_o[v] + acc_d[v]
                out_node_flow[v] += val
                if journal:
                    writes.append(("node", int(v), val))

    delivered = float(acc_d[dest_virtual_node])
    if journal:
        # per-OD delta = this OD's writes replayed into zero buffers
        delta_AB = np.zeros_like(out_AB)
        delta_BA = np.zeros_like(out_BA)
        delta_node = np.zeros_like(out_node_flow)
        buckets = {"AB": delta_AB, "BA": delta_BA, "node": delta_node}
        for kind, idx, val in writes:
            buckets[kind][idx] += val
        journal_data.update({
            "early_return": None,
            "reach": reach.copy(),
            "reach_decisions": reach_decisions,
            "n_r": n_r,
            "reach_nodes": reach_nodes.copy(),
            "order_o": order_o.copy(),
            "order_d": order_d.copy(),
            "cont_o": cont_o.copy(),
            "cont_d": cont_d.copy(),
            "pass1_arcs": pass1,
            "q_sum": q_sum,
            "scale": scale,
            "seeds": seeds,
            "writes": writes,
            "acc_o": acc_o.copy(),
            "acc_d": acc_d.copy(),
            "out_AB": delta_AB,
            "out_BA": delta_BA,
            "out_node_flow": delta_node,
            "out_AB_cumulative": out_AB.copy(),
            "out_BA_cumulative": out_BA.copy(),
            "delivered": delivered,
        })
    return out_AB, out_BA, out_node_flow, delivered, journal_data


def trace_origin_loop(b0, engine, settings, K, journal_od=True,
                      stale_scratch=False, stripe_shift=0,
                      final_sum_order="ascending"):
    """Serial replica of _process_origins_aggregate for a FIXED K.

    Returns dict with per-origin/per-OD journals, per-stripe partials,
    and the final folded arrays. Compiled equivalence is established by
    the tests against engine.Centrality outputs at the same K.

    Mutant knobs (negative tests only):
      stale_scratch  — do not reset dd_buf/pd_buf between ODs.
      stripe_shift   — stripe assignment becomes range(slot+shift, n, K).
      final_sum_order — 'descending' folds slots K-1..0 instead of 0..K-1.
    """
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

    ns = engine._prepare_params(settings)
    from scipy.sparse.csgraph import dijkstra as _scipy_dijkstra

    curve_name = ns["path_penalty"]
    if curve_name == "equal":
        decay_curve_id = b0._DECAY_EQUAL
    elif curve_name == "exponential":
        decay_curve_id = b0._DECAY_EXPONENTIAL
    else:
        decay_curve_id = b0._DECAY_LOGISTIC
    decay_beta = float(ns["route_beta"])
    decay_midpoint = float(ns["route_midpoint"]) if ns["route_midpoint"] > 0 else 200.0

    radius = float(ns["search_radius"])
    gravity_beta = float(ns["beta"])
    use_nearest = bool(ns["closest_dest"])
    decay_on = bool(ns["decay"])
    decay_curve_dcy = ns["decay_curve"]
    use_o_weights = bool(ns["use_o_weights"])
    use_d_weights = bool(ns["use_d_weights"])
    dest_weights = np.asarray(dest.node_weight, dtype=np.float64)
    dest_node_ids = np.arange(n_net, n_net + n_dest, dtype=np.int64)
    dest_edge_ids = np.asarray(dest.nearest_edge_id, dtype=np.int64)
    mode = ns["mode"]
    ratio = float(ns["ratio"])
    buffer_ = float(ns["buffer"])
    grad_limit = float(b0._cutoff_for_shortest(
        radius, mode, ratio, buffer_))

    n_edges = engine.edge_flow_AB.shape[0]
    local_AB = [np.zeros(n_edges, dtype=np.float64) for _ in range(K)]
    local_BA = [np.zeros(n_edges, dtype=np.float64) for _ in range(K)]
    if engine.node_flow is not None:
        local_node = [np.zeros(n_net, dtype=np.float64) for _ in range(K)]
    else:
        local_node = [np.zeros(0, dtype=np.float64) for _ in range(K)]

    origin_journals = []
    od_journals = []
    for slot in range(K):
        buf_AB = local_AB[slot]
        buf_BA = local_BA[slot]
        buf_node = local_node[slot]
        dd_buf = np.full(n_total_nodes, np.inf, dtype=np.float64)
        pd_buf = np.full(n_total_nodes, -9999, dtype=np.int32)

        for o_pos in range(slot + stripe_shift, n_origins, K):
            o_weight = float(origins.node_weight[o_pos])
            o_rec = {"o_pos": o_pos, "slot": slot, "o_weight": o_weight,
                     "skipped_zero_weight": bool(use_o_weights and o_weight == 0.0)}
            origin_journals.append(o_rec)
            if use_o_weights and o_weight == 0.0:
                continue
            if not use_o_weights:
                o_weight = 1.0

            origin_virtual = int(engine._first_origin_node + o_pos)
            o_edge_id = int(origins.nearest_edge_id[o_pos])

            d_o, pred_o = _scipy_dijkstra(
                csr_fwd, directed=True, indices=origin_virtual,
                limit=grad_limit, return_predecessors=True,
            )
            o_rec["origin_virtual"] = origin_virtual
            o_rec["o_edge_id"] = o_edge_id
            o_rec["d_o"] = np.asarray(d_o, dtype=np.float64).copy()
            o_rec["pred_o"] = np.asarray(pred_o).copy()

            d_shortest_arr = d_o[dest_node_ids]
            trip_vols = b0._compute_trip_volumes(
                o_weight, dest_weights, d_shortest_arr,
                radius, gravity_beta, decay_on, decay_curve_dcy,
                use_nearest, use_d_weights,
                ns["decay_method"], float(ns["gravity_cap"]),
            )
            o_rec["trip_vols"] = np.asarray(trip_vols, dtype=np.float64).copy()

            for d_idx in range(n_dest):
                d_shortest = float(d_shortest_arr[d_idx])
                trip_vol = float(trip_vols[d_idx])
                if trip_vol <= 0.0 or not np.isfinite(d_shortest):
                    continue
                if d_shortest > radius:
                    continue
                budget = float(b0._cutoff_for_shortest(d_shortest, mode, ratio, buffer_))

                s0, s1 = g_indptr[d_idx], g_indptr[d_idx + 1]
                cols = g_nodes[s0:s1]
                dd_buf[cols] = g_dist[s0:s1]
                pd_buf[cols] = g_pred[s0:s1]

                _, _, _, delivered, od_journal = trace_accumulate_od_flow(
                    b0._decay, b0._find_arc,
                    csr_indptr, csr_indices, csr_weights,
                    csr_edge_id, csr_direction,
                    d_o, dd_buf, pred_o, pd_buf,
                    origin_virtual, int(dest_node_ids[d_idx]),
                    o_edge_id, int(dest_edge_ids[d_idx]),
                    d_shortest, budget,
                    decay_curve_id, decay_beta, decay_midpoint,
                    trip_vol, n_net,
                    out_AB=buf_AB, out_BA=buf_BA, out_node_flow=buf_node,
                    journal=journal_od,
                )
                od_journal.update({
                    "o_pos": o_pos, "slot": slot, "d_idx": d_idx,
                    "d_shortest": d_shortest, "trip_vol": trip_vol,
                    "budget": budget,
                    "gradient_cols": cols.copy(),
                    "gradient_dist": g_dist[s0:s1].copy(),
                    "gradient_pred": g_pred[s0:s1].copy(),
                    "delivered": delivered,
                    "dd_buf_before_reset": dd_buf.copy() if journal_od else None,
                })
                od_journals.append(od_journal)

                if not stale_scratch:
                    dd_buf[cols] = np.inf
                    pd_buf[cols] = -9999

    # Final reduction: fold partials in REQUIRED slot order.
    n_slots = K
    order = range(n_slots - 1, -1, -1) if final_sum_order == "descending" else range(n_slots)
    final_AB = np.zeros(n_edges, dtype=np.float64)
    final_BA = np.zeros(n_edges, dtype=np.float64)
    final_node = (np.zeros(n_net, dtype=np.float64)
                  if engine.node_flow is not None else np.zeros(0, dtype=np.float64))
    fold_sequence = []
    for slot in order:
        final_AB += local_AB[slot]
        final_BA += local_BA[slot]
        if engine.node_flow is not None:
            final_node += local_node[slot]
        fold_sequence.append(int(slot))

    partials = {
        "local_AB": [a.copy() for a in local_AB],
        "local_BA": [a.copy() for a in local_BA],
        "local_node": [a.copy() for a in local_node],
        "stripe_members": [
            list(range(slot + stripe_shift, n_origins, K)) for slot in range(K)
        ],
        "fold_sequence": fold_sequence,
    }
    return {
        "origin_journals": origin_journals,
        "od_journals": od_journals,
        "partials": partials,
        "final_AB": final_AB,
        "final_BA": final_BA,
        "final_node": final_node,
    }
