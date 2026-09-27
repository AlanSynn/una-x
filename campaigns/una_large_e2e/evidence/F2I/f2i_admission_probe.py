#!/usr/bin/env python
"""F2I admission probe — local flow overlap + complete scratch reset.

Measures, on the CURRENT selected source (f3r_cand_wheel site == HEAD
worktree, byte-verified in-run), against the frozen H05 O3_FLOW capture:

  1. per-OD fixed-cost decomposition of AggregateFlow._accumulate_od_flow
     (two full-V' scans + five zeroed n_nodes allocations) — the removable
     side of the dossier mechanism;
  2. a statement-for-statement tracked PROTOTYPE of the F2 candidate
     (cols-restricted scan over the destination's finite gradient slice +
     preallocated stripe scratch + touched-list recording of EVERY scratch
     write + integer reset epilogue), differentially compared BITWISE
     against the real kernel on every sampled OD;
  3. the nonfinite predicate characterization the dossier demands
     (ascending/unique slices; outside-slice +inf sentinel behavior under
     the ACTUAL compiled fastmath predicate; behavioral proof via the
     bitwise baseline-vs-prototype equality);
  4. gate arithmetic vs campaign.json gates.minimum_removable_fraction.

REV-9 quiet window is stamped in-child at start and end (refusal = exit 2).
A2I-era rules: bounded wall budget, seeded sampling, min-of-reps timing,
append-only disclosure of superseded runs.

Run:
  NUMBA_CACHE_DIR=<fresh> python f2i_admission_probe.py \
      --out <record.json> --site-expected <site-packages>
"""
import argparse
import datetime
import hashlib
import json
import os
import sys
import time

import numpy as np
import numba as nb
import psutil

WT = "/Users/alansynn/orca/workspaces/una-x/wt-large-e2e"
CAMPAIGN = f"{WT}/campaigns/una_large_e2e"
CAPTURE = ("/Users/alansynn/orca/workspaces/una-x/campaign_data/"
           "h05_captures/O3_FLOW/flow_arrays.npz")
SESSION = ("/Users/alansynn/orca/workspaces/una-x/campaign_data/"
           "h05_sessions/FLOW_sel1024_od.json")
SESSION_CHOSEN = ("/Users/alansynn/orca/workspaces/una-x/campaign_data/"
                  "h05_sessions/FLOW_chosen_od.json")
STAGE_MODEL = f"{CAMPAIGN}/evidence/H05/stage_model.json"
CAMPAIGN_JSON = f"{CAMPAIGN}/campaign.json"

WALL_BUDGET_S = 600
N_SAMPLED_ORIGINS = 32
MAX_ODS_PER_ORIGIN = 16
SEED = 20260925
REPS = 7
REV9_FLOOR_BYTES = 5_368_709_120
REV9_LOADAVG_MAX = 8.0

# Decay-curve codes — VERBATIM mirror of AggregateFlow's module constants.
_DECAY_EQUAL = 0
_DECAY_EXPONENTIAL = 1
_DECAY_LOGISTIC = 2


@nb.njit(cache=True, inline='always')
def _find_arc(indptr, indices, u, v):
    """VERBATIM mirror of AggregateFlow._find_arc."""
    for ai in range(indptr[u], indptr[u + 1]):
        if indices[ai] == v:
            return ai
    return -1


@nb.njit(cache=True, inline='always')
def _decay(curve_id, beta, midpoint, excess):
    """VERBATIM mirror of AggregateFlow._decay."""
    if curve_id == _DECAY_EQUAL:
        return 1.0
    if curve_id == _DECAY_EXPONENTIAL:
        if excess <= 0.0:
            return 1.0
        return np.exp(-beta * excess)
    k = np.log(99.0) / midpoint
    return 1.0 / (1.0 + np.exp(k * excess))


def utc():
    return datetime.datetime.now(datetime.timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ")


def window_stamp():
    vm = psutil.virtual_memory()
    la = psutil.getloadavg()
    return {"utc": utc(), "available_bytes": vm.available,
            "loadavg": list(la),
            "rev9_quiet_floor_bytes": REV9_FLOOR_BYTES,
            "rev9_loadavg_max": REV9_LOADAVG_MAX,
            "rev9_admitted": bool(vm.available >= REV9_FLOOR_BYTES
                                  and la[0] < REV9_LOADAVG_MAX)}


def sha256_file(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


# ======================================================================
# Kernels
# ======================================================================

@nb.njit(cache=True, fastmath=True, nogil=True)
def _scan2_replica_virtual(d_o, d_d, budget, origin_virtual_node,
                           dest_virtual_node):
    """Byte-semantics mirror of baseline kernel sections 1-2a: the reach
    alloc+scan, the virtual early-return check, and the reach_nodes
    alloc+fill. Attribution leg for the fixed full-V' scan+alloc cost."""
    n_nodes = d_o.shape[0]
    reach = np.zeros(n_nodes, dtype=nb.boolean)
    n_r = 0
    for v in range(n_nodes):
        dov = d_o[v]
        ddv = d_d[v]
        if dov < np.inf and ddv < np.inf and dov + ddv <= budget:
            reach[v] = True
            n_r += 1
    if not reach[dest_virtual_node] or not reach[origin_virtual_node]:
        return n_r, reach, np.empty(0, dtype=np.int64)
    reach_nodes = np.empty(n_r, dtype=np.int64)
    p = 0
    for v in range(n_nodes):
        if reach[v]:
            reach_nodes[p] = v
            p += 1
    return n_r, reach, reach_nodes


@nb.njit(cache=True, fastmath=True, nogil=True)
def _alloc4_replica(n_nodes):
    """Attribution leg: the cont_o/cont_d/acc_o/acc_d alloc+zero cost the
    baseline pays inside every call (reach/reach_nodes allocs sit inside
    _scan2_replica_virtual)."""
    cont_o = np.zeros(n_nodes, dtype=nb.boolean)
    cont_d = np.zeros(n_nodes, dtype=nb.boolean)
    acc_o = np.zeros(n_nodes, dtype=np.float64)
    acc_d = np.zeros(n_nodes, dtype=np.float64)
    return cont_o[0], cont_d[0], acc_o[0], acc_d[0]


@nb.njit(cache=True, fastmath=True, nogil=True)
def _cols_scan(cols, d_o, d_d, budget, reach, reach_nodes,
               origin_virtual_node, dest_virtual_node):
    """F2 replacement for baseline sections 1-2: iterate the destination's
    ascending finite slice instead of all V'. Fills the PREALLOCATED reach
    mask and reach_nodes in ascending global order (baseline order).
    Returns n_r, or -1 on the baseline early-return path."""
    n_r = 0
    for i in range(cols.shape[0]):
        v = cols[i]
        dov = d_o[v]
        ddv = d_d[v]
        if dov < np.inf and ddv < np.inf and dov + ddv <= budget:
            reach[v] = True
            reach_nodes[n_r] = v
            n_r += 1
    if not reach[dest_virtual_node] or not reach[origin_virtual_node]:
        return -1
    return n_r


@nb.njit(cache=True, fastmath=True, nogil=True)
def _pred_fastmath(dov, ddv, budget):
    """The ACTUAL compiled predicate under fastmath, isolated for the
    nonfinite characterization leg."""
    return dov < np.inf and ddv < np.inf and dov + ddv <= budget


@nb.njit(cache=True, fastmath=True, nogil=True)
def _f2_tracked_kernel(
    indptr, indices, weights, edge_id_of_arc, dir_of_arc,
    d_o, d_d, pred_o, pred_d,
    origin_virtual_node, dest_virtual_node,
    o_edge_id, d_edge_id,
    d_shortest, budget,
    decay_curve_id, decay_beta, decay_midpoint,
    trip_volume,
    n_net,
    out_AB, out_BA, out_node_flow,
    cols, reach_nodes,
    reach, cont_o, cont_d, acc_o, acc_d,
    t_reach, t_cont_o, t_cont_d, t_acc_o, t_acc_d,
    cap_reach, cap_cont_o, cap_cont_d, cap_acc_o, cap_acc_d,
):
    """Statement-for-statement F2 candidate prototype of
    AggregateFlow._accumulate_od_flow. Sections 2b-7 are IDENTICAL to the
    baseline; sections 1-2 are the cols-restricted scan; every scratch
    write records its index into a touched list (duplicates allowed,
    dossier-sanctioned); scratch is preallocated and pristine on entry
    (the reset epilogue restores it after each OD)."""
    of = 0  # overflow flag

    # ── 1-2. cols-restricted reach build (ascending global order) ─────
    n_r = 0
    n_tr = 0
    for i in range(cols.shape[0]):
        v = cols[i]
        dov = d_o[v]
        ddv = d_d[v]
        if dov < np.inf and ddv < np.inf and dov + ddv <= budget:
            reach[v] = True
            if n_tr < cap_reach:
                t_reach[n_tr] = v
                n_tr += 1
            else:
                of = 1
            reach_nodes[n_r] = v
            n_r += 1

    if not reach[dest_virtual_node] or not reach[origin_virtual_node]:
        return (0.0, of, n_r, n_tr, 0, 0, 0, 0, 0, 0)

    # OD argsorts — retained baseline cost (n_r-sized, statement-identical).
    rn = reach_nodes[:n_r]
    order_o = rn[np.argsort(d_o[rn])]   # ascending d_o
    order_d = rn[np.argsort(d_d[rn])]   # ascending d_d

    # ── 2b. Leg contamination marking (identical + records) ───────────
    n_tco = 0
    n_tcd = 0
    for i in range(n_r):
        v = order_o[i]
        pv = pred_o[v]
        if pv < 0:
            continue
        if cont_o[pv]:
            cont_o[v] = True
            if n_tco < cap_cont_o:
                t_cont_o[n_tco] = v
                n_tco += 1
            else:
                of = 1
        elif v < n_net and pv < n_net:
            ai = _find_arc(indptr, indices, pv, v)
            if ai >= 0 and edge_id_of_arc[ai] == d_edge_id:
                cont_o[v] = True
                if n_tco < cap_cont_o:
                    t_cont_o[n_tco] = v
                    n_tco += 1
                else:
                    of = 1
    for i in range(n_r):
        v = order_d[i]
        pv = pred_d[v]
        if pv < 0:
            continue
        if cont_d[pv]:
            cont_d[v] = True
            if n_tcd < cap_cont_d:
                t_cont_d[n_tcd] = v
                n_tcd += 1
            else:
                of = 1
        elif v < n_net and pv < n_net:
            ai = _find_arc(indptr, indices, v, pv)
            if ai >= 0 and edge_id_of_arc[ai] == o_edge_id:
                cont_d[v] = True
                if n_tcd < cap_cont_d:
                    t_cont_d[n_tcd] = v
                    n_tcd += 1
                else:
                    of = 1

    # ── 3. Pass 1 — total decay weight (identical; reads only) ────────
    q_sum = 0.0
    n_p1 = 0
    for i in range(n_r):
        u = reach_nodes[i]
        if cont_o[u]:
            continue
        for ai in range(indptr[u], indptr[u + 1]):
            n_p1 += 1
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
            q_sum += _decay(decay_curve_id, decay_beta, decay_midpoint, excess)

    if q_sum <= 0.0:
        return (0.0, of, n_r, n_tr, n_tco, n_tcd, 0, 0, n_p1, 0)

    scale = trip_volume / q_sum

    # ── 4. Pass 2 — seed via-arc shares (identical + records) ─────────
    n_tao = 0
    n_tad = 0
    n_p2 = 0
    for i in range(n_r):
        u = reach_nodes[i]
        if cont_o[u]:
            continue
        for ai in range(indptr[u], indptr[u + 1]):
            n_p2 += 1
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
            q = scale * _decay(decay_curve_id, decay_beta, decay_midpoint, excess)

            if eid >= 0:
                if dir_of_arc[ai] == 0:
                    out_AB[eid] += q
                else:
                    out_BA[eid] += q
            acc_o[u] += q
            acc_d[x] += q
            if n_tao < cap_acc_o:
                t_acc_o[n_tao] = u
                n_tao += 1
            else:
                of = 1
            if n_tad < cap_acc_d:
                t_acc_d[n_tad] = x
                n_tad += 1
            else:
                of = 1

    # ── 5. Origin legs (identical + pv records) ───────────────────────
    for i in range(n_r - 1, -1, -1):
        v = order_o[i]
        f = acc_o[v]
        if f <= 0.0:
            continue
        pv = pred_o[v]
        if pv < 0:
            continue
        ai = _find_arc(indptr, indices, pv, v)
        if ai >= 0:
            eid = edge_id_of_arc[ai]
            if eid >= 0:
                if dir_of_arc[ai] == 0:
                    out_AB[eid] += f
                else:
                    out_BA[eid] += f
        acc_o[pv] += f
        if n_tao < cap_acc_o:
            t_acc_o[n_tao] = pv
            n_tao += 1
        else:
            of = 1

    # ── 6. Destination legs (identical + pv records) ──────────────────
    for i in range(n_r - 1, -1, -1):
        v = order_d[i]
        f = acc_d[v]
        if f <= 0.0:
            continue
        pv = pred_d[v]
        if pv < 0:
            continue
        ai = _find_arc(indptr, indices, v, pv)
        if ai >= 0:
            eid = edge_id_of_arc[ai]
            if eid >= 0:
                if dir_of_arc[ai] == 0:
                    out_AB[eid] += f
                else:
                    out_BA[eid] += f
        acc_d[pv] += f
        if n_tad < cap_acc_d:
            t_acc_d[n_tad] = pv
            n_tad += 1
        else:
            of = 1

    # ── 7. Node flow (identical; reads only) ──────────────────────────
    if out_node_flow.shape[0] > 0:
        for i in range(n_r):
            v = reach_nodes[i]
            if v < n_net:
                out_node_flow[v] += acc_o[v] + acc_d[v]

    return (acc_d[dest_virtual_node], of, n_r, n_tr, n_tco, n_tcd,
            n_tao, n_tad, n_p1, n_p2)


@nb.njit(cache=True, nogil=True)
def _f2_reset(reach, cont_o, cont_d, acc_o, acc_d,
              t_reach, n_reach, t_cont_o, n_co, t_cont_d, n_cd,
              t_acc_o, n_ao, t_acc_d, n_ad):
    """Dossier step 6: one cleanup epilogue restoring exact original
    False/+0.0 at every recorded scratch index."""
    for i in range(n_reach):
        reach[t_reach[i]] = False
    for i in range(n_co):
        cont_o[t_cont_o[i]] = False
    for i in range(n_cd):
        cont_d[t_cont_d[i]] = False
    for i in range(n_ao):
        acc_o[t_acc_o[i]] = 0.0
    for i in range(n_ad):
        acc_d[t_acc_d[i]] = 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--site-expected", required=True)
    args = ap.parse_args()

    deadline = time.monotonic() + WALL_BUDGET_S
    rec = {"probe": "F2I_admission", "task": "F2I", "utc_started": utc(),
           "wall_budget_s": WALL_BUDGET_S}

    ws = window_stamp()
    rec["window_at_start"] = ws
    if not ws["rev9_admitted"]:
        print("[f2i-probe] REFUSED: REV-9 quiet window not admitted at start")
        print(json.dumps(ws))
        sys.exit(2)

    sys.path.insert(0, args.site_expected)
    from urban_network_analysis.Engines import AggregateFlow as AF

    # ---- identity receipts ------------------------------------------------
    site = args.site_expected
    eng = f"{site}/urban_network_analysis/Engines"
    wt_eng = f"{WT}/src/urban_network_analysis/Engines"
    id_rows = {}
    for fn in ("AggregateFlow.py", "_large_flow_workspace.py"):
        s1 = sha256_file(f"{eng}/{fn}")
        s2 = sha256_file(f"{wt_eng}/{fn}")
        id_rows[fn] = {"site_sha256": s1, "worktree_sha256": s2,
                       "match": s1 == s2}
    rec["identity"] = {"site": site, "files": id_rows,
                       "identity_ok": all(r["match"] for r in id_rows.values())}

    # ---- inputs -----------------------------------------------------------
    rec["capture_path"] = CAPTURE
    rec["capture_sha256_before"] = sha256_file(CAPTURE)
    cap = dict(np.load(CAPTURE))
    session = json.load(open(SESSION))
    asm = session["settings"]["assembled"]
    radius = float(asm["search_radius"])
    mode = asm["flow_detour_mode"]
    ratio = float(asm["flow_detour_ratio"])
    penalty = asm["flow_path_detour_penalty"]
    route_beta = float(asm["flow_route_enumeration_beta"])
    route_mid = float(asm["flow_route_enumeration_logistic_midpoint"])
    rec["settings_receipt"] = {
        "session": SESSION, "search_radius": radius, "detour_mode": mode,
        "detour_ratio": ratio,
        "detour_buffer": float(asm["flow_detour_buffer"]),
        "path_penalty": penalty, "route_beta": route_beta,
        "route_midpoint": route_mid, "turns": asm["turns"],
    }
    assert mode == "ratio" and penalty == "logistic" and not asm["turns"]

    indptr = cap["csr_indptr"]; indices = cap["csr_indices"]
    weights = cap["csr_weights"]; eid_arr = cap["csr_edge_id"]
    dr = cap["csr_direction"]
    fwd_indptr = cap["csr_fwd_indptr"]; fwd_indices = cap["csr_fwd_indices"]
    fwd_data = cap["csr_fwd_data"]
    g_indptr = cap["grad_indptr"]; g_nodes = cap["grad_nodes"]
    g_dist = cap["grad_dist"]; g_pred = cap["grad_pred"]
    n_net = int(cap["n_network_nodes"]); n_dest = int(cap["n_destinations"])
    n_orig = int(cap["n_origins"]); first_o = int(cap["first_origin_node"])
    n_total = indptr.shape[0] - 1
    n_arcs = weights.shape[0]
    rec["dims"] = {"n_total_V_prime": n_total, "n_net": n_net,
                   "n_dest": n_dest, "n_orig": n_orig,
                   "n_arcs": int(n_arcs)}

    decay_curve_id = _DECAY_LOGISTIC  # frozen flow_path_detour_penalty
    grad_limit = radius * ratio  # _gradient_limit_for: cutoff(radius, ratio)
    rec["grad_limit"] = grad_limit

    # ---- predicate characterization (dossier Admission) -------------------
    lens = np.diff(g_indptr.astype(np.int64))
    asc_viol = 0
    for d in range(n_dest):
        s = g_nodes[g_indptr[d]:g_indptr[d + 1]]
        if not np.all(np.diff(s) > 0):
            asc_viol += 1
    dv_lo, dv_hi = n_net, n_net + n_dest
    ov_lo, ov_hi = first_o, first_o + n_orig
    rec["slice_characterization"] = {
        "slices_checked": int(n_dest),
        "strictly_ascending_violations": int(asc_viol),
        "len_min": int(lens.min()), "len_median": float(np.median(lens)),
        "len_mean": float(lens.mean()), "len_max": int(lens.max()),
        "len_over_V_prime_mean": float(lens.mean() / n_total),
        "members_net": int(np.count_nonzero(g_nodes < n_net)),
        "members_dest_virtual": int(np.count_nonzero(
            (g_nodes >= dv_lo) & (g_nodes < dv_hi))),
        "members_origin_virtual": int(np.count_nonzero(
            (g_nodes >= ov_lo) & (g_nodes < ov_hi))),
        "note": ("origin virtuals DO appear in finite slices, so the "
                 "baseline early-return path is live and the cols scan "
                 "must cover virtual nodes - it does, by construction"),
    }
    pred_leg = {
        "finite_finite_over_budget": bool(_pred_fastmath(1.0, 2.0, 2.0)),
        "finite_inf": bool(_pred_fastmath(1.0, np.inf, 500.0)),
        "inf_finite": bool(_pred_fastmath(np.inf, 1.0, 500.0)),
        "inf_inf": bool(_pred_fastmath(np.inf, np.inf, 500.0)),
    }
    pred_leg["fastmath_predicate_ok"] = (
        pred_leg["finite_inf"] is False and pred_leg["inf_finite"] is False
        and pred_leg["inf_inf"] is False
        and pred_leg["finite_finite_over_budget"] is False)
    rec["predicate_characterization"] = pred_leg

    # ---- OD sampling (driver-faithful reconstruction) ---------------------
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import dijkstra as sp_dijkstra
    csr_fwd = csr_matrix((fwd_data, fwd_indices, fwd_indptr),
                         shape=(n_total, n_total))

    def connector_edge_id(v):
        rows = eid_arr[indptr[v]:indptr[v + 1]]
        tg = indices[indptr[v]:indptr[v + 1]]
        uniq = np.unique(rows)
        return (int(uniq[0]) if uniq.shape[0] == 1 else -1,
                int(rows.shape[0]), bool(np.all(tg < n_net)))

    rng = np.random.default_rng(SEED)
    sampled = sorted(int(x) for x in
                     rng.choice(n_orig,
                                size=min(N_SAMPLED_ORIGINS, n_orig),
                                replace=False).tolist())
    rec["sampled_origins"] = sampled

    ods = []
    n_dijkstra = 0
    for o_pos in sampled:
        if (time.monotonic() > deadline
                or len(ods) >= N_SAMPLED_ORIGINS * MAX_ODS_PER_ORIGIN):
            break
        ov = first_o + o_pos
        o_eid, o_deg, o_ok = connector_edge_id(ov)
        if o_eid < 0 or not o_ok:
            continue
        dist_o, pred_o = sp_dijkstra(csr_fwd, directed=True, indices=ov,
                                     limit=grad_limit,
                                     return_predecessors=True)
        n_dijkstra += 1
        d_o = np.ascontiguousarray(dist_o, dtype=np.float64)
        pred_o32 = np.ascontiguousarray(pred_o, dtype=np.int32)
        dv_all = np.arange(n_net, n_net + n_dest)
        d_short_all = d_o[dv_all]
        cand = np.where(np.isfinite(d_short_all)
                        & (d_short_all <= radius))[0]
        order_k = cand[np.argsort(d_short_all[cand])]  # nearest-first
        taken = 0
        for d_try in order_k:
            if taken >= MAX_ODS_PER_ORIGIN or time.monotonic() > deadline:
                break
            d_eid, d_deg, d_ok = connector_edge_id(n_net + int(d_try))
            if d_eid < 0 or not d_ok:
                continue
            s0, s1 = int(g_indptr[d_try]), int(g_indptr[d_try + 1])
            if s1 <= s0:
                continue
            dd_buf = np.full(n_total, np.inf, dtype=np.float64)
            pd_buf = np.full(n_total, -9999, dtype=np.int32)
            cols64 = np.ascontiguousarray(g_nodes[s0:s1].astype(np.int64))
            dd_buf[cols64] = g_dist[s0:s1]
            pd_buf[cols64] = g_pred[s0:s1]
            d_shortest = float(d_short_all[d_try])
            budget = d_shortest * ratio  # mode 'ratio' (frozen)
            ods.append({
                "o_pos": o_pos, "ov": int(ov), "d_idx": int(d_try),
                "dv": int(n_net + d_try),
                "o_edge_id": o_eid, "d_edge_id": d_eid,
                "d_shortest": d_shortest, "budget": budget,
                "d_o": d_o, "pred_o": pred_o32,
                "dd_buf": dd_buf, "pd_buf": pd_buf, "cols": cols64,
                "C_d": int(s1 - s0),
            })
            taken += 1
    rec["od_search"] = {
        "n_dijkstra_runs": n_dijkstra, "n_od_tasks": len(ods),
        "gate": ("finite d_shortest AND d_shortest <= search_radius "
                 "(SUPERSET of the driver's gate, which also requires "
                 "trip_volume > 0; destination node weights are not in "
                 "the capture, so the true driver OD set cannot be "
                 "reproduced - disclosed)"),
        "trip_volume_convention": ("1.0 (H05 F1-probe convention; timing "
                                   "probe, not a numerical oracle)"),
        "snap_edge_ids": ("REAL ids reconstructed from connector CSR rows "
                          "(uniqueness verified per row), so snap-edge "
                          "exclusions are LIVE on both routes"),
    }

    # ---- per-OD measurement ----------------------------------------------
    per_od = []
    eq_fail = 0
    reach_fail = 0
    pristine_fail = 0
    overflow_count = 0
    zero_path_count = 0
    for t in ods:
        if time.monotonic() > deadline:
            break
        C_d = t["C_d"]
        cap = 16 * C_d + 4096
        reach = np.zeros(n_total, dtype=np.bool_)
        cont_o = np.zeros(n_total, dtype=np.bool_)
        cont_d = np.zeros(n_total, dtype=np.bool_)
        acc_o = np.zeros(n_total, dtype=np.float64)
        acc_d = np.zeros(n_total, dtype=np.float64)
        reach_nodes = np.empty(cap, dtype=np.int64)
        t_reach = np.empty(cap, dtype=np.int64)
        t_cont_o = np.empty(cap, dtype=np.int64)
        t_cont_d = np.empty(cap, dtype=np.int64)
        t_acc_o = np.empty(cap, dtype=np.int64)
        t_acc_d = np.empty(cap, dtype=np.int64)
        scratch = (reach, cont_o, cont_d, acc_o, acc_d)
        pristine = tuple(a.copy() for a in scratch)

        def seg(call, prep=None, reps=REPS):
            ts = []
            for _ in range(reps):
                if prep is not None:
                    prep()
                t0 = time.perf_counter_ns()
                call()
                ts.append(time.perf_counter_ns() - t0)
            return float(np.min(ts))

        out_AB0 = np.zeros(n_arcs, dtype=np.float64)
        out_BA0 = np.zeros(n_arcs, dtype=np.float64)
        out_N0 = np.zeros(n_net, dtype=np.float64)

        def fill0():
            out_AB0.fill(0.0); out_BA0.fill(0.0); out_N0.fill(0.0)

        base_args = (indptr, indices, weights, eid_arr, dr,
                     t["d_o"], t["dd_buf"], t["pred_o"], t["pd_buf"],
                     t["ov"], t["dv"], t["o_edge_id"], t["d_edge_id"],
                     t["d_shortest"], t["budget"],
                     decay_curve_id, route_beta, route_mid,
                     1.0, n_net, out_AB0, out_BA0, out_N0)
        t_base = seg(lambda: AF._accumulate_od_flow(*base_args), prep=fill0)

        def run_track():
            return _f2_tracked_kernel(
                indptr, indices, weights, eid_arr, dr,
                t["d_o"], t["dd_buf"], t["pred_o"], t["pd_buf"],
                t["ov"], t["dv"], t["o_edge_id"], t["d_edge_id"],
                t["d_shortest"], t["budget"],
                decay_curve_id, route_beta, route_mid,
                1.0, n_net, out_AB0, out_BA0, out_N0,
                t["cols"], reach_nodes,
                reach, cont_o, cont_d, acc_o, acc_d,
                t_reach, t_cont_o, t_cont_d, t_acc_o, t_acc_d,
                cap, cap, cap, cap, cap)

        t_track = seg(run_track, prep=fill0)

        # restore pristine scratch (untimed) then canonical tracked run
        reach.fill(False); cont_o.fill(False); cont_d.fill(False)
        acc_o.fill(0.0); acc_d.fill(0.0)
        out_A1 = np.zeros(n_arcs, dtype=np.float64)
        out_B1 = np.zeros(n_arcs, dtype=np.float64)
        out_N1 = np.zeros(n_net, dtype=np.float64)

        def run_track_cmp():
            return _f2_tracked_kernel(
                indptr, indices, weights, eid_arr, dr,
                t["d_o"], t["dd_buf"], t["pred_o"], t["pd_buf"],
                t["ov"], t["dv"], t["o_edge_id"], t["d_edge_id"],
                t["d_shortest"], t["budget"],
                decay_curve_id, route_beta, route_mid,
                1.0, n_net, out_A1, out_B1, out_N1,
                t["cols"], reach_nodes,
                reach, cont_o, cont_d, acc_o, acc_d,
                t_reach, t_cont_o, t_cont_d, t_acc_o, t_acc_d,
                cap, cap, cap, cap, cap)

        res = run_track_cmp()
        (delivered1, of, n_r, n_tr, n_tco, n_tcd, n_tao, n_tad,
         n_p1, n_p2) = (res[0], int(res[1]), int(res[2]), int(res[3]),
                        int(res[4]), int(res[5]), int(res[6]), int(res[7]),
                        int(res[8]), int(res[9]))
        # snapshot the single-contribution canonical outputs NOW — the
        # reset-timing prep below re-runs the tracked kernel into these
        # same buffers and must not contaminate the comparison
        snap_A1 = out_A1.tobytes()
        snap_B1 = out_B1.tobytes()
        snap_N1 = out_N1.tobytes()
        snap_d1 = np.float64(delivered1).tobytes()

        def run_reset():
            _f2_reset(reach, cont_o, cont_d, acc_o, acc_d,
                      t_reach, n_tr, t_cont_o, n_tco, t_cont_d, n_tcd,
                      t_acc_o, n_tao, t_acc_d, n_tad)
        t_reset = seg(run_reset, prep=run_track_cmp)

        # canonical baseline outputs for the bitwise legs
        out_A0 = np.zeros(n_arcs, dtype=np.float64)
        out_B0 = np.zeros(n_arcs, dtype=np.float64)
        out_N0c = np.zeros(n_net, dtype=np.float64)
        delivered0 = AF._accumulate_od_flow(
            indptr, indices, weights, eid_arr, dr,
            t["d_o"], t["dd_buf"], t["pred_o"], t["pd_buf"],
            t["ov"], t["dv"], t["o_edge_id"], t["d_edge_id"],
            t["d_shortest"], t["budget"],
            decay_curve_id, route_beta, route_mid,
            1.0, n_net, out_A0, out_B0, out_N0c)

        ok_bits = (out_A0.tobytes() == snap_A1
                   and out_B0.tobytes() == snap_B1
                   and out_N0c.tobytes() == snap_N1
                   and np.float64(delivered0).tobytes() == snap_d1)

        # reach mask/list equality vs the baseline-mirror replica
        n_r0, reach0, rnodes0 = _scan2_replica_virtual(
            t["d_o"], t["dd_buf"], t["budget"], t["ov"], t["dv"])
        reach1 = np.zeros(n_total, dtype=np.bool_)
        rn1 = np.empty(max(1, n_r), dtype=np.int64)
        _cols_scan(t["cols"], t["d_o"], t["dd_buf"], t["budget"],
                   reach1, rn1, t["ov"], t["dv"])
        reach_eq = (n_r0 == n_r
                    and reach0.tobytes() == reach1.tobytes()
                    and (n_r == 0 or
                         np.array_equal(rnodes0, rn1[:n_r])))

        # pristine epilogue check
        pr_ok = all(scratch[i].tobytes() == pristine[i].tobytes()
                    for i in range(5))
        # scatter discipline outside cols
        mask = np.ones(n_total, dtype=bool)
        mask[t["cols"]] = False
        scat_ok = (bool(np.all(t["dd_buf"][mask] == np.inf))
                   and bool(np.all(t["pd_buf"][mask] == -9999)))

        if not ok_bits:
            eq_fail += 1
        if not reach_eq:
            reach_fail += 1
        if not pr_ok:
            pristine_fail += 1
        if of:
            overflow_count += 1
        if delivered0 == 0.0:
            zero_path_count += 1

        # attribution legs (separate buffers; not in the net arithmetic)
        reach_r = np.zeros(n_total, dtype=np.bool_)
        rn_r = np.empty(cap, dtype=np.int64)
        t_scan2 = seg(lambda: _scan2_replica_virtual(
            t["d_o"], t["dd_buf"], t["budget"], t["ov"], t["dv"]))
        t_alloc4 = seg(lambda: _alloc4_replica(n_total))
        t_cols = seg(lambda: _cols_scan(
            t["cols"], t["d_o"], t["dd_buf"], t["budget"],
            reach_r, rn_r, t["ov"], t["dv"]))

        cand_side = t_track + t_reset
        row = {
            "o_pos": t["o_pos"], "d_idx": t["d_idx"], "C_d": C_d,
            "n_r": n_r, "n_r_over_C_d": (float(n_r) / C_d) if C_d else None,
            "t_base_ns": t_base, "t_track_ns": t_track,
            "t_reset_ns": t_reset,
            "t_scan2_ns": t_scan2, "t_alloc4_ns": t_alloc4,
            "t_cols_ns": t_cols,
            "net_ns": int(cand_side - t_base),
            "net_share": (cand_side - t_base) / t_base,
            "appends": {"reach": n_tr, "cont_o": n_tco, "cont_d": n_tcd,
                        "acc_o": n_tao, "acc_d": n_tad},
            "appends_total": int(n_tr + n_tco + n_tcd + n_tao + n_tad),
            "unique_total": int(len(set(t_reach[:n_tr].tolist())
                                    | set(t_cont_o[:n_tco].tolist())
                                    | set(t_cont_d[:n_tcd].tolist())
                                    | set(t_acc_o[:n_tao].tolist())
                                    | set(t_acc_d[:n_tad].tolist()))),
            "n_arc_checks_p1": n_p1, "n_arc_checks_p2": n_p2,
            "bitwise_outputs_ok": bool(ok_bits),
            "reach_mask_list_eq": bool(reach_eq),
            "pristine_after_reset": bool(pr_ok),
            "scatter_outside_cols_ok": bool(scat_ok),
            "overflow": bool(of),
        }
        per_od.append(row)

    rec["per_od"] = per_od

    # ---- synthetic early-return differential -------------------------------
    # The sampled ODs all delivered > 0 (zero_path_ods == 0), so the
    # baseline's virtual early-return branch was not exercised above.
    # Prove it here: halve the budget of the first OD so neither virtual
    # node can be reachable, then compare both kernels bitwise and check
    # the epilogue on an all-zero touched-list reset.
    if ods:
        t0 = ods[0]
        cap_e = 16 * t0["C_d"] + 4096
        reach_e = np.zeros(n_total, dtype=np.bool_)
        co_e = np.zeros(n_total, dtype=np.bool_)
        cd_e = np.zeros(n_total, dtype=np.bool_)
        ao_e = np.zeros(n_total, dtype=np.float64)
        ad_e = np.zeros(n_total, dtype=np.float64)
        rn_e = np.empty(cap_e, dtype=np.int64)
        tre = np.empty(cap_e, dtype=np.int64)
        tcoe = np.empty(cap_e, dtype=np.int64)
        tcde = np.empty(cap_e, dtype=np.int64)
        taoe = np.empty(cap_e, dtype=np.int64)
        tade = np.empty(cap_e, dtype=np.int64)
        outAe = np.zeros(n_arcs, dtype=np.float64)
        outBe = np.zeros(n_arcs, dtype=np.float64)
        outNe = np.zeros(n_net, dtype=np.float64)
        budget_e = t0["d_shortest"] * 0.5
        base_e = (indptr, indices, weights, eid_arr, dr,
                  t0["d_o"], t0["dd_buf"], t0["pred_o"], t0["pd_buf"],
                  t0["ov"], t0["dv"], t0["o_edge_id"], t0["d_edge_id"],
                  t0["d_shortest"], budget_e,
                  decay_curve_id, route_beta, route_mid,
                  1.0, n_net, outAe, outBe, outNe)
        d0e = AF._accumulate_od_flow(*base_e)
        r_e = _f2_tracked_kernel(
            indptr, indices, weights, eid_arr, dr,
            t0["d_o"], t0["dd_buf"], t0["pred_o"], t0["pd_buf"],
            t0["ov"], t0["dv"], t0["o_edge_id"], t0["d_edge_id"],
            t0["d_shortest"], budget_e,
            decay_curve_id, route_beta, route_mid,
            1.0, n_net, outAe, outBe, outNe,
            t0["cols"], rn_e,
            reach_e, co_e, cd_e, ao_e, ad_e,
            tre, tcoe, tcde, taoe, tade, cap_e, cap_e, cap_e, cap_e, cap_e)
        _f2_reset(reach_e, co_e, cd_e, ao_e, ad_e,
                  tre, int(r_e[3]), tcoe, int(r_e[4]), tcde, int(r_e[5]),
                  taoe, int(r_e[6]), tade, int(r_e[7]))
        pr_e = all(x.tobytes() == y.tobytes() for x, y in
                   ((reach_e, np.zeros(n_total, dtype=np.bool_)),
                    (co_e, np.zeros(n_total, dtype=np.bool_)),
                    (cd_e, np.zeros(n_total, dtype=np.bool_)),
                    (ao_e, np.zeros(n_total, dtype=np.float64)),
                    (ad_e, np.zeros(n_total, dtype=np.float64))))
        rec["early_return_leg"] = {
            "od_index_used": 0, "budget_e": budget_e,
            "budget_e_lt_d_shortest": bool(budget_e < t0["d_shortest"]),
            "baseline_delivered": float(d0e),
            "tracked_delivered": float(r_e[0]),
            "tracked_n_r": int(r_e[2]),
            "outputs_untouched_bitwise": bool(
                outAe.tobytes() == b"\x00" * outAe.nbytes
                and outBe.tobytes() == b"\x00" * outBe.nbytes
                and outNe.tobytes() == b"\x00" * outNe.nbytes),
            "delivered_bitwise_equal": bool(
                np.float64(d0e).tobytes() == np.float64(r_e[0]).tobytes()),
            "pristine_after_reset": bool(pr_e),
            "note": ("synthetic branch-coverage leg: not part of the "
                     "sampled distribution or the gate arithmetic"),
        }
    rec["validation"] = {
        "od_compared": len(per_od),
        "bitwise_output_failures": eq_fail,
        "reach_mask_or_list_failures": reach_fail,
        "pristine_epilogue_failures": pristine_fail,
        "overflow_events": overflow_count,
        "zero_path_ods": zero_path_count,
        "scatter_outside_cols_all_ok": all(
            r["scatter_outside_cols_ok"] for r in per_od),
    }

    # ---- aggregates --------------------------------------------------------
    if per_od:
        def med(k):
            return float(np.median([r[k] for r in per_od]))
        rec["walls_summary_ns"] = {
            "reps": REPS, "stat": "min-of-reps per OD, median across ODs",
            "median_t_base": med("t_base_ns"),
            "median_t_track": med("t_track_ns"),
            "median_t_reset": med("t_reset_ns"),
            "median_cand_side": med("t_track_ns") + med("t_reset_ns"),
            "median_net": med("net_ns"),
            "median_net_share": float(np.median(
                [r["net_share"] for r in per_od])),
            "median_t_scan2": med("t_scan2_ns"),
            "median_t_alloc4": med("t_alloc4_ns"),
            "median_t_cols": med("t_cols_ns"),
            "median_fixed_attr": med("t_scan2_ns") + med("t_alloc4_ns"),
        }
        rec["coverage_summary"] = {
            "C_d_median": float(np.median([r["C_d"] for r in per_od])),
            "C_d_max": int(np.max([r["C_d"] for r in per_od])),
            "n_r_median": med("n_r"),
            "n_r_max": int(np.max([r["n_r"] for r in per_od])),
            "n_r_over_C_d_median": float(np.median(
                [r["n_r_over_C_d"] for r in per_od])),
            "appends_total_median": med("appends_total"),
            "unique_total_median": med("unique_total"),
            "arc_checks_p1_median": med("n_arc_checks_p1"),
            "arc_checks_p2_median": med("n_arc_checks_p2"),
        }

    # ---- gate arithmetic (CPU-domain, disclosure-preserving) --------------
    cam = json.load(open(CAMPAIGN_JSON))
    GATE = cam["gates"]["minimum_removable_fraction"]
    sm = json.load(open(STAGE_MODEL))
    cell = sm["cells"]["FLOW_sel1024_od"]
    stage_wall = cell["top_stage_inclusive_s"]["flow_engine_centrality"]
    window_wall = cell["window_s"]
    agg = session["od_kernel_aggregate"]
    kernel_cpu = agg["sum_ns"] / 1e9
    chosen = json.load(open(SESSION_CHOSEN))
    agg_c = chosen["od_kernel_aggregate"]
    stage_wall_c = (sm["cells"]["FLOW_chosen_od"]["top_stage_inclusive_s"]
                    ["flow_engine_centrality"])
    window_wall_c = sm["cells"]["FLOW_chosen_od"]["window_s"]

    arith = {
        "measured_cell": "FLOW_sel1024_od",
        "gate": GATE,
        "gate_source": "campaign.json gates.minimum_removable_fraction",
        "session_kernel_cpu_s": kernel_cpu,
        "session_kernel_calls": agg["n_calls"],
        "session_kernel_mean_ns": agg["mean_ns"],
        "session_kernel_min_ns": agg["min_ns"],
        "stage_wall_s": stage_wall,
        "window_wall_s": window_wall,
        "kernel_cpu_over_stage_wall": kernel_cpu / stage_wall,
        "concurrency_note": (
            "the session kernel CPU aggregate EXCEEDS the stage wall "
            "(see kernel_cpu_over_stage_wall > 1), forcing mean kernel "
            "concurrency >= that ratio; therefore the wall saving from "
            "removing a kernel-CPU fraction f is bounded by "
            "f * stage_wall, giving removable_wall_share_of_window <= "
            "f * stage_wall/window - a rigorous ceiling needing no "
            "concurrency model"),
    }
    if per_od:
        f_share = -float(np.median([r["net_share"] for r in per_od]))
        arith["removable_kernel_share_measured"] = f_share
        arith["removable_wall_share_ceiling"] = f_share * (stage_wall
                                                           / window_wall)
        arith["above_gate_ceiling"] = bool(
            arith["removable_wall_share_ceiling"] >= GATE)
        # chosen-cell context ceiling from the measured fixed cost
        fixed = med("t_scan2_ns") + med("t_alloc4_ns")
        r_chosen_max = min(1.0, fixed / agg_c["mean_ns"])
        serial_share_c = min(1.0, (agg_c["sum_ns"] / 1e9) / stage_wall_c)
        arith["chosen_cell_context"] = {
            "session": SESSION_CHOSEN,
            "kernel_mean_ns": agg_c["mean_ns"],
            "kernel_min_ns": agg_c["min_ns"],
            "kernel_cpu_s": agg_c["sum_ns"] / 1e9,
            "stage_wall_s": stage_wall_c, "window_wall_s": window_wall_c,
            "measured_fixed_cost_ns_here": fixed,
            "r_ceiling_from_fixed": r_chosen_max,
            "kernel_cpu_over_stage_wall": (agg_c["sum_ns"] / 1e9)
            / stage_wall_c,
            "removable_wall_share_ceiling": (r_chosen_max * serial_share_c
                                             * (stage_wall_c
                                                / window_wall_c)),
            "note": ("no per-OD capture exists for this session; the "
                     "ceiling uses the measured fixed cost at the SAME "
                     "V' (both sessions share workload sel1024: same "
                     "V/E/O/D counts and the same 8217 calls) and, since "
                     "this cell's kernel CPU (0.639 s) is BELOW its "
                     "stage wall (3.821 s), assumes kernels may be fully "
                     "serial (serial_share = kernelCPU/stageWall); "
                     "still a ceiling, not a measurement"),
        }
        arith["session_discrepancy_flag"] = (
            "same frozen workload and the same 8217 calls in both "
            "sessions yet mean kernel service differs ~27x (see the two "
            "session_kernel_mean_ns figures in this record); this probe's "
            "single-thread min-of-7 measurements at HEAD are the primary "
            "anchor; the discrepancy is flagged for review")
    rec["admission_arithmetic"] = arith

    # ---- close out ---------------------------------------------------------
    rec["capture_sha256_after"] = sha256_file(CAPTURE)
    rec["inputs_immutable"] = (rec["capture_sha256_before"]
                               == rec["capture_sha256_after"])
    rec["wall_used_s"] = round(WALL_BUDGET_S - (deadline - time.monotonic()), 3)
    rec["window_at_end"] = window_stamp()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(rec, f, indent=1, default=str)
    print(f"[f2i-probe] wrote {args.out}")

    print("[f2i-probe] identity_ok:", rec["identity"]["identity_ok"],
          "| inputs_immutable:", rec["inputs_immutable"])
    print("[f2i-probe] od_compared:", rec["validation"]["od_compared"],
          "| bitwise output failures:",
          rec["validation"]["bitwise_output_failures"],
          "| reach failures:",
          rec["validation"]["reach_mask_or_list_failures"],
          "| pristine failures:",
          rec["validation"]["pristine_epilogue_failures"],
          "| overflow:", rec["validation"]["overflow_events"])
    print("[f2i-probe] fastmath_predicate_ok:",
          rec["predicate_characterization"]["fastmath_predicate_ok"],
          "| slice ascending violations:",
          rec["slice_characterization"]["strictly_ascending_violations"])
    if "walls_summary_ns" in rec:
        for k in ("median_t_base", "median_t_track", "median_t_reset",
                  "median_cand_side", "median_net", "median_net_share",
                  "median_t_scan2", "median_t_alloc4", "median_t_cols",
                  "median_fixed_attr"):
            print(f"[f2i-probe] {k}: {rec['walls_summary_ns'][k]}")
    if "coverage_summary" in rec:
        for k in ("C_d_median", "C_d_max", "n_r_median", "n_r_max",
                  "n_r_over_C_d_median", "appends_total_median",
                  "unique_total_median", "arc_checks_p1_median",
                  "arc_checks_p2_median"):
            print(f"[f2i-probe] {k}: {rec['coverage_summary'][k]}")
    for k in ("removable_kernel_share_measured",
              "removable_wall_share_ceiling", "above_gate_ceiling",
              "kernel_cpu_over_stage_wall"):
        if k in arith:
            print(f"[f2i-probe] arith {k}: {arith[k]}")
    if "chosen_cell_context" in arith:
        for k in ("r_ceiling_from_fixed", "removable_wall_share_ceiling",
                  "kernel_cpu_over_stage_wall"):
            print(f"[f2i-probe] chosen {k}: "
                  f"{arith['chosen_cell_context'][k]}")
    print("[f2i-probe] window_at_end:", rec["window_at_end"])


if __name__ == "__main__":
    main()
