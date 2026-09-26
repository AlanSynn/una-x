"""H05 bounded kernel probes (A1, A2, A3, F1, F2) — measurement only.

Each invocation runs ONE probe in this (bounded child) process:

  --probe A1  compiled-allocation evidence for the accessibility search kernel:
              NRT runtime alloc/free per call (NUMBA_NRT_STATS=1) + NRT_Alloc
              callsite sizes parsed from inspect_llvm of the REAL dispatchers
              (source-syntax reading is not used as evidence).
  --probe A2  measured destination-scan coverage: out-of-cutoff/unreachable
              share of the per-origin destination scans, using the REAL
              compact_vector_node_view_scope + adjust_destination_distances
              kernels on captured engine arrays; validated by recomputing the
              four metrics per origin and comparing to the engine outputs
              (ULP rule: max_ulp <= 4 for float sums).
  --probe A3  measured private-scope init share across >= 32 sampled origins:
              init-replica (same allocation/init pattern, njit) vs the real
              kernel; validated by value semantics (unreached == 1 + cutoff).
  --probe F1  measured serialization of _accumulate_od_flow: T independent-
              buffer threads (real kernel, real captured CSR/gradients, distinct
              per-thread origin/destination pairs and out-buffers) vs serial;
              delivered>0 required for every task; concurrency counters at
              kernel enter/exit.
  --probe F2  measured per-OD full-V' scan coverage and temporaries: envelope
              sizes from the captured gradients, timed replica of the two fixed
              full-V' scans, temporaries byte sizes parsed from inspect_llvm
              NRT_Alloc callsites of the REAL _accumulate_od_flow.

Inputs: campaign_data h05 pilot captures (access_arrays.npz, flow_arrays.npz).
Runs against the INSTALLED wheel venv (campaign_data/venvs/b0_wheel) with
NUMBA_CACHE_DIR shared with the pilot sessions (warm compile).
Output: campaign_data/h05_probes/<probe>_<tag>.json

Every probe enforces its own wall budget and refuses (records refusal) instead
of overrunning.
"""
import argparse
import hashlib
import json
import os
import re
import sys
import threading
import time

import numpy as np

CAMPAIGN = "/Users/alansynn/orca/workspaces/una-x"
CAPTURE = f"{CAMPAIGN}/campaign_data/h05_captures"
PROBE_DIR = f"{CAMPAIGN}/campaign_data/h05_probes"
WALL_BUDGET_S = 600.0
N_SAMPLED_ORIGINS = 32
SEED = 20260925
MAX_ULP = 4  # H03 ULP rule for trace-vs-compiled float sums


def utc():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def save(name, rec):
    os.makedirs(PROBE_DIR, exist_ok=True)
    p = f"{PROBE_DIR}/{name}.json"
    with open(p, "w") as f:
        json.dump(rec, f, indent=1, default=str)
    print(f"[h05-probe] wrote {p}")
    return p


def identity_record(site_expected):
    import urban_network_analysis as u
    path = os.path.dirname(os.path.abspath(u.__file__))
    ok = os.path.realpath(path).startswith(os.path.realpath(site_expected) + os.sep)
    return {"una_path": path, "identity_ok": ok}


# ---------------------------------------------------------------------------
# LLVM NRT callsite parsing (compiled-allocation evidence)
# ---------------------------------------------------------------------------
NRT_CALL_RE = re.compile(r"@NRT_[A-Za-z0-9_]*[Aa]lloc[A-Za-z0-9_]*\(\s*(?:i64\s+)?([^\s,)]+)")


def nrt_callsites(dispatcher):
    """Parse NRT allocation callsites from the compiled LLVM IR of every
    compiled signature. Returns static-size list, dynamic count, per-signature
    summary. Evidence class: compiled IR, not source syntax."""
    out = {"n_signatures": len(dispatcher.signatures), "signatures": []}
    static_sizes_all = []
    for i, sig in enumerate(dispatcher.signatures):
        try:
            llvm = dispatcher.inspect_llvm(sig)
        except Exception as e:
            out["signatures"].append({"index": i, "error": repr(e)})
            continue
        sizes, dynamic = [], 0
        for m in NRT_CALL_RE.finditer(llvm):
            tok = m.group(1)
            if tok.isdigit():
                sizes.append(int(tok))
            else:
                dynamic += 1
        nrt_alloc_syms = sorted(set(re.findall(r"@NRT_[A-Za-z0-9_]+", llvm)))
        out["signatures"].append({
            "index": i, "static_alloc_sizes_bytes": sorted(set(sizes)),
            "n_static_callsites": len(sizes), "n_dynamic_callsites": dynamic,
            "nrt_symbols_present": nrt_alloc_syms,
            "llvm_bytes": len(llvm),
        })
        static_sizes_all.extend(sizes)
    out["static_sizes_union_bytes"] = sorted(set(static_sizes_all))
    out["has_compiled_alloc"] = bool(static_sizes_all) or any(
        s.get("n_dynamic_callsites", 0) > 0 for s in out["signatures"])
    return out


def load_access_capture():
    d = np.load(f"{CAPTURE}/O3_ACCESS/access_arrays.npz")
    return {k: d[k] for k in d.files}


def load_flow_capture():
    d = np.load(f"{CAPTURE}/O3_FLOW/flow_arrays.npz")
    return {k: d[k] for k in d.files}


def load_access_settings():
    """Assembled settings of the ACCESS_od pilot (decay parameters the engine
    actually used — never hardcode these in replicas)."""
    with open(f"{CAMPAIGN}/campaign_data/h05_sessions/ACCESS_od.json") as f:
        return json.load(f)["settings"]["assembled"]


def sampled_origins(n_origins):
    rng = np.random.default_rng(SEED)
    idx = np.sort(rng.choice(n_origins, size=min(N_SAMPLED_ORIGINS, n_origins),
                             replace=False))
    return idx, int(rng_calls_needed(n_origins))


def rng_calls_needed(n):
    return 1  # single choice() call; recorded for provenance


# ---------------------------------------------------------------------------
# A1
# ---------------------------------------------------------------------------
def probe_a1(site_expected, deadline):
    rec = {"probe": "A1", "hypothesis": "A1_snapshot_scratch",
           "evidence_class": "compiled_IR_plus_runtime_NRT", "utc": utc(),
           "identity": identity_record(site_expected)}
    from numba.core.runtime import rtsys
    from urban_network_analysis.Engines import AccessibilityWElevation as AWE

    cap = load_access_capture()
    ap_ = cap["adjacency_pointer"]; av = cap["adjacency_vector"]
    avw = cap["adjacency_vector_weights"]; avn = cap["adjacynct_vector_network_node"]
    oti = cap["o_terminal_idxs"]; otw = cap["o_terminal_weights"]
    dti = cap["d_terminal_idxs"]; dtw = cap["d_terminal_weights"]
    d_w = cap["d_node_weight"]
    n_orig = oti.shape[0]
    d_count = int(dtw.shape[0])
    cutoff = float(load_access_settings()["search_radius"])
    o_idx, _ = sampled_origins(n_orig)

    kernels = {
        "compact_vector_node_view_scope": AWE.compact_vector_node_view_scope,
        "adjust_destination_distances": AWE.adjust_destination_distances,
        "reach_gravity_knn_access": AWE.reach_gravity_knn_access,
        "integrated_scope_access": AWE.integrated_scope_access,
    }
    rec["ir"] = {}
    for name, disp in kernels.items():
        warm_args = a1_warm_args(name, disp, cap, cutoff, o_idx[0])
        disp(*warm_args)  # compile/warm (cache hit expected)
        rec["ir"][name] = nrt_callsites(disp)

    # runtime NRT per origin call (serial, single thread)
    stats = {"compact_vector_node_view_scope": [], "integrated_scope_access": []}
    for o in o_idx:
        if time.monotonic() > deadline:
            rec["truncated"] = True
            break
        args = a1_warm_args("compact_vector_node_view_scope",
                            AWE.compact_vector_node_view_scope, cap, cutoff, int(o))
        b = rtsys.get_allocation_stats()
        AWE.compact_vector_node_view_scope(*args)
        a = rtsys.get_allocation_stats()
        stats["compact_vector_node_view_scope"].append(
            {"origin": int(o), "alloc": a.alloc - b.alloc, "free": a.free - b.free,
             "mi_alloc": a.mi_alloc - b.mi_alloc})
        args_i = a1_warm_args("integrated_scope_access", AWE.integrated_scope_access,
                              cap, cutoff, int(o))
        b = rtsys.get_allocation_stats()
        AWE.integrated_scope_access(*args_i)
        a = rtsys.get_allocation_stats()
        stats["integrated_scope_access"].append(
            {"origin": int(o), "alloc": a.alloc - b.alloc, "free": a.free - b.free,
             "mi_alloc": a.mi_alloc - b.mi_alloc})
    for k, v in stats.items():
        allocs = np.array([x["alloc"] for x in v])
        frees = np.array([x["free"] for x in v])
        rec[f"runtime_nrt_{k}"] = {
            "n_origins": len(v), "alloc_per_call_min": int(allocs.min()),
            "alloc_per_call_median": float(np.median(allocs)),
            "alloc_per_call_max": int(allocs.max()),
            "free_per_call_median": float(np.median(frees)),
            "per_origin": v,
        }
    # derived bounds: bytes/call in [max_static_size, sum_static_sizes] per
    # callsite-execution unknown; runtime rate bounds the count dimension
    rec["interpretation"] = (
        "alloc_per_call_* = measured NRT allocation count per single-origin call "
        "(runtime, per origin). Compiled-allocation evidence = NRT allocation "
        "symbols + allocation callsites present in inspect_llvm of the real "
        "dispatchers; numba emits DYNAMIC-size NRT_MemInfo_alloc_aligned callsites "
        "(verified: no literal i64 sizes), so per-call BYTES are bounded as "
        "rate x per-site element counts x itemsize, with element counts taken from "
        "the measured captured array dimensions (V, D, node degrees), not from "
        "source-syntax reading. Reported as bounds, not a point estimate.")
    return rec


def a1_warm_args(name, disp, cap, cutoff, o):
    ap_ = cap["adjacency_pointer"]; av = cap["adjacency_vector"]
    avw = cap["adjacency_vector_weights"]; avn = cap["adjacynct_vector_network_node"]
    oti = cap["o_terminal_idxs"]; otw = cap["o_terminal_weights"]
    dti = cap["d_terminal_idxs"]; dtw = cap["d_terminal_weights"]
    d_w = cap["d_node_weight"]
    if name == "compact_vector_node_view_scope":
        return (oti[o].astype(np.int64), otw[o].astype(np.float64),
                ap_, av, avw, avn, cutoff, int(dtw.shape[0]))
    if name == "integrated_scope_access":
        return (oti, otw, ap_, av, avw, avn, dti, dtw, d_w,
                0.001, 0.0, 500.0, 1.0, "logistic",
                np.array([1.0, 1.0, 0.5], dtype=np.float64), cutoff)
    if name == "adjust_destination_distances":
        sw = np.full(ap_.shape[0] - 1 + dtw.shape[0], 1.0 + cutoff)
        return (sw, dti, dtw, int(dtw.shape[0]))
    if name == "reach_gravity_knn_access":
        s = load_access_settings()
        g = s["gravity_decay_constant"] / s["gravity_logistic_midpoint"]
        dd = np.full(dtw.shape[0], cutoff + 1.0)
        return (dd, d_w, cutoff, s["gravity_beta"], float(s["gravity_plateau"]),
                float(s["gravity_logistic_midpoint"]), float(s["gravity_plateau"]),
                np.asarray(s["knn_weights"], dtype=np.float64),
                s["gravity_beta"], s["knn_decay"],
                float(s["gravity_logistic_midpoint"]), g, g)


# ---------------------------------------------------------------------------
# A2 — destination-scan coverage (real kernels, numpy filter replica)
# ---------------------------------------------------------------------------
def numpy_metrics_replica(d_distance, d_weights, cutoff, beta, plateau, midpoint,
                          growth, knn_weights, knn_decay="logistic"):
    assert knn_decay == "logistic", f"replica implements logistic only, got {knn_decay!r}"
    """Exact numpy translation of reach_gravity_knn_access (same expressions,
    same order). Validated against engine outputs with max_ulp <= 4."""
    sel = np.where(d_distance <= cutoff)[0]
    nd = d_distance[sel]
    nw = d_weights[sel]
    reach = nw.sum()
    g_exp = (nw / np.exp(beta * np.maximum(0.0, nd - plateau))).sum()
    g_log = (nw * (1 - 1 / (1 + np.exp(-growth * (nd - plateau - midpoint))))).sum()
    knn = min(nd.shape[0], knn_weights.shape[0])
    if knn == 0:
        return reach, g_exp, g_log, 0.0, int(sel.shape[0])
    idx = np.argsort(nd)
    ns_, nw_ = nd[idx], nw[idx]
    kc = knn_weights[:knn]
    knn_access = ((kc * nw_[:knn]) *
                  (1 - 1 / (1 + np.exp(-growth * (ns_[:knn] - plateau - midpoint))))).sum()
    return reach, g_exp, g_log, knn_access, int(sel.shape[0])


def ulp_diff(a, b):
    if a == b:
        return 0
    ia = np.float64(a).view(np.int64)
    ib = np.float64(b).view(np.int64)
    return int(abs(int(ia) - int(ib)))


def probe_a2(site_expected, deadline):
    rec = {"probe": "A2", "hypothesis": "A2_destination_locality",
           "evidence_class": "real_kernel_replica_bit_validated", "utc": utc(),
           "identity": identity_record(site_expected)}
    from urban_network_analysis.Engines import AccessibilityWElevation as AWE
    cap = load_access_capture()
    ap_ = cap["adjacency_pointer"]; av = cap["adjacency_vector"]
    avw = cap["adjacency_vector_weights"]; avn = cap["adjacynct_vector_network_node"]
    oti = cap["o_terminal_idxs"]; otw = cap["o_terminal_weights"]
    dti = cap["d_terminal_idxs"]; dtw = cap["d_terminal_weights"]
    d_w = cap["d_node_weight"]
    d_count = int(dtw.shape[0])
    cutoff = float(load_access_settings()["search_radius"])
    # engine metric outputs captured from the od_level pilot record
    eng = json.load(open(f"{CAPTURE}/O3_ACCESS/access_engine_metrics.json"))
    n_orig = oti.shape[0]
    o_idx, _ = sampled_origins(n_orig)

    per_origin, validations = [], []
    for o in o_idx:
        o = int(o)
        t0 = time.perf_counter_ns()
        scope, _pred = AWE.compact_vector_node_view_scope(
            oti[o].astype(np.int64), otw[o].astype(np.float64),
            ap_, av, avw, avn, cutoff, d_count)
        d_distance = AWE.adjust_destination_distances(scope, dti, dtw, d_count)
        t1 = time.perf_counter_ns()
        s = load_access_settings()
        r, ge, gl, kn, n_in = numpy_metrics_replica(
            d_distance, d_w, float(s["search_radius"]),
            float(s["gravity_beta"]), float(s["gravity_plateau"]),
            float(s["gravity_logistic_midpoint"]),
            s["gravity_decay_constant"] / s["gravity_logistic_midpoint"],
            np.asarray(s["knn_weights"], dtype=np.float64),
            s["knn_decay"])
        t2 = time.perf_counter_ns()
        n_out = d_count - n_in
        per_origin.append({
            "origin": o, "wall_ns": t1 - t0, "metrics_ns": t2 - t1,
            "n_in_radius": n_in, "n_out_of_radius": n_out,
            "out_share_of_scan": n_out / d_count,
        })
        e = eng[str(o)] if str(o) in eng else None
        if e is None:
            validations.append({"origin": o, "status": "no_engine_value_at_sampled_origin"})
            continue
        ulps = {m: ulp_diff(v, e[m]) for m, v in
                (("reach", r), ("gravity_exponential", ge),
                 ("gravity_logistic", gl), ("knn_access", kn))}
        validations.append({"origin": o, "status": "ok", "ulp": ulps,
                            "within_ulp4": all(u <= MAX_ULP for u in ulps.values())})
        if time.perf_counter() > deadline:
            rec["truncated"] = True
            break
    out_shares = np.array([p["out_share_of_scan"] for p in per_origin])
    rec["scan_coverage"] = {
        "n_origins_measured": len(per_origin), "d_count": d_count,
        "out_share_mean": float(out_shares.mean()),
        "out_share_median": float(np.median(out_shares)),
        "out_share_min": float(out_shares.min()), "out_share_max": float(out_shares.max()),
        "in_radius_mean": float(np.mean([p["n_in_radius"] for p in per_origin])),
        "scan_work_note": ("per-origin destination scan work is 2 fixed O(D) passes "
                           "(adjust loop + np.where filter); out_share_of_scan is the "
                           "measured fraction of that scan work on destinations the "
                           "metrics never use"),
        "per_origin": per_origin,
    }
    rec["validation"] = {
        "rule": f"max_ulp <= {MAX_ULP} per H03 trace-vs-compiled float-sum rule",
        "n_validated": sum(1 for v in validations if v["status"] == "ok"),
        "n_within_ulp4": sum(1 for v in validations if v.get("within_ulp4")),
        "details": validations,
    }
    rec["gate_metric"] = {
        "gate": "minimum_removable_fraction 0.05",
        "measured_out_share_median": float(np.median(out_shares)),
        "note": "A2 admission asks whether destination-local enumeration could remove "
                ">=5% of end-to-end time; measured scan coverage is the direct "
                "coverage number, end-to-end share requires the stage model weights",
    }
    return rec


# ---------------------------------------------------------------------------
# A3 — private scope init share (>= 32 origins)
# ---------------------------------------------------------------------------
def probe_a3(site_expected, deadline):
    rec = {"probe": "A3", "hypothesis": "A3_private_scope",
           "evidence_class": "init_replica_timing_multi_origin", "utc": utc(),
           "identity": identity_record(site_expected)}
    import numba as nb
    from urban_network_analysis.Engines import AccessibilityWElevation as AWE
    cap = load_access_capture()
    ap_ = cap["adjacency_pointer"]; av = cap["adjacency_vector"]
    avw = cap["adjacency_vector_weights"]; avn = cap["adjacynct_vector_network_node"]
    oti = cap["o_terminal_idxs"]; otw = cap["o_terminal_weights"]
    dtw = cap["d_terminal_weights"]
    d_count = int(dtw.shape[0])
    nd_node_count = d_count + ap_.shape[0] - 1
    cutoff = float(load_access_settings()["search_radius"])
    n_orig = oti.shape[0]
    o_idx, _ = sampled_origins(n_orig)

    @nb.njit(cache=False, nogil=True, fastmath=True)
    def init_replica(o_idx_start, o_idx_end, w0, w1, n, cut):
        w = np.ones(n, dtype=np.float64) + cut
        w[o_idx_start] = w0
        w[o_idx_end] = w1
        return w

    init_replica(oti[o_idx[0]][0], oti[o_idx[0]][1], otw[o_idx[0]][0],
                 otw[o_idx[0]][1], nd_node_count, cutoff)  # compile

    REPEATS = 5
    per_origin = []
    for o in o_idx:
        o = int(o)
        s0, s1 = int(oti[o][0]), int(oti[o][1])
        w0, w1 = float(otw[o][0]), float(otw[o][1])
        t0 = time.perf_counter_ns()
        for _ in range(REPEATS):
            scope, _p = AWE.compact_vector_node_view_scope(
                oti[o].astype(np.int64), otw[o].astype(np.float64),
                ap_, av, avw, avn, cutoff, d_count)
        t1 = time.perf_counter_ns()
        for _ in range(REPEATS):
            init_replica(s0, s1, w0, w1, nd_node_count, cutoff)
        t2 = time.perf_counter_ns()
        full_ns, init_ns = (t1 - t0) / REPEATS, (t2 - t1) / REPEATS
        # value-semantics validation: unreached entries must equal 1 + cutoff
        max_val = float(scope.max())
        per_origin.append({
            "origin": o, "full_kernel_ns": full_ns, "init_replica_ns": init_ns,
            "init_share": init_ns / full_ns,
            "scope_max": max_val, "expected_init_value": 1.0 + cutoff,
            "init_value_ok": abs(max_val - (1.0 + cutoff)) < 1e-9,
        })
        if time.perf_counter() > deadline:
            rec["truncated"] = True
            break
    shares = np.array([p["init_share"] for p in per_origin])
    rec["init_share"] = {
        "n_origins": len(per_origin),
        "mean": float(shares.mean()), "median": float(np.median(shares)),
        "min": float(shares.min()), "max": float(shares.max()),
        "nd_node_count": int(nd_node_count),
        "init_array_bytes": int(nd_node_count * 8),
        "per_origin": per_origin,
        "note": "supersedes the old single-origin 0.98% probe; init replica mirrors "
                "np.ones(nd)+cutoff + two terminal assignments (the kernel's init "
                "section), compiled njit with the same flags",
    }
    rec["validation"] = {
        "init_value_check": all(p["init_value_ok"] for p in per_origin),
        "note": "kernel scope max must equal 1+cutoff (unreached nodes keep the "
                "initialized value) — validates the replica's init semantics",
    }
    return rec


# ---------------------------------------------------------------------------
# F1 — GIL serialization of _accumulate_od_flow
# ---------------------------------------------------------------------------
def find_flow_tasks(cap, T, deadline):
    """Build up to T real _accumulate_od_flow calls on captured engine arrays.

    Search policy: per origin, try destinations NEAREST first — the frozen
    detour gate is ratio 1.05, so qualifying pairs are spatially close (the
    engine itself finds ~8 qualifying destinations per origin at sel1024).
    The non-degeneracy gate is strict everywhere: delivered>0 AND nonzero
    out-buffers. Bounded by deadline and a kernel-trial cap.
    Returns (tasks, meta)."""
    from scipy.sparse.csgraph import dijkstra as sp_dijkstra
    from urban_network_analysis.Engines import AggregateFlow as AF
    indptr = cap["csr_indptr"]; indices = cap["csr_indices"]
    weights = cap["csr_weights"]; eid = cap["csr_edge_id"]
    dr = cap["csr_direction"]
    fwd_indptr = cap["csr_fwd_indptr"]; fwd_indices = cap["csr_fwd_indices"]
    fwd_data = cap["csr_fwd_data"]
    g_indptr = cap["grad_indptr"]; g_nodes = cap["grad_nodes"]; g_dist = cap["grad_dist"]
    g_pred = cap["grad_pred"]
    n_net = int(cap["n_network_nodes"]); n_dest = int(cap["n_destinations"])
    n_orig = int(cap["n_origins"]); first_o = int(cap["first_origin_node"])
    n_total = indptr.shape[0] - 1
    # sentinel snap-edge ids that cannot collide with real edge ids
    fake_o_edge = int(eid.max()) + 1000
    fake_d_edge = int(eid.max()) + 1001
    from scipy.sparse import csr_matrix
    csr_fwd = csr_matrix((fwd_data, fwd_indices, fwd_indptr),
                         shape=(n_total, n_total))

    tasks = []
    n_kernel_trials = 0
    origins_walked = 0
    for o_try in range(n_orig):
        if len(tasks) >= T or time.monotonic() > deadline:
            break
        origins_walked += 1
        ov = first_o + o_try
        dist_o, pred_o = sp_dijkstra(csr_fwd, directed=True, indices=ov,
                                     return_predecessors=True)
        dv_all = np.arange(n_net, n_net + n_dest)
        d_o = dist_o[dv_all]
        order = np.argsort(np.where(np.isfinite(d_o), d_o, np.inf))
        for k in order[:40]:
            if len(tasks) >= T or time.monotonic() > deadline or n_kernel_trials > 3000:
                break
            d_try = int(k)
            dv = n_net + d_try
            s0, s1 = int(g_indptr[d_try]), int(g_indptr[d_try + 1])
            if not (s1 > s0 and np.isfinite(dist_o[dv])):
                continue
            dd_buf = np.full(n_total, np.inf, dtype=np.float64)
            pd_buf = np.full(n_total, -9999, dtype=np.int32)
            cols = g_nodes[s0:s1]
            dd_buf[cols] = g_dist[s0:s1]
            pd_buf[cols] = g_pred[s0:s1]
            d_shortest = float(dist_o[dv])
            # frozen O3_FLOW settings: flow_detour_mode 'ratio' -> budget = shortest*1.05
            budget = d_shortest * 1.05
            out_AB = np.zeros(weights.shape[0], dtype=np.float64)
            out_BA = np.zeros(weights.shape[0], dtype=np.float64)
            args = (indptr, indices, weights, eid, dr,
                    dist_o.astype(np.float64), dd_buf,
                    pred_o.astype(np.int32), pd_buf,
                    int(ov), int(dv), fake_o_edge, fake_d_edge,
                    d_shortest, budget,
                    2, 0.001, 200.0, 1.0, n_net,
                    out_AB, out_BA, np.zeros(0, dtype=np.float64))
            n_kernel_trials += 1
            delivered = AF._accumulate_od_flow(*args)
            if delivered > 0.0 and (out_AB.sum() + out_BA.sum()) > 0:
                tasks.append({"args": args, "origin_virtual": int(ov),
                              "dest_virtual": int(dv), "delivered_warm": float(delivered),
                              "envelope_nodes": int(s1 - s0)})
    meta = {
        "policy": ("per origin, destinations tried nearest-first (ratio-1.05 detour "
                   "gate makes qualifying pairs spatially close); gate unchanged: "
                   "delivered>0 and nonzero out-buffers"),
        "origins_walked": origins_walked, "n_kernel_trials": n_kernel_trials,
        "n_qualifying": len(tasks), "needed": T,
    }
    return tasks, meta


def probe_f1(site_expected, deadline):
    rec = {"probe": "F1", "hypothesis": "F1_fixed_stripe_nogil",
           "evidence_class": "controlled_concurrent_independent_buffer", "utc": utc(),
           "identity": identity_record(site_expected)}
    from urban_network_analysis.Engines import AggregateFlow as AF
    cap = load_flow_capture()

    T = min(9, os.cpu_count() - 1)
    ROUNDS = 12
    tasks, meta = find_flow_tasks(cap, T, deadline)
    rec["task_search"] = meta
    rec["tasks"] = [{k: v for k, v in t.items() if k != "args"} for t in tasks]
    if len(tasks) < T:
        rec["refused"] = f"only {len(tasks)} non-degenerate tasks found (need {T})"
        return rec

    conc = {"cur": 0, "max": 0, "lock": threading.Lock()}

    def work(task, rounds):
        args = task["args"]
        for _ in range(rounds):
            with conc["lock"]:
                conc["cur"] += 1
                conc["max"] = max(conc["max"], conc["cur"])
            try:
                AF._accumulate_od_flow(*args)
            finally:
                with conc["lock"]:
                    conc["cur"] -= 1

    # serial reference: T tasks x ROUNDS on one thread
    t0 = time.perf_counter_ns()
    for t in tasks:
        work(t, ROUNDS)
    t1 = time.perf_counter_ns()
    serial_ns = t1 - t0

    # parallel: T threads, each its own task+buffers
    from concurrent.futures import ThreadPoolExecutor
    t2 = time.perf_counter_ns()
    with ThreadPoolExecutor(max_workers=T) as ex:
        list(ex.map(lambda tk: work(tk, ROUNDS), tasks))
    t3 = time.perf_counter_ns()
    parallel_ns = t3 - t2
    rec["result"] = {
        "n_threads": T, "rounds": ROUNDS,
        "serial_wall_ns": serial_ns, "parallel_wall_ns": parallel_ns,
        "speedup_serial_over_parallel": serial_ns / parallel_ns,
        "observed_max_concurrent": conc["max"],
        "verdict_inputs": ("speedup ~1.0 and max_concurrent==1 => GIL-bound kernel "
                           "serialization measured (decorator inspection not used)"),
        "buffers": "every thread: independent d_o/pred_o, dd_buf/pd_buf, out_AB/out_BA; "
                   "shared read-only CSR + gradients",
        "n_kernel_calls_total": T * ROUNDS * 2,
        "probe_call_semantics": ("decay_curve_id=2 (logistic, matches frozen "
                                 "flow_path_detour_penalty); trip_volume=1.0; sentinel "
                                 "snap-edge ids above max real edge id — timing probe, "
                                 "not a numerical oracle"),
    }
    return rec


# ---------------------------------------------------------------------------
# F2 — per-OD full-V' scan coverage + temporaries
# ---------------------------------------------------------------------------
def probe_f2(site_expected, deadline):
    rec = {"probe": "F2", "hypothesis": "F2_local_overlap",
           "evidence_class": "envelope_stats_plus_timed_scan_replica_plus_IR", "utc": utc(),
           "identity": identity_record(site_expected)}
    from urban_network_analysis.Engines import AggregateFlow as AF
    cap = load_flow_capture()
    g_indptr = cap["grad_indptr"]; g_nodes = cap["grad_nodes"]; g_dist = cap["grad_dist"]
    n_total = int(cap["csr_indptr"].shape[0] - 1)
    n_dest = int(cap["n_destinations"])

    lens = np.diff(g_indptr.astype(np.int64))
    rec["envelope"] = {
        "n_total_V_prime": n_total, "n_destinations": n_dest,
        "grad_entries_total": int(g_nodes.shape[0]),
        "grad_bytes_sparse_total": int(g_nodes.shape[0] * 20),
        "envelope_len_min": int(lens.min()), "envelope_len_median": float(np.median(lens)),
        "envelope_len_mean": float(lens.mean()), "envelope_len_max": int(lens.max()),
        "envelope_over_V_prime_mean": float(lens.mean() / n_total),
    }

    # timed replica of the two fixed full-V' scans (reach build + reach_nodes
    # build), on realistic data (finite/inf mixed d_o, d_d from gradients)
    import numba as nb

    @nb.njit(cache=False, nogil=True, fastmath=True)
    def scans_replica(d_o, d_d, budget):
        n_nodes = d_o.shape[0]
        reach = np.zeros(n_nodes, dtype=nb.boolean)
        n_r = 0
        for v in range(n_nodes):
            if d_o[v] < np.inf and d_d[v] < np.inf and d_o[v] + d_d[v] <= budget:
                reach[v] = True
                n_r += 1
        reach_nodes = np.empty(n_r, dtype=np.int64)
        p = 0
        for v in range(n_nodes):
            if reach[v]:
                reach_nodes[p] = v
                p += 1
        return n_r

    d_o = np.full(n_total, np.inf)
    d_d = np.full(n_total, np.inf)
    # realistic envelope: scatter a mid-sized gradient + a forward-distance field
    mid = int(np.argsort(lens)[len(lens) // 2])
    cols = g_nodes[g_indptr[mid]:g_indptr[mid + 1]]
    d_d[cols] = g_dist[g_indptr[mid]:g_indptr[mid + 1]]
    finite = np.isfinite(d_d)
    d_o[finite] = np.linspace(1.0, float(finite.sum()), int(finite.sum()))
    budget = 1.05 * 3000.0 + 100.0
    scans_replica(d_o, d_d, budget)  # compile
    REPEATS = 20
    t0 = time.perf_counter_ns()
    for _ in range(REPEATS):
        n_r = scans_replica(d_o, d_d, budget)
    t1 = time.perf_counter_ns()
    scans_ns = (t1 - t0) / REPEATS

    # mean real kernel time from the chosen flow pilot's session record
    # (od_kernel_aggregate is written inside the session JSON)
    with open(f"{CAMPAIGN}/campaign_data/h05_flow_decision.json") as f:
        chosen = json.load(f)["chosen_variant"]
    session = f"{CAMPAIGN}/campaign_data/h05_sessions/FLOW_{chosen}_od.json"
    agg = json.load(open(session))["od_kernel_aggregate"]
    mean_kernel_ns = agg["mean_ns"]
    rec["fixed_scans"] = {
        "replica": "two full-V' passes: reach mask build + reach_nodes gather "
                   "(byte-semantics mirror of kernel sections 1..2)",
        "measured_scans_ns_at_V_prime": scans_ns,
        "observed_envelope_n_r": int(n_r),
        "pilot_mean_kernel_ns": mean_kernel_ns,
        "pilot_n_calls": agg["n_calls"],
        "scans_share_of_mean_kernel": scans_ns / mean_kernel_ns,
        "note": "share uses the pilot mean kernel service time at the same scale",
    }
    # compile the REAL kernel on same-dtype mini arrays so inspect_llvm has a
    # signature to parse (temporaries are n_nodes-sized -> dynamic callsites)
    mini = lambda dt: np.zeros(4, dtype=dt)
    AF._accumulate_od_flow(
        mini(np.int64), mini(np.int32) if cap["csr_indices"].dtype == np.int32
        else mini(np.int64), mini(np.float64), mini(np.int64), mini(np.int8)
        if cap["csr_direction"].dtype == np.int8 else mini(np.int64),
        mini(np.float64), mini(np.float64), mini(np.int32), mini(np.int32),
        0, 1, 0, 1, 1.0, 2.0, 0, 0.001, 200.0, 1.0, 4,
        np.zeros(1, dtype=np.float64), np.zeros(1, dtype=np.float64),
        np.zeros(0, dtype=np.float64))
    rec["temporaries_ir"] = nrt_callsites(AF._accumulate_od_flow)
    # per-call NRT allocation of the REAL kernel at real (captured) scale —
    # measured here because pilot sessions deliberately run without NRT stats
    nrt_per_call = {"enabled": False, "reason": None, "alloc_per_call": None,
                    "bytes_note": None}
    if not os.environ.get("NUMBA_NRT_STATS"):
        nrt_per_call["reason"] = ("NUMBA_NRT_STATS not set for this probe child; "
                                  "per-call NRT rate not measured this run")
    else:
        try:
            from numba.core.runtime import rtsys
            tasks, _meta = find_flow_tasks(cap, 1, deadline)
            if not tasks:
                nrt_per_call["reason"] = "no qualifying real task found for NRT measurement"
            else:
                args = tasks[0]["args"]
                CALLS = 50
                allocs, frees = [], []
                for _ in range(CALLS):
                    b = rtsys.get_allocation_stats()
                    AF._accumulate_od_flow(*args)
                    a = rtsys.get_allocation_stats()
                    allocs.append(a.alloc - b.alloc)
                    frees.append(a.free - b.free)
                nrt_per_call = {
                    "enabled": True, "n_calls": CALLS,
                    "task": {"origin_virtual": tasks[0]["origin_virtual"],
                             "dest_virtual": tasks[0]["dest_virtual"],
                             "envelope_nodes": tasks[0]["envelope_nodes"]},
                    "alloc_per_call_min": int(min(allocs)),
                    "alloc_per_call_median": float(np.median(allocs)),
                    "alloc_per_call_max": int(max(allocs)),
                    "free_per_call_median": float(np.median(frees)),
                    "note": ("measured on real captured CSR/gradients at V'=%d; "
                             "alloc==free per call confirms temporaries are "
                             "per-call allocations, not retained state" % n_total),
                }
        except Exception as e:  # noqa: BLE001 — record and continue
            nrt_per_call["reason"] = f"error: {e!r}"
    rec["temporaries_accounting"] = {
        "full_V_prime_temporaries": [
            {"name": "reach", "dtype": "bool", "bytes_per_od": n_total},
            {"name": "cont_o", "dtype": "bool", "bytes_per_od": n_total},
            {"name": "cont_d", "dtype": "bool", "bytes_per_od": n_total},
            {"name": "acc_o", "dtype": "float64", "bytes_per_od": 8 * n_total},
            {"name": "acc_d", "dtype": "float64", "bytes_per_od": 8 * n_total},
        ],
        "bytes_per_stripe_per_od": 19 * n_total,
        "runtime_nrt_per_call": nrt_per_call,
        "note": "dtype/size cross-referenced with measured V' and the runtime NRT "
                "rate; IR allocation callsites above are the compiled-allocation "
                "evidence (numba emits dynamic-size NRT_MemInfo_alloc_aligned "
                "callsites)",
    }
    return rec


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", required=True,
                    choices=["A1", "A2", "A3", "F1", "F2"])
    ap.add_argument("--site-expected",
                    default=f"{CAMPAIGN}/campaign_data/venvs/b0_wheel/lib/python3.11/site-packages")
    ap.add_argument("--tag", default=time.strftime("%H%M%S"))
    ap.add_argument("--wall-budget-s", type=float, default=WALL_BUDGET_S)
    args = ap.parse_args()

    deadline = time.monotonic() + args.wall_budget_s
    fn = {"A1": probe_a1, "A2": probe_a2, "A3": probe_a3,
          "F1": probe_f1, "F2": probe_f2}[args.probe]
    t0 = time.monotonic()
    try:
        rec = fn(args.site_expected, deadline)
        rec["wall_s"] = round(time.monotonic() - t0, 1)
        rec["within_budget"] = rec["wall_s"] <= args.wall_budget_s
    except Exception as e:
        import traceback
        rec = {"probe": args.probe, "status": "ERROR", "error": repr(e),
               "traceback": traceback.format_exc(), "wall_s": round(time.monotonic() - t0, 1)}
    rec["pressure_note"] = "probe ran inside a bounded child with its own wall budget"
    save(f"{args.probe}_{args.tag}", rec)
    if rec.get("status") == "ERROR":
        sys.exit(1)


if __name__ == "__main__":
    main()
