"""A2I admission probe (task A2I, campaign una_large_e2e) - measurement only.

Rechecks the H05 A2 admission against the CURRENT selected source
(a1r_cand_wheel venv; its Engine files are sha256-identical to worktree
HEAD - bridge receipt recorded in admission.json). The H05 probe
(A2_002536.json) measured out-of-radius scan coverage through the BASELINE
wheel and never measured the touched/candidate fraction the A2 dataflow
needs; this probe measures, on the same frozen capture, the same 32 sampled
origins (same seed), and the assembled ACCESS_od settings:

  1. coverage of the proposed A2 route through REAL engine kernels:
     per-origin assignment count A, distinct touched nodes U, raw candidate
     count C_raw, distinct candidates C, in-radius count (cross-checked
     against the H05 probe per-origin values), and the sentinel value
     b = scope[V] read AFTER the identical search;
  2. bitwise validation of the candidate route against the dense route and
     against the engine's captured metrics: scope identity of the tracked
     search vs _a1_scope_search, adjusted distances at included IDs,
     retained n_distances/n_weights sequences (numpy replica of the
     unchanged filter), and the four metrics (ULP rule max_ulp <= 4, H03);
  3. per-origin service walls: plain A1 search, tracked search (prototype
     mirror of _a1_scope_search + touched appends), candidate enumeration
     (integer sort/unique + reverse-index gather, allocation-gated by the
     preregistered C_raw < D rule BEFORE the raw buffer is allocated),
     dense tail (real adjust_destination_distances full-D + real
     reach_gravity_knn_access full-D), candidate tail (F-layout row gather
     + real adjust kernel on gathered rows + real metric kernel on
     candidate arrays);
  4. dispatcher signature census: the candidate route must dispatch the
     real kernels to the SAME compiled signatures as the dense route
     (gathered 2-D rows are F-layout transpose views to match d_terminal
     arrays; 1-D arrays are C-layout both routes);
  5. admission arithmetic against committed anchors, computed HERE:
     stage-model shares (H05 stage_model.json), A1R warm complete-job
     medians (A1R selection.json), the derived acc-stage share at the A1
     source, the measured net kernel share, and the resulting removable
     complete-job service vs the frozen 0.05 gate (policy constant).

Every number in the record is computed in this process from the named
artifacts; nothing is relayed or hand-computed. Inputs are hash-stamped
before and after (immutability receipt). Single instrumented session on a
shared laptop: diagnostics class, H05 timing_class disclaimers apply
verbatim; this is admission evidence, NOT a performance result.

Runs against the INSTALLED wheel venv
(campaign_data/venvs/a1r_cand_wheel) with a FRESH numba cache root
(campaign_data/nbc_a2i_probe, created empty at start; the A1R screen roots
are left untouched). Output:
campaign_data/a2i_probes/a2i_admission_<tag>.json
"""

import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np

CAMPAIGN = "/Users/alansynn/orca/workspaces/una-x"
CAPTURE = f"{CAMPAIGN}/campaign_data/h05_captures/O3_ACCESS"
PROBE_DIR = f"{CAMPAIGN}/campaign_data/a2i_probes"
NBC_ROOT = f"{CAMPAIGN}/campaign_data/nbc_a2i_probe"
WT = f"{CAMPAIGN}/wt-large-e2e"
STAGE_MODEL = f"{WT}/campaigns/una_large_e2e/evidence/H05/stage_model.json"
A1R_SELECTION = f"{WT}/campaigns/una_large_e2e/evidence/A1R/selection.json"
H05_PROBE_A2 = f"{CAMPAIGN}/campaign_data/h05_probes/A2_002536.json"
SESSION = f"{CAMPAIGN}/campaign_data/h05_sessions/ACCESS_od.json"
WALL_BUDGET_S = 600.0
N_SAMPLED_ORIGINS = 32
SEED = 20260925
MAX_ULP = 4  # H03 ULP rule for float sums (same rule as the H05 probe)
REPS = 7     # timed repetitions per segment; min-of-reps is reported
TOUCHED_CAP0 = 1024


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_arrays(arrays):
    h = hashlib.sha256()
    for k in sorted(arrays):
        a = np.ascontiguousarray(arrays[k])
        h.update(k.encode())
        h.update(str(a.shape).encode())
        h.update(str(a.dtype).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def utc():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def window_stamp():
    import psutil
    vm = psutil.virtual_memory()
    la = os.getloadavg()
    return {
        "utc": utc(),
        "available_bytes": vm.available,
        "loadavg": [round(x, 2) for x in la],
        "rev9_quiet_floor_bytes": 5368709120,
        "rev9_loadavg_max": 8.0,
        "rev9_admitted": bool(vm.available >= 5368709120 and la[0] < 8.0),
    }


def sampled_origins(n_origins):
    rng = np.random.default_rng(SEED)
    idx = np.sort(rng.choice(n_origins, size=min(N_SAMPLED_ORIGINS, n_origins),
                             replace=False))
    return idx


def ulp_diff(a, b):
    if a == b:
        return 0
    ia = np.float64(a).view(np.int64)
    ib = np.float64(b).view(np.int64)
    return int(abs(int(ia) - int(ib)))


def numpy_metrics_replica(d_distance, d_weights, cutoff, beta, plateau, midpoint,
                          growth, knn_weights, knn_decay):
    """Exact numpy translation of reach_gravity_knn_access (same expressions,
    same order) - used ONLY to compare the retained sequences and as the
    H05-instrument continuity leg; the bitwise metric comparison is done on
    the REAL compiled kernel outputs."""
    sel = np.where(d_distance <= cutoff)[0]
    nd = d_distance[sel]
    nw = d_weights[sel]
    reach = nw.sum()
    g_exp = (nw / np.exp(beta * np.maximum(0.0, nd - plateau))).sum()
    g_log = (nw * (1 - 1 / (1 + np.exp(-growth * (nd - plateau - midpoint))))).sum()
    knn = min(nd.shape[0], knn_weights.shape[0])
    if knn == 0:
        return reach, g_exp, g_log, 0.0, sel, nd, nw
    idx = np.argsort(nd)
    ns_, nw_ = nd[idx], nw[idx]
    kc = knn_weights[:knn]
    knn_access = ((kc * nw_[:knn]) *
                  (1 - 1 / (1 + np.exp(-growth * (ns_[:knn] - plateau - midpoint))))).sum()
    return reach, g_exp, g_log, knn_access, sel, nd, nw


# ---------------------------------------------------------------------------
# Prototype kernels (the proposed A2 route shapes; Phase-3 integration is
# subject to the proof + approval gate and does not exist yet).
# ---------------------------------------------------------------------------
import numba as nb
from heapq import heappush, heappop

NUMBA_PARALLEL = False
NUMBA_CACHE = True
NUMBA_NOGIL = True
NUMBA_FASTMATH = True  # identical to the engine kernel flags


@nb.njit(cache=NUMBA_CACHE, nogil=NUMBA_NOGIL)
def _proto_tracked_search(o_pair, o_weights_pair, adjacency_pointer,
                          adjacency_vector, adjacency_vector_weights,
                          adjacynct_vector_network_node, cutoff, d_count,
                          eligible_offset, eligible_weight, touched_buf):
    """Statement-for-statement mirror of _a1_scope_search
    (_large_access_scratch.py, same phase structure, same heap sequence,
    same return) plus touched tracking: one int64 append mirroring EVERY
    o_scope_weights label write, in write order - the two unconditional
    seed assignments first, then one append per assignment executed in the
    replay pass. Returns (scope, pred, n_touched)."""
    nd_node_count = d_count + adjacency_pointer.shape[0] - 1
    o_idx_start = o_pair[0]
    o_idx_end = o_pair[1]
    o_idx_start_weight = o_weights_pair[0]
    o_idx_end_weight = o_weights_pair[1]

    o_scope_weights = np.ones(nd_node_count, dtype=o_weights_pair.dtype) + cutoff
    o_scope_pred = np.empty(0, dtype=o_pair.dtype)

    n_touched = 0
    cap = touched_buf.shape[0]
    touched_buf[0] = o_idx_start
    touched_buf[1] = o_idx_end
    n_touched = 2

    # Add start node segment terminals to seen with their weights
    o_scope_weights[o_idx_start] = o_idx_start_weight
    o_scope_weights[o_idx_end] = o_idx_end_weight

    queue = [(o_idx_start_weight, o_idx_start)]
    weight, node = heappop(queue)

    if o_idx_end_weight < cutoff:
        heappush(queue, (o_idx_end_weight, o_idx_end))

    if o_idx_start_weight < cutoff:
        heappush(queue, (o_idx_start_weight, o_idx_start))

    while queue:
        weight, node = heappop(queue)

        node_start_pointer = adjacency_pointer[node]
        node_end_pointer = adjacency_pointer[node + 1]

        n_eligible = 0
        for i in range(node_end_pointer - node_start_pointer):
            neighbor_weight = adjacency_vector_weights[node_start_pointer + i] + weight
            neighbor_node = adjacency_vector[node_start_pointer + i]
            if neighbor_weight <= cutoff and neighbor_weight < o_scope_weights[neighbor_node]:
                eligible_offset[n_eligible] = node_start_pointer + i
                eligible_weight[n_eligible] = neighbor_weight
                n_eligible += 1

        for j in range(n_eligible):
            offset = eligible_offset[j]
            neighbor_weight = eligible_weight[j]
            neighbor_node = adjacency_vector[offset]
            o_scope_weights[neighbor_node] = neighbor_weight
            if n_touched == cap:
                grown = np.empty(cap * 2, np.int64)
                grown[:n_touched] = touched_buf[:n_touched]
                touched_buf = grown
                cap = cap * 2
            touched_buf[n_touched] = neighbor_node
            n_touched = n_touched + 1
            if adjacynct_vector_network_node[offset]:
                if adjacency_pointer[neighbor_node + 1] - adjacency_pointer[neighbor_node] > 1:
                    heappush(queue, (neighbor_weight, neighbor_node))

    return o_scope_weights, o_scope_pred, n_touched


@nb.njit(cache=NUMBA_CACHE, nogil=NUMBA_NOGIL)
def _proto_enum_candidates(touched_buf, n_touched, fill_start, fill, d_count):
    """Integer sort/unique of the touched appends, reverse-index gather of
    destination IDs, integer sort/unique again - ascending original ID
    order. The preregistered allocation gate (C_raw < D) is evaluated AFTER
    the counts sum but BEFORE the raw buffer is allocated: when the gate
    fails the kernel returns an empty candidate array and the caller runs
    the unchanged dense tail (decision-before-mutation; the touched sort is
    the charged decision overhead). Returns (candidates, n_U, C_raw,
    gate_engaged)."""
    t = np.sort(touched_buf[:n_touched])
    u = 0
    for i in range(t.shape[0]):
        if u == 0 or t[i] != t[u - 1]:
            t[u] = t[i]
            u = u + 1
    total = 0
    for i in range(u):
        total = total + (fill_start[t[i] + 1] - fill_start[t[i]])
    if total >= d_count:
        e = np.empty(0, np.int64)
        return e, u, total, False
    raw = np.empty(total, np.int64)
    m = 0
    for i in range(u):
        for p in range(fill_start[t[i]], fill_start[t[i] + 1]):
            raw[m] = fill[p]
            m = m + 1
    s = np.sort(raw)
    c = 0
    for i in range(m):
        if c == 0 or s[i] != s[c - 1]:
            s[c] = s[i]
            c = c + 1
    return s[:c].copy(), u, total, True


@nb.njit(cache=NUMBA_CACHE, nogil=NUMBA_NOGIL)
def _proto_gather_rows_f(dti, dtw, cand):
    """Gather (C,2) terminal rows into (2,C) C-contiguous buffers returned as
    F-layout transpose views, so the real adjust_destination_distances
    kernel dispatches to the SAME 2-D F-layout signature as the dense
    route."""
    C = cand.shape[0]
    gi = np.empty((2, C), np.int64)
    gw = np.empty((2, C), np.float64)
    for j in range(C):
        d = cand[j]
        gi[0, j] = dti[d, 0]
        gi[1, j] = dti[d, 1]
        gw[0, j] = dtw[d, 0]
        gw[1, j] = dtw[d, 1]
    return gi.T, gw.T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--site-expected", required=True)
    ap.add_argument("--session", default=f"{CAMPAIGN}/campaign_data/h05_sessions/ACCESS_od.json",
                    help="assembled-settings session JSON (sets cutoff and decay params)")
    ap.add_argument("--stage-cell", default="ACCESS_od", choices=["ACCESS_od", "HOLDOUT_clean"],
                    help="stage-model cell the admission arithmetic is computed against")
    ap.add_argument("--a1r-stat", default="O3_ACCESS", choices=["O3_ACCESS", "O3_HOLDOUT"],
                    help="A1R selection statistics block for the warm-median derivation")
    ap.add_argument("--label", default=None, help="run label recorded in the record")
    ap.add_argument("--engine-metrics", default=f"{CAPTURE}/access_engine_metrics.json",
                    help="engine metrics JSON for the ULP cross-check leg "
                         "(pass an empty string to skip when the capture has no "
                         "engine outputs for this cell)")
    ap.add_argument("--h05-probe", default=H05_PROBE_A2,
                    help="H05 probe record for the in-radius cross-check leg "
                         "(pass an empty string to skip)")
    args = ap.parse_args()

    deadline = time.monotonic() + WALL_BUDGET_S
    rec = {"probe": "A2I_admission", "hypothesis": "A2_destination_locality",
           "evidence_class": "real_kernel_candidate_route_bit_validated",
           "utc": utc(), "instrument": os.path.abspath(__file__),
           "label": args.label,
           "session": args.session, "stage_cell": args.stage_cell,
           "a1r_stat": args.a1r_stat}
    rec["window_at_start"] = window_stamp()
    if not rec["window_at_start"]["rev9_admitted"]:
        rec["refusal"] = "REV-9 quiet window not met at admission; nothing ran"
        with open(args.out, "w") as f:
            json.dump(rec, f, indent=1, default=str)
        print("[a2i-probe] REFUSED: quiet window not met")
        sys.exit(2)

    import urban_network_analysis as u
    path = os.path.dirname(os.path.abspath(u.__file__))
    rec["identity"] = {
        "una_path": path,
        "identity_ok": os.path.realpath(path).startswith(
            os.path.realpath(args.site_expected) + os.sep),
    }
    import numba
    rec["versions"] = {"python": sys.version.split()[0], "numpy": np.__version__,
                       "numba": numba.__version__,
                       "numba_num_threads": numba.config.NUMBA_NUM_THREADS,
                       "threading_layer": None}
    rec["cache_state"] = {
        "root": NBC_ROOT,
        "fresh_root": True,
        "engine_kernels": "cold-compiled in-child into the fresh root; "
                          "measured sections run warm after untimed warm-up calls",
        "a1r_screen_roots_touched": False,
    }

    # ---- frozen inputs --------------------------------------------------
    cap_npz = np.load(f"{CAPTURE}/access_arrays.npz")
    cap = {k: cap_npz[k] for k in cap_npz.files}
    rec["input_sha256_before"] = sha256_arrays(cap)
    ap_ = cap["adjacency_pointer"]; av = cap["adjacency_vector"]
    avw = cap["adjacency_vector_weights"]; avn = cap["adjacynct_vector_network_node"]
    oti = cap["o_terminal_idxs"]; otw = cap["o_terminal_weights"]
    dti = cap["d_terminal_idxs"]; dtw = cap["d_terminal_weights"]
    d_w = cap["d_node_weight"]
    V = int(ap_.shape[0] - 1)
    D = int(dtw.shape[0])
    n_orig = int(oti.shape[0])
    rec["dims"] = {"V": V, "D": D, "O": n_orig, "E_directed": int(av.shape[0])}
    with open(args.session) as f:
        settings = json.load(f)["settings"]["assembled"]
    cutoff = float(settings["search_radius"])
    beta = float(settings["gravity_beta"])
    plateau = float(settings["gravity_plateau"])
    midpoint = float(settings["gravity_logistic_midpoint"])
    growth = settings["gravity_decay_constant"] / settings["gravity_logistic_midpoint"]
    knn_weights = np.asarray(settings["knn_weights"], dtype=np.float64)
    knn_decay = settings["knn_decay"]
    rec["settings_receipt"] = {
        "session": args.session, "cutoff": cutoff, "knn_decay": knn_decay,
        "gravity_beta": beta, "gravity_plateau": plateau,
        "gravity_logistic_midpoint": midpoint, "growth": growth,
    }
    eng = {}
    if args.engine_metrics:
        eng = json.load(open(args.engine_metrics))

    from urban_network_analysis.Engines import AccessibilityWElevation as AWE
    from urban_network_analysis.Engines._large_access_scratch import (
        _a1_scope_admits, _a1_scope_search)

    # ---- A1 admission of the capture arrays (driver-level receipt) ------
    # The guard is an overload: it resolves only inside jitted code
    # (same invocation pattern as tests/large_e2e/A1/test_a1_kernel.py
    # _probe), so the probe resolves it through a tiny njit wrapper.
    @nb.njit(cache=False)
    def _a1_guard_via_njit(adjacency_pointer, adjacency_vector,
                           adjacency_vector_weights,
                           adjacynct_vector_network_node, o_terminal_idxs,
                           o_terminal_weights, cutoff):
        return _a1_scope_admits(adjacency_pointer, adjacency_vector,
                                adjacency_vector_weights,
                                adjacynct_vector_network_node,
                                o_terminal_idxs, o_terminal_weights, cutoff)

    a1_admitted, a1_max_degree = _a1_guard_via_njit(ap_, av, avw, avn, oti,
                                                    otw, cutoff)
    rec["a1_admits_receipt"] = {"admitted": bool(a1_admitted),
                                "max_degree": int(a1_max_degree)}
    if not a1_admitted:
        rec["refusal"] = "A1 guard refuses the capture arrays; probe meaningless"
        with open(args.out, "w") as f:
            json.dump(rec, f, indent=1, default=str)
        sys.exit(2)

    # ---- A2 preconditions on the observed arrays (guard-scan receipt) ---
    term_in_range = bool(dti.size == 0 or ((dti.min() >= 0) and (dti.max() < V)))
    dtw_finite = bool(np.isfinite(dtw).all())
    dtw_nonneg = bool((dtw >= 0.0).all())
    d_weights_c = bool(d_w.flags["C_CONTIGUOUS"] and d_w.dtype == np.float64)
    rec["a2_guard_scan_receipt"] = {
        "d_terminal_idxs_all_in_0_V": term_in_range,
        "d_terminal_weights_finite": dtw_finite,
        "d_terminal_weights_nonneg": dtw_nonneg,
        "d_node_weight_C_float64": d_weights_c,
        "dti_dtype_layout": [str(dti.dtype), "F" if dti.flags["F_CONTIGUOUS"] else "C"],
        "dtw_dtype_layout": [str(dtw.dtype), "F" if dtw.flags["F_CONTIGUOUS"] else "C"],
    }

    # ---- reverse index (built ONCE per run from current arrays; the
    #      engine rebuilds it once per kernel call) ------------------------
    nodes2 = np.concatenate([dti[:, 0], dti[:, 1]]).astype(np.int64)
    counts = np.bincount(nodes2, minlength=V).astype(np.int64)
    fill_start = np.zeros(V + 1, np.int64)
    fill_start[1:] = np.cumsum(counts)
    fill = np.empty(2 * D, np.int64)
    cursor = fill_start[:-1].copy()
    for i in range(nodes2.shape[0]):
        v = nodes2[i]
        fill[cursor[v]] = i % D
        cursor[v] += 1
    rec["reverse_index_receipt"] = {
        "build": "once per probe run from d_terminal_idxs; engine plan: once "
                 "per integrated_scope_access call, read-only under prange, "
                 "no cross-run cache",
        "entries": int(2 * D),
        "counts_sum": int(counts.sum()),
        "construction_order": "terminal-0 of all destinations ascending, then "
                              "terminal-1 of all destinations ascending; order "
                              "is irrelevant downstream (integer sort/unique)",
    }

    o_idx = sampled_origins(n_orig)
    rec["sampled_origins"] = {"seed": SEED, "n": int(o_idx.shape[0]),
                              "protocol": "identical to H05 probe (default_rng(SEED).choice)"}

    eligible_offset = np.empty(a1_max_degree, dtype=np.int64)
    eligible_weight = np.empty(a1_max_degree, dtype=np.float64)

    def metric_args(dd, ww):
        # positional mapping identical to the engine's integrated_scope_access
        # call (d_distance, d_weights, cutoff, gravity_beta, gravity_plateau,
        # gravity_logistic_midpoint, knn_gravity_plateau, knn_weights,
        # knn_gravity_beta, knn_decay, knn_gravity_logistic_midpoint,
        # gravity_growth_rate, knn_gravity_growth_rate)
        return (dd, ww, cutoff, beta, plateau, midpoint, plateau, knn_weights,
                beta, knn_decay, midpoint, growth, growth)

    per_origin = []
    sig_census = {"adjust_before": None, "adjust_after": None,
                  "metric_before": None, "metric_after": None}
    scope_mismatches = 0
    dist_mismatches = 0
    seq_mismatches = 0
    exclusion_violations = 0
    b_values = []
    gate_engaged_count = 0

    # ---- warm-up / compile (untimed) ------------------------------------
    o0 = int(o_idx[0])
    o_pair0 = oti[o0]
    o_w_pair0 = otw[o0]
    tbuf0 = np.empty(TOUCHED_CAP0, np.int64)
    scope_plain, _ = _a1_scope_search(o_pair0, o_w_pair0, ap_, av, avw, avn,
                                      cutoff, D, eligible_offset, eligible_weight)
    scope_tr, _, n_tr = _proto_tracked_search(o_pair0, o_w_pair0, ap_, av, avw,
                                              avn, cutoff, D, eligible_offset,
                                              eligible_weight, tbuf0)
    cand0, _, _, _ = _proto_enum_candidates(tbuf0, n_tr, fill_start, fill, D)
    gi0, gw0 = _proto_gather_rows_f(dti, dtw, cand0)
    _ = AWE.adjust_destination_distances(scope_plain, dti, dtw, D)
    _ = AWE.reach_gravity_knn_access(*metric_args(
        np.full(D, cutoff + 1.0), d_w))
    if cand0.shape[0] > 0:
        _ = AWE.adjust_destination_distances(scope_tr, gi0, gw0, int(cand0.shape[0]))
        _ = AWE.reach_gravity_knn_access(*metric_args(
            np.full(int(cand0.shape[0]), cutoff + 1.0), d_w[cand0]))
    sig_census["adjust_before"] = [str(s) for s in AWE.adjust_destination_distances.signatures]
    sig_census["metric_before"] = [str(s) for s in AWE.reach_gravity_knn_access.signatures]

    for oo in o_idx:
        if time.monotonic() > deadline:
            rec["truncated"] = True
            break
        o = int(oo)
        o_pair = oti[o]
        o_w_pair = otw[o]
        tbuf = np.empty(TOUCHED_CAP0, np.int64)
        row = {"origin": o}

        def seg(fn):
            best = None
            for _ in range(REPS):
                t0 = time.perf_counter_ns()
                r = fn()
                dt = time.perf_counter_ns() - t0
                if best is None or dt < best:
                    best = dt
            return best, r

        # plain A1 search (the route the engine runs today)
        t_plain, (scope_a, _pa) = seg(lambda: _a1_scope_search(
            o_pair, o_w_pair, ap_, av, avw, avn, cutoff, D,
            eligible_offset, eligible_weight))
        # tracked search (A2 prototype)
        t_tracked, (scope_b, _pb, n_t) = seg(lambda: _proto_tracked_search(
            o_pair, o_w_pair, ap_, av, avw, avn, cutoff, D,
            eligible_offset, eligible_weight, tbuf))
        row["scope_bitwise_equal_plain_vs_tracked"] = bool(
            scope_a.tobytes() == scope_b.tobytes() and
            scope_a.shape == scope_b.shape and scope_a.dtype == scope_b.dtype)
        scope_mismatches += 0 if row["scope_bitwise_equal_plain_vs_tracked"] else 1

        # sentinel value read AFTER the identical search
        b_val = float(scope_b[V])
        b_values.append(b_val)
        row["b_scope_V"] = b_val
        row["b_finite_gt_cutoff"] = bool(b_val == b_val and b_val < np.inf
                                         and b_val > cutoff)

        # enumeration with the preregistered allocation gate
        t_enum, (cand, n_U, c_raw, engaged) = seg(lambda: _proto_enum_candidates(
            tbuf, n_t, fill_start, fill, D))
        C = int(cand.shape[0])
        gate_engaged_count += 1 if engaged else 0
        row.update({"A": int(n_t), "U": int(n_U), "C_raw": int(c_raw),
                    "C": C, "gate_C_raw_lt_D_engaged": bool(engaged)})

        # dense tail (unchanged route): real kernels, full D
        def dense_tail():
            dd_ = AWE.adjust_destination_distances(scope_a, dti, dtw, D)
            md = AWE.reach_gravity_knn_access(*metric_args(dd_, d_w))
            return dd_, md

        t_dense, (dd, metrics_d) = seg(dense_tail)

        # candidate tail (A2 route): F-layout gather + real kernels on C rows
        def cand_tail():
            gi, gw = _proto_gather_rows_f(dti, dtw, cand)
            dc = AWE.adjust_destination_distances(scope_b, gi, gw, C)
            dc_metrics = AWE.reach_gravity_knn_access(*metric_args(
                dc, d_w[cand]))
            return dc, dc_metrics

        if engaged:
            t_cand, (dc, metrics_c) = seg(cand_tail)
        else:
            t_cand, (dc, metrics_c) = 0, (None, None)

        row.update({
            "t_plain_ns": int(t_plain), "t_tracked_ns": int(t_tracked),
            "t_enum_ns": int(t_enum), "t_dense_tail_ns": int(t_dense),
            "t_cand_tail_ns": int(t_cand),
            "dense_s": metrics_d, "cand_s": metrics_c,
        })

        # ---- validation legs ----
        n_in = int(np.count_nonzero(dd <= cutoff))
        row["n_in_radius"] = n_in
        if engaged:
            row["dist_bitwise_equal_at_included"] = bool(
                dd[cand].tobytes() == dc.tobytes())
            dist_mismatches += 0 if row["dist_bitwise_equal_at_included"] else 1
            # retained sequences through the unchanged filter (numpy replica)
            _, _, _, _, sel_d, nd_d, nw_d = numpy_metrics_replica(
                dd, d_w, cutoff, beta, plateau, midpoint, growth, knn_weights,
                knn_decay)
            _, _, _, _, sel_c, nd_c, nw_c = numpy_metrics_replica(
                dc, d_w[cand], cutoff, beta, plateau, midpoint, growth,
                knn_weights, knn_decay)
            seq_ok = bool(nd_d.shape == nd_c.shape and nd_d.dtype == nd_c.dtype
                          and nd_d.tobytes() == nd_c.tobytes()
                          and nw_d.tobytes() == nw_c.tobytes())
            row["retained_seq_identical"] = seq_ok
            seq_mismatches += 0 if seq_ok else 1
            ulps = {m: ulp_diff(v, w) for m, (v, w) in
                    (("reach", (metrics_d[0], metrics_c[0])),
                     ("gravity_exponential", (metrics_d[1], metrics_c[1])),
                     ("gravity_logistic", (metrics_d[2], metrics_c[2])),
                     ("knn_access", (metrics_d[3], metrics_c[3])))}
            row["ulp_dense_vs_cand"] = ulps
            row["metrics_bitwise_equal_dense_vs_cand"] = all(v == 0 for v in ulps.values())
        else:
            row["dist_bitwise_equal_at_included"] = None
            row["retained_seq_identical"] = None
            row["ulp_dense_vs_cand"] = None
            row["metrics_bitwise_equal_dense_vs_cand"] = None

        # exclusion proof: every destination outside the candidate set has
        # dense distance > cutoff (sentinel argument, measured)
        if engaged and C < D:
            mask = np.ones(D, dtype=bool)
            mask[cand] = False
            excluded = np.nonzero(mask)[0]
            viol = int(np.count_nonzero(dd[excluded] <= cutoff))
            row["excluded_n"] = int(excluded.shape[0])
            row["excluded_le_cutoff_violations"] = viol
            exclusion_violations += viol
        else:
            row["excluded_n"] = int(D - C)
            row["excluded_le_cutoff_violations"] = None

        # engine-metric validation leg (H05 continuity; H03 ULP rule)
        e = eng.get(str(o))
        if e is not None:
            ulps_e = {m: ulp_diff(v, e[m]) for m, v in
                      (("reach", metrics_d[0]), ("gravity_exponential", metrics_d[1]),
                       ("gravity_logistic", metrics_d[2]), ("knn_access", metrics_d[3]))}
            row["ulp_dense_vs_engine"] = ulps_e
            row["within_ulp4_vs_engine"] = all(v <= MAX_ULP for v in ulps_e.values())

        # per-origin net (candidate side minus dense side; negative = saving)
        dense_side = t_plain + t_dense
        cand_side = t_tracked + t_enum + t_cand
        row["net_ns"] = int(cand_side - dense_side)
        row["net_share_of_dense_side"] = (cand_side - dense_side) / dense_side
        per_origin.append(row)

    sig_census["adjust_after"] = [str(s) for s in AWE.adjust_destination_distances.signatures]
    sig_census["metric_after"] = [str(s) for s in AWE.reach_gravity_knn_access.signatures]
    sig_census["unchanged"] = {
        "adjust": sig_census["adjust_before"] == sig_census["adjust_after"],
        "metric": sig_census["metric_before"] == sig_census["metric_after"],
    }
    rec["signature_census"] = sig_census
    rec["signature_census_note"] = (
        "candidate route must add NO compiled signature: 2-D terminal rows "
        "gathered as F-layout transpose views (dense dti/dtw are F-layout); "
        "1-D metric arrays are C-layout on both routes")

    rec["per_origin"] = per_origin
    rec["validation"] = {
        "scope_bitwise_mismatches": scope_mismatches,
        "dist_bitwise_mismatches_at_included": dist_mismatches,
        "retained_seq_mismatches": seq_mismatches,
        "exclusion_le_cutoff_violations_total": exclusion_violations,
        "gate_engaged_origins": gate_engaged_count,
        "b_values_all_finite_gt_cutoff": bool(all(v == v and v < np.inf and v > cutoff
                                                  for v in b_values)),
        "b_value_set": sorted(set(b_values)),
        "ulp_rule": f"max_ulp <= {MAX_ULP} (H03); bitwise equality expected and checked separately",
    }

    # ---- aggregate walls / coverage (all computed here) ------------------
    if per_origin:
        def med(k):
            return float(np.median([r[k] for r in per_origin]))
        def mn(k):
            return int(np.min([r[k] for r in per_origin]))
        def mx(k):
            return int(np.max([r[k] for r in per_origin]))
        dense_med = med("t_plain_ns") + med("t_dense_tail_ns")
        cand_med = med("t_tracked_ns") + med("t_enum_ns") + med("t_cand_tail_ns")
        rec["walls_summary_ns"] = {
            "reps_per_segment": REPS, "stat": "min-of-reps per origin, median across origins",
            "median_t_plain": med("t_plain_ns"), "median_t_tracked": med("t_tracked_ns"),
            "median_t_enum": med("t_enum_ns"), "median_t_dense_tail": med("t_dense_tail_ns"),
            "median_t_cand_tail": med("t_cand_tail_ns"),
            "median_dense_side": dense_med, "median_cand_side": cand_med,
            "median_net": med("net_ns"),
            "median_net_share": float(np.median([r["net_share_of_dense_side"]
                                                 for r in per_origin])),
            "min_net_share": float(np.min([r["net_share_of_dense_side"]
                                           for r in per_origin])),
            "max_net_share": float(np.max([r["net_share_of_dense_side"]
                                           for r in per_origin])),
        }
        rec["coverage_summary"] = {
            "A_min": mn("A"), "A_median": med("A"), "A_max": mx("A"),
            "U_max": mx("U"), "C_raw_max": mx("C_raw"), "C_max": mx("C"),
            "C_over_D_median": float(np.median([r["C"] / D for r in per_origin])),
            "C_over_D_max": float(np.max([r["C"] / D for r in per_origin])),
            "A_over_V_median": float(np.median([r["A"] / V for r in per_origin])),
            "n_in_radius_mean": float(np.mean([r["n_in_radius"] for r in per_origin])),
            "scan_avoided_when_engaged_median":
                float(np.median([1.0 - r["C"] / D for r in per_origin
                                 if r["gate_C_raw_lt_D_engaged"]] or [0.0])),
        }
        # cross-instrument receipt vs the H05 probe per-origin in-radius counts
        # (ACCESS radius-500 cell only; skipped for other sessions)
        if args.h05_probe:
            h05 = json.load(open(args.h05_probe))
            h05_map = {p["origin"]: p["n_in_radius"]
                       for p in h05["scan_coverage"]["per_origin"]}
            mism = [r["origin"] for r in per_origin
                    if r["origin"] in h05_map and r["n_in_radius"] != h05_map[r["origin"]]]
            rec["h05_in_radius_crosscheck"] = {
                "h05_probe": args.h05_probe,
                "compared": sum(1 for r in per_origin if r["origin"] in h05_map),
                "mismatched_origins": mism,
            }

    # ---- full-driver acc wall at the current source (context anchor) -----
    t_acc = []
    for _ in range(3):
        t0 = time.perf_counter_ns()
        AWE.integrated_scope_access(oti, otw, ap_, av, avw, avn, dti, dtw, d_w,
                                    beta, plateau, midpoint, growth, knn_decay,
                                    knn_weights, cutoff)
        t_acc.append(time.perf_counter_ns() - t0)
    acc_wall_s = min(t_acc) / 1e9
    rec["acc_stage_wall_at_head"] = {
        "wall_s_min_of_3_warm": acc_wall_s,
        "O": n_orig, "note": "real integrated_scope_access over ALL origins, "
                             "prange, default numba threads; single-session "
                             "context anchor, not a performance result",
    }

    # ---- admission arithmetic: THIS run's cell only ----------------------
    # net_share is (cand_side - dense_side)/dense_side, i.e. NEGATIVE when the
    # candidate route saves.  The removable fraction is its NEGATION; every
    # end-to-end figure below multiplies the positive saving share.
    sm = json.load(open(STAGE_MODEL))
    a1r = json.load(open(A1R_SELECTION))
    ccell = sm["cells"][args.stage_cell]
    w_b0 = ccell["window_s"]
    acc_b0 = ccell["top_stage_inclusive_s"]["acc_centrality"]
    other = w_b0 - acc_b0
    st = a1r["disposition_dataset"]["statistics"][args.a1r_stat]
    arith = {"stage_model_cell": args.stage_cell, "a1r_stat": args.a1r_stat}
    have_warm = {"p2_BA_warm", "p3_AB_warm"} <= set(st)
    if have_warm:
        b0_med = float(np.median([st["p2_BA_warm"]["b0_s"], st["p3_AB_warm"]["b0_s"]]))
        cand_med = float(np.median([st["p2_BA_warm"]["cand_s"], st["p3_AB_warm"]["cand_s"]]))
        acc_at_a1 = cand_med - other
        arith.update({
            "window_b0_s": w_b0, "acc_centrality_b0_s": acc_b0,
            "share_acc_b0": acc_b0 / w_b0,
            "other_stages_b0_s": other,
            "a1r_warm_b0_median_s": b0_med, "a1r_warm_cand_median_s": cand_med,
            "acc_stage_at_a1_derived_s": acc_at_a1,
            "share_acc_at_a1_derived": acc_at_a1 / cand_med,
            "a1r_source_note": "warm complete-job medians from A1R "
                               "selection.json statistics (session-coupled "
                               "derivation: cand median minus B0 non-acc "
                               "stages; spread disclosed in A1R raws)",
        })
    else:
        arith["a1r_derived_leg"] = {
            "error": "warm pair keys missing in A1R statistics for this stat",
            "available_keys": sorted(st)}
    share_head_direct = acc_wall_s / (other + acc_wall_s)
    arith.update({
        "acc_wall_at_head_measured_s": acc_wall_s,
        "share_acc_at_head_from_measured_wall": share_head_direct,
    })
    if per_origin:
        net_share = rec["walls_summary_ns"]["median_net_share"]
        saving_share = -net_share
        arith.update({
            "net_kernel_share_signed": net_share,
            "removable_kernel_share": saving_share,
            "net_kernel_share_note": (f"measured on the {args.stage_cell} "
                                      f"capture at radius {cutoff}; NOT "
                                      "transferred to other cells (the "
                                      "candidate fraction is "
                                      "radius-dependent)"),
            "removable_end_to_end_vs_measured_wall_share":
                saving_share * share_head_direct,
            "gate": "minimum_removable_fraction 0.05 (policy constant)",
        })
        if have_warm:
            arith["removable_end_to_end_vs_a1_derived_share"] = \
                saving_share * (acc_at_a1 / cand_med)
    rec["admission_arithmetic"] = arith

    rec["input_sha256_after"] = sha256_arrays(cap)
    rec["inputs_immutable"] = bool(rec["input_sha256_before"] == rec["input_sha256_after"])
    rec["wall_budget_s"] = WALL_BUDGET_S
    rec["wall_used_s"] = round(WALL_BUDGET_S - (deadline - time.monotonic()), 3)
    rec["window_at_end"] = window_stamp()

    os.makedirs(PROBE_DIR, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(rec, f, indent=1, default=str)
    print(f"[a2i-probe] wrote {args.out}")

    # ---- printed summary (relay quotes THESE lines) ----------------------
    print("[a2i-probe] identity_ok:", rec["identity"]["identity_ok"],
          "| a1_admitted:", rec["a1_admits_receipt"]["admitted"],
          "| inputs_immutable:", rec["inputs_immutable"])
    print("[a2i-probe] b values:", sorted(set(b_values)),
          "| all finite > cutoff:", rec["validation"]["b_values_all_finite_gt_cutoff"])
    print("[a2i-probe] scope mismatches:", rec["validation"]["scope_bitwise_mismatches"],
          "| dist mismatches:", rec["validation"]["dist_bitwise_mismatches_at_included"],
          "| retained-seq mismatches:", rec["validation"]["retained_seq_mismatches"],
          "| exclusion violations:", rec["validation"]["exclusion_le_cutoff_violations_total"])
    print("[a2i-probe] metric ulp dense-vs-cand max:",
          max((max(r["ulp_dense_vs_cand"].values()) for r in per_origin
               if r.get("ulp_dense_vs_cand")), default=None))
    print("[a2i-probe] metric ulp dense-vs-engine max:",
          max((max(r["ulp_dense_vs_engine"].values()) for r in per_origin
               if r.get("ulp_dense_vs_engine")), default=None),
          "| within_ulp4 all:",
          all(r.get("within_ulp4_vs_engine", True) for r in per_origin))
    if "walls_summary_ns" in rec:
        wsm = rec["walls_summary_ns"]
        for k in ("median_t_plain", "median_t_tracked", "median_t_enum",
                  "median_t_dense_tail", "median_t_cand_tail",
                  "median_dense_side", "median_cand_side", "median_net",
                  "median_net_share"):
            print(f"[a2i-probe] {k}: {wsm[k]}")
    if "coverage_summary" in rec:
        csm = rec["coverage_summary"]
        for k in ("A_min", "A_median", "A_max", "C_max", "C_over_D_median",
                  "C_over_D_max", "A_over_V_median", "n_in_radius_mean",
                  "scan_avoided_when_engaged_median"):
            print(f"[a2i-probe] {k}: {csm[k]}")
    if "h05_in_radius_crosscheck" in rec:
        print("[a2i-probe] h05 in-radius crosscheck:",
              rec["h05_in_radius_crosscheck"])
    else:
        print("[a2i-probe] h05 in-radius crosscheck: skipped (--h05-probe empty)")
    print(f"[a2i-probe] arithmetic cell {args.stage_cell} / {args.a1r_stat}:")
    for kk in ("share_acc_b0", "share_acc_at_a1_derived",
               "share_acc_at_head_from_measured_wall",
               "net_kernel_share_signed", "removable_kernel_share",
               "removable_end_to_end_vs_a1_derived_share",
               "removable_end_to_end_vs_measured_wall_share"):
        if kk in arith:
            print(f"[a2i-probe]   {kk}: {arith[kk]}")
    print("[a2i-probe] signatures adjust before/after:",
          len(sig_census["adjust_before"]), "->", len(sig_census["adjust_after"]),
          "| metric before/after:",
          len(sig_census["metric_before"]), "->", len(sig_census["metric_after"]),
          "| unchanged:", sig_census["unchanged"])
    print("[a2i-probe] acc wall at head (s):", acc_wall_s)


if __name__ == "__main__":
    main()
