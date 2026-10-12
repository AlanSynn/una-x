"""S12 independent hostile review probe (reviewer-authored, not implementer's).

Reference discipline: the ONLY references are the compiled baseline
`integrated_scope_access` (run TWICE per case, jitter-gated) and, for tiny
schedule-law cases, the frozen kernels oracle `scope_search_schedule`
(domain="a3").  The candidate module is never used as its own reference.
"""
import dataclasses
import hashlib
import json
import sys
import threading

import numpy as np

SRC = "/storage/scratch1/1/dsynn6/una-x/.claude/worktrees/s12-review-r1/src"
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from urban_network_analysis.Engines.AccessibilityWElevation import (
    integrated_scope_access)
from urban_network_analysis.Engines import _sparse_access_scratch as SP

FIELDS = ("reach", "gravity_exponential", "gravity_logistic", "knn_access")
RESULTS = []


def bits(a):
    return np.ascontiguousarray(a).view(np.uint64)


def case_hash(args):
    h = hashlib.sha256()
    for a in args:
        if isinstance(a, np.ndarray):
            h.update(str(a.dtype).encode())
            h.update(str(a.shape).encode())
            h.update(a.tobytes())
        else:
            h.update(repr(a).encode())
        h.update(b"|")
    return h.hexdigest()


def run_baseline_twice(args):
    b1 = integrated_scope_access(*args)
    b2 = integrated_scope_access(*args)
    jitter = all(bits(x).tobytes() == bits(y).tobytes()
                 for x, y in zip(b1, b2))
    return b1, jitter


def parity(res, ref):
    out = {}
    for f, b in zip(FIELDS, ref):
        r = getattr(res, f)
        ok = (bits(r).tobytes() == bits(b).tobytes()
              and np.asarray(r).dtype == np.asarray(b).dtype
              and np.asarray(r).shape == np.asarray(b).shape)
        out[f] = bool(ok)
    return out


def record(cid, attack, law, *, route=None, refusal=None, counters=None,
           par=None, jitter=None, status="PASS", extra=None, dump=None,
           args=None):
    row = dict(id=cid, attack=attack, expected_law=law, status=status,
               route=route, refusal=refusal,
               counters=dataclasses.asdict(counters) if counters is not None
               else None,
               parity=par, baseline_jitter_free=jitter, extra=extra or {})
    if dump is not None:
        row["dump"] = dump
    if args is not None:
        row["input_sha256"] = case_hash(args)
    print(json.dumps(row, default=str))
    RESULTS.append(row)


def fail_bytes(res, ref):
    d = {}
    for f, b in zip(FIELDS, ref):
        r = getattr(res, f)
        d[f] = dict(candidate=[hex(int(x)) for x in bits(r).ravel()[:8]],
                    baseline=[hex(int(x)) for x in bits(b).ravel()[:8]])
    return d


def check(cid, attack, law, args, *, expect_route="sparse_epoch_a3",
          expect_refusal="", epoch_rollover=None):
    kw = {} if epoch_rollover is None else dict(epoch_rollover=epoch_rollover)
    try:
        b1, jitter = run_baseline_twice(args)
        res = SP.sparse_scope_access(*args, **kw)
    except Exception as e:
        record(cid, attack, law, status="ERROR",
               extra=dict(exception=type(e).__name__, msg=str(e)[:300]),
               args=args)
        return None
    par = parity(res, b1)
    ok = (all(par.values()) and jitter
          and res.route == expect_route and res.refusal == expect_refusal)
    record(cid, attack, law, route=res.route, refusal=res.refusal,
           counters=res.counters, par=par, jitter=jitter,
           status="PASS" if ok else "FAIL",
           dump=None if ok else fail_bytes(res, b1), args=args)
    return res


def ring_plus(n, extra, rng, wlo=0.1, whi=0.9):
    src = np.concatenate([np.arange(n, dtype=np.int64),
                          rng.integers(0, n, extra).astype(np.int64)])
    dst = np.concatenate([((np.arange(n) + 1) % n).astype(np.int64),
                          rng.integers(0, n, extra).astype(np.int64)])
    order = np.argsort(src, kind="stable")
    src, dst = src[order], dst[order]
    ptr = np.zeros(n + 1, dtype=np.int64)
    ptr[1:] = np.cumsum(np.bincount(src, minlength=n))
    vec = np.ascontiguousarray(dst)
    wts = np.ascontiguousarray(rng.uniform(wlo, whi, vec.shape[0]))
    flag = np.ones(vec.shape[0], dtype=bool)
    return (np.ascontiguousarray(ptr), vec, wts, flag)


MET = (1.5, 2.0, 3.0, 0.8, "exponential",
       np.array([1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]))


def dargs(o, ow, ptr, vec, wts, flag, d, dw, dwt, cutoff, met=MET):
    return (o, ow, ptr, vec, wts, flag, d, dw, dwt) + met + (cutoff,)


def tiny(n, arcs, weights, flags):
    rows = {v: [] for v in range(n)}
    for k, (a, b) in enumerate(arcs):
        rows[a].append(k)
    ptr = [0]
    vec = []
    ws = []
    fl = []
    for v in range(n):
        for k in rows[v]:
            vec.append(arcs[k][1])
            ws.append(weights[k])
            fl.append(flags[k])
        ptr.append(len(vec))
    return (np.ascontiguousarray(np.array(ptr, dtype=np.int64)),
            np.ascontiguousarray(np.array(vec, dtype=np.int64)),
            np.ascontiguousarray(np.array(ws, dtype=np.float64)),
            np.ascontiguousarray(np.array(fl, dtype=bool)))


n = 60
ptr, vec, wts, flag = ring_plus(n, 60, np.random.default_rng(1))
o = np.tile(np.array([[3, 4]], dtype=np.int64), (48, 1))
ow = np.tile(np.array([[0.05, 0.1]]), (48, 1))
d = np.stack([np.random.default_rng(2).integers(0, n, 40),
              np.random.default_rng(3).integers(0, n, 40)],
             axis=1).astype(np.int64)
dw = np.random.default_rng(4).uniform(0.0, 0.4, (40, 2))
dwt = np.random.default_rng(5).uniform(0.25, 1.0, 40)
res_h1 = None

o9 = np.arange(9, dtype=np.int64)
o9 = np.stack([o9, ((o9 + 5) % n)], axis=1).astype(np.int64)
ow9 = np.full((9, 2), 0.05)
for rollover in (1, 2, 3, 4):
    check(f"H2/H3-rollover={rollover}",
          f"9 diverse origins, epoch_rollover={rollover} (dirty arenas "
          f"across rollover; lanes=4)",
          "stamps+marks cleared BEFORE rollover; bitwise vs baseline "
          "regardless of rollover width",
          dargs(o9, ow9, ptr, vec, wts, flag, d, dw, dwt, 1.2),
          epoch_rollover=rollover)
check("H2-rollover=huge", "epoch_rollover=2**62 default path",
      "no clearing; bitwise", dargs(o9, ow9, ptr, vec, wts, flag, d, dw,
                                    dwt, 1.2), epoch_rollover=1 << 62)

d_at = np.array([[3, 4], [4, 3], [3, 3], [0, 59], [59, 0]], dtype=np.int64)
dw_at = np.array([[0.0, 0.0], [0.3, 0.1], [0.5, 0.5], [0.2, 0.2],
                  [0.0, 0.4]])
dwt_at = np.array([1.0, 0.5, 0.25, 0.75, 1.5])
o4 = np.tile(np.array([[3, 4]], dtype=np.int64), (6, 1))
ow4 = np.tile(np.array([[0.05, 0.1]]), (6, 1))
check("H4", "destination t0/t1 equal origin seed nodes (t0==t1 and "
      "crossed pairs)",
      "seed labels project directly into dist; ascending-id order; bitwise",
      dargs(o4, ow4, ptr, vec, wts, flag, d_at, dw_at, dwt_at, 1.2))

ptr_i = np.array([0, 2, 4, 6, 8, 8, 10, 12, 14, 16, 18], dtype=np.int64)
vec_i = np.array([1, 2, 0, 2, 0, 1, 0, 1, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9],
                 dtype=np.int64)
wts_i = np.full(18, 0.25)
flag_i = np.ones(18, dtype=bool)
o_mixed = np.array([[0, 1], [5, 6], [0, 1], [5, 6], [2, 2], [5, 6]],
                   dtype=np.int64)
ow_mixed = np.full((6, 2), 0.0)
d_mixed = np.array([[5, 6], [6, 5], [0, 1], [1, 0], [5, 5], [9, 9]],
                   dtype=np.int64)
dw_mixed = np.array([[0.0, 0.0], [0.1, 0.2], [0.0, 0.0], [0.3, 0.0],
                     [0.5, 0.5], [0.0, 0.25]])
dwt_mixed = np.array([1.0, 0.5, 0.25, 0.75, 1.5, 2.0])
check("H5a", "cohort mixing empty-candidate origins and kept==0 origins",
      "empty prefix -> exact 0.0 rows (fold called for EVERY origin); "
      "kept==0 -> 0.0 rows; bitwise vs baseline",
      dargs(o_mixed, ow_mixed, ptr_i, vec_i, wts_i, flag_i, d_mixed,
            dw_mixed, dwt_mixed, 0.6))
check("H5b", "tiny cutoff: origins with only seed-touched candidates and "
      "origins with zero candidates",
      "all-zero rows exact 0.0 bitwise", dargs(o_mixed, ow_mixed, ptr_i,
                                               vec_i, wts_i, flag_i,
                                               d_mixed, dw_mixed, dwt_mixed,
                                               0.05))

ptr_t, vec_t, wts_t, flag_t = tiny(3, [(0, 1), (1, 2)],
                                   [0.5, 0.5], [True, True])
d_b = np.array([[1, 2], [2, 0], [1, 1]], dtype=np.int64)
dw_b = np.array([[0.0, 0.0], [0.25, 0.0], [0.5, 0.5]])
dwt_b = np.array([1.0, 0.5, 0.75])
for cut, tag in ((1.0, "seed==cutoff"), (0.5, "cand==cutoff"),
                 (np.nextafter(1.0, 2.0), "nextafter-up"),
                 (np.nextafter(0.5, 0.0), "nextafter-down-at-arc"),
                 (-0.0, "cutoff=-0.0"), (0.0, "cutoff=0.0")):
    check(f"H6[{tag}]", f"cutoff={cut!r} ({tag})",
          "strict < push gate; closed <= membership; bitwise",
          dargs(np.array([[0, 2]], dtype=np.int64),
                np.array([[1.0, 4.0]]), ptr_t, vec_t, wts_t, flag_t,
                d_b, dw_b, dwt_b, cut))

import numba
lanes = int(numba.config.NUMBA_NUM_THREADS)
V_under = (268435456 // 8 // lanes - 8) // 3
V_over = (268435456 // 8 // lanes) // 3 + 1
for tag, V in (("just-under", V_under), ("just-over", V_over)):
    ptr_c = np.zeros(V + 1, dtype=np.int64)
    vec_c = np.zeros(0, dtype=np.int64)
    wts_c = np.zeros(0, dtype=np.float64)
    flag_c = np.zeros(0, dtype=bool)
    d_c = np.array([[0, 1], [1, 0]], dtype=np.int64)
    dw_c = np.array([[0.25, 0.5], [0.5, 0.25]])
    dwt_c = np.array([1.0, 0.5])
    exp_route = ("sparse_epoch_a3" if tag == "just-under"
                 else "compact_fallback")
    exp_ref = "" if tag == "just-under" else "capacity"
    check(f"H7[{tag}]", f"V={V} ({tag} the 256MiB arena cap; D=2, md=0)",
          "admitted iff 8*lanes*(3V+4D+2md) <= cap; refusal 'capacity' "
          "BEFORE allocation; fallback bitwise",
          dargs(np.array([[0, 1]], dtype=np.int64),
                np.array([[0.5, 0.25]]), ptr_c, vec_c, wts_c, flag_c,
                d_c, dw_c, dwt_c, 1.0),
          expect_route=exp_route, expect_refusal=exp_ref)

case8 = dargs(o4, ow4, ptr, vec, wts, flag, d_at, dw_at, dwt_at, 1.2)
names8 = ["o_terminal_idxs", "o_terminal_weights", "adjacency_pointer",
          "adjacency_vector", "adjacency_vector_weights",
          "adjacynct_vector_network_node", "d_terminal_idxs",
          "d_terminal_weights", "d_weights", "gravity_beta",
          "gravity_plateau", "gravity_logistic_midpoint",
          "gravity_growth_rate", "knn_decay", "knn_weights", "cutoff"]
try:
    b8, j8 = run_baseline_twice(case8)
    res_kw = SP.sparse_scope_access(**dict(zip(names8, case8)))
    res_pos = SP.sparse_scope_access(*case8)
    res_17 = SP.sparse_scope_access(*case8, 4611686018427387904)
    par_kw = all(parity(res_kw, b8).values())
    par_pos = all(parity(res_pos, b8).values())
    par_17 = all(parity(res_17, b8).values())
    bad8 = case8[:15] + (np.float32(1.2),)
    res_fbk = SP.sparse_scope_access(**dict(zip(names8, bad8)))
    ok = par_kw and par_pos and par_17 and res_fbk.route == "compact_fallback"
    record("H8", "16-param kwarg binding, positional binding, 17th "
          "epoch_rollover positional, kwarg-bound REFUSAL path",
          "identical bitwise outputs on all four bindings; refusal path "
          "binds the fallback correctly (a1_guard)",
          route=res_kw.route, refusal=res_fbk.refusal,
          counters=res_kw.counters,
          par={"kw": par_kw, "pos": par_pos, "pos17": par_17},
          jitter=j8, status="PASS" if ok and res_fbk.refusal == "a1_guard"
          else "FAIL", args=case8)
except Exception as e:
    record("H8", "kwarg/positional binding", "bitwise", status="ERROR",
           extra=dict(exception=type(e).__name__, msg=str(e)[:300]))

import importlib
try:
    b9, _ = run_baseline_twice(case8)
    r_a = SP.sparse_scope_access(*case8)
    importlib.reload(SP)
    r_b = SP.sparse_scope_access(*case8)
    importlib.reload(SP)
    r_c = SP.sparse_scope_access(*case8)
    oks = {k: bits(getattr(r_a, k)).tobytes()
           == bits(getattr(r_b, k)).tobytes()
           == bits(getattr(r_c, k)).tobytes() == bits(b).tobytes()
           for k, b in zip(FIELDS, b9)}
    record("H9", "module reloaded twice; results across reloads",
           "reload determinism + baseline parity bitwise", par=oks,
           status="PASS" if all(oks.values()) else "FAIL")
except Exception as e:
    record("H9", "reload determinism", "bitwise", status="ERROR",
           extra=dict(exception=type(e).__name__, msg=str(e)[:300]))

c = res_h1.counters if res_h1 is not None else None
if c is not None:
    inv = dict(touched_le_OxV=c.touched_total <= 48 * n,
               cand_le_OxD=c.candidates_total <= 48 * 40,
               stamps_eq_lanes_x_VpD=c.stamps_zeroed == lanes * (n + 40),
               dense_eq_OxV=c.dense_init_avoided == 48 * n,
               lanes_gt_0=c.lanes >= 1)
    record("H10", "counter invariants on H1 cohort",
           "touched<=O*V, candidates<=O*D, stamps=lanes*(V+D), "
           "dense_init=O*V", counters=c,
           status="PASS" if all(inv.values()) else "FAIL", extra=inv)
badv = case8[:15] + (np.float32(1.2),)
rf = SP.sparse_scope_access(*badv)
zc = (rf.counters.touched_total == 0 and rf.counters.candidates_total == 0
      and rf.counters.stamps_zeroed == 0
      and rf.counters.dense_init_avoided == 0
      and rf.counters.origins == 0 and rf.counters.lanes == 0)
record("H10-fallback", "refused call counters", "all zero on fallback",
       route=rf.route, refusal=rf.refusal, counters=rf.counters,
       status="PASS" if zc and rf.route == "compact_fallback" else "FAIL")

cases11 = []
for k in range(4):
    r = np.random.default_rng(100 + k)
    p, v, w, fl = ring_plus(50 + 10 * k, 30 + 10 * k, r)
    o_k = np.stack([r.integers(0, 50 + 10 * k, 12),
                    r.integers(0, 50 + 10 * k, 12)],
                   axis=1).astype(np.int64)
    ow_k = r.uniform(0.0, 0.1, (12, 2))
    dd = np.stack([r.integers(0, 50 + 10 * k, 25),
                   r.integers(0, 50 + 10 * k, 25)],
                  axis=1).astype(np.int64)
    cases11.append(dargs(o_k, ow_k, p, v, w, fl, dd,
                         r.uniform(0.0, 0.4, (25, 2)),
                         r.uniform(0.25, 1.0, 25), 0.8 + 0.3 * k))
refs11 = [run_baseline_twice(a) for a in cases11]
out11 = {}
bar = threading.Barrier(4)


def go11(i):
    bar.wait()
    out11[i] = SP.sparse_scope_access(*cases11[i])


ths = [threading.Thread(target=go11, args=(i,)) for i in range(4)]
for t in ths:
    t.start()
for t in ths:
    t.join()
par11 = {i: all(parity(out11[i], refs11[i][0]).values()) for i in range(4)}
jit11 = all(r[1] for r in refs11)
record("H11", "4 concurrent calls with 4 DIFFERENT graphs/cutoffs at a "
      "barrier (4 lanes each)",
       "each equals its own serial baseline bitwise (no cross-call state)",
       par={str(k): v for k, v in par11.items()}, jitter=jit11,
       status="PASS" if all(par11.values()) and jit11 else "FAIL")

V_big = (268435456 // 8 // lanes) // 3 + 1
d_c2 = np.array([[0, 1], [1, 0]], dtype=np.int64)
big_case = dargs(np.array([[0, 1]], dtype=np.int64),
                 np.array([[0.5, 0.25]]),
                 np.zeros(V_big + 1, dtype=np.int64),
                 np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.float64),
                 np.zeros(0, dtype=bool), d_c2,
                 np.array([[0.25, 0.5], [0.5, 0.25]]),
                 np.array([1.0, 0.5]), 1.0)
nan_case = list(cases11[1])
nan_case[1] = nan_case[1].copy()
nan_case[1][0, 0] = np.nan
nan_case = tuple(nan_case)
mixed = {0: cases11[2], 1: big_case, 2: nan_case, 3: cases11[3]}
refm = {0: run_baseline_twice(cases11[2]), 2: run_baseline_twice(nan_case),
        3: run_baseline_twice(cases11[3])}
outm = {}
barm = threading.Barrier(4)


def gom(i):
    barm.wait()
    outm[i] = SP.sparse_scope_access(*mixed[i])


ths = [threading.Thread(target=gom, args=(i,)) for i in range(4)]
for t in ths:
    t.start()
for t in ths:
    t.join()
okm = (outm[0].route == "sparse_epoch_a3"
       and outm[1].route == "compact_fallback"
       and outm[1].refusal == "capacity"
       and outm[2].route == "compact_fallback"
       and outm[2].refusal == "values"
       and outm[3].route == "sparse_epoch_a3"
       and all(parity(outm[i], refm[i][0]).values()) for i in (0, 2, 3))
record("H11b", "concurrent MIXED routes: admitted + capacity-refused + "
      "values-refused + admitted",
       "each thread takes its own route; parity per thread",
       status="PASS" if okm else "FAIL",
       par={str(i): (all(parity(outm[i], refm[i][0]).values())
                     if i in refm else outm[i].refusal) for i in outm})

d_dup = np.array([[2, 2], [2, 5], [5, 2], [5, 5], [2, 2]], dtype=np.int64)
dw_dup = np.array([[0.3, 0.7], [0.1, 0.0], [0.0, 0.4], [0.25, 0.25],
                   [0.6, 0.0]])
dwt_dup = np.array([1.0, 0.5, 0.25, 0.75, 1.25])
check("H12", "t0==t1 destinations, shared terminals across destinations, "
      "duplicates across both columns",
      "reverse incidence keeps both entries; mark dedup; ascending ORIGINAL "
      "id; dist=min per dossier 03; bitwise",
      dargs(np.array([[2, 5], [0, 1], [2, 5]], dtype=np.int64),
            np.array([[0.0, 0.0], [0.1, 0.2], [0.3, 0.4]]),
            ptr, vec, wts, flag, d_dup, dw_dup, dwt_dup, 1.2))

b13 = 1.0 + 1.2
for tag, seeds in (
        ("equal-seeds", np.array([[0.7, 0.7]])),
        ("seed==b", np.array([[b13, b13]])),
        ("seed==cutoff", np.array([[1.2, 1.2]])),
        ("seed>cutoff", np.array([[7.5, 9.25]])),
        ("seeds-b-and-cutoff", np.array([[b13, 1.2]]))):
    check(f"H13[{tag}]", f"seed weights {seeds.tolist()} cutoff=1.2 "
          f"(b={b13})",
          "strict < gates: seed>=cutoff never pushed but still "
          "touched+enumerated; bitwise",
          dargs(np.array([[2, 5]], dtype=np.int64), seeds, ptr, vec, wts,
                flag, d_dup, dw_dup, dwt_dup, 1.2))

ptr_z, vec_z, wts_z, flag_z = tiny(3, [(0, 1), (1, 0)],
                                   [-0.0, 0.5], [True, True])
check("H14", "-0.0 arc weights, -0.0 seeds, -0.0 terminal costs, -0.0 "
      "destination weights",
      "-0.0 admitted (not negative) and retained; parity bitwise incl. "
      "any -0.0 in outputs",
      dargs(np.array([[0, 1]], dtype=np.int64), np.array([[-0.0, -0.0]]),
            ptr_z, vec_z, wts_z, flag_z,
            np.array([[0, 1], [1, 0]], dtype=np.int64),
            np.array([[-0.0, -0.0], [-0.0, 0.0]]),
            np.array([-0.0, -0.0]), 1.0))

arcs15 = [(0, 1), (0, 1), (0, 1), (1, 0), (0, 0), (1, 1), (0, 2), (2, 0),
          (2, 2), (1, 2)]
ptr15, vec15, wts15, flag15 = tiny(
    3, arcs15,
    [0.5, 0.5, 0.5, 0.25, 0.0, 0.0, 0.75, 0.25, 0.0, 0.5],
    [True] * 10)
d15 = np.array([[0, 2], [1, 2], [2, 2], [0, 0], [0, 1]], dtype=np.int64)
dw15 = np.array([[0.25, 0.0], [0.0, 0.25], [0.5, 0.5], [0.0, 0.0],
                 [0.75, 0.75]])
dwt15 = np.array([1.0, 0.5, 0.25, 0.75, 1.5])
r15 = check("H15", "parallel duplicate arcs (unequal+equal weights), "
            "self-loops with zero cost, zero-cost 2-cycle, tied rows",
            "schedule law preserved (S10/VALIDATION multigraph rules); "
            "bitwise",
            dargs(np.array([[0, 1], [1, 2], [0, 2]], dtype=np.int64),
                  np.array([[0.0, 0.0], [0.25, 0.5], [0.5, 0.5]]),
                  ptr15, vec15, wts15, flag15, d15, dw15, dwt15, 1.0))
try:
    from urban_network_analysis.kernels import scope_access as SA
    dense = SA.scope_search_schedule(
        np.array([0, 1], dtype=np.int64), np.array([0.0, 0.0]),
        ptr15, vec15, wts15, flag15, 1.0, d_count=0, domain="a3")
    b15, _ = run_baseline_twice(
        dargs(np.array([[0, 1]], dtype=np.int64), np.array([[0.0, 0.0]]),
              ptr15, vec15, wts15, flag15, d15, dw15, dwt15, 1.0))
    ok = (r15 is not None and all(parity(r15, b15).values())
          and int(np.asarray(dense).shape[0]) == 3)
    record("H15-oracle", "frozen oracle scope_search_schedule on the H15 "
          "multigraph (dense labels) exists and kernel parity holds",
          "oracle is a third authority; dense labels shape V",
          status="PASS" if ok else "FAIL")
except Exception as e:
    record("H15-oracle", "oracle cross-check", "-", status="ERROR",
           extra=dict(exception=type(e).__name__, msg=str(e)[:200]))

base16 = dargs(np.array([[2, 5]], dtype=np.int64), np.array([[0.0, 0.1]]),
               ptr, vec, wts, flag, d_dup, dw_dup, dwt_dup, 1.2)
for tag, met in (
        ("knn-empty", (1.5, 2.0, 3.0, 0.8, "exponential",
                       np.array([], dtype=np.float64))),
        ("knn-len1", (1.5, 2.0, 3.0, 0.8, "other",
                      np.array([2.0]))),
        ("knn-float32", (1.5, 2.0, 3.0, 0.8, "logistic",
                         np.array([1.0, 0.5, 0.25], dtype=np.float32))),
        ("gravity-zero", (0.0, 0.0, 0.0, 0.0, "exponential",
                          np.array([1.0, 0.5]))),
        ("gravity-huge", (700.0, 300.0, 250.0, 80.0, "exponential",
                          np.array([1.0, 0.5]))),
        ("gravity-float32-scalars", (np.float32(1.5), np.float32(2.0),
                                     np.float32(3.0), np.float32(0.8),
                                     "logistic", np.array([1.0, 0.5])))):
    args16 = base16[:9] + met + (1.2,)
    check(f"H16[{tag}]", f"metric params: {tag}",
          "fold receives identical objects both routes; parity bitwise",
          args16)

check("H17a", "7 origins on 4 lanes (non-divisible)",
      "lane striding origin, lane+lanes, ...; bitwise",
      dargs(o9[:7], ow9[:7], ptr, vec, wts, flag, d, dw, dwt, 1.2))
check("H17b", "2 origins on 4 lanes (idle lanes)",
      "idle lanes contribute zero counters; bitwise",
      dargs(o9[:2], ow9[:2], ptr, vec, wts, flag, d, dw, dwt, 1.2))
check("H17c", "1 origin", "single origin; bitwise",
      dargs(o9[:1], ow9[:1], ptr, vec, wts, flag, d, dw, dwt, 1.2))

ref18 = None
ok18 = True
for tag, cut in (("int", 1), ("float", 1.0), ("np-int64", np.int64(1)),
                 ("np-float64", np.float64(1.0))):
    args18 = dargs(np.array([[2, 5]], dtype=np.int64),
                   np.array([[0.0, 0.1]]), ptr, vec, wts, flag,
                   d_dup, dw_dup, dwt_dup, cut)
    b18, j18 = run_baseline_twice(args18)
    r18 = SP.sparse_scope_access(*args18)
    sig = (bits(r18.reach).tobytes(), bits(r18.gravity_exponential).tobytes(),
           bits(r18.knn_access).tobytes())
    if ref18 is None:
        ref18 = sig
    ok18 = ok18 and sig == ref18 and all(parity(r18, b18).values()) and j18
record("H18", "cutoff as int/float/np.int64/np.float64 of the same value",
       "b=fl(1.0+cutoff) identical for all four; outputs bitwise EQUAL "
       "across representations and vs baseline",
       status="PASS" if ok18 else "FAIL",
       extra={"cross_representation_equal": bool(ok18)})


def exc_info(fn):
    try:
        fn()
        return ("no-exception", "")
    except Exception as e:
        return (type(e).__name__, str(e)[:160])


base19 = dargs(np.array([[2, 5]], dtype=np.int64), np.array([[0.0, 0.1]]),
               ptr, vec, wts, flag, d_dup, dw_dup, dwt_dup, 1.2)
dwt19 = np.array(dwt_dup)
mut19 = {
    "d_weights-2d": lambda a: a[:8]
    + (np.ascontiguousarray(dwt19[:, None]),) + a[9:],
    "d_terminal_weights-1d": lambda a: a[:7] + (np.ascontiguousarray(dw_dup[:, 0]),)
    + (dwt19,) + a[9:],
    "d_weights-0d": lambda a: a[:8] + (np.array(0.5),) + a[9:],
}
for tag, mutate in mut19.items():
    a19 = mutate(base19)
    eb = exc_info(lambda: integrated_scope_access(*a19))
    es = exc_info(lambda: SP.sparse_scope_access(*a19))
    record(f"H19[{tag}]", f"malformed {tag} (gate-admit-then-raise or "
          f"gate-raises class)",
          "claimed contract: gate 'raises nothing', every refusal falls "
          "back UNCHANGED; observed divergence recorded as finding",
          status="DIVERGENCE-RECORDED",
          extra=dict(baseline=eb, sparse=es), args=a19)

ptr20 = np.array([0], dtype=np.int64)
a20 = (np.zeros((0, 2), dtype=np.int64), np.zeros((0, 2)), ptr20,
       np.zeros(0, dtype=np.int64), np.zeros(0), np.zeros(0, dtype=bool),
       np.zeros((0, 2), dtype=np.int64), np.zeros((0, 2)),
       np.zeros(0)) + MET + (1.0,)
check("H20", "V=0, O=0, D=0 (bincount minlength=0 corner)",
      "empty outputs, dtypes preserved; bitwise", a20)

W = 100
ptr21 = np.zeros(4 + 1, dtype=np.int64)
ptr21[1] = W
ptr21[2:] = W
vec21 = np.zeros(W, dtype=np.int64)
wts21 = np.full(W, 0.5)
flag21 = np.ones(W, dtype=bool)
d21 = np.array([[0, 1], [1, 1], [0, 0]], dtype=np.int64)
dw21 = np.array([[0.0, 0.25], [0.5, 0.5], [0.0, 0.0]])
dwt21 = np.array([1.0, 0.5, 0.75])
check("H21", "100 parallel equal-cost arcs in one row (ties, twice-"
      "accepted replay), leaf targets",
      "row buffer max_degree; ascending row order; bitwise",
      dargs(np.array([[3, 3], [0, 0]], dtype=np.int64),
            np.array([[0.0, 0.0], [0.1, 0.1]]), ptr21, vec21, wts21,
            flag21, d21, dw21, dwt21, 2.0))

check("H22", "cutoff=np.nextafter(1.2, 2.0) boundary on a rich graph",
      "bitwise both routes",
      dargs(np.array([[2, 5]], dtype=np.int64), np.array([[0.0, 0.1]]),
            ptr, vec, wts, flag, d_dup, dw_dup, dwt_dup,
            np.nextafter(1.2, 2.0)))

with open("/storage/scratch1/1/dsynn6/una-x/.claude/worktrees/"
          "s12-review-r1/evidence/tasks/S12/raw/hostile_fix_results.json",
          "w") as f:
    json.dump(RESULTS, f, indent=1, default=str)
print("TOTAL", len(RESULTS), "FAIL",
      sum(1 for r in RESULTS if r["status"] == "FAIL"), "ERROR",
      sum(1 for r in RESULTS if r["status"] == "ERROR"))
