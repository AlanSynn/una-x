"""S12 third-authority + findings-reproduction probe (reviewer-authored).

A: For one-origin cases, reconstruct the full result chain from the FROZEN
   kernels oracle (scope_search_schedule labels -> adjust_destination_
   distances_plan -> the engine's own fold) and compare bitwise against
   BOTH the compiled baseline and the sparse route.  This is independent
   of the implementer's L0 mirror (different construction path).
B: Reproduce the two recorded implementer findings minimally:
   P4 float32 origin seeds -> ORIGINAL driver TypingError at
   _large_access_scratch.py:422; sparse gate refuses a1_guard.
   P1 fix class -> empty-prefix and kept==0-with-candidates rows are
   exact zeros, matching the baseline.
"""
import json
import sys
import traceback

import numpy as np

SRC = "/storage/scratch1/1/dsynn6/una-x/.claude/worktrees/s12-review-r1/src"
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from urban_network_analysis.Engines.AccessibilityWElevation import (
    integrated_scope_access, reach_gravity_knn_access)
from urban_network_analysis.Engines import _sparse_access_scratch as SP
from urban_network_analysis.kernels import scope_access as SA
from urban_network_analysis.kernels.reductions import (
    adjust_destination_distances_plan)

FIELDS = ("reach", "gravity_exponential", "gravity_logistic", "knn_access")
RESULTS = []
MET = (1.5, 2.0, 3.0, 0.8, "exponential",
       np.array([1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]))


def bits(a):
    return np.ascontiguousarray(a).view(np.uint64).tobytes()


def tiny(n, arcs, weights, flags):
    rows = {v: [] for v in range(n)}
    for k, (a, b) in enumerate(arcs):
        rows[a].append(k)
    ptr, vec, ws, fl = [0], [], [], []
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


def record(cid, attack, law, status, extra=None):
    row = dict(id=cid, attack=attack, expected_law=law, status=status,
               extra=extra or {})
    print(json.dumps(row, default=str))
    RESULTS.append(row)


def oracle_chain(o_row, ow_row, ptr, vec, wts, flag, d, dw, dwt, cutoff,
                 met=MET):
    labels = SA.scope_search_schedule(
        o_row, ow_row, ptr, vec, wts, flag, cutoff, d_count=0, domain="a3")
    dist = adjust_destination_distances_plan(
        labels, d, np.ascontiguousarray(dw, dtype=np.float64))
    return reach_gravity_knn_access(
        d_distance=np.ascontiguousarray(dist),
        d_weights=np.ascontiguousarray(dwt, dtype=np.float64),
        cutoff=cutoff, gravity_beta=met[0], gravity_plateau=met[1],
        gravity_logistic_midpoint=met[2], gravity_growth_rate=met[3],
        knn_gravity_plateau=met[1], knn_weights=met[5],
        knn_gravity_beta=met[0], knn_decay=met[4],
        knn_gravity_logistic_midpoint=met[2],
        knn_gravity_growth_rate=met[3])


def authority_case(cid, o, ow, ptr, vec, wts, flag, d, dw, dwt, cutoff,
                   met=MET, note=""):
    args = (o, ow, ptr, vec, wts, flag, d, dw, dwt) + met + (cutoff,)
    base = integrated_scope_access(*args)
    res = SP.sparse_scope_access(*args)
    chain = []
    for i in range(o.shape[0]):
        chain.append(oracle_chain(np.ascontiguousarray(o[i]),
                                  np.ascontiguousarray(ow[i]), ptr, vec,
                                  wts, flag, d, dw, dwt, cutoff, met))
    ok = res.route == "sparse_epoch_a3" and res.refusal == ""
    per = {}
    for fi, f in enumerate(FIELDS):
        r_u = bits(np.asarray(getattr(res, f)))
        b_u = bits(np.asarray(base[fi]))
        o_u = bits(np.asarray(np.array([c[fi] for c in chain],
                                       dtype=np.asarray(base[fi]).dtype)))
        per[f] = dict(vs_baseline=r_u == b_u, oracle_vs_baseline=o_u == b_u,
                      sparse_vs_oracle=r_u == o_u)
        if not (r_u == b_u == o_u):
            ok = False
    record(cid, note or "oracle chain vs baseline vs sparse",
           "frozen-oracle chain, compiled baseline and sparse route agree "
           "bitwise on all four outputs", "PASS" if ok else "FAIL",
           extra=dict(per_field=per, route=res.route))


# --- A1: multigraph (H15 graph) -------------------------------------------
arcs15 = [(0, 1), (0, 1), (0, 1), (1, 0), (0, 0), (1, 1), (0, 2), (2, 0),
          (2, 2), (1, 2)]
ptr15, vec15, wts15, flag15 = tiny(
    3, arcs15,
    [0.5, 0.5, 0.5, 0.25, 0.0, 0.0, 0.75, 0.25, 0.0, 0.5], [True] * 10)
d15 = np.array([[0, 2], [1, 2], [2, 2], [0, 0], [0, 1]], dtype=np.int64)
dw15 = np.array([[0.25, 0.0], [0.0, 0.25], [0.5, 0.5], [0.0, 0.0],
                 [0.75, 0.75]])
dwt15 = np.array([1.0, 0.5, 0.25, 0.75, 1.5])
authority_case("A1-multigraph",
               np.array([[0, 1], [1, 2], [0, 2]], dtype=np.int64),
               np.array([[0.0, 0.0], [0.25, 0.5], [0.5, 0.5]]),
               ptr15, vec15, wts15, flag15, d15, dw15, dwt15, 1.0)

# --- A2: signed zero and boundaries --------------------------------------
ptr_a2, vec_a2, wts_a2, flag_a2 = tiny(3, [(0, 1), (1, 0)], [-0.0, 0.5],
                                       [True, True])
authority_case("A2-signed-zero-boundaries",
               np.array([[0, 1], [0, 2], [2, 2]], dtype=np.int64),
               np.array([[-0.0, -0.0], [1.0, 4.0], [0.5, 0.5]]),
               ptr_a2, vec_a2, wts_a2, flag_a2,
               np.array([[0, 1], [1, 0], [2, 2]], dtype=np.int64),
               np.array([[-0.0, -0.0], [-0.0, 0.0], [0.4, 0.4]]),
               np.array([-0.0, -0.0, 1.0]), 1.0)

# --- A3: random graph, several origins -----------------------------------
rng = np.random.default_rng(42)
n = 80
src = np.concatenate([np.arange(n, dtype=np.int64),
                      rng.integers(0, n, 120).astype(np.int64)])
dst = np.concatenate([((np.arange(n) + 1) % n).astype(np.int64),
                      rng.integers(0, n, 120).astype(np.int64)])
order = np.argsort(src, kind="stable")
src, dst = src[order], dst[order]
ptr_a3 = np.zeros(n + 1, dtype=np.int64)
ptr_a3[1:] = np.cumsum(np.bincount(src, minlength=n))
vec_a3 = np.ascontiguousarray(dst)
wts_a3 = np.ascontiguousarray(rng.uniform(0.05, 0.9, dst.shape[0]))
flag_a3 = (rng.uniform(size=dst.shape[0]) > 0.2)
d_a3 = np.stack([rng.integers(0, n, 30), rng.integers(0, n, 30)],
                axis=1).astype(np.int64)
dw_a3 = rng.uniform(0.0, 0.5, (30, 2))
dwt_a3 = rng.uniform(0.25, 1.5, 30)
authority_case("A3-random-80",
               np.stack([np.arange(6), (np.arange(6) * 7 + 3) % n],
                        axis=1).astype(np.int64),
               rng.uniform(0.0, 0.2, (6, 2)),
               np.ascontiguousarray(ptr_a3), vec_a3, wts_a3, flag_a3,
               d_a3, dw_a3, dwt_a3, 1.4)

# --- B1: kept==0 with NONEMPTY candidates + empty-prefix rows ------------
# dest at touched node with dw 0.9 > cutoff 0.6: candidate enumerated,
# dist 4.9 > cutoff -> kept 0 -> exact zeros.
authority_case("B1-kept0-nonempty-cand",
               np.array([[0, 2]], dtype=np.int64),
               np.array([[1.0, 4.0]]),
               ptr_a2, vec_a2, wts_a2, flag_a2,
               np.array([[2, 1]], dtype=np.int64),
               np.array([[0.9, 0.9]]), np.array([2.0]), 0.6)
row = SP.sparse_scope_access(
    np.array([[0, 2]], dtype=np.int64), np.array([[1.0, 4.0]]),
    ptr_a2, vec_a2, wts_a2, flag_a2,
    np.array([[2, 1]], dtype=np.int64), np.array([[0.9, 0.9]]),
    np.array([2.0]), *MET, 0.6)
z_ok = (int(row.reach[0]) == 0 and bits(row.gravity_exponential)
        == bits(np.array([0.0])) and bits(row.knn_access)
        == bits(np.array([0.0])))
record("B1-zeros", "kept==0 nonempty-candidates row values",
       "exact 0.0/0 rows (uninitialized-memory fix class)",
       "PASS" if z_ok else "FAIL",
       extra=dict(reach=int(row.reach[0]),
                  grav_u64=hex(int(np.asarray(row.gravity_exponential
                                              ).view(np.uint64)[0])),
                  knn_u64=hex(int(np.asarray(row.knn_access
                                             ).view(np.uint64)[0]))))

# --- B2 (P4): float32 origin seeds, ORIGINAL driver ----------------------
o_f32 = np.array([[0, 1]], dtype=np.int64)
ow_f32 = np.array([[0.1, 0.2]], dtype=np.float32)
args_b2 = (o_f32, ow_f32, ptr_a2, vec_a2, wts_a2, flag_a2,
           np.array([[0, 1]], dtype=np.int64),
           np.array([[0.25, 0.5]]), np.array([1.0])) + MET + (1.0,)


def try_call(fn):
    try:
        fn()
        return ("no-exception", "")
    except Exception as e:
        tb = traceback.format_exc()
        idx = tb.find("_large_access_scratch.py")
        site = tb[idx:idx + 40].splitlines()[0] if idx >= 0 else ""
        return (type(e).__name__, site)


eb2 = try_call(lambda: integrated_scope_access(*args_b2))
try:
    rb2 = SP.sparse_scope_access(*args_b2)
    sb2 = (rb2.route, rb2.refusal)
except Exception as e:
    sb2 = ("raised", type(e).__name__)
record("B2-float32-seeds",
       "float32 o_terminal_weights: original driver compile vs sparse gate",
       "ENGINE FINDING recorded not patched: original driver TypingError "
       "at _large_access_scratch.py:422; sparse refuses a1_guard then the "
       "fallback reproduces the crash",
       "PASS" if (eb2[0].endswith("Error") and sb2[0] in
                  ("compact_fallback", "raised")) else "FAIL",
       extra=dict(baseline=eb2, sparse=sb2))

# direct minimal reproduction of the recorded finding (no sparse module):
try:
    integrated_scope_access(*args_b2)
    direct = "no-exception"
except Exception as e:
    tb = traceback.format_exc()
    lines = [l for l in tb.splitlines() if "_large_access_scratch.py"
             in l]
    direct = f"{type(e).__name__} at {lines[-1].strip() if lines else '?'}"
record("B2-direct", "minimal P4 reproduction on the ORIGINAL driver alone",
       "TypingError raised at _large_access_scratch.py:422 (heappush "
       "float32/int64 list typing)", "PASS" if "422" in direct else "FAIL",
       extra=dict(observed=direct))

with open("/storage/scratch1/1/dsynn6/una-x/.claude/worktrees/"
          "s12-review-r1/evidence/tasks/S12/raw/authority_results.json",
          "w") as f:
    json.dump(RESULTS, f, indent=1, default=str)
print("TOTAL", len(RESULTS), "FAIL",
      sum(1 for r in RESULTS if r["status"] == "FAIL"))
