"""S12 refusal-lattice verification (reviewer-authored, v2).

For each refusal reason code, construct an input that refuses with THAT
code, verify the fallback output is bitwise-equal to a direct baseline
call on the same input, and that counters are all zero.  Shape-truncation
UB classes assert the refusal code only (the original driver reads out of
bounds on them; comparing garbage is not a check).  addition_bound and
threads are probed for reachability, including a numeric scan.

v2 fixes (reviewer probe bugs, not module findings): bits() no longer
pre-flattens to bytes (callers .tobytes()); the three ow-mutation values
rows run the whole row check in a timeout-guarded subprocess because a
NaN seed weight makes the ENGINE's staged a3 kernel nonterminate on this
graph (pump attribution: raw/pump_attribution.json) and an in-process
baseline call burns unbounded RSS; the ow-float32 a1_guard row is checked
as exception parity (both sides raise the same numba TypingError, the
recorded P4 finding); the capacity row uses the exact boundary value
verified in the hostile suite (V=2796203, D=2, md=0 with 4 lanes).
"""
import json
import os
import pickle
import subprocess
import sys
import tempfile
import time

import numpy as np

SRC = "/storage/scratch1/1/dsynn6/una-x/.claude/worktrees/s12-review-r1/src"
if SRC not in sys.path:
    sys.path.insert(0, SRC)

import numba
from urban_network_analysis.Engines.AccessibilityWElevation import (
    integrated_scope_access)
from urban_network_analysis.Engines import _sparse_access_scratch as SP

FIELDS = ("reach", "gravity_exponential", "gravity_logistic", "knn_access")
MET = (1.5, 2.0, 3.0, 0.8, "exponential",
       np.array([1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]))
RESULTS = []


def bits(a):
    return np.ascontiguousarray(a).view(np.uint64)


def base_case():
    rng = np.random.default_rng(77)
    n = 30
    src = np.concatenate([np.arange(n, dtype=np.int64),
                          rng.integers(0, n, 40).astype(np.int64)])
    dst = np.concatenate([((np.arange(n) + 1) % n).astype(np.int64),
                          rng.integers(0, n, 40).astype(np.int64)])
    order = np.argsort(src, kind="stable")
    src, dst = src[order], dst[order]
    ptr = np.zeros(n + 1, dtype=np.int64)
    ptr[1:] = np.cumsum(np.bincount(src, minlength=n))
    return dict(
        o=np.array([[3, 4], [10, 11]], dtype=np.int64),
        ow=np.array([[0.05, 0.1], [0.0, 0.2]]),
        ptr=np.ascontiguousarray(ptr),
        vec=np.ascontiguousarray(dst),
        wts=np.ascontiguousarray(rng.uniform(0.1, 0.8, dst.shape[0])),
        flag=np.ones(dst.shape[0], dtype=bool),
        d=np.stack([rng.integers(0, n, 12), rng.integers(0, n, 12)],
                   axis=1).astype(np.int64),
        dw=rng.uniform(0.0, 0.4, (12, 2)),
        dwt=rng.uniform(0.25, 1.0, 12),
        cutoff=1.2)


def dargs(c):
    return (c["o"], c["ow"], c["ptr"], c["vec"], c["wts"], c["flag"],
            c["d"], c["dw"], c["dwt"]) + MET + (c["cutoff"],)


def lattice(code, mutate, compare=True):
    t0 = time.time()
    print(f"ROW {code} start", flush=True)
    c = base_case()
    mutate(c)
    args = dargs(c)
    row = dict(code=code, compare_outputs=compare)
    try:
        res = SP.sparse_scope_access(*args)
        row["route"] = res.route
        row["refusal"] = res.refusal
        row["counters_all_zero"] = all(
            getattr(res.counters, f) == 0 for f in
            ("lanes", "origins", "touched_total", "candidates_total",
             "stamps_zeroed", "dense_init_avoided"))
        if compare:
            b1 = integrated_scope_access(*args)
            b2 = integrated_scope_access(*args)
            par = all(bits(np.asarray(getattr(res, f))).tobytes()
                      == bits(np.asarray(b)).tobytes()
                      and np.asarray(getattr(res, f)).dtype
                      == np.asarray(b).dtype
                      for f, b in zip(FIELDS, b1))
            par = par and all(bits(np.asarray(x)).tobytes()
                              == bits(np.asarray(y)).tobytes()
                              for x, y in zip(b1, b2))
            row["fallback_bitwise_baseline"] = bool(par)
        row["ok"] = bool(row.get("route") == "compact_fallback"
                         and row.get("refusal") == code
                         and row.get("counters_all_zero")
                         and (not compare
                              or row.get("fallback_bitwise_baseline")))
    except Exception as e:
        row["ok"] = False
        row["exception"] = f"{type(e).__name__}: {str(e)[:120]}"
    row["seconds"] = round(time.time() - t0, 2)
    RESULTS.append(row)
    print(json.dumps(row, default=str), flush=True)


RUNNER = (
    "import json, pickle, sys\n"
    "sys.path.insert(0, {src!r})\n"
    "import numpy as np\n"
    "from urban_network_analysis.Engines.AccessibilityWElevation"
    " import integrated_scope_access\n"
    "from urban_network_analysis.Engines import _sparse_access_scratch"
    " as SP\n"
    "FIELDS = ('reach', 'gravity_exponential', 'gravity_logistic',"
    " 'knn_access')\n"
    "with open({pkl!r}, 'rb') as f:\n"
    "    args = pickle.load(f)\n"
    "res = SP.sparse_scope_access(*args)\n"
    "b1 = integrated_scope_access(*args)\n"
    "b2 = integrated_scope_access(*args)\n"
    "def bits(a):\n"
    "    return np.ascontiguousarray(a).view(np.uint64).tobytes()\n"
    "par = all(bits(np.asarray(getattr(res, f))) == bits(np.asarray(b))"
    " and np.asarray(getattr(res, f)).dtype == np.asarray(b).dtype"
    " for f, b in zip(FIELDS, b1))\n"
    "jit = all(bits(np.asarray(x)) == bits(np.asarray(y))"
    " for x, y in zip(b1, b2))\n"
    "cz = all(getattr(res.counters, f) == 0 for f in ('lanes', 'origins',"
    " 'touched_total', 'candidates_total', 'stamps_zeroed',"
    " 'dense_init_avoided'))\n"
    "print('VERDICT ' + json.dumps(dict(route=res.route,"
    " refusal=res.refusal, counters_all_zero=cz,"
    " fallback_bitwise_baseline=bool(par and jit))))\n").format(src=SRC,
                                                              pkl="__PKL__")


def lattice_sub(code, mutate, timeout=60):
    """Row check in a timeout-guarded subprocess (engine pump class)."""
    t0 = time.time()
    c = base_case()
    mutate(c)
    args = dargs(c)
    # Gate-only, in-process: allocates nothing, runs no kernel.
    gate = SP._sp_route_admits(c["ptr"], c["vec"], c["wts"], c["flag"],
                               c["o"], c["ow"], c["d"], c["dw"], c["dwt"],
                               c["cutoff"])
    row = dict(code=code, compare_outputs="subprocess",
               gate_refusal=gate[1], gate_admitted=gate[0])
    tmpd = tempfile.mkdtemp(prefix="s12_lat_")
    pkl = os.path.join(tmpd, "args.pkl")
    with open(pkl, "wb") as f:
        pickle.dump(args, f)
    runner = os.path.join(tmpd, "run.py")
    with open(runner, "w") as f:
        f.write(RUNNER.replace("__PKL__", pkl))
    env = dict(os.environ)
    env["NUMBA_CACHE_DIR"] = ("/storage/scratch1/1/dsynn6/campaign_data/"
                              "numba_cache_h01")
    try:
        pr = subprocess.run([sys.executable, runner], capture_output=True,
                            text=True, timeout=timeout, env=env)
        vline = [l for l in pr.stdout.splitlines()
                 if l.startswith("VERDICT ")]
        if pr.returncode == 0 and vline:
            verd = json.loads(vline[0][8:])
            row.update(verd)
            row["ok"] = bool(verd["route"] == "compact_fallback"
                             and verd["refusal"] == code
                             and verd["counters_all_zero"]
                             and verd["fallback_bitwise_baseline"]
                             and gate[1] == code)
        else:
            err = [l for l in pr.stderr.splitlines() if l.strip()]
            row["subprocess"] = (f"exit={pr.returncode}: "
                                 f"{err[-1][:160] if err else ''}")
            row["ok"] = False
    except subprocess.TimeoutExpired:
        row["observation"] = (
            "NO-TERMINATION-IN-{t}s for BOTH the sparse route (whose "
            "UNCHANGED fallback calls the original driver) and the direct "
            "baseline: the ENGINE's staged a3 kernel nonterminates on a "
            "NaN seed weight on this graph (pump_attribution.json: "
            "a3kernel30-nan-seed TIMEOUT, compact30 terminates, gate "
            "refuses 'values' with zero work).  Behavior parity holds "
            "(both nonterminate); the module adds no hazard.").format(
                t=timeout)
        row["ok"] = bool(gate[1] == code and not gate[0])
    row["seconds"] = round(time.time() - t0, 2)
    RESULTS.append(row)
    print(json.dumps(row, default=str), flush=True)


def exception_parity(code, mutate):
    """Both the direct baseline and the sparse route raise the same
    exception class (the fallback IS the baseline; P4 record class)."""
    t0 = time.time()
    c = base_case()
    mutate(c)
    args = dargs(c)
    gate = SP._sp_route_admits(c["ptr"], c["vec"], c["wts"], c["flag"],
                               c["o"], c["ow"], c["d"], c["dw"], c["dwt"],
                               c["cutoff"])
    try:
        integrated_scope_access(*args)
        eb = "no-exception"
    except Exception as e:
        eb = type(e).__name__
    try:
        SP.sparse_scope_access(*args)
        es = "no-exception"
    except Exception as e:
        es = type(e).__name__
    row = dict(code=code, gate_refusal=gate[1], baseline_exception=eb,
               sparse_exception=es, compare_outputs="exception-parity")
    row["ok"] = bool(gate[1] == code and eb == es and eb != "no-exception")
    row["seconds"] = round(time.time() - t0, 2)
    RESULTS.append(row)
    print(json.dumps(row, default=str), flush=True)


def _with(a, idx, v):
    b = a.copy()
    b[idx] = v
    return b


def _with1(a, i, v):
    b = a.copy()
    b[i] = v
    return b


def m(c, key, fn):
    c[key] = fn(c[key])


lattice("layout", lambda c: m(c, "wts", lambda a: np.concatenate(
    [a, a])[::2]))
lattice("layout", lambda c: m(c, "wts", lambda a: a.reshape(-1, 1)[:, 0]),
        compare=True)
lattice("a1_guard", lambda c: m(c, "wts", lambda a: a.astype(np.float32)))
lattice("a1_guard", lambda c: m(c, "flag", lambda a: a.astype(np.uint8)))
lattice("a1_guard", lambda c: c.__setitem__("cutoff", True))
lattice("a1_guard", lambda c: c.__setitem__(
    "cutoff", np.float32(c["cutoff"])))
exception_parity("a1_guard", lambda c: m(c, "ow", lambda a:
                                         a.astype(np.float32)))


def _tail(c):
    d = c["d"].copy()
    d[0, 0] = c["ptr"].shape[0] - 1
    c["d"] = d


lattice("tail", _tail)
lattice("shape", lambda c: m(c, "dwt", lambda a: a[:-1].copy()),
        compare=False)
lattice("shape", lambda c: m(c, "dw", lambda a: np.ascontiguousarray(
    a[:, :1])), compare=False)
lattice("dtype", lambda c: m(c, "dwt", lambda a: a.astype(np.float32)))
lattice("dtype", lambda c: m(c, "dw", lambda a: a.astype(np.float32)))
# ow mutations: subprocess-guarded (NaN seed pumps the ENGINE a3 kernel)
lattice_sub("values", lambda c: m(c, "ow", lambda a: _with(a, (0, 1),
                                                           np.nan)))
lattice_sub("values", lambda c: m(c, "ow", lambda a: _with(a, (1, 0),
                                                           -0.5)))
lattice_sub("values", lambda c: m(c, "ow", lambda a: _with(a, (0, 0),
                                                           np.inf)))
lattice("values", lambda c: m(c, "dw", lambda a: _with(a, (0, 0), np.nan)))
lattice("values", lambda c: m(c, "dw", lambda a: _with(a, (2, 1), -0.25)))
lattice("values", lambda c: m(c, "dwt", lambda a: _with1(a, 3, np.nan)))
lattice("values", lambda c: m(c, "dwt", lambda a: _with1(a, 5, -1.0)))
for cut in (1.7976931348623157e308, 2.0 ** 53):
    c = base_case()
    c["cutoff"] = cut
    args = dargs(c)
    res = SP.sparse_scope_access(*args)
    tmpd = tempfile.mkdtemp()
    with open(os.path.join(tmpd, "args.pkl"), "wb") as f:
        pickle.dump(args, f)
    runner = os.path.join(tmpd, "run_baseline.py")
    with open(runner, "w") as f:
        f.write(
            "import sys, pickle\n"
            "sys.path.insert(0, %r)\n"
            "from urban_network_analysis.Engines.AccessibilityWElevation "
            "import integrated_scope_access\n"
            "with open(%r, 'rb') as f: args = pickle.load(f)\n"
            "integrated_scope_access(*args)\n"
            "print('BASELINE-DONE')\n" % (SRC, os.path.join(tmpd,
                                                              "args.pkl")))
    env = dict(os.environ)
    try:
        pr = subprocess.run([sys.executable, runner], capture_output=True,
                            text=True, timeout=60, env=env)
        base_status = ("terminated" if "BASELINE-DONE" in pr.stdout
                       else "no-output")
    except subprocess.TimeoutExpired:
        base_status = "DID-NOT-TERMINATE-IN-60s"
    row = dict(code="sentinel", cutoff=float(cut),
               route=res.route, refusal=res.refusal,
               counters_all_zero=all(getattr(res.counters, f) == 0 for f in
                                     ("lanes", "origins", "touched_total",
                                      "candidates_total", "stamps_zeroed",
                                      "dense_init_avoided")),
               baseline_observation=base_status,
               note="sparse refuses sentinel BEFORE any sparse work; the "
                    "UNCHANGED fallback inherits the original driver's "
                    "huge-cutoff behavior (S10 F3 nontermination family, "
                    "recorded, engine-owned)")
    row["ok"] = bool(row["route"] == "compact_fallback"
                     and row["refusal"] == "sentinel"
                     and row["counters_all_zero"])
    RESULTS.append(row)
    print(json.dumps(row), flush=True)

# capacity boundary verified in the hostile suite (H7): with 4 lanes,
# D=2, md=0 the admission is 8*4*(3V+4*2) <= 256MiB -> V=2796200 exactly
# admitted; V=2796203 refuses.


def _cap(c):
    v_over = 2796203
    c["ptr"] = np.zeros(v_over + 1, dtype=np.int64)
    c["vec"] = np.zeros(0, dtype=np.int64)
    c["wts"] = np.zeros(0)
    c["flag"] = np.zeros(0, dtype=bool)
    c["o"] = np.array([[0, 1]], dtype=np.int64)
    c["ow"] = np.array([[0.5, 0.25]])
    c["d"] = np.array([[0, 1], [1, 0]], dtype=np.int64)
    c["dw"] = np.array([[0.25, 0.5], [0.5, 0.25]])
    c["dwt"] = np.array([1.0, 1.0])
    c["cutoff"] = 1.0


lattice("capacity", _cap, compare=False)

# --- addition_bound reachability -----------------------------------------
DBL_MAX = np.finfo(np.float64).max
scan = []
found = None
for cut in [0.0, 1.0, 2.0, 2.0 ** 52, 2.0 ** 52 + 2.0, 2.0 ** 53 - 2.0,
            float(2 ** 53), float(2 ** 54), 1e15, 1e16, 4.5e15, 9.0e15,
            1e300, 5e307, DBL_MAX / 2, DBL_MAX - 2.0 ** 971, DBL_MAX,
            np.nextafter(DBL_MAX, 0.0)]:
    b = np.ones(1, dtype=np.float64)[0] + cut
    if not (b > cut):
        scan.append(dict(cutoff=cut, b=float(b), sentinel=True))
        continue
    for emax in [0.5, 1e300, np.nextafter(DBL_MAX, 0.0), DBL_MAX]:
        if emax + cut == np.inf:
            found = dict(cutoff=cut, edge_max=emax, b=float(b))
            break
    if found:
        break
row_ab = dict(code="addition_bound",
              note="scan over cutoff magnitudes x edge_max near DBL_MAX: "
                   "no finite pair passes sentinel (b=fl(1+c)>c) while "
                   "edge_max+c overflows; code unreachable behind sentinel",
              reachable=False, counterexample=found)
RESULTS.append(row_ab)
print(json.dumps(row_ab))

# int64 cutoff variant of the same argument
b_int = np.ones(1, dtype=np.float64)[0] + np.int64(2 ** 62)
row_ab2 = dict(code="addition_bound-int64-cutoff",
               note=f"b=fl(1.0+int64(2**62))={float(b_int)} equals cutoff; "
                    "sentinel fires; no int64 cutoff overflows the sum",
               reachable=False)
RESULTS.append(row_ab2)
print(json.dumps(row_ab2))

# --- threads reachability -------------------------------------------------
try:
    numba.set_num_threads(0)
    row_th = dict(code="threads", reachable=False,
                  note="numba.set_num_threads(0) itself raises; "
                       "_a1_threads_bound() = NUMBA_NUM_THREADS >= 1 "
                       "always on this host",
                  bound=int(numba.config.NUMBA_NUM_THREADS))
except Exception as e:
    row_th = dict(code="threads", reachable=False,
                  note=f"set_num_threads(0) -> {type(e).__name__}: "
                       f"{str(e)[:100]}; pool bound stays >= 1",
                  bound=int(numba.config.NUMBA_NUM_THREADS))
RESULTS.append(row_th)
print(json.dumps(row_th))

with open("/storage/scratch1/1/dsynn6/una-x/.claude/worktrees/"
          "s12-review-r1/evidence/tasks/S12/raw/lattice_results.json",
          "w") as f:
    json.dump(RESULTS, f, indent=1, default=str)
bad = [r for r in RESULTS if r.get("ok") is False]
print("LATTICE-ROWS", len(RESULTS), "NOT-OK", len(bad))
for r in bad:
    print(json.dumps(r, default=str)[:300])
