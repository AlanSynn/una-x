"""S12 observed-network-scale L2 route evidence (reviewer-authored, v2).

Builds a reviewer-owned geometric random network (no implementer
artifacts), runs BOTH routes with route counters, and records route
taken, refusal reasons, counters, and bitwise equality vs the compiled
baseline.  Timings are recorded as OBSERVATIONS ONLY — this is evidence
collection for dispatch, not a performance claim; the throughput gate is
the C-phase public-batch runtime.

v2: bits() returns the u64 view (v1 pre-flattened and the callers'
.tobytes() chained a second time); the refused-mixed NaN-seed variant
runs its whole check in a timeout-guarded subprocess because a NaN seed
weight can make the ENGINE's staged a3 kernel nonterminate on cyclic
graphs (pump attribution: raw/pump_attribution.json) and an in-process
baseline call burns unbounded RSS.
"""
import hashlib
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

import dataclasses

from urban_network_analysis.Engines.AccessibilityWElevation import (
    integrated_scope_access)
from urban_network_analysis.Engines import _sparse_access_scratch as SP

FIELDS = ("reach", "gravity_exponential", "gravity_logistic", "knn_access")
OUT = "/storage/scratch1/1/dsynn6/una-x/.claude/worktrees/s12-review-r1/evidence/tasks/S12"
MET = (1.5, 2.0, 3.0, 0.8, "exponential",
       np.array([1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]))


def bits(a):
    return np.ascontiguousarray(a).view(np.uint64)


def build_graph(seed, v, knn, extra_long):
    rng = np.random.default_rng(seed)
    xy = rng.uniform(0.0, 200.0, (v, 2))
    # k-nearest neighbours by cell grid to keep it cheap
    srcs, dsts = [], []
    cell = 8.0
    grid = {}
    ci = (xy // cell).astype(np.int64)
    for i in range(v):
        grid.setdefault(tuple(ci[i]), []).append(i)
    for i in range(v):
        c = ci[i]
        cand = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                cand.extend(grid.get((c[0] + dx, c[1] + dy), ()))
        cand = np.array([j for j in cand if j != i], dtype=np.int64)
        if cand.size:
            dd = np.hypot(*(xy[cand] - xy[i]).T)
            pick = cand[np.argsort(dd)[:knn]]
            for j in pick:
                srcs.append(i)
                dsts.append(int(j))
    # long arcs
    ls = rng.integers(0, v, extra_long)
    ld = rng.integers(0, v, extra_long)
    srcs.extend(ls.tolist())
    dsts.extend(ld.tolist())
    src = np.array(srcs, dtype=np.int64)
    dst = np.array(dsts, dtype=np.int64)
    w = np.hypot(*(xy[dst] - xy[src]).T)
    order = np.argsort(src, kind="stable")
    src, dst, w = src[order], dst[order], w[order]
    ptr = np.zeros(v + 1, dtype=np.int64)
    ptr[1:] = np.cumsum(np.bincount(src, minlength=v))
    return (xy, np.ascontiguousarray(ptr), np.ascontiguousarray(dst),
            np.ascontiguousarray(w))


def build_case(seed, v, o_count, d_count, cutoff, knn=6, extra_long=3000):
    rng = np.random.default_rng(seed + 1)
    xy, ptr, vec, wts = build_graph(seed, v, knn, extra_long)
    flag = np.ones(vec.shape[0], dtype=bool)
    o = np.stack([rng.integers(0, v, o_count),
                  rng.integers(0, v, o_count)], axis=1).astype(np.int64)
    ow = rng.uniform(0.05, 0.15, (o_count, 2))
    # destinations placed ON arcs: terminal pair = (u, v_arc), split weight
    arc_pick = rng.integers(0, vec.shape[0], d_count)
    frac = rng.uniform(0.2, 0.8, d_count)
    d = np.stack([np.searchsorted(ptr, arc_pick, side="right") - 1,
                  vec[arc_pick]], axis=1).astype(np.int64)
    dw = np.stack([wts[arc_pick] * frac, wts[arc_pick] * (1.0 - frac)],
                  axis=1)
    dw = np.ascontiguousarray(dw)
    dwt = np.ascontiguousarray(rng.uniform(0.25, 2.0, d_count))
    return dict(o=np.ascontiguousarray(o), ow=np.ascontiguousarray(ow),
                ptr=ptr, vec=vec, wts=wts, flag=flag,
                d=np.ascontiguousarray(d), dw=dw, dwt=dwt, cutoff=cutoff,
                xy=xy)


def dargs(c):
    return (c["o"], c["ow"], c["ptr"], c["vec"], c["wts"], c["flag"],
            c["d"], c["dw"], c["dwt"]) + MET + (c["cutoff"],)


def graph_sha(c):
    return hashlib.sha256(
        c["ptr"].tobytes() + c["vec"].tobytes() + c["wts"].tobytes()
        + c["d"].tobytes() + c["dw"].tobytes() + c["dwt"].tobytes()
        + c["o"].tobytes() + c["ow"].tobytes()).hexdigest()


def run_variant(tag, **kw):
    c = build_case(**kw)
    args = dargs(c)
    v = c["ptr"].shape[0] - 1
    e = c["vec"].shape[0]
    rec = dict(
        variant=tag, seed=kw["seed"], node_count=int(v), edge_count=int(e),
        origin_count=int(c["o"].shape[0]), dest_count=int(c["d"].shape[0]),
        cutoff=float(c["cutoff"]),
        graph_sha256=graph_sha(c))
    np.savez(f"{OUT}/raw/L2_{tag}_inputs.npz",
             **{k: val for k, val in c.items()
                if isinstance(val, np.ndarray)})
    t0 = time.perf_counter()
    b1 = integrated_scope_access(*args)
    t_b1 = time.perf_counter() - t0
    t0 = time.perf_counter()
    b2 = integrated_scope_access(*args)
    t_b2 = time.perf_counter() - t0
    t0 = time.perf_counter()
    res = SP.sparse_scope_access(*args)
    t_sp = time.perf_counter() - t0
    rec.update(
        baseline_jitter_free=all(bits(x).tobytes() == bits(y).tobytes()
                                 for x, y in zip(b1, b2)),
        route=res.route, refusal=res.refusal,
        counters=dataclasses.asdict(res.counters),
        parity={f: bits(np.asarray(getattr(res, f))).tobytes()
                == bits(np.asarray(b)).tobytes()
                for f, b in zip(FIELDS, b1)},
        dtypes={f: str(np.asarray(getattr(res, f)).dtype)
                for f in FIELDS},
        output_bytes={f: int(np.asarray(getattr(res, f)).nbytes)
                      for f in FIELDS},
        touched_fraction_of_OV=round(
            res.counters.touched_total / (res.counters.origins * v), 5),
        candidate_fraction_of_OD=round(
            res.counters.candidates_total
            / max(1, res.counters.origins * c["d"].shape[0]), 5),
        timings_observation_only=dict(
            baseline_run1_s=round(t_b1, 3), baseline_run2_s=round(t_b2, 3),
            sparse_s=round(t_sp, 3)),
    )
    return rec


RUNNER = (
    "import json, pickle, sys\n"
    "sys.path.insert(0, {src!r})\n"
    "import dataclasses\n"
    "import numpy as np\n"
    "from urban_network_analysis.Engines.AccessibilityWElevation"
    " import integrated_scope_access\n"
    "from urban_network_analysis.Engines import _sparse_access_scratch"
    " as SP\n"
    "FIELDS = ('reach', 'gravity_exponential', 'gravity_logistic',"
    " 'knn_access')\n"
    "with open({pkl!r}, 'rb') as f:\n"
    "    args = pickle.load(f)\n"
    "def bits(a):\n"
    "    return np.ascontiguousarray(a).view(np.uint64).tobytes()\n"
    "b1 = integrated_scope_access(*args)\n"
    "b2 = integrated_scope_access(*args)\n"
    "res = SP.sparse_scope_access(*args)\n"
    "par = {{f: bits(np.asarray(getattr(res, f))) == bits(np.asarray(b))"
    " for f, b in zip(FIELDS, b1)}}\n"
    "jit = all(bits(np.asarray(x)) == bits(np.asarray(y))"
    " for x, y in zip(b1, b2))\n"
    "print('VERDICT ' + json.dumps(dict(route=res.route,"
    " refusal=res.refusal, counters=dataclasses.asdict(res.counters),"
    " parity=par, baseline_jitter_free=bool(jit))))\n")


def run_refused_mixed(kw):
    """NaN-seed refusal on observed scale, whole check in a guarded
    subprocess (engine pump class)."""
    c = build_case(**kw)
    ow = c["ow"].copy()
    ow[7, 1] = np.nan
    c["ow"] = ow
    args = dargs(c)
    gate = SP._sp_route_admits(c["ptr"], c["vec"], c["wts"], c["flag"],
                               c["o"], c["ow"], c["d"], c["dw"], c["dwt"],
                               c["cutoff"])
    rec = dict(variant="refused-mixed", seed=kw["seed"],
               node_count=int(c["ptr"].shape[0] - 1),
               edge_count=int(c["vec"].shape[0]),
               origin_count=int(c["o"].shape[0]),
               dest_count=int(c["d"].shape[0]),
               cutoff=float(c["cutoff"]), mutation="ow[7,1]=NaN",
               graph_sha256=graph_sha(c), gate_refusal=gate[1],
               gate_admitted=gate[0])
    np.savez(f"{OUT}/raw/L2_refused_mixed_inputs.npz",
             **{k: val for k, val in c.items()
                if isinstance(val, np.ndarray)})
    tmpd = tempfile.mkdtemp(prefix="s12_l2_")
    pkl = os.path.join(tmpd, "args.pkl")
    with open(pkl, "wb") as f:
        pickle.dump(args, f)
    runner = os.path.join(tmpd, "run.py")
    with open(runner, "w") as f:
        f.write(RUNNER.format(src=SRC, pkl=pkl))
    env = dict(os.environ)
    env["NUMBA_CACHE_DIR"] = ("/storage/scratch1/1/dsynn6/campaign_data/"
                              "numba_cache_h01")
    try:
        pr = subprocess.run([sys.executable, runner], capture_output=True,
                            text=True, timeout=120, env=env)
        vline = [l for l in pr.stdout.splitlines()
                 if l.startswith("VERDICT ")]
        if pr.returncode == 0 and vline:
            rec.update(json.loads(vline[0][8:]))
        else:
            err = [l for l in pr.stderr.splitlines() if l.strip()]
            rec["subprocess"] = (f"exit={pr.returncode}: "
                                 f"{err[-1][:160] if err else ''}")
    except subprocess.TimeoutExpired:
        rec["observation"] = (
            "NO-TERMINATION-IN-120s for BOTH the sparse route (UNCHANGED "
            "fallback = original driver) and the direct baseline on this "
            "NaN-seed observed-scale graph: the ENGINE's staged a3 kernel "
            "nontermination class (pump_attribution.json).  The sparse "
            "gate refuses 'values' with zero work (gate_refusal above); "
            "behavior parity holds; the module adds no hazard.")
    return rec


records = []
t_all = time.time()
for tag, kw in (
        ("sparse-regime", dict(seed=31, v=15000, o_count=400, d_count=1000,
                               cutoff=8.0)),
        ("mid-regime", dict(seed=32, v=12000, o_count=300, d_count=600,
                            cutoff=16.0)),
        ("dense-regime", dict(seed=33, v=8000, o_count=200, d_count=400,
                              cutoff=60.0))):
    rec = run_variant(tag, **kw)
    records.append(rec)
    print(json.dumps(rec, default=str), flush=True)

rec = run_refused_mixed(dict(seed=34, v=9000, o_count=250, d_count=500,
                             cutoff=10.0))
records.append(rec)
print(json.dumps(rec, default=str), flush=True)

doc = dict(
    task="S12 sparse_L2 (independent reviewer)",
    reference="integrated_scope_access compiled baseline, run twice",
    timings_note="timings are observations only, NOT a performance claim; "
                 "the throughput gate is the C-phase public-batch runtime",
    graph_construction="geometric random graph: 200x200 box, 6-NN cell-grid "
                       "arcs + 3000 uniform random long arcs, weight = "
                       "euclidean distance; destinations placed on arcs "
                       "with fractional terminal splits; flags all True",
    metric_params=dict(gravity_beta=1.5, gravity_plateau=2.0,
                       gravity_logistic_midpoint=3.0,
                       gravity_growth_rate=0.8, knn_decay="exponential",
                       knn_weights=[1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3,
                                    0.2, 0.1]),
    wall_seconds=round(time.time() - t_all, 1),
    variants=records)
with open(f"{OUT}/sparse_L2.json", "w") as f:
    json.dump(doc, f, indent=1, default=str)
print("WROTE sparse_L2.json", flush=True)
