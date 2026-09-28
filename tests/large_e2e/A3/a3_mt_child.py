"""T6 bounded child: the od and integrated drivers under a 4-thread
numba pool (NUMBA_NUM_THREADS=4 is set by the parent's env for this
process). Writes one JSON of output digests; the parent compares them
against its own 2-thread runs.

Usage: a3_mt_child.py ROOT OUT_JSON
"""
from __future__ import annotations

import hashlib
import json
import os
import sys

import numpy as np

root, out_json = sys.argv[1], sys.argv[2]
sys.path.insert(0, root)

import numba

import urban_network_analysis.Engines.AccessibilityWElevation as acwe


def digest(arr):
    arr = np.ascontiguousarray(arr)
    h = hashlib.sha256()
    h.update(str(arr.dtype).encode())
    h.update(str(arr.shape).encode())
    h.update(arr.tobytes())
    return h.hexdigest()


def graph(seed=20260925, v=120):
    rng = np.random.default_rng(seed)
    pointer = [0]
    nbrs, weights, flags = [], [], []
    for node in range(v):
        targets = {(node + 1) % v, (node + 7) % v, (node + 29) % v}
        deg = int(rng.integers(1, 4))
        for t in sorted(targets)[:deg]:
            nbrs.append(t)
            weights.append(round(float(rng.uniform(0.5, 9.0)), 6))
            flags.append(bool(t % 5 != 3))
        pointer.append(len(nbrs))
    o_count, d_count = 5, 6
    oti = np.stack(
        [(np.arange(o_count) * 11) % v, (np.arange(o_count) * 11 + 1) % v],
        axis=1).astype(np.int64)
    otw = np.round(rng.uniform(0.0, 2.0, (o_count, 2)), 6)
    dti = np.stack(
        [(np.arange(d_count) * 13 + 3) % v,
         (np.arange(d_count) * 13 + 5) % v], axis=1).astype(np.int64)
    dtw = np.round(rng.uniform(0.0, 2.0, (d_count, 2)), 6)
    return dict(
        pointer=np.asarray(pointer, np.int64),
        nbrs=np.asarray(nbrs, np.int64),
        weights=np.asarray(weights, np.float64),
        flags=np.asarray(flags, np.bool_),
        oti=oti, otw=otw, dti=dti, dtw=dtw,
        d_w=np.round(rng.uniform(0.5, 3.0, d_count), 6),
        cutoff=12.5,
    )


g = graph()
od = acwe.od_compact_vector_node_view_scope(
    g["oti"], g["otw"], g["pointer"], g["nbrs"], g["weights"], g["flags"],
    g["cutoff"], g["dti"].shape[0], g["dti"], g["dtw"])
reach, grav_exp, grav_log, knn = acwe.integrated_scope_access(
    g["oti"], g["otw"], g["pointer"], g["nbrs"], g["weights"], g["flags"],
    g["dti"], g["dtw"], g["d_w"],
    0.05, 0.0, 5.0, float(np.log(99.0) / 5.0), "logistic",
    np.array([1.0, 0.6, 0.3]), g["cutoff"])

result = {
    "numba_num_threads": numba.get_num_threads(),
    "od": digest(od),
    "reach": digest(reach),
    "gravity_exponential": digest(grav_exp),
    "gravity_logistic": digest(grav_log),
    "knn_access": digest(knn),
}
with open(out_json, "w") as fh:
    json.dump(result, fh, indent=1, sort_keys=True)
print(json.dumps(result, sort_keys=True))
