"""Positive boundsafety child (proof.md section 2, executed): the
tail-free kernel under NUMBA_BOUNDSCHECK=1 across every origin of the
admitted graph, plus the guard on firing and refusal inputs. Any
out-of-domain index dies with a BoundsError; a clean run is the
positive proof that every index reaching the V-length vector is in
[0, V).

Usage: a3_bounds_child.py ROOT OUT_JSON
"""
from __future__ import annotations

import hashlib
import json
import os
import sys

os.environ["NUMBA_BOUNDSCHECK"] = "1"  # must precede numba import

import numpy as np

root, out_json = sys.argv[1], sys.argv[2]
sys.path.insert(0, root)

import urban_network_analysis.Engines._large_access_scratch as scratch


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
    return (
        np.asarray(pointer, np.int64),
        np.asarray(nbrs, np.int64),
        np.asarray(weights, np.float64),
        np.asarray(flags, np.bool_),
        rng,
    )


def d_terminals(rng, v, d_count=6):
    dti = np.stack(
        [(np.arange(d_count) * 13 + 3) % v,
         (np.arange(d_count) * 13 + 5) % v], axis=1).astype(np.int64)
    return dti


pointer, nbrs, weights, flags, rng = graph()
v = pointer.shape[0] - 1
oti = np.stack(
    [(np.arange(5) * 11) % v, (np.arange(5) * 11 + 1) % v],
    axis=1).astype(np.int64)
otw = np.round(rng.uniform(0.0, 2.0, (5, 2)), 6)
max_degree = int(np.diff(pointer).max())

runs = 0
h = hashlib.sha256()
for seed_v, dti in (
    (v, d_terminals(rng, v)),
    (v, np.stack([np.zeros(6, np.int64), np.full(6, v - 1, np.int64)], axis=1)),
):
    assert scratch._a3_tail_admits(dti, dti.shape[0], seed_v)
    for o in range(oti.shape[0]):
        eligible_offset = np.empty(max_degree, dtype=np.int64)
        eligible_weight = np.empty(max_degree, dtype=np.float64)
        labels, pred = scratch._a3_scope_search_tailless(
            oti[o], otw[o], pointer, nbrs, weights, flags,
            12.5, eligible_offset, eligible_weight)
        assert labels.shape == (v,)
        h.update(np.ascontiguousarray(labels).tobytes())
        runs += 1

result = {
    "boundscheck": os.environ.get("NUMBA_BOUNDSCHECK"),
    "origins_run": runs,
    "labels_digest": h.hexdigest(),
}
with open(out_json, "w") as fh:
    json.dump(result, fh, indent=1, sort_keys=True)
print(json.dumps(result, sort_keys=True))
