"""T12 bounded child: run integrated + od drivers under the parent's
NUMBA_NUM_THREADS on a deterministic mid-size graph, twice, asserting
in-process bit-identity, and save all outputs for cross-arm byte
comparison. One arm per child (one import root each), like
compare_arms.py.
"""
from __future__ import annotations

import os
import sys

import numpy as np

root, out_npz = sys.argv[1], sys.argv[2]
sys.path.insert(0, root)

import urban_network_analysis.Engines.AccessibilityWElevation as acwe
import urban_network_analysis.Engines.Accessibility as acc

rng = np.random.default_rng(20260925)
V = 600
pointer = [0]
nbrs, weights, flags = [], [], []
for node in range(V):
    targets = {(node + 1) % V, (node + 7) % V, (node + 61) % V,
               (node + 211) % V}
    deg = int(rng.integers(1, 5))
    for t in sorted(targets)[:deg]:
        nbrs.append(t)
        weights.append(round(float(rng.uniform(0.5, 9.0)), 6))
        flags.append(bool(t % 7 != 5))
    pointer.append(len(nbrs))
ptr_a = np.asarray(pointer, np.int64)
nbr_a = np.asarray(nbrs, np.int64)
wgt_a = np.asarray(weights, np.float64)
flg_a = np.asarray(flags, np.bool_)

O, D = 16, 12
oti = np.stack([(np.arange(O) * 37) % V, (np.arange(O) * 37 + 1) % V],
               axis=1).astype(np.int64)
otw = np.round(rng.uniform(0.0, 2.0, (O, 2)), 6)
dti = np.stack([(np.arange(D) * 53 + 3) % V,
                (np.arange(D) * 53 + 7) % V], axis=1).astype(np.int64)
dtw = np.round(rng.uniform(0.0, 2.0, (D, 2)), 6)
d_w = np.round(rng.uniform(0.5, 3.0, D), 6)
cutoff = 25.0

common = dict(
    gravity_beta=0.05, gravity_plateau=0.0, gravity_logistic_midpoint=5.0,
    gravity_growth_rate=float(np.log(99.0) / 5.0),
    knn_decay="logistic", knn_weights=np.array([1.0, 0.6, 0.3]),
)

payload = {}
for tag, mod in (("acce", acwe), ("acc", acc)):
    for rep in range(2):
        outs = mod.integrated_scope_access(
            oti, otw, ptr_a, nbr_a, wgt_a, flg_a,
            dti, dtw, d_w, common["gravity_beta"], common["gravity_plateau"],
            common["gravity_logistic_midpoint"], common["gravity_growth_rate"],
            common["knn_decay"], common["knn_weights"], cutoff,
        )
        od = mod.od_compact_vector_node_view_scope(
            oti, otw, ptr_a, nbr_a, wgt_a, flg_a, cutoff, D, dti, dtw)
        if rep == 0:
            for name, arr in zip(("reach", "gexp", "glog", "knn"), outs):
                payload[f"{tag}_{name}"] = arr
            payload[f"{tag}_od"] = od
        else:
            for name, arr in zip(("reach", "gexp", "glog", "knn"), outs):
                assert arr.tobytes() == payload[f"{tag}_{name}"].tobytes()
            assert od.tobytes() == payload[f"{tag}_od"].tobytes()

np.savez(out_npz, **payload)
print("MT_CHILD_OK", os.environ.get("NUMBA_NUM_THREADS"), out_npz)
