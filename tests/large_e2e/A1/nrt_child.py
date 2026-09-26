"""T13/T14 bounded child: per-call NRT allocation counts for the
integrated driver on an admitted input and on a value-refusing input
(NUMBA_NRT_STATS=1, set before numba import). One arm per child.

Usage: nrt_child.py ROOT OUT_JSON
"""
from __future__ import annotations

import json
import os

os.environ["NUMBA_NRT_STATS"] = "1"  # must precede numba import

import sys

import numpy as np

root, out_json = sys.argv[1], sys.argv[2]
sys.path.insert(0, root)

import urban_network_analysis.Engines.AccessibilityWElevation as acwe
from numba.core.runtime import rtsys

rng = np.random.default_rng(20260925)
V = 400
pointer = [0]
nbrs, weights, flags = [], [], []
for node in range(V):
    for t in ((node + 1) % V, (node + 17) % V, (node + 61) % V):
        nbrs.append(t)
        weights.append(round(float(rng.uniform(0.5, 3.0)), 6))
        flags.append(True)
    pointer.append(len(nbrs))
ptr_a = np.asarray(pointer, np.int64)
nbr_a = np.asarray(nbrs, np.int64)
wgt_a = np.asarray(weights, np.float64)
flg_a = np.asarray(flags, np.bool_)
O = 4
oti = np.stack([np.zeros(O, np.int64), np.ones(O, np.int64)], axis=1)
otw = np.zeros((O, 2))
D = 5
dti = np.stack([(np.arange(D, dtype=np.int64) * 7) % V,
                ((np.arange(D, dtype=np.int64) * 7 + 1)) % V], axis=1)
dtw = np.zeros((D, 2))
d_w = np.ones(D)
cutoff = 30.0
ARGS = (oti, otw, ptr_a, nbr_a, wgt_a, flg_a, dti, dtw, d_w,
        0.05, 0.0, 5.0, 0.919, "logistic", np.array([1.0, 0.6, 0.3]), cutoff)

result = {"root": root, "nrt_stats": True}

# warm-up first: the NRT stats become readable only after the NRT
# initializes at the first compiled allocation
acwe.integrated_scope_access(*ARGS)


def per_call_allocs(args, n=8):
    out = []
    for _ in range(n):
        b = rtsys.get_allocation_stats()
        acwe.integrated_scope_access(*args)
        a = rtsys.get_allocation_stats()
        out.append(int(a.alloc - b.alloc))
    return out


result["admitted_allocs"] = per_call_allocs(ARGS)

wgt_bad = wgt_a.copy()
wgt_bad[3] = -1.0  # refusal class 10 (negative cost); same typing
ARGS_BAD = (oti, otw, ptr_a, nbr_a, wgt_bad, flg_a, dti, dtw, d_w,
            0.05, 0.0, 5.0, 0.919, "logistic", np.array([1.0, 0.6, 0.3]),
            cutoff)
acwe.integrated_scope_access(*ARGS_BAD)  # warm the (same) specialization
result["refused_allocs"] = per_call_allocs(ARGS_BAD)

with open(out_json, "w") as fh:
    json.dump(result, fh, indent=2)
print("NRT_CHILD_OK", out_json)
