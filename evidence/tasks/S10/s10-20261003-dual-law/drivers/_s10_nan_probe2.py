"""Extend the NaN investigation to the STAGED kernel (_a1_scope_search,
the production A1 route) — does IT follow the fast law (NaN passes
gates) or the IEEE behavior the compact kernel just showed?"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "src"))

from urban_network_analysis.kernels.scope_access import (
    scope_search_schedule)
from urban_network_analysis.Engines.AccessibilityWElevation import (
    compact_vector_node_view_scope)
from urban_network_analysis.Engines._large_access_scratch import (
    _a1_scope_search)

ptr = np.array([0, 1, 2, 3], dtype=np.int64)
adj = np.array([1, 2, 0], dtype=np.int64)
wts = np.array([0.5, 0.5, 1.0], dtype=np.float64)
flag = np.ones(3, dtype=np.bool_)
max_deg = 1

for (w0, w1) in ((0.1, float("nan")), (float("nan"), 0.5)):
    o_idx = np.array([0, 2], dtype=np.int64)
    o_w = np.array([w0, w1], dtype=np.float64)
    lab_k = scope_search_schedule(o_idx, o_w, ptr, adj, wts, flag, 1.5)
    lab_c = compact_vector_node_view_scope(
        o_idx, o_w, ptr, adj, wts, flag, 1.5, 0)[0]
    off = np.empty(max_deg + 1, dtype=np.int64)
    wt = np.empty(max_deg + 1, dtype=np.float64)
    lab_a1 = _a1_scope_search(o_idx, o_w, ptr, adj, wts, flag, 1.5, 0,
                              off, wt)
    print(f"seed=({w0!r},{w1!r})")
    print("  oracle :", [float(x) for x in lab_k])
    print("  compact:", [float(x) for x in lab_c])
    print("  a1     :", [float(x) for x in lab_a1[0]])
