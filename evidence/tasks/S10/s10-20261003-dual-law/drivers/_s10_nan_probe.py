"""Investigate the two bitwise FAILs: e4/warts/0.1/nan and e4/warts/nan/0.5.
Compare oracle vs engine label arrays bit-for-bit with hex dumps."""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "src"))

from urban_network_analysis.kernels.scope_access import (
    scope_search_schedule)
from urban_network_analysis.Engines.AccessibilityWElevation import (
    compact_vector_node_view_scope)

ptr = np.array([0, 1, 2, 3], dtype=np.int64)
adj = np.array([1, 2, 0], dtype=np.int64)
wts = np.array([0.5, 0.5, 1.0], dtype=np.float64)
flag = np.ones(3, dtype=np.bool_)

for (w0, w1) in ((0.1, float("nan")), (float("nan"), 0.5)):
    o_idx = np.array([0, 2], dtype=np.int64)
    o_w = np.array([w0, w1], dtype=np.float64)
    lab_k = scope_search_schedule(o_idx, o_w, ptr, adj, wts, flag, 1.5)
    lab_e = compact_vector_node_view_scope(
        o_idx, o_w, ptr, adj, wts, flag, 1.5, 0)[0]
    same = bool(np.array_equal(lab_k.view(np.uint64),
                               lab_e.view(np.uint64)))
    print(f"seed=({w0!r},{w1!r}) bitwise_equal={same}")
    print("  oracle:", [float(x) for x in lab_k],
          [hex(v) for v in lab_k.view(np.uint64)])
    print("  engine:", [float(x) for x in lab_e],
          [hex(v) for v in lab_e.view(np.uint64)])
