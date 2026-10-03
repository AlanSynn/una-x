"""S10 smoke: 3 targeted cases + e5 pump, then dump CASES."""
import importlib.util
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROBE = os.path.join(
    HERE, "campaigns", "una_10x", "evidence", "proofs",
    "S10_sparse_projection", "probe_sparse_projection_tiny.py")
spec = importlib.util.spec_from_file_location("s10probe", PROBE)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

import numpy as np  # noqa: E402

# case 1: finite 4-node, degree-2 node (exercises A1 scratch sizing)
ptr, adj, w = m.csr(4, [(0, 1), (0, 2), (1, 3), (2, 3), (3, 0)],
                    [0.5, 0.75, 0.25, 0.5, 1.25])
o, wv = m.ot(0, 3, 0.1, 0.9)
m.case("smoke/finite", o, wv, ptr, adj, w, np.float64(1.5))

# case 2: NaN seed end, 3-node (law-divergent domain; fast triple holds)
ptr, adj, w = m.csr(3, [(0, 1), (1, 2), (0, 2)], [0.5, 0.5, 1.0])
o, wv = m.ot(0, 2, 0.1, np.nan)
m.case("smoke/nan_seed_end", o, wv, ptr, adj, w, np.float64(1.5))

# case 3: NaN seed start
o, wv = m.ot(0, 2, np.nan, 0.5)
m.case("smoke/nan_seed_start", o, wv, ptr, adj, w, np.float64(1.5))

# case 4: same-node seeds with NaN
ptr, adj, w = m.csr(2, [(0, 1), (1, 0)], [0.5, 0.5])
o, wv = m.ot(0, 0, np.nan, 0.5)
m.case("smoke/same_node_nan", o, wv, ptr, adj, w, np.float64(1.5))

m.e5_nan_pump()

print("passed:", m.CASES["passed"])
print("failed:", m.CASES["failed"])
print("counts:", m.CASES["counts"])
