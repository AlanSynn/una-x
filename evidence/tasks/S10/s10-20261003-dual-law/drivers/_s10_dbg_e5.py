"""Debug e5/ieee_projection failure after the M1 fix."""
import importlib.util
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location(
    "s10probe", os.path.join(
        HERE, "campaigns", "una_10x", "evidence", "proofs",
        "S10_sparse_projection", "probe_sparse_projection_tiny.py"))
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

ptr, adj, w = m.csr(3, [(0, 1), (0, 1), (1, 2), (1, 2), (2, 0), (2, 0)],
                    [0.5] * 6)
o, wv = m.ot(0, 1, np.nan, np.nan)
fl = m.flags_all(adj.shape[0])

vi, evi, _ = m.kp_ieee(o, wv, ptr, adj, w, fl, 1.5,
                       arenas=m.fresh_arenas(ptr.shape[0] - 1))
mi, evm, _ = m.trace_ieee(o, wv, ptr, adj, w, fl, 1.5, stamp=True,
                          arenas=m.fresh_arenas(ptr.shape[0] - 1))
print("vi :", [float(x) for x in vi], [hex(v) for v in vi.view(np.uint64)])
print("mi :", [float(x) for x in mi], [hex(v) for v in mi.view(np.uint64)])
print("bit_eq:", m.bit_eq(vi, mi))
print("evi:", m.canon_ev(evi))
print("evm:", m.canon_ev(evm))
print("ev equal:", m.canon_ev(evi) == m.canon_ev(evm))
