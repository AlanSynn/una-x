"""Prove the E7 coverage hole: e7's fixture touches all nodes every
epoch (masks the reconstruction defect); a minimal touched-then-
untouched sequence on the SAME shared arenas diverges with the probe's
own Kp_schedule and passes with the epoch-filtered reconstruction."""
import importlib.util
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location(
    's10_probe', os.path.join(HERE, '..', 'probe_sparse_projection_tiny.py'))
P = importlib.util.module_from_spec(spec)
spec.loader.exec_module(P)
import urban_network_analysis.kernels.scope_access as sa

# --- the exact e7 fixture
ptr, adj, w = P.csr(3, [(0, 1), (1, 2), (2, 0), (0, 2), (1, 0)],
                    [0.5, 0.25, 0.75, 1.25, 0.1])
fl = P.flags_all(adj.shape[0])
co = np.float64(1.5)
origins = [P.ot(k % 3, (k + 1) % 3, 0.05 * k, 0.9 - 0.05 * k)
           for k in range(9)]
arenas = P.fresh_arenas(3)
masking = True
seen = None
for k, (o, wv) in enumerate(origins):
    n0 = len(arenas[2])
    P.Kp_schedule(o, wv, ptr, adj, w, fl, co, arenas=arenas, epoch=k + 1)
    new = set(arenas[2][n0:])
    if seen is not None and not new >= seen:
        masking = False
    seen = new
print('e7 fixture touches EVERY node every epoch:', masking,
      '| final stamps:', [int(x) for x in arenas[1]])

# --- minimal touched-then-untouched sequence on shared arenas:
#     epoch 1 seeds node 0 and 1 (0 reachable); epoch 2 seeds 1 and 2 so
#     that node 0 is NOT touched this epoch (verify via stamps after).
ptr2, adj2, w2 = P.csr(3, [(0, 1), (1, 2), (2, 0)], [0.5, 0.5, 0.25])
fl2 = P.flags_all(adj2.shape[0])
arenas2 = P.fresh_arenas(3)
o1, wv1 = P.ot(0, 1, 0.1, 0.9)
P.Kp_schedule(o1, wv1, ptr2, adj2, w2, fl2, co, arenas=arenas2, epoch=1)
o2, wv2 = P.ot(1, 2, 0.1, 0.9)   # closure: 1 -> 2 -> 0 (cycle! touches 0)
ref2 = sa.scope_search_schedule(o2, wv2, ptr2, adj2, w2, fl2, co)
got2, _, _ = P.Kp_schedule(o2, wv2, ptr2, adj2, w2, fl2, co,
                           arenas=arenas2, epoch=2)
print('cycle fixture (0 reachable in ep2) probe-vs-oracle:',
      P.bit_eq(got2, ref2), '| stamps:', [int(x) for x in arenas2[1]])

# To make node 0 untouched in epoch 2 the closure must exclude it: use a
# DAG 1->2 (no cycle through 0).
ptr3, adj3, w3 = P.csr(4, [(0, 1), (1, 2), (2, 3)], [0.5, 0.5, 0.25])
fl3 = P.flags_all(adj3.shape[0])
arenas3 = P.fresh_arenas(4)
o1, wv1 = P.ot(0, 1, 0.1, 0.9)   # touches {0,1,2,3}
P.Kp_schedule(o1, wv1, ptr3, adj3, w3, fl3, co, arenas=arenas3, epoch=1)
o2, wv2 = P.ot(2, 2, 0.1, 0.9)   # touches {2,3} only; node 0,1 untouched
ref2 = sa.scope_search_schedule(o2, wv2, ptr3, adj3, w3, fl3, co)
got2, _, _ = P.Kp_schedule(o2, wv2, ptr3, adj3, w3, fl3, co,
                           arenas=arenas3, epoch=2)
print('touched-then-untouched (nodes 0,1): probe Kp-vs-oracle:',
      P.bit_eq(got2, ref2),
      '| got:', [float(x) for x in got2], 'ref:', [float(x) for x in ref2])
