"""Debug H10 epoch attribution: which epoch diverges and why."""
import importlib.util
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location(
    's10_probe', os.path.join(HERE, '..', 'probe_sparse_projection_tiny.py'))
P = importlib.util.module_from_spec(spec)
spec.loader.exec_module(P)
import urban_network_analysis.kernels.scope_access as sa

ptrA, adjA, wA = P.csr(4, [(0, 1), (1, 2), (2, 0), (0, 2), (1, 3)],
                       [0.5, 0.25, 0.75, 1.25, 0.1])
flA = P.flags_all(adjA.shape[0])
origins = [(0, 1, 0.05, 0.9), (1, 2, 0.15, 0.8), (2, 3, 0.25, 0.7),
           (3, 0, 0.35, 0.6), (0, 2, 0.45, 0.5)]
arenas = P.fresh_arenas(4)
arenas[1][:] = 97
arenas[0][:] = -1.0
for k, (s, e, w0, w1) in enumerate(origins):
    ep = k + 1
    o, wv = P.ot(s, e, w0, w1)
    ref = sa.scope_search_schedule(o, wv, ptrA, adjA, wA, flA,
                                   np.float64(1.5))
    got, _, _ = P.Kp_schedule(o, wv, ptrA, adjA, wA, flA, np.float64(1.5),
                              arenas=arenas, epoch=ep)
    same = P.bit_eq(got, ref)
    print(f'k={k} epoch={ep} origin=({s},{e}) match={same}')
    if not same:
        print('  ref :', [float(x) for x in ref], [hex(v) for v in
                                                   ref.view(np.uint64)])
        print('  got :', [float(x) for x in got], [hex(v) for v in
                                                   got.view(np.uint64)])
        print('  stamps after:', list(arenas[1]))
        print('  touched:', list(arenas[2]))
print('labels after:', [float(x) for x in arenas[0]])
