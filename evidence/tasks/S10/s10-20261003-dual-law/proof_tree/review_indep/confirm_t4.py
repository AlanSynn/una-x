"""Confirm the T4/reconstruction defect and both repairs.

Repro: shared arenas, epoch k touches node 0; epoch k+1 does NOT touch
node 0. Probe reconstruction copies the stale label. Two proof-faithful
repairs (stamp==epoch filter; per-epoch touched-suffix) both fix it.
"""
import importlib.util
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location(
    's10_probe', os.path.join(HERE, '..', 'probe_sparse_projection_tiny.py'))
P = importlib.util.module_from_spec(spec)
spec.loader.exec_module(P)
import urban_network_analysis.kernels.scope_access as sa

ptr, adj, w = P.csr(4, [(0, 1), (1, 2), (2, 0), (0, 2), (1, 3)],
                    [0.5, 0.25, 0.75, 1.25, 0.1])
fl = P.flags_all(adj.shape[0])
co = np.float64(1.5)

# --- reproduction with the probe's own Kp_schedule
arenas = P.fresh_arenas(4)
o1, w1 = P.ot(0, 1, 0.05, 0.9)   # touches 0 (and more)
sa_ref1 = sa.scope_search_schedule(o1, w1, ptr, adj, w, fl, co)
g1, _, _ = P.Kp_schedule(o1, w1, ptr, adj, w, fl, co, arenas=arenas,
                         epoch=1)
o2, w2 = P.ot(1, 2, 0.15, 0.8)   # node 0 NOT touched this epoch
sa_ref2 = sa.scope_search_schedule(o2, w2, ptr, adj, w, fl, co)
g2, _, _ = P.Kp_schedule(o2, w2, ptr, adj, w, fl, co, arenas=arenas,
                         epoch=2)
print('epoch1 probe-vs-oracle :', P.bit_eq(g1, sa_ref1))
print('epoch2 probe-vs-oracle :', P.bit_eq(g2, sa_ref2),
      '(stale node0:', float(g2[0]), 'vs sentinel', float(sa_ref2[0]), ')')
print('touched list (cumulative):', list(arenas[2]))

# --- proof-faithful reconstruction A: stamp==epoch filter over touched


def kp_repair_filter(o, wv, ptr_, adj_, wts_, flag_, cutoff, arenas,
                     epoch):
    label, stamps, touched = arenas
    arena, ev, capped = P.trace_schedule(
        o, wv, ptr_, adj_, wts_, flag_, cutoff, stamp=True,
        arenas=arenas, epoch=epoch)
    sentinel = (np.ones(1, dtype=arena.dtype) + cutoff)[0]
    dense = np.full(arena.shape[0], sentinel, dtype=arena.dtype)
    if touched:
        idx = np.array([v for v in touched if stamps[v] == epoch],
                       dtype=np.int64)
        if idx.size:
            dense[idx] = arena[idx]
    return dense


# --- proof-faithful reconstruction B: per-epoch touched suffix (count
#     snapshot taken BEFORE the origin runs, exactly SPEC/proof "count")
def kp_repair_suffix(o, wv, ptr_, adj_, wts_, flag_, cutoff, arenas,
                     epoch):
    label, stamps, touched = arenas
    n0 = len(touched)
    arena, ev, capped = P.trace_schedule(
        o, wv, ptr_, adj_, wts_, flag_, cutoff, stamp=True,
        arenas=arenas, epoch=epoch)
    sentinel = (np.ones(1, dtype=arena.dtype) + cutoff)[0]
    dense = np.full(arena.shape[0], sentinel, dtype=arena.dtype)
    if len(touched) > n0:
        idx = np.array(touched[n0:], dtype=np.int64)
        dense[idx] = arena[idx]
    return dense


arenas = P.fresh_arenas(4)
a1_ = kp_repair_filter(o1, w1, ptr, adj, w, fl, co, arenas, 1)
b1_ = kp_repair_suffix(o1, w1, ptr, adj, w, fl, co, arenas, 1)
a2_ = kp_repair_filter(o2, w2, ptr, adj, w, fl, co, arenas, 2)
b2_ = kp_repair_suffix(o2, w2, ptr, adj, w, fl, co, arenas, 2)
print('repairA ep1 vs oracle:', P.bit_eq(a1_, sa_ref1),
      '| ep2:', P.bit_eq(a2_, sa_ref2))
print('repairB ep1 vs oracle:', P.bit_eq(b1_, sa_ref1),
      '| ep2:', P.bit_eq(b2_, sa_ref2))

# --- full H10-style sequence under both repairs, unique epochs,
#     including a mid-origin interrupt continued on the same arenas
class Trap(np.ndarray):
    budget = [2]

    def __getitem__(self, idx):
        if isinstance(idx, (int, np.integer)):
            Trap.budget[0] -= 1
            if Trap.budget[0] < 0:
                raise RuntimeError('mid-origin interrupt')
        return super().__getitem__(idx)


origins = [(0, 1, 0.05, 0.9), (1, 2, 0.15, 0.8), (2, 3, 0.25, 0.7),
           (3, 0, 0.35, 0.6), (0, 2, 0.45, 0.5)]
for name, fn in (('filter', kp_repair_filter),
                 ('suffix', kp_repair_suffix)):
    arenas = P.fresh_arenas(4)
    arenas[1][:] = 97
    arenas[0][:] = -1.0
    bad = []
    ep = 0
    for k, (s, e, w0, ww) in enumerate(origins):
        o, wv = P.ot(s, e, w0, ww)
        if k == 2:
            ep += 1
            Trap.budget[0] = 2
            try:
                fn(o, wv, ptr, adj, w, fl, co, arenas, ep)
                bad.append(f'e{ep}: no interrupt')
            except RuntimeError:
                pass
        ep += 1
        ref = sa.scope_search_schedule(o, wv, ptr, adj, w, fl, co)
        got = fn(o, wv, ptr, adj, w, fl, co, arenas, ep)
        if not P.bit_eq(got, ref):
            bad.append(f'e{ep}')
    print(f'H10 with repair {name}:', 'ALL OK' if not bad else f'FAIL {bad}')
