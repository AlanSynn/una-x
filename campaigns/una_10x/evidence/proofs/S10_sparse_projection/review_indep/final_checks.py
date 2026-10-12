"""Final confirmation: interrupt + dirty-arena continuation with repair A,
Trap correctly wired; plus mechanical mirror-vs-oracle source diff."""
import ast
import importlib.util
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location(
    's10_probe', os.path.join(HERE, '..', 'probe_sparse_projection_tiny.py'))
P = importlib.util.module_from_spec(spec)
spec.loader.exec_module(P)
import urban_network_analysis.kernels.scope_access as sa

# ---------- part 1: interrupt + continuation under repair A
ptr, adj, w = P.csr(4, [(0, 1), (1, 2), (2, 0), (0, 2), (1, 3)],
                    [0.5, 0.25, 0.75, 1.25, 0.1])
fl = P.flags_all(adj.shape[0])
co = np.float64(1.5)


class Trap(np.ndarray):
    budget = [2]

    def __getitem__(self, idx):
        if isinstance(idx, (int, np.integer)):
            Trap.budget[0] -= 1
            if Trap.budget[0] < 0:
                raise RuntimeError('mid-origin interrupt')
        return super().__getitem__(idx)


P_arenas = None


def kp_repair(o, wv, wts_, epoch):
    label, stamps, touched = P_arenas
    arena, ev, capped = P.trace_schedule(o, wv, ptr, adj, wts_, fl, co,
                                         stamp=True, arenas=P_arenas,
                                         epoch=epoch)
    sentinel = (np.ones(1, dtype=arena.dtype) + co)[0]
    dense = np.full(arena.shape[0], sentinel, dtype=arena.dtype)
    if touched:
        idx = np.array([v for v in touched if stamps[v] == epoch],
                       dtype=np.int64)
        if idx.size:
            dense[idx] = arena[idx]
    return dense


origins = [(0, 1, 0.05, 0.9), (1, 2, 0.15, 0.8), (2, 3, 0.25, 0.7),
           (3, 0, 0.35, 0.6), (0, 2, 0.45, 0.5)]
P_arenas = P.fresh_arenas(4)
P_arenas[1][:] = 97
P_arenas[0][:] = -1.0
bad = []
ep = 0
interrupted = False
for k, (s, e, w0, ww) in enumerate(origins):
    o, wv = P.ot(s, e, w0, ww)
    use = w.view(Trap) if k == 2 else w
    if k == 2:
        ep += 1
        Trap.budget[0] = 2
        try:
            kp_repair(o, wv, use, ep)
            bad.append(f'e{ep}: interrupt did not fire')
        except RuntimeError:
            interrupted = True
    ep += 1
    ref = sa.scope_search_schedule(o, wv, ptr, adj, w, fl, co)
    if k == 2:
        Trap.budget[0] = 10 ** 9  # continuation must run to completion
    got = kp_repair(o, wv, use, ep)
    if not P.bit_eq(got, ref):
        bad.append(f'e{ep} diverged')
print(f'repair-A H10: interrupted_fired={interrupted} '
      f'bad={bad if bad else "NONE"}')

# ---------- part 2: source-skeleton diff, mirror(stamp=False) vs oracle
probe_path = os.path.join(HERE, '..', 'probe_sparse_projection_tiny.py')
oracle_path = os.path.join(os.path.dirname(sa.__file__), 'scope_access.py')


def norm_fn(path, name):
    tree = ast.parse(open(path).read())
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f'{name} not found in {path}')


mf = norm_fn(probe_path, 'trace_schedule')
oc = norm_fn(oracle_path, 'scope_search_schedule')


def skeleton(fn):
    kinds = []
    for node in ast.walk(fn):
        if isinstance(node, (ast.If, ast.For, ast.While, ast.Return,
                             ast.Assign, ast.Expr, ast.FunctionDef)):
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
                f = node.value.func
                if isinstance(f, ast.Attribute) and f.attr == 'append' \
                        and isinstance(f.value, ast.Name) \
                        and f.value.id == 'ev':
                    continue  # pure event observation
            kinds.append(type(node).__name__)
    return kinds


sk_m = skeleton(mf)
sk_o = skeleton(oc)
print('skeleton mirror:', sk_m)
print('skeleton oracle:', sk_o)
print('skeletons equal (events excluded):', sk_m == sk_o)
# the eligibility expression must be textually identical under the fast law
import re
m_src = ast.get_source_segment(open(probe_path).read(), mf)
o_src = ast.get_source_segment(open(oracle_path).read(), oc)
gate_m = re.search(r'if fast_le\(c, cutoff\) and fast_lt\(c, L\(v\)\)', m_src)
gate_o = re.search(r'if fast_le\(neighbor_weight, cutoff\) and '
                   r'fast_lt\(neighbor_weight, labels\[neighbor_node\]\)',
                   o_src)
push_o = re.search(r'if fast_lt\(o_idx_end_weight, cutoff\)', o_src)
push_m = re.search(r'if fast_lt\(w1, cutoff\)', m_src)
leaf_o = re.search(r'adjacency_pointer\[neighbor_node \+ 1\] - '
                   r'adjacency_pointer\[neighbor_node\] > 1', o_src)
leaf_m = re.search(r'ptr\[v \+ 1\] - ptr\[v\] > 1', m_src)
print('eligibility gate present mirror/oracle:',
      bool(gate_m), bool(gate_o))
print('seed push gate (strict <, end then start) mirror/oracle:',
      bool(push_m), bool(push_o),
      '| end-first order mirror:', m_src.index('fast_lt(w1') <
      m_src.index('fast_lt(w0'),
      '| oracle:', o_src.index('fast_lt(o_idx_end_weight') <
      o_src.index('fast_lt(o_idx_start_weight'))
print('leaf degree gate mirror/oracle:', bool(leaf_m), bool(leaf_o))
case_src = open(probe_path).read()
assert "bit_eq(r['mf'], r['oracle'])" in case_src, 'mirror check missing'
assert "chk(bit_eq(r['kpf'], r['oracle'])" in case_src, 'kpf check missing'
print('per-case mirror-vs-oracle + Kp-vs-oracle bitwise checks: PRESENT')
