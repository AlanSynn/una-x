"""Isolate the h5b hang: which sub-search of run_all nonterminates?"""
import threading
import traceback

import numpy as np

import importlib.util
import os
HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location(
    's10_probe', os.path.join(HERE, '..', 'probe_sparse_projection_tiny.py'))
P = importlib.util.module_from_spec(spec)
spec.loader.exec_module(P)

import urban_network_analysis.kernels.scope_access as sa

NAN = float('nan')
ptr, adj, w = P.csr(2, [(0, 1), (0, 1), (1, 0), (1, 0)], [NAN, 0.5, NAN, 0.5])
o, wv = P.ot(0, 1, NAN, 0.5)
fl = P.flags_all(adj.shape[0])
co = np.float64(1.5)


def watch(name, fn):
    box = {}

    def go():
        try:
            box['r'] = fn()
        except Exception as e:  # noqa: BLE001
            box['r'] = ('EXC', repr(e))
    t = threading.Thread(target=go, daemon=True)
    t.start()
    t.join(timeout=20)
    if t.is_alive():
        print(f'{name}: STILL RUNNING after 20s (nonterminating)')
        return None
    print(f'{name}: done -> {str(box.get("r"))[:120]}')
    return box.get('r')


watch('oracle', lambda: sa.scope_search_schedule(o, wv, ptr, adj, w, fl, co))
watch('compact', lambda: P.compact_vector_node_view_scope(
    o, wv, ptr, adj, w, fl, co, 0)[0])
deg = int((ptr[1:] - ptr[:-1]).max())
off = np.empty(deg + 1, dtype=np.int64)
wt = np.empty(deg + 1, dtype=np.float64)
watch('a1', lambda: P._a1_scope_search(o, wv, ptr, adj, w, fl, co, 0, off,
                                       wt)[0])
watch('trace_fast', lambda: P.trace_schedule(o, wv, ptr, adj, w, fl, co,
                                             stamp=False))
watch('Kp_fast', lambda: P.Kp_schedule(o, wv, ptr, adj, w, fl, co,
                                       arenas=P.fresh_arenas(2)))
watch('trace_ieee', lambda: P.trace_ieee(o, wv, ptr, adj, w, fl, co,
                                         stamp=False))
watch('Kp_ieee', lambda: P.kp_ieee(o, wv, ptr, adj, w, fl, co,
                                   arenas=P.fresh_arenas(2)))
print('isolation done')
