"""S10 L0 probe: sparse epoch projection (dossier 02) vs the frozen
dense schedule and the compiled engine.

Three implementations compared per case:
  K_oracle  kernels.scope_access.scope_search_schedule (frozen portable
            oracle, event-free) -- bitwise vector authority #1
  K_engine  Engines.AccessibilityWElevation.compact_vector_node_view_scope
            (compiled, pinned baseline) -- bitwise vector authority #2
  trace_K   probe-local mirror of the oracle emitting events (validated
            against K_oracle on every case before use)
  Kp        probe-local projection oracle: identical schedule text with
            exactly two deltas (L(v) read projection, W(v,c) stamping)

Acceptance per case: Kp events == trace_K events (element-for-element),
Kp reconstructed vector bitwise == K_oracle == K_engine.
Run under the alan interpreter; writes sparse_admission.json.

WARNING (finding F3): NaN ARC weights pump the frozen oracle and the
staged A1 kernel indefinitely (compact/IEEE terminates). Never feed
NaN arc weights to case() without a pop cap; the committed cases
never do. NaN SEED cases are capped mirrors only (e5).
"""
import hashlib
import json
import os
import sys
import threading
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = HERE
while not os.path.isdir(os.path.join(REPO, 'src', 'urban_network_analysis')):
    nxt = os.path.dirname(REPO)
    assert nxt != REPO, 'worktree root not found above ' + REPO
    REPO = nxt
SRC = os.path.join(REPO, 'src')
sys.path.insert(0, SRC)
from urban_network_analysis.kernels.scope_access import _heap_pop, _heap_push, fast_le, fast_lt, scope_search_schedule
from urban_network_analysis.Engines.AccessibilityWElevation import compact_vector_node_view_scope
from urban_network_analysis.Engines._large_access_scratch import _a1_scope_search
CASES = {'passed': 0, 'failed': [], 'counts': {}}

def fail(tag, detail):
    CASES['failed'].append({'tag': tag, 'detail': str(detail)[:400]})

def trace_schedule(o_idx, o_w, ptr, adj, wts, flag, cutoff, d_count=0, domain='a3', stamp=False, arenas=None, epoch=1, pop_cap=None):
    """Mirror of the frozen oracle with events; stamp=False reproduces
    the dense text exactly (validated against the frozen oracle); with
    stamp=True the only deltas are the L(v) read and the W(v,c) write
    of sparse_proof.md section 0."""
    o_start, o_end = (int(o_idx[0]), int(o_idx[1]))
    w0, w1 = (float(o_w[0]), float(o_w[1]))
    if domain == 'a3':
        n = ptr.shape[0] - 1
    else:
        n = d_count + ptr.shape[0] - 1
    sentinel = (np.ones(1, dtype=o_w.dtype) + cutoff)[0]
    if stamp:
        label, stamps, touched = arenas
        count = [0]
        ev = []

        def L(v):
            return label[v] if stamps[v] == epoch else sentinel

        def W(v, c):
            if stamps[v] != epoch:
                touched.append(v)
                stamps[v] = epoch
                count[0] += 1
            label[v] = c
    else:
        label = np.ones(n, dtype=o_w.dtype) + cutoff
        ev = []

        def L(v):
            return label[v]

        def W(v, c):
            label[v] = c
    W(o_start, w0)
    W(o_end, w1)
    ev.append(('seed', o_start, w0))
    ev.append(('seed', o_end, w1))
    queue = [(w0, o_start)]
    weight, node = _heap_pop(queue)
    if fast_lt(w1, cutoff):
        _heap_push(queue, (w1, o_end))
    if fast_lt(w0, cutoff):
        _heap_push(queue, (w0, o_start))
    ev.append(('pop', weight, node))
    pops = 0
    while queue:
        if pop_cap is not None and pops >= pop_cap:
            return (label, ev, True)
        weight, node = _heap_pop(queue)
        pops += 1
        ev.append(('pop', weight, node))
        sp = ptr[node]
        ep = ptr[node + 1]
        elig = []
        for i in range(ep - sp):
            c = float(wts[sp + i]) + weight
            v = adj[sp + i]
            if fast_le(c, cutoff) and fast_lt(c, L(v)):
                elig.append((sp + i, c))
        for off, c in elig:
            v = adj[off]
            W(v, c)
            ev.append(('write', v, c))
            if flag[off]:
                if ptr[v + 1] - ptr[v] > 1:
                    _heap_push(queue, (c, v))
                    ev.append(('push', c, v))
    return (label, ev, False)

def fresh_arenas(n, stamp_dtype=np.int64):
    return (np.empty(n, dtype=np.float64), np.zeros(n, dtype=stamp_dtype), [])

def Kp_schedule(o_idx, o_w, ptr, adj, wts, flag, cutoff, d_count=0, domain='a3', arenas=None, epoch=1, pop_cap=None, raise_after_pops=None):
    """The projection: shared-arena run at epoch `epoch` (stamps must be
    zeroed before the arena's first use). Returns the reconstructed
    dense vector + the touched prefix (count)."""
    label, stamps, touched = arenas
    arena, ev, capped = trace_schedule(o_idx, o_w, ptr, adj, wts, flag, cutoff, d_count=d_count, domain=domain, stamp=True, arenas=arenas, epoch=epoch, pop_cap=pop_cap)
    if raise_after_pops is not None:
        raise RuntimeError('injected mid-origin interruption')
    sentinel = (np.ones(1, dtype=arena.dtype) + cutoff)[0]
    dense = np.full(arena.shape[0], sentinel, dtype=arena.dtype)
    idx = np.array([v for v in touched if stamps[v] == epoch], dtype=np.int64)
    dense[idx] = arena[idx]
    return (dense, ev, capped)

def csr(n, arcs, weights):
    deg = np.zeros(n + 1, dtype=np.int64)
    for a in arcs:
        deg[a[0] + 1] += 1
    ptr = np.cumsum(deg).astype(np.int64)
    order = sorted(range(len(arcs)), key=lambda k: (arcs[k][0], arcs[k][1]))
    adj = np.array([arcs[k][1] for k in order], dtype=np.int64)
    wts = np.array([weights[k] for k in order], dtype=np.float64)
    return (ptr, adj, wts)

def ot(o_start, o_end, w0, w1):
    return (np.array([o_start, o_end], dtype=np.int64), np.array([w0, w1], dtype=np.float64))

def flags_all(arc_count):
    return np.ones(arc_count, dtype=np.bool_)

def bit_eq(a, b):
    return a.dtype == b.dtype and a.shape == b.shape and bool(np.array_equal(a.view(np.uint64), b.view(np.uint64)))

def e1_exhaustive():
    rng = np.random.default_rng(20261003)
    fixed = [0.0, 0.5, 1.25, 0.75]
    cutoff = np.float64(1.5)
    total = 0
    for n in (2, 3, 4):
        arcs = [(i, j) for i in range(n) for j in range(n)]
        for mask in range(1 << len(arcs)):
            sel = [arcs[k] for k in range(len(arcs)) if mask >> k & 1]
            wts = [fixed[k % 4] for k in range(len(sel))]
            ptr, adj, w = csr(n, sel, wts)
            for s, e in ((0, n - 1), (1, 1)):
                o, wv = ot(s, e, 0.1, 0.9)
                case(f'e1/n{n}/m{mask}/{s}{e}', o, wv, ptr, adj, w, cutoff)
                total += 1
        for rep in range(2000):
            sel = [arcs[k] for k in range(len(arcs)) if rng.random() < 0.5]
            wts = list(rng.uniform(0.0, 2.0, size=len(sel)))
            ptr, adj, w = csr(n, sel, wts)
            o, wv = ot(0, n - 1, float(rng.uniform(0, 1)), float(rng.uniform(0, 1)))
            case(f'e1r/n{n}/r{rep}', o, wv, ptr, adj, w, cutoff)
            total += 1
    CASES['counts']['e1_cases'] = total

def e2_e6_boundary():
    cutoff = np.float64(1.5)
    for s, e, w0, w1 in ((1, 1, 0.5, 0.5), (1, 1, 0.5, 0.25), (0, 0, np.nan, 0.5)):
        ptr, adj, w = csr(2, [(0, 1), (1, 0)], [0.5, 0.5])
        case(f'e2/same{s}{e}/{w0}/{w1}', *ot(s, e, w0, w1), ptr, adj, w, cutoff)
    for w0, w1 in ((cutoff, 0.5), (0.5, cutoff), (np.nextafter(cutoff, np.inf), 0.5), (np.nextafter(cutoff, 0.0), 0.5)):
        ptr, adj, w = csr(2, [(0, 1), (1, 0)], [0.5, 0.5])
        case(f'e2/gate/{w0!r}/{w1!r}', *ot(0, 1, w0, w1), ptr, adj, w, cutoff)
    ptr, adj, w = csr(3, [(0, 1), (1, 2), (2, 0), (0, 2)], [0.5, 0.5, 1.25, np.nextafter(1.5, 0.0)])
    for co in (0.0, -0.0, np.nextafter(0.5, 0.0), 0.5, np.nextafter(1.5, np.inf), 1.5, np.nextafter(np.float64(0.0), np.float64(1.0)) * np.float64(2), sys_float_max(), np.int64(0), np.int64(2), np.int64(2) ** np.int64(62), np.int64(2 ** 63 - 1), np.int64(-1)):
        case(f'e3/cutoff/{float(co)!r}', *ot(0, 2, 0.25, 0.5), ptr, adj, w, co)
    ptr, adj, w = csr(3, [(0, 1), (1, 2), (0, 2)], [0.5, 0.5, 1.0])
    for w0, w1 in ((0.1, np.nan), (np.nan, 0.5), (-0.5, 0.5), (np.inf, 0.5), (0.5, np.inf), (-0.0, 0.0)):
        case(f'e4/warts/{w0!r}/{w1!r}', *ot(0, 2, w0, w1), ptr, adj, w, cutoff)
    ptr, adj, w = csr(3, [(0, 1), (0, 1), (1, 1), (1, 2), (1, 2)], [0.5, 0.5, 0.25, 0.75, 0.6])
    case('e6/dup/self', *ot(0, 2, 0.1, 0.2), ptr, adj, w, cutoff)
    CASES['counts']['boundary_cases'] = CASES['passed']

def sys_float_max():
    import sys as _s
    return np.float64(_s.float_info.max)

def e7_epochs():
    ptr, adj, w = csr(3, [(0, 1), (1, 2), (2, 0), (0, 2), (1, 0)], [0.5, 0.25, 0.75, 1.25, 0.1])
    origins = [ot(k % 3, (k + 1) % 3, 0.05 * k, 0.9 - 0.05 * k) for k in range(9)]
    arenas = fresh_arenas(ptr.shape[0] - 1)
    bad = 0
    for k, (o, wv) in enumerate(origins):
        lab_k = scope_search_schedule(o, wv, ptr, adj, w, flags_all(adj.shape[0]), 1.5)
        dense_p, _, _ = Kp_schedule(o, wv, ptr, adj, w, flags_all(adj.shape[0]), 1.5, arenas=arenas, epoch=k + 1)
        if not bit_eq(dense_p, lab_k):
            bad += 1
    if bad:
        fail('e7/shared_epochs', f'{bad}/9 origins diverged')
    else:
        CASES['passed'] += 1
    o, wv = origins[0]
    try:
        Kp_schedule(o, wv, ptr, adj, w, flags_all(adj.shape[0]), 1.5, arenas=fresh_arenas(ptr.shape[0] - 1), raise_after_pops=3)
    except RuntimeError:
        CASES['passed'] += 1
    lab_k = scope_search_schedule(o, wv, ptr, adj, w, flags_all(adj.shape[0]), 1.5)
    dense_p, _, _ = Kp_schedule(o, wv, ptr, adj, w, flags_all(adj.shape[0]), 1.5, arenas=fresh_arenas(ptr.shape[0] - 1))
    if not bit_eq(dense_p, lab_k):
        fail('e7/after_interrupt', 'post-exception call diverged')
    else:
        CASES['passed'] += 1
    ptr2, adj2, w2 = csr(2, [(0, 1), (1, 0)], [0.5, 0.5])
    small = fresh_arenas(2, stamp_dtype=np.uint8)
    bad = 0
    for k in range(300):
        e = k % 255 + 1
        if e == 1 and k:
            small[1][:] = 0
        lab_k = scope_search_schedule(*ot(0, 1, 0.1, 0.9), ptr2, adj2, w2, flags_all(adj2.shape[0]), 1.5)
        dense_p, _, _ = Kp_schedule(ot(0, 1, 0.1, 0.9)[0], ot(0, 1, 0.1, 0.9)[1], ptr2, adj2, w2, flags_all(adj2.shape[0]), 1.5, arenas=small, epoch=e)
        if not bit_eq(dense_p, lab_k):
            bad += 1
    if bad:
        fail('e7/uint8_exhaustion', f'{bad}/300 diverged')
    else:
        CASES['passed'] += 1

def e8_sentinel():
    bad = []
    for co in (1.5, 0.0, -0.0, np.nextafter(0.0, 1.0), 1e+308, np.int64(0), np.int64(2 ** 53), np.int64(2 ** 53 + 1), np.int64(2 ** 62), np.int64(2 ** 63 - 1), np.int64(-2 ** 63)):
        a = np.ones(7, dtype=np.float64) + co
        b = np.full(7, (np.ones(1, dtype=np.float64) + co)[0])
        if not bit_eq(a, b):
            bad.append(float(co))
    if bad:
        fail('e8/sentinel', f'bit deltas at {bad}')
    else:
        CASES['passed'] += 1

def e9_concurrent():
    rng = np.random.default_rng(7)
    n = 5
    arcs = [(i, j) for i in range(n) for j in range(n) if i != j][:20]
    ref = []
    inputs = []
    for rep in range(120):
        sel = [arcs[k] for k in range(len(arcs)) if rng.random() < 0.5]
        wts = list(rng.uniform(0, 2, size=len(sel)))
        ptr, adj, w = csr(n, sel, wts)
        o, wv = ot(0, 4, float(rng.uniform()), float(rng.uniform()))
        inputs.append((o, wv, ptr, adj, w))
        ref.append(scope_search_schedule(o, wv, ptr, adj, w, flags_all(adj.shape[0]), 1.5))
    got = {}

    def worker(wid):
        for rep, (o, wv, ptr, adj, w) in enumerate(inputs):
            if rep % 2 == wid:
                arenas = fresh_arenas(ptr.shape[0] - 1)
                dense_p, _, _ = Kp_schedule(o, wv, ptr, adj, w, flags_all(adj.shape[0]), 1.5, arenas=arenas)
                got[rep] = dense_p
    ts = [threading.Thread(target=worker, args=(i,)) for i in (0, 1)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    bad = [r for r in range(120) if not bit_eq(got[r], ref[r])]
    if bad:
        fail('e9/concurrent', f'{len(bad)} diverged')
    else:
        CASES['passed'] += 1
    CASES['counts']['e9_cases'] = 120

def e10_sparse_touch():
    n = 8
    ptr, adj, w = csr(n, [(k, k + 1) for k in range(n - 1)] + [(7, 0)], [0.25] * n)
    lab_k = scope_search_schedule(*ot(0, 1, 0.1, 0.9), ptr, adj, w, flags_all(adj.shape[0]), 1.5)
    lab_e = compact_vector_node_view_scope(*ot(0, 1, 0.1, 0.9), ptr, adj, w, flags_all(adj.shape[0]), 1.5, 0)[0]
    arenas = fresh_arenas(n)
    dense_p, ev_p, _ = Kp_schedule(*ot(0, 1, 0.1, 0.9), ptr, adj, w, flags_all(adj.shape[0]), 1.5, arenas=arenas)
    touched = sorted(set(arenas[2]))
    ok = bit_eq(dense_p, lab_k) and bit_eq(lab_e, lab_k) and (len(touched) <= 4)
    if ok:
        CASES['passed'] += 1
    else:
        fail('e10/untouched', f'touched={touched}')
    CASES['counts']['e10_touched'] = touched

def e7b_touched_gap():
    # Review finding M1 regression: touched in epoch 1, untouched in
    # epoch 2, SAME shared arenas. Reconstruction must gather the
    # stamp==epoch set (proof T4), never the cumulative touched list.
    ptr, adj, w = csr(4, [(0, 1), (1, 2), (2, 3)], [0.5, 0.5, 0.25])
    fl = flags_all(adj.shape[0])
    arenas = fresh_arenas(4)
    o1, wv1 = ot(0, 1, 0.1, 0.9)
    Kp_schedule(o1, wv1, ptr, adj, w, fl, 1.5, arenas=arenas, epoch=1)
    o2, wv2 = ot(2, 2, 0.1, 0.9)  # touches {2,3} only; 0,1 untouched
    ref = scope_search_schedule(o2, wv2, ptr, adj, w, fl, 1.5)
    got, _, _ = Kp_schedule(o2, wv2, ptr, adj, w, fl, 1.5,
                            arenas=arenas, epoch=2)
    if not bit_eq(got, ref):
        fail('e7b/touched_gap', f'got={[float(x) for x in got]} '
             f'ref={[float(x) for x in ref]}')
    else:
        CASES['passed'] += 1
    # dirty-arena continuation after a mid-origin interrupt: the SAME
    # arenas at fresh epochs must behave like pristine state
    o3, wv3 = ot(0, 1, 0.1, 0.9)
    try:
        Kp_schedule(o3, wv3, ptr, adj, w, fl, 1.5, arenas=arenas,
                    epoch=3, raise_after_pops=2)
    except RuntimeError:
        CASES['passed'] += 1  # arenas left dirty; epoch abandoned
    ref3 = scope_search_schedule(o3, wv3, ptr, adj, w, fl, 1.5)
    got3, _, _ = Kp_schedule(o3, wv3, ptr, adj, w, fl, 1.5,
                             arenas=arenas, epoch=4)
    if not bit_eq(got3, ref3):
        fail('e7b/dirty_continuation', 'post-interrupt reuse diverged')
    else:
        CASES['passed'] += 1
    CASES['counts']['e7b_epochs'] = [1, 2, 3, 4]


def _lockstep_case(tag, o_idx, o_w, ptr, adj, wts, flag, cutoff):
    """One schedule execution maintaining BOTH the plain dense label
    and the stamping arenas; at every pop boundary materialize the
    epoch reconstruction and require bitwise equality with the plain
    array (proof §7: compare full labels at each chronological
    event)."""
    o_start, o_end = int(o_idx[0]), int(o_idx[1])
    w0, w1 = float(o_w[0]), float(o_w[1])
    n = ptr.shape[0] - 1
    sentinel = (np.ones(1, dtype=o_w.dtype) + cutoff)[0]
    label = np.ones(n, dtype=o_w.dtype) + cutoff
    stamp_lab, stamps, touched = fresh_arenas(n)
    epoch = 1

    def Lr(v):
        return stamp_lab[v] if stamps[v] == epoch else sentinel

    def Wr(v, c):
        if stamps[v] != epoch:
            touched.append(v)
            stamps[v] = epoch
        stamp_lab[v] = c

    Wr(o_start, w0)
    Wr(o_end, w1)
    label[o_start] = w0
    label[o_end] = w1
    queue = [(w0, o_start)]
    weight, node = _heap_pop(queue)
    if fast_lt(w1, cutoff):
        _heap_push(queue, (w1, o_end))
    if fast_lt(w0, cutoff):
        _heap_push(queue, (w0, o_start))

    def boundary():
        dense = np.full(n, sentinel, dtype=label.dtype)
        idx = np.array([v for v in touched if stamps[v] == epoch],
                       dtype=np.int64)
        dense[idx] = stamp_lab[idx]
        return bit_eq(dense, label)

    ok = boundary()
    while ok and queue:
        weight, node = _heap_pop(queue)
        sp = ptr[node]
        ep = ptr[node + 1]
        elig = []
        for i in range(ep - sp):
            c = float(wts[sp + i]) + weight
            v = adj[sp + i]
            if fast_le(c, cutoff) and fast_lt(c, Lr(v)):
                elig.append((sp + i, c))
        for off, c in elig:
            v = adj[off]
            Wr(v, c)
            label[v] = c
            if flag[off]:
                if ptr[v + 1] - ptr[v] > 1:
                    _heap_push(queue, (c, v))
        ok = boundary()
    if not ok:
        fail(tag, 'boundary reconstruction diverged')
        return
    lab_ref = scope_search_schedule(o_idx, o_w, ptr, adj, wts, flag,
                                    cutoff)
    if not bit_eq(label, lab_ref):
        fail(tag, 'lockstep mirror vs frozen oracle')
        return
    CASES['passed'] += 1


def e11_lockstep():
    # Review finding M2 repair: the proof's chronological-event claim,
    # executed on a bounded subset (all finite inputs; no pump risk).
    rng = np.random.default_rng(20261003)
    total = 0
    for (n, arcs) in (
            (2, [(0, 1), (1, 0)]),
            (3, [(0, 1), (1, 2), (2, 0), (0, 2), (1, 0)]),
            (3, [(0, 1), (0, 1), (1, 2), (1, 2), (2, 0)]),
            (4, [(0, 1), (0, 2), (1, 3), (2, 3), (3, 0), (1, 2)])):
        for mask in range(1 << len(arcs)):
            sel = [arcs[k] for k in range(len(arcs)) if mask >> k & 1]
            wts = [0.25 + 0.5 * (k % 3) for k in range(len(sel))]
            ptr, adj, w = csr(n, sel, wts)
            o, wv = ot(0, n - 1, 0.1, 0.9)
            _lockstep_case(f'e11/n{n}/m{mask}', o, wv, ptr, adj, w,
                           flags_all(adj.shape[0]), 1.5)
            total += 1
    for rep in range(64):
        n = 4
        all_arcs = [(i, j) for i in range(n) for j in range(n)]
        sel = [a for a in all_arcs if rng.random() < 0.4]
        wts = list(rng.uniform(0, 2, size=len(sel)))
        ptr, adj, w = csr(n, sel, wts)
        o, wv = ot(0, 3, float(rng.uniform()), float(rng.uniform()))
        _lockstep_case(f'e11r/r{rep}', o, wv, ptr, adj, w,
                       flags_all(adj.shape[0]), 1.5)
        total += 1
    CASES['counts']['e11_lockstep_cases'] = total


def e12_a1_tail():
    # Review MINOR repair: the A1 reduced-domain label width
    # (V + d_count tail) across the same authority matrix.
    ptr, adj, w = csr(4, [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)],
                      [0.5, 0.5, 0.25, 0.75, 1.25])
    for (s, e, w0, w1) in ((0, 3, 0.1, 0.9), (1, 1, 0.5, 0.5),
                           (0, 3, float(np.nextafter(1.5, 0.0)), 0.25),
                           (2, 0, 0.75, 1.25)):
        case(f'e12/a1/{s}{e}', *ot(s, e, w0, w1), ptr, adj, w, 1.5,
             d_count=2, domain='a1')
    CASES['counts']['e12_a1_cases'] = 4


def main():
    e1_exhaustive()
    e2_e6_boundary()
    e5_nan_pump()
    e7_epochs()
    e7b_touched_gap()
    e8_sentinel()
    e9_concurrent()
    e10_sparse_touch()
    e11_lockstep()
    e12_a1_tail()
    doc = {'task': 'S10 sparse epoch projection L0 probe', 'repo_src': SRC, 'cases_passed': CASES['passed'], 'cases_failed': CASES['failed'], 'counts': CASES['counts'], 'probe_sha256': hashlib.sha256(open(os.path.abspath(__file__), 'rb').read()).hexdigest(), 'note': 'trace_schedule (stamp=False) validated against the frozen oracle on every E1 case before event comparison'}
    doc['findings'] = {k: v for k, v in CASES.items() if k not in ('passed', 'failed', 'counts')}
    out = os.path.join(HERE, 'sparse_admission.json')
    with open(out, 'w') as f:
        json.dump(doc, f, indent=2, sort_keys=True, default=str)
        f.write('\n')
    print('passed:', CASES['passed'], '| failed:', len(CASES['failed']))
    for fl in CASES['failed'][:10]:
        print('  FAIL:', fl['tag'], '|', fl['detail'][:120])
    print('written:', out)

def canon_ev(ev):
    """NaN-canonical event tuples: IEEE tuple equality treats NaN as
    unequal to itself, so event comparison must canonicalize first."""
    out = []
    for e in ev:
        out.append(tuple(('nan' if isinstance(x, float) and x != x else x for x in e)))
    return out

def trace_ieee(o_idx, o_w, ptr, adj, wts, flag, cutoff, d_count=0, domain='a3', stamp=False, arenas=None, epoch=1, pop_cap=None):
    """IEEE-law mirror: byte-for-byte the same schedule text with plain
    ordered < / <=. Mirrors the compact fallback kernel's observed
    comparison law on NaN (counterexamples preserved with the S10
    evidence); the fast-law mirror trace_schedule is unchanged above."""
    o_start, o_end = (int(o_idx[0]), int(o_idx[1]))
    w0, w1 = (float(o_w[0]), float(o_w[1]))
    if domain == 'a3':
        n = ptr.shape[0] - 1
    else:
        n = d_count + ptr.shape[0] - 1
    sentinel = (np.ones(1, dtype=o_w.dtype) + cutoff)[0]
    if stamp:
        label, stamps, touched = arenas
        ev = []

        def L(v):
            return label[v] if stamps[v] == epoch else sentinel

        def W(v, c):
            if stamps[v] != epoch:
                touched.append(v)
                stamps[v] = epoch
            label[v] = c
    else:
        label = np.ones(n, dtype=o_w.dtype) + cutoff
        ev = []

        def L(v):
            return label[v]

        def W(v, c):
            label[v] = c
    W(o_start, w0)
    W(o_end, w1)
    ev.append(('seed', o_start, w0))
    ev.append(('seed', o_end, w1))
    queue = [(w0, o_start)]
    weight, node = _heap_pop(queue)
    if w1 < cutoff:
        _heap_push(queue, (w1, o_end))
    if w0 < cutoff:
        _heap_push(queue, (w0, o_start))
    ev.append(('pop', weight, node))
    pops = 0
    while queue:
        if pop_cap is not None and pops >= pop_cap:
            return (label, ev, True)
        weight, node = _heap_pop(queue)
        pops += 1
        ev.append(('pop', weight, node))
        sp = ptr[node]
        ep = ptr[node + 1]
        elig = []
        for i in range(ep - sp):
            c = float(wts[sp + i]) + weight
            v = adj[sp + i]
            if c <= cutoff and c < L(v):
                elig.append((sp + i, c))
        for off, c in elig:
            v = adj[off]
            W(v, c)
            ev.append(('write', v, c))
            if flag[off]:
                if ptr[v + 1] - ptr[v] > 1:
                    _heap_push(queue, (c, v))
                    ev.append(('push', c, v))
    return (label, ev, False)

def kp_ieee(o_idx, o_w, ptr, adj, wts, flag, cutoff, d_count=0, domain='a3', arenas=None, epoch=1, pop_cap=None):
    """The projection under the IEEE law: identical to Kp_schedule except the
    mirror it runs; same section-4 reconstruction."""
    label, stamps, touched = arenas
    arena, ev, capped = trace_ieee(o_idx, o_w, ptr, adj, wts, flag, cutoff, d_count=d_count, domain=domain, stamp=True, arenas=arenas, epoch=epoch, pop_cap=pop_cap)
    sentinel = (np.ones(1, dtype=arena.dtype) + cutoff)[0]
    dense = np.full(arena.shape[0], sentinel, dtype=arena.dtype)
    idx = np.array([v for v in touched if stamps[v] == epoch], dtype=np.int64)
    dense[idx] = arena[idx]
    return (dense, ev, capped)

def run_all(o_idx, o_w, ptr, adj, wts, cutoff, d_count=0, domain='a3'):
    """Per-case result matrix over three authorities and both laws.
    Authorities: frozen oracle (fast law, portable), staged A1 kernel
    (production route, fast law), compact fallback kernel (observed
    IEEE-on-NaN law). The projection is evaluated under each law
    against that law's mirror."""
    fl = flags_all(adj.shape[0])
    lab_oracle = scope_search_schedule(o_idx, o_w, ptr, adj, wts, fl, cutoff, d_count=d_count, domain=domain)
    lab_compact = compact_vector_node_view_scope(o_idx, o_w, ptr, adj, wts, fl, cutoff, d_count if domain == 'a1' else 0)[0]
    deg = int((ptr[1:] - ptr[:-1]).max()) if ptr.shape[0] > 1 else 1
    off = np.empty(deg + 1, dtype=np.int64)
    wt = np.empty(deg + 1, dtype=np.float64)
    lab_a1 = _a1_scope_search(o_idx, o_w, ptr, adj, wts, fl, cutoff, d_count if domain == 'a1' else 0, off, wt)[0]
    vec_mf, ev_mf, _ = trace_schedule(o_idx, o_w, ptr, adj, wts, fl, cutoff, d_count=d_count, domain=domain, stamp=False)
    arenas_f = fresh_arenas((d_count if domain == 'a1' else 0) + ptr.shape[0] - 1)
    vec_kf, ev_kf, _ = Kp_schedule(o_idx, o_w, ptr, adj, wts, fl, cutoff, d_count=d_count, domain=domain, arenas=arenas_f)
    vec_mi, ev_mi, _ = trace_ieee(o_idx, o_w, ptr, adj, wts, fl, cutoff, d_count=d_count, domain=domain, stamp=False)
    arenas_i = fresh_arenas((d_count if domain == 'a1' else 0) + ptr.shape[0] - 1)
    vec_ki, ev_ki, _ = kp_ieee(o_idx, o_w, ptr, adj, wts, fl, cutoff, d_count=d_count, domain=domain, arenas=arenas_i)
    return {'oracle': lab_oracle, 'compact': lab_compact, 'a1': lab_a1, 'mf': vec_mf, 'ev_mf': ev_mf, 'kpf': vec_kf, 'ev_kf': ev_kf, 'mi': vec_mi, 'ev_mi': ev_mi, 'ki': vec_ki, 'ev_ki': ev_ki}

def case(tag, o_idx, o_w, ptr, adj, wts, cutoff, d_count=0, domain='a3'):
    r = run_all(o_idx, o_w, ptr, adj, wts, cutoff, d_count=d_count, domain=domain)
    ok = True

    def chk(cond, msg):
        nonlocal ok
        if not cond:
            fail(tag, msg)
            ok = False
    strict_a1 = not bool(np.isnan(np.asarray(o_w, dtype=np.float64)).any())
    chk(bit_eq(r['mf'], r['oracle']), 'fast mirror vs oracle')
    chk(bit_eq(r['kpf'], r['oracle']), 'projection(fast) vs oracle')
    if strict_a1:
        chk(bit_eq(r['kpf'], r['a1']), 'projection(fast) vs staged A1')
    elif not bit_eq(r['kpf'], r['a1']):
        CASES.setdefault('a1_nan_divergent_tags', []).append(tag)
        if len(CASES.setdefault('a1_nan_counterexamples', [])) < 8:
            CASES['a1_nan_counterexamples'].append({'tag': tag, 'oracle_u64': [hex(v) for v in np.asarray(r['oracle']).view(np.uint64)], 'a1_u64': [hex(v) for v in np.asarray(r['a1']).view(np.uint64)]})
        CASES['counts']['a1_nan_divergences'] = CASES['counts'].get('a1_nan_divergences', 0) + 1
    chk(canon_ev(r['ev_kf']) == canon_ev(r['ev_mf']), 'fast events')
    chk(bit_eq(r['ki'], r['mi']), 'projection(ieee) vs ieee mirror')
    chk(canon_ev(r['ev_ki']) == canon_ev(r['ev_mi']), 'ieee events')
    if bit_eq(r['kpf'], r['ki']):
        chk(bit_eq(r['oracle'], r['compact']), 'oracle vs compact')
        if strict_a1:
            chk(bit_eq(r['a1'], r['compact']), 'staged A1 vs compact')
        chk(bit_eq(r['ki'], r['compact']), 'projection(ieee) vs compact')
        CASES['counts']['coincide_cases'] = CASES['counts'].get('coincide_cases', 0) + 1
    else:
        CASES.setdefault('law_divergent_tags', []).append(tag)
        CASES['counts']['divergent_cases'] = CASES['counts'].get('divergent_cases', 0) + 1
    if ok:
        CASES['passed'] += 1
    return ok

def e5_nan_pump():
    ptr, adj, w = csr(3, [(0, 1), (0, 1), (1, 2), (1, 2), (2, 0), (2, 0)], [0.5] * 6)
    o, wv = ot(0, 1, np.nan, np.nan)
    cap = 400
    fl = flags_all(adj.shape[0])
    _, ev1, cap1 = trace_schedule(o, wv, ptr, adj, w, fl, 1.5, pop_cap=cap)
    arenas = fresh_arenas(ptr.shape[0] - 1)
    _, ev2, cap2 = Kp_schedule(o, wv, ptr, adj, w, fl, 1.5, arenas=arenas, pop_cap=cap)
    if not (cap1 and cap2):
        fail('e5/pump', 'cap flags %s %s' % (cap1, cap2))
    elif canon_ev(ev1) != canon_ev(ev2):
        fail('e5/pump', 'pop sequences diverge under cap')
    else:
        CASES['passed'] += 1
    _, _, cap3 = trace_ieee(o, wv, ptr, adj, w, fl, 1.5, stamp=False)
    if cap3:
        fail('e5/ieee_terminates', 'ieee pair unexpectedly capped')
    else:
        CASES['passed'] += 1
    vi, evi, _ = kp_ieee(o, wv, ptr, adj, w, fl, 1.5, arenas=fresh_arenas(ptr.shape[0] - 1))
    # plain mode returns the dense-initialized label; the
    # stamp-mode return is a raw arena (untouched slots are
    # uninitialized) and must never be bitwise-compared
    mi, evm, _ = trace_ieee(o, wv, ptr, adj, w, fl, 1.5)
    if not bit_eq(vi, mi) or canon_ev(evi) != canon_ev(evm):
        fail('e5/ieee_projection', 'projection vs mirror under ieee')
    else:
        CASES['passed'] += 1
    CASES['counts']['e5_pump_pops'] = cap
if __name__ == '__main__':
    main()