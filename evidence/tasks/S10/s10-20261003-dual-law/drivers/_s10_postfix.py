"""S10 post-review repairs (binding conditions from review_indep):
M1: reconstruction gathers the stamp==epoch set, never the cumulative
    touched list (T4); E7b pins touched-then-untouched + dirty-arena
    continuation after a mid-origin interrupt.
M2: E11 lockstep boundary sweep — full label arrays compared at EVERY
    pop boundary (the proof's chronological-event claim, bounded set).
E12: A1 reduced-domain tail (d_count>0) across the authority matrix.
F3 warning: NaN ARC weights pump the frozen oracle + staged A1
    indefinitely — never feed them to case() uncapped.
"""
import ast
import os

HERE = os.path.dirname(os.path.abspath(__file__))
P = os.path.join(HERE, "campaigns", "una_10x", "evidence", "proofs",
                 "S10_sparse_projection",
                 "probe_sparse_projection_tiny.py")
src = open(P).read()

PATCHES = []

# --- M1: stamp-filtered reconstruction (both Kp bodies)
PATCHES.append((
    """    if touched:
        idx = np.array(touched, dtype=np.int64)
        dense[idx] = arena[idx]""",
    """    idx = np.array([v for v in touched if stamps[v] == epoch], dtype=np.int64)
    dense[idx] = arena[idx]""", 2))

# --- F3 docstring warning
PATCHES.append((
    """Run under the alan interpreter; writes sparse_admission.json.
\"\"\"""",
    """Run under the alan interpreter; writes sparse_admission.json.

WARNING (finding F3): NaN ARC weights pump the frozen oracle and the
staged A1 kernel indefinitely (compact/IEEE terminates). Never feed
NaN arc weights to case() without a pop cap; the committed cases
never do. NaN SEED cases are capped mirrors only (e5).
\"\"\"""", 1))

# --- new E-functions + main call list
OLD_MAIN = """def main():
    e1_exhaustive()
    e2_e6_boundary()
    e5_nan_pump()
    e7_epochs()
    e8_sentinel()
    e9_concurrent()
    e10_sparse_touch()
"""
NEW_MAIN = '''def e7b_touched_gap():
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
'''
PATCHES.append((OLD_MAIN, NEW_MAIN, 1))

for old, new, want in PATCHES:
    got = src.count(old)
    assert got == want, ("anchor count", got, "want", want,
                         old[:80])
    src = src.replace(old, new)

ast.parse(src)
open(P, "w").write(src)
print("post-repair patches applied:", len(PATCHES))
