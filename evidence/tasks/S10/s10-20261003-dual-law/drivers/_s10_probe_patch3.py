"""Probe patch 3: implement the proof's section-4 reconstruction.
Kp must return np.full(n, S) with touched slots copied from the arena —
not the raw arena label (the previous run's bitwise FAILs were this
probe bug: uninitialized np.empty slots exposed). Touched records move
out of the event stream (kept on the arena for reconstruction/E10)."""
import ast

P = ("campaigns/una_10x/evidence/proofs/S10_sparse_projection/"
     "probe_sparse_projection_tiny.py")
src = open(P).read()

PATCHES = [
    # arenas gain a touched-list slot
    ("""def fresh_arenas(n, stamp_dtype=np.int64):
    return (np.empty(n, dtype=np.float64),
            np.zeros(n, dtype=stamp_dtype))""",
     """def fresh_arenas(n, stamp_dtype=np.int64):
    return (np.empty(n, dtype=np.float64),
            np.zeros(n, dtype=stamp_dtype),
            [])""", 1),
    # stamp branch: unpack touched, drop touched from events
    ("""    if stamp:
        label, stamps = arenas
        count = [0]
        ev = []

        def L(v):
            return label[v] if stamps[v] == epoch else sentinel

        def W(v, c):
            if stamps[v] != epoch:
                ev.append(("touched", v))
                stamps[v] = epoch
                count[0] += 1
            label[v] = c""",
     """    if stamp:
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
            label[v] = c""", 1),
    # Kp: reconstruct per proof section 4 before returning
    ("""    label, stamps = arenas
    dense, ev, capped = trace_schedule(
        o_idx, o_w, ptr, adj, wts, flag, cutoff, d_count=d_count,
        domain=domain, stamp=True, arenas=arenas, epoch=epoch,
        pop_cap=pop_cap)
    if raise_after_pops is not None:
        raise RuntimeError("injected mid-origin interruption")
    return dense, ev, capped""",
     """    label, stamps, touched = arenas
    arena, ev, capped = trace_schedule(
        o_idx, o_w, ptr, adj, wts, flag, cutoff, d_count=d_count,
        domain=domain, stamp=True, arenas=arenas, epoch=epoch,
        pop_cap=pop_cap)
    if raise_after_pops is not None:
        raise RuntimeError("injected mid-origin interruption")
    # proof section 4: materialize the public array from the private
    # arenas; untouched slots take the sentinel, touched slots copy
    sentinel = (np.ones(1, dtype=arena.dtype) + cutoff)[0]
    dense = np.full(arena.shape[0], sentinel, dtype=arena.dtype)
    if touched:
        idx = np.array(touched, dtype=np.int64)
        dense[idx] = arena[idx]
    return dense, ev, capped""", 1),
    # E10: touched from the arena, not the event stream
    ("""    touched = sorted({e[1] for e in ev_p if e[0] == "touched"})""",
     """    touched = sorted(set(arenas[2]))""", 1),
]
for old, new, want in PATCHES:
    got = src.count(old)
    assert got == want, (old[:70], "count", got, "want", want)
    src = src.replace(old, new)

ast.parse(src)
open(P, "w").write(src)
print("reconstruction patched:", len(PATCHES), "sites")
