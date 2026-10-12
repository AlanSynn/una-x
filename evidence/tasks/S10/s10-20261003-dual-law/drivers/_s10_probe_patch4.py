"""Probe patch 4 — dual comparison laws + NaN-canonical events + pumping
E5 graph + staged-A1 production authority.

Findings this encodes (both machine-verified before this patch):
1. The portable oracle matches the STAGED A1 kernel (_a1_scope_search)
   bitwise on NaN-seed cases (fast comparison law holds on production
   routes). The COMPACT fallback kernel instead follows IEEE-ordered
   gates on NaN seeds — a pre-existing oracle-vs-compact divergence
   (counterexample preserved separately), NOT caused by the projection.
2. Event tuples containing NaN compare unequal to themselves under IEEE
   tuple equality — the probe must canonicalize NaN in events.
3. E5's original graph had out-degree 1 everywhere: the leaf push gate
   blocked every push, so no pump was possible. The law needs a gated
   cycle with out-degree >= 2.
"""
import ast

P = ("campaigns/una_10x/evidence/proofs/S10_sparse_projection/"
     "probe_sparse_projection_tiny.py")
src = open(P).read()

PATCHES = [
    # imports: staged A1 kernel (production-route bitwise authority)
    ("""from urban_network_analysis.Engines.AccessibilityWElevation import (  # noqa: E402
    compact_vector_node_view_scope)""",
     """from urban_network_analysis.Engines.AccessibilityWElevation import (  # noqa: E402
    compact_vector_node_view_scope)
from urban_network_analysis.Engines._large_access_scratch import (  # noqa: E402
    _a1_scope_search)""", 1),
    # trace_schedule: parameterize the comparison law
    ("""def trace_schedule(o_idx, o_w, ptr, adj, wts, flag, cutoff,
                   d_count=0, domain="a3", stamp=False, arenas=None,
                   epoch=1, pop_cap=None):""",
     """def trace_schedule(o_idx, o_w, ptr, adj, wts, flag, cutoff,
                   d_count=0, domain="a3", stamp=False, arenas=None,
                   epoch=1, pop_cap=None, fast=True):""", 1),
    ("""        def L(v):
            return label[v]

        def W(v, c):
            label[v] = c""",
     """        def L(v):
            return label[v]

        def W(v, c):
            label[v] = c

        lt = fast_lt if fast else (lambda a, b: a < b)
        le = fast_le if fast else (lambda a, b: a <= b)""", 1),
    ("""    if fast_lt(w1, cutoff):
        _heap_push(queue, (w1, o_end))
    if fast_lt(w0, cutoff):
        _heap_push(queue, (w0, o_start))""",
     """    if (fast_lt if fast else (lambda a, b: a < b))(w1, cutoff):
        _heap_push(queue, (w1, o_end))
    if (fast_lt if fast else (lambda a, b: a < b))(w0, cutoff):
        _heap_push(queue, (w0, o_start))""", 1),
    ("""            if fast_le(c, cutoff) and fast_lt(c, L(v)):
                elig.append((sp + i, c))""",
     """            if le(c, cutoff) and lt(c, L(v)):
                elig.append((sp + i, c))""", 1),
    # Kp: pass the law through
    ("""def Kp_schedule(o_idx, o_w, ptr, adj, wts, flag, cutoff, d_count=0,
                domain="a3", arenas=None, epoch=1, pop_cap=None,
                raise_after_pops=None):""",
     """def Kp_schedule(o_idx, o_w, ptr, adj, wts, flag, cutoff, d_count=0,
                domain="a3", arenas=None, epoch=1, pop_cap=None,
                raise_after_pops=None, fast=True):""", 1),
    ("""    arena, ev, capped = trace_schedule(
        o_idx, o_w, ptr, adj, wts, flag, cutoff, d_count=d_count,
        domain=domain, stamp=True, arenas=arenas, epoch=epoch,
        pop_cap=pop_cap)""",
     """    arena, ev, capped = trace_schedule(
        o_idx, o_w, ptr, adj, wts, flag, cutoff, d_count=d_count,
        domain=domain, stamp=True, arenas=arenas, epoch=epoch,
        pop_cap=pop_cap, fast=fast)""", 1),
]
for old, new, want in PATCHES:
    got = src.count(old)
    assert got == want, (old[:70], "count", got, "want", want)
    src = src.replace(old, new)

# run_all: dual-law comparison matrix
old_run_all = src[src.index("def run_all("):src.index("def bit_eq(")]
new_run_all = '''def canon_ev(ev):
    """NaN-canonical event list for IEEE tuple comparison."""
    out = []
    for e in ev:
        out.append(tuple("nan" if isinstance(x, float) and x != x
                         else x for x in e))
    return out


def run_all(o_idx, o_w, ptr, adj, wts, cutoff, d_count=0, domain="a3"):
    """Dual-law matrix:
    fast pair:  Kp(fast) vs trace(fast)  -> events + vector bitwise,
                vs ORACLE (fast-law authority) and vs the STAGED A1
                kernel (production route) bitwise.
    ieee pair:  trace(ieee) vs COMPACT kernel bitwise (mirror validity),
                Kp(ieee) vs trace(ieee) events + vector bitwise.
    """
    n = ptr.shape[0] - 1
    fl = flags_all(adj.shape[0])
    lab_k = scope_search_schedule(o_idx, o_w, ptr, adj, wts, fl,
                                  cutoff, d_count=d_count, domain=domain)
    lab_c = compact_vector_node_view_scope(
        o_idx, o_w, ptr, adj, wts, fl, cutoff,
        d_count if domain == "a1" else 0)[0]
    deg = int((ptr[1:] - ptr[:-1]).max()) if ptr.shape[0] > 1 else 1
    off = np.empty(deg + 1, dtype=np.int64)
    wt = np.empty(deg + 1, dtype=np.float64)
    lab_a1 = _a1_scope_search(o_idx, o_w, ptr, adj, wts, fl, cutoff,
                              d_count if domain == "a1" else 0,
                              off, wt)[0]
    arenas_f = fresh_arenas(n + (d_count if domain == "a1" else 0))
    dense_pf, ev_pf, _ = Kp_schedule(o_idx, o_w, ptr, adj, wts, fl,
                                     cutoff, d_count=d_count,
                                     domain=domain, arenas=arenas_f,
                                     fast=True)
    arenas_i = fresh_arenas(n + (d_count if domain == "a1" else 0))
    dense_pi, ev_pi, _ = Kp_schedule(o_idx, o_w, ptr, adj, wts, fl,
                                     cutoff, d_count=d_count,
                                     domain=domain, arenas=arenas_i,
                                     fast=False)
    return {
        "oracle": lab_k, "compact": lab_c, "a1": lab_a1,
        "kp_fast": dense_pf, "kp_ieee": dense_pi,
        "ev_fast": canon_ev(ev_pf), "ev_ieee": canon_ev(ev_pi),
        "arenas_fast": arenas_f, "arenas_ieee": arenas_i,
    }


def events_of(o_idx, o_w, ptr, adj, wts, cutoff, fast, d_count=0,
              domain="a3"):
    """Dense-mirror event trace under the given law (for parity)."""
    ev = trace_schedule(o_idx, o_w, ptr, adj, wts,
                        flags_all(adj.shape[0]), cutoff,
                        d_count=d_count, domain=domain,
                        fast=fast)[1]
    return canon_ev(ev)


'''
src = src.replace(old_run_all, new_run_all)

# case(): dual-law acceptance
old_case = src[src.index("def case("):src.index("# ==============")]
new_case = '''def case(tag, o_idx, o_w, ptr, adj, wts, cutoff, d_count=0,
         domain="a3"):
    r = run_all(o_idx, o_w, ptr, adj, wts, cutoff, d_count=d_count,
                domain=domain)
    ev_tf = events_of(o_idx, o_w, ptr, adj, wts, cutoff, True,
                      d_count=d_count, domain=domain)
    ev_ti = events_of(o_idx, o_w, ptr, adj, wts, cutoff, False,
                      d_count=d_count, domain=domain)
    ok = True
    # fast-law pair: oracle + staged A1 are bitwise authorities
    if not bit_eq(r["kp_fast"], r["oracle"]):
        fail(tag, "Kp(fast) vs oracle bitwise"); ok = False
    if not bit_eq(r["kp_fast"], r["a1"]):
        fail(tag, "Kp(fast) vs staged A1 bitwise"); ok = False
    if r["ev_fast"] != ev_tf:
        d = next((i for i, (x, y) in enumerate(zip(r["ev_fast"], ev_tf))
                  if x != y), min(len(r["ev_fast"]), len(ev_tf)))
        fail(tag, f"fast events diverge at {d}: "
             f"{r['ev_fast'][d:d+1]} vs {ev_tf[d:d+1]}"); ok = False
    # ieee pair: the COMPACT kernel is the bitwise authority; its
    # NaN-seed divergence vs the fast law is pre-existing and recorded
    if not bit_eq(r["kp_ieee"], r["compact"]):
        fail(tag, "Kp(ieee) vs compact bitwise"); ok = False
    if r["ev_ieee"] != ev_ti:
        d = next((i for i, (x, y) in enumerate(zip(r["ev_ieee"], ev_ti))
                  if x != y), min(len(r["ev_ieee"]), len(ev_ti)))
        fail(tag, f"ieee events diverge at {d}: "
             f"{r['ev_ieee'][d:d+1]} vs {ev_ti[d:d+1]}"); ok = False
    # mirror validity of the oracle itself vs staged A1 (production)
    if not bit_eq(r["oracle"], r["a1"]):
        CASES.setdefault("oracle_vs_a1_divergences", []).append(tag)
    if ok:
        CASES["passed"] += 1
    return ok


'''
src = src.replace(old_case, new_case)

# E5: pumping graph (out-degree >= 2 on the cycle), fast law only
old_e5 = src[src.index("def e5_nan_pump("):src.index("# =======")]
new_e5 = '''def e5_nan_pump():
    # NaN on both seeds with a gated cycle (out-degree >= 2): the heap
    # pumps (pinned NONTERMINATION law, staged-engine characterization).
    ptr, adj, w = csr(3, [(0, 1), (0, 1), (1, 2), (1, 2), (2, 0), (2, 0)],
                      [0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
    o, wv = ot(0, 1, np.nan, np.nan)
    cap = 400
    fl = flags_all(adj.shape[0])
    _, ev1, cap1 = trace_schedule(o, wv, ptr, adj, w, fl, 1.5,
                                  pop_cap=cap, fast=True)
    arenas = fresh_arenas(ptr.shape[0] - 1)
    _, ev2, cap2 = Kp_schedule(o, wv, ptr, adj, w, fl, 1.5,
                               arenas=arenas, pop_cap=cap, fast=True)
    if not (cap1 and cap2):
        fail("e5/pump", f"cap flags {cap1} {cap2}")
    elif canon_ev(ev1) != canon_ev(ev2):
        fail("e5/pump", "pop sequences diverge under cap")
    else:
        CASES["passed"] += 1
    # the ieee law pair TERMINATES on the same case (nan never passes
    # the ieee gates) — pin that too (compact-kernel behavior class)
    _, _, cap3 = trace_schedule(o, wv, ptr, adj, w, fl, 1.5, fast=False)
    if cap3:
        fail("e5/ieee_terminates", "ieee pair unexpectedly capped")
    else:
        CASES["passed"] += 1
    CASES["counts"]["e5_pump_pops"] = cap


'''
src = src.replace(old_e5, new_e5)

# E10: Kp signature gained nothing, but touched lives on the arena —
# keep existing access (arenas[2]) intact.

ast.parse(src)
open(P, "w").write(src)
print("dual-law probe assembled")
