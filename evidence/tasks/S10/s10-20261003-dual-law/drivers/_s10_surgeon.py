"""S10 probe surgeon — AST-based rewrite, token-flip-immune.

All identifiers are discovered from AST truth (difflib), never trusted
from hand-typed text. Replaces run_all / case / e5_nan_pump; adds
trace_ieee + kp_ieee (IEEE-law mirror pair) + canon_ev; imports the
staged A1 kernel as a third bitwise authority; adds the law-coincide
domain gate so the compact engine is compared only where the two
comparison laws coincide bitwise (the NaN-seed divergence becomes a
recorded finding, not a false failure).
"""
import ast
import difflib
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
PROBE = os.path.join(
    HERE, "campaigns", "una_10x", "evidence", "proofs",
    "S10_sparse_projection", "probe_sparse_projection_tiny.py")

# ---- repo root by walk-up (same rule as the probe itself)
REPO = HERE
while not os.path.isdir(os.path.join(REPO, "src", "urban_network_analysis")):
    nxt = os.path.dirname(REPO)
    assert nxt != REPO, "repo root not found above " + REPO
    REPO = nxt
SRC = os.path.join(REPO, "src")
PKG = os.path.join(SRC, "urban_network_analysis")

src = open(PROBE).read()
tree = ast.parse(src)

fnames = [n.name for n in tree.body if isinstance(n, ast.FunctionDef)]
assign_names = [t.id for n in tree.body if isinstance(n, ast.Assign)
                for t in n.targets if isinstance(t, ast.Name)]
import_aliases = {}  # local name -> imported symbol
for n in tree.body:
    if isinstance(n, ast.ImportFrom):
        for a in n.names:
            import_aliases[a.asname or a.name] = a.name


def disc(want, pool, cutoff=0.72):
    m = difflib.get_close_matches(want, pool, n=1, cutoff=cutoff)
    assert m, ("no discovery match", want, pool)
    return m[0]


NM = {}
for role, want in [
    ("TRACE", "trace_schedule"), ("KP", "Kp_schedule"),
    ("FRESH", "fresh_arenas"), ("FLAGS", "flags_all"),
    ("BITEQ", "bit_eq"), ("FAIL", "fail"), ("CSR", "csr"), ("OT", "ot"),
]:
    NM[role] = disc(want, fnames)
for role, want in [
    ("HEAPPOP", "_heap_pop"), ("HEAPPUSH", "_heap_push"),
    ("ORACLE", "scope_search_schedule"),
    ("COMPACT", "compact_vector_node_view_scope"),
]:
    NM[role] = disc(want, list(import_aliases.keys()))
NM["CASES"] = disc("CASES", assign_names)

fdef_of = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}


def params_of(fn_node):
    return [a.arg for a in fn_node.args.args]


def kw_of(params, want):
    m = difflib.get_close_matches(want, params, n=1, cutoff=0.55)
    assert m, ("no param match", want, params)
    return m[0]


TR_P = params_of(fdef_of[NM["TRACE"]])
KP_P = params_of(fdef_of[NM["KP"]])
TR = {w: kw_of(TR_P, w) for w in
      ("d_count", "domain", "stamp", "arenas", "epoch", "pop_cap")}
KP = {w: kw_of(KP_P, w) for w in ("d_count", "domain", "arenas",
                                  "epoch", "pop_cap")}

# ---- staged A1 kernel: resolve module + signature from AST truth
a1_symbol = "_a1_scope_search"
eng_dir = os.path.join(PKG, "Engines")
hits = [f for f in sorted(os.listdir(eng_dir))
        if f.endswith(".py")
        and "def " + a1_symbol in open(os.path.join(eng_dir, f)).read()]
assert len(hits) == 1, ("A1 def module not unique", hits)
a1_mod_stem = hits[0][:-3]
a1_tree = ast.parse(open(os.path.join(eng_dir, hits[0])).read())
a1_fn = None
for n in ast.walk(a1_tree):
    if isinstance(n, ast.FunctionDef) and n.name == a1_symbol:
        a1_fn = n
assert a1_fn is not None, "A1 def not found in " + hits[0]
A1_P = [a.arg for a in a1_fn.args.args]

# oracle param names (for kwargs) from the frozen kernel source
kern_dir = os.path.join(PKG, "kernels")
khits = [f for f in sorted(os.listdir(kern_dir))
         if f.endswith(".py")
         and "def " + NM["ORACLE"] in open(os.path.join(kern_dir, f)).read()]
assert len(khits) == 1, ("oracle module not unique", khits)
k_tree = ast.parse(open(os.path.join(kern_dir, khits[0])).read())
k_fn = None
for n in ast.walk(k_tree):
    if isinstance(n, ast.FunctionDef) and n.name == NM["ORACLE"]:
        k_fn = n
assert k_fn is not None
K_P = [a.arg for a in k_fn.args.args]
KO = {w: kw_of(K_P, w) for w in ("d_count", "domain")}

# ---- A1 argument order: positions proven by the executed nan-probe call
# (o_idx, o_w, ptr, adj, wts, flag, cutoff, d_count, off, wt); sanity-
# check the discovered names per slot instead of fuzzy-matching all.
assert len(A1_P) == 10, ("A1 arity", A1_P)
for slot, want in ((2, "pointer"), (3, "vector"), (4, "weights"),
                   (6, "cutoff"), (7, "d_count"), (8, "offset"),
                   (9, "weight")):
    assert want in A1_P[slot], ("A1 slot mismatch", slot, want, A1_P)
a1_args = ["o_idx", "o_w", "ptr", "adj", "wts", "fl", "cutoff",
           'd_count if domain == "a1" else 0', "off", "wt"]

SUB = {
    "@TRACE@": NM["TRACE"], "@KP@": NM["KP"], "@FRESH@": NM["FRESH"],
    "@FLAGS@": NM["FLAGS"], "@BITEQ@": NM["BITEQ"], "@FAIL@": NM["FAIL"],
    "@CSR@": NM["CSR"], "@OT@": NM["OT"], "@HEAPPOP@": NM["HEAPPOP"],
    "@HEAPPUSH@": NM["HEAPPUSH"], "@ORACLE@": NM["ORACLE"],
    "@COMPACT@": NM["COMPACT"], "@CASES@": NM["CASES"],
    "@A1MOD@": "urban_network_analysis.Engines." + a1_mod_stem,
    "@A1@": a1_symbol,
    # template already spells slots 0-6 positionally; join the tail only
    "@A1ARGS@": ", ".join(a1_args[7:]),
    "@TR_DC@": TR["d_count"], "@TR_DOM@": TR["domain"],
    "@TR_STAMP@": TR["stamp"], "@TR_AR@": TR["arenas"],
    "@TR_EP@": TR["epoch"], "@TR_CAP@": TR["pop_cap"],
    "@KP_DC@": KP["d_count"], "@KP_DOM@": KP["domain"],
    "@KP_AR@": KP["arenas"], "@KP_EP@": KP["epoch"],
    "@KP_CAP@": KP["pop_cap"],
    "@KO_DC@": KO["d_count"], "@KO_DOM@": KO["domain"],
}

NEW = '''
# ------------------------- dual-law authorities (surgeon-generated) ----

def canon_ev(ev):
    """NaN-canonical event tuples: IEEE tuple equality treats NaN as
    unequal to itself, so event comparison must canonicalize first."""
    out = []
    for e in ev:
        out.append(tuple("nan" if isinstance(x, float) and x != x
                         else x for x in e))
    return out


def trace_ieee(o_idx, o_w, ptr, adj, wts, flag, cutoff, d_count=0,
               domain="a3", stamp=False, arenas=None, epoch=1,
               pop_cap=None):
    """IEEE-law mirror: byte-for-byte the same schedule text with plain
    ordered < / <=. Mirrors the compact fallback kernel's observed
    comparison law on NaN (counterexamples preserved with the S10
    evidence); the fast-law mirror @TRACE@ is unchanged above."""
    o_start, o_end = int(o_idx[0]), int(o_idx[1])
    w0, w1 = float(o_w[0]), float(o_w[1])
    if domain == "a3":
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
    ev.append(("seed", o_start, w0))
    ev.append(("seed", o_end, w1))

    queue = [(w0, o_start)]
    weight, node = @HEAPPOP@(queue)
    if w1 < cutoff:
        @HEAPPUSH@(queue, (w1, o_end))
    if w0 < cutoff:
        @HEAPPUSH@(queue, (w0, o_start))
    ev.append(("pop", weight, node))

    pops = 0
    while queue:
        if pop_cap is not None and pops >= pop_cap:
            return label, ev, True
        weight, node = @HEAPPOP@(queue)
        pops += 1
        ev.append(("pop", weight, node))
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
            ev.append(("write", v, c))
            if flag[off]:
                if ptr[v + 1] - ptr[v] > 1:
                    @HEAPPUSH@(queue, (c, v))
                    ev.append(("push", c, v))
    return label, ev, False


def kp_ieee(o_idx, o_w, ptr, adj, wts, flag, cutoff, d_count=0,
            domain="a3", arenas=None, epoch=1, pop_cap=None):
    """The projection under the IEEE law: identical to @KP@ except the
    mirror it runs; same section-4 reconstruction."""
    label, stamps, touched = arenas
    arena, ev, capped = trace_ieee(
        o_idx, o_w, ptr, adj, wts, flag, cutoff, d_count=d_count,
        domain=domain, stamp=True, arenas=arenas, epoch=epoch,
        pop_cap=pop_cap)
    sentinel = (np.ones(1, dtype=arena.dtype) + cutoff)[0]
    dense = np.full(arena.shape[0], sentinel, dtype=arena.dtype)
    if touched:
        idx = np.array(touched, dtype=np.int64)
        dense[idx] = arena[idx]
    return dense, ev, capped


def run_all(o_idx, o_w, ptr, adj, wts, cutoff, d_count=0, domain="a3"):
    """Per-case result matrix over three authorities and both laws.
    Authorities: frozen oracle (fast law, portable), staged A1 kernel
    (production route, fast law), compact fallback kernel (observed
    IEEE-on-NaN law). The projection is evaluated under each law
    against that law's mirror."""
    fl = @FLAGS@(adj.shape[0])
    lab_oracle = @ORACLE@(o_idx, o_w, ptr, adj, wts, fl, cutoff,
                          @KO_DC@=d_count, @KO_DOM@=domain)
    lab_compact = @COMPACT@(o_idx, o_w, ptr, adj, wts, fl, cutoff,
                            d_count if domain == "a1" else 0)[0]
    deg = int((ptr[1:] - ptr[:-1]).max()) if ptr.shape[0] > 1 else 1
    off = np.empty(deg + 1, dtype=np.int64)
    wt = np.empty(deg + 1, dtype=np.float64)
    lab_a1 = @A1@(o_idx, o_w, ptr, adj, wts, fl, cutoff,
                  @A1ARGS@)[0]
    vec_mf, ev_mf, _ = @TRACE@(o_idx, o_w, ptr, adj, wts, fl, cutoff,
                               @TR_DC@=d_count, @TR_DOM@=domain,
                               @TR_STAMP@=False)
    arenas_f = @FRESH@((d_count if domain == "a1" else 0)
                       + ptr.shape[0] - 1)
    vec_kf, ev_kf, _ = @KP@(o_idx, o_w, ptr, adj, wts, fl, cutoff,
                            @KP_DC@=d_count, @KP_DOM@=domain,
                            @KP_AR@=arenas_f)
    vec_mi, ev_mi, _ = trace_ieee(o_idx, o_w, ptr, adj, wts, fl, cutoff,
                                  d_count=d_count, domain=domain,
                                  stamp=False)
    arenas_i = @FRESH@((d_count if domain == "a1" else 0)
                       + ptr.shape[0] - 1)
    vec_ki, ev_ki, _ = kp_ieee(o_idx, o_w, ptr, adj, wts, fl, cutoff,
                               d_count=d_count, domain=domain,
                               arenas=arenas_i)
    return {"oracle": lab_oracle, "compact": lab_compact, "a1": lab_a1,
            "mf": vec_mf, "ev_mf": ev_mf,
            "kpf": vec_kf, "ev_kf": ev_kf,
            "mi": vec_mi, "ev_mi": ev_mi,
            "ki": vec_ki, "ev_ki": ev_ki}


def case(tag, o_idx, o_w, ptr, adj, wts, cutoff, d_count=0, domain="a3"):
    r = run_all(o_idx, o_w, ptr, adj, wts, cutoff, d_count=d_count,
                domain=domain)
    ok = True

    def chk(cond, msg):
        nonlocal ok
        if not cond:
            @FAIL@(tag, msg)
            ok = False

    # fast law: projection == mirror text == oracle (+ staged A1 where
    # its seed-write collision law is in contract: NaN-free inputs)
    strict_a1 = not bool(np.isnan(np.asarray(o_w,
                                             dtype=np.float64)).any())
    chk(@BITEQ@(r["mf"], r["oracle"]), "fast mirror vs oracle")
    chk(@BITEQ@(r["kpf"], r["oracle"]), "projection(fast) vs oracle")
    if strict_a1:
        chk(@BITEQ@(r["kpf"], r["a1"]), "projection(fast) vs staged A1")
    elif not @BITEQ@(r["kpf"], r["a1"]):
        @CASES@.setdefault("a1_nan_divergent_tags", []).append(tag)
        if len(@CASES@.setdefault("a1_nan_counterexamples", [])) < 8:
            @CASES@["a1_nan_counterexamples"].append({
                "tag": tag,
                "oracle_u64": [hex(v) for v in np.asarray(
                    r["oracle"]).view(np.uint64)],
                "a1_u64": [hex(v) for v in np.asarray(
                    r["a1"]).view(np.uint64)]})
        @CASES@["counts"]["a1_nan_divergences"] = \\
            @CASES@["counts"].get("a1_nan_divergences", 0) + 1
    chk(canon_ev(r["ev_kf"]) == canon_ev(r["ev_mf"]), "fast events")
    # ieee law: projection == mirror text
    chk(@BITEQ@(r["ki"], r["mi"]), "projection(ieee) vs ieee mirror")
    chk(canon_ev(r["ev_ki"]) == canon_ev(r["ev_mi"]), "ieee events")
    # compact engine: compared only on the law-coinciding domain
    if @BITEQ@(r["kpf"], r["ki"]):
        chk(@BITEQ@(r["oracle"], r["compact"]), "oracle vs compact")
        if strict_a1:
            chk(@BITEQ@(r["a1"], r["compact"]), "staged A1 vs compact")
        chk(@BITEQ@(r["ki"], r["compact"]), "projection(ieee) vs compact")
        @CASES@["counts"]["coincide_cases"] = \\
            @CASES@["counts"].get("coincide_cases", 0) + 1
    else:
        @CASES@.setdefault("law_divergent_tags", []).append(tag)
        @CASES@["counts"]["divergent_cases"] = \\
            @CASES@["counts"].get("divergent_cases", 0) + 1
    if ok:
        @CASES@["passed"] += 1
    return ok


def e5_nan_pump():
    # NaN on both seeds + gated cycle (out-degree >= 2): the fast-law
    # pair pumps the heap (pinned NONTERMINATION law -> capped at 400);
    # the IEEE pair never pushes and terminates. The uncapped kernels
    # are deliberately NOT invoked on this graph.
    ptr, adj, w = @CSR@(3, [(0, 1), (0, 1), (1, 2), (1, 2), (2, 0),
                            (2, 0)],
                        [0.5] * 6)
    o, wv = @OT@(0, 1, np.nan, np.nan)
    cap = 400
    fl = @FLAGS@(adj.shape[0])
    _, ev1, cap1 = @TRACE@(o, wv, ptr, adj, w, fl, 1.5, pop_cap=cap)
    arenas = @FRESH@(ptr.shape[0] - 1)
    _, ev2, cap2 = @KP@(o, wv, ptr, adj, w, fl, 1.5, arenas=arenas,
                        pop_cap=cap)
    if not (cap1 and cap2):
        @FAIL@("e5/pump", "cap flags %s %s" % (cap1, cap2))
    elif canon_ev(ev1) != canon_ev(ev2):
        @FAIL@("e5/pump", "pop sequences diverge under cap")
    else:
        @CASES@["passed"] += 1
    _, _, cap3 = trace_ieee(o, wv, ptr, adj, w, fl, 1.5, stamp=False)
    if cap3:
        @FAIL@("e5/ieee_terminates", "ieee pair unexpectedly capped")
    else:
        @CASES@["passed"] += 1
    vi, evi, _ = kp_ieee(o, wv, ptr, adj, w, fl, 1.5,
                         arenas=@FRESH@(ptr.shape[0] - 1))
    mi, evm, _ = trace_ieee(o, wv, ptr, adj, w, fl, 1.5, stamp=True,
                            arenas=@FRESH@(ptr.shape[0] - 1))
    if not @BITEQ@(vi, mi) or canon_ev(evi) != canon_ev(evm):
        @FAIL@("e5/ieee_projection", "projection vs mirror under ieee")
    else:
        @CASES@["passed"] += 1
    @CASES@["counts"]["e5_pump_pops"] = cap
'''

for tok, val in SUB.items():
    NEW = NEW.replace(tok, val)
assert "@" not in NEW, "unsubstituted placeholder remains"

new_nodes = ast.parse(NEW).body
drop = {id(fdef_of[NM_ORIG]) for NM_ORIG in ()}  # placeholder, unused
targets = {nd.name for nd in new_nodes
           if isinstance(nd, ast.FunctionDef)}
kept, added = [], []
for node in tree.body:
    if isinstance(node, ast.FunctionDef) and node.name in targets:
        continue  # replaced below
    kept.append(node)
main_idx = next(i for i, n in enumerate(kept)
                if isinstance(n, ast.FunctionDef) and n.name == "main")
a1_mod_full = SUB["@A1MOD@"]
have_a1_import = any(
    isinstance(n, ast.ImportFrom) and n.module == a1_mod_full
    and any(a.name == SUB["@A1@"] for a in n.names) for n in kept)
last_imp = max(i for i, n in enumerate(kept)
               if isinstance(n, (ast.Import, ast.ImportFrom)))
if not have_a1_import:
    kept.insert(
        last_imp + 1,
        ast.ImportFrom(module=a1_mod_full,
                       names=[ast.alias(name=SUB["@A1@"], asname=None)],
                       level=0))
for i, nd in enumerate(new_nodes):
    kept.insert(main_idx + 1 + i, nd)
ast.fix_missing_locations(kept[0])
# serialize non-count findings into the admission JSON (idempotent)
main_node = next(n for n in kept if isinstance(n, ast.FunctionDef)
                 and difflib.get_close_matches(n.name, ["main"],
                                               n=1, cutoff=0.8))
cases_nm = NM["CASES"]
has_findings = any('"findings"' in ast.unparse(st)
                   for st in main_node.body)
if not has_findings:
    doc_idx = next(
        i for i, st in enumerate(main_node.body)
        if isinstance(st, ast.Assign)
        and any(isinstance(t, ast.Name)
                and difflib.get_close_matches(t.id, ["doc"], n=1,
                                              cutoff=0.8)
                for t in st.targets))
    fsrc = ('doc["findings"] = {{k: v for k, v in {}.items() '
            'if k not in ("passed", "failed", "counts")}}'
            ).format(cases_nm)
    main_node.body.insert(doc_idx + 1,
                          ast.parse(fsrc).body[0])
out = ast.unparse(ast.Module(body=kept, type_ignores=[]))
ast.parse(out)

open(PROBE, "w").write(out)
# deny duplicated defs
final = ast.parse(open(PROBE).read())
seen = [n.name for n in final.body if isinstance(n, ast.FunctionDef)]
assert len(seen) == len(set(seen)), ("duplicate defs", seen)
print("surgery ok; A1 module:", a1_mod_stem)
print("A1 params:", A1_P)
print("oracle kwargs:", KO, "| trace params:", TR_P)
print("functions:", seen)
