"""Byte-exact patcher for the S10 probe (token-flip-safe): replaces the
flag-length bug, removes the E9 dead loop, fixes the subnormal literal.
Every replacement is count-asserted; ast.parse must pass afterwards."""
import ast

P = ("campaigns/una_10x/evidence/proofs/S10_sparse_projection/"
     "probe_sparse_projection_tiny.py")
src = open(P).read()

PATCHES = [
    # flag length = arc count E, everywhere (uniform identifier)
    ("flags_all(ptr.shape[0])", "flags_all(adj.shape[0])", 12),
    ("flags_all(ptr2.shape[0])", "flags_all(adj2.shape[0])", 2),
    # e9: drop the leftover dead loop body
    ("""    def worker(wid):
        for rep in range(120):
            if wid == rep % 2:  # interleave
                continue
        for rep, (o, wv, ptr, adj, w) in enumerate(inputs):
""",
     """    def worker(wid):
        for rep, (o, wv, ptr, adj, w) in enumerate(inputs):
""", 1),
    # e3: real subnormal boundary value
    ("np.float64(np.sin(1e-308))",
     "np.nextafter(np.float64(0.0), np.float64(1.0)) * np.float64(2)", 1),
]
for old, new, want in PATCHES:
    got = src.count(old)
    assert got == want, (old[:60], "count", got, "want", want)
    src = src.replace(old, new)

ast.parse(src)
open(P, "w").write(src)
print("patched:", P)
for old, new, want in PATCHES:
    print(f"  {want}x: {old[:52]!r} -> {new[:52]!r}")
