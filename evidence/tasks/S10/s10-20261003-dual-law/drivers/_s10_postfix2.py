"""e5/ieee_projection repair: compare the reconstruction against the
PLAIN-mode ieee mirror (dense-initialized label), never the stamp-mode
raw arena whose untouched slots are uninitialized np.empty bytes (the
previous "pass" was a garbage-recycling coincidence)."""
import ast
import os

HERE = os.path.dirname(os.path.abspath(__file__))
P = os.path.join(HERE, "campaigns", "una_10x", "evidence", "proofs",
                 "S10_sparse_projection",
                 "probe_sparse_projection_tiny.py")
src = open(P).read()

OLD = ("    mi, evm, _ = trace_ieee(o, wv, ptr, adj, w, fl, 1.5, "
       "stamp=True, arenas=fresh_arenas(ptr.shape[0] - 1))\n")
NEW = ("    # plain mode returns the dense-initialized label; the\n"
       "    # stamp-mode return is a raw arena (untouched slots are\n"
       "    # uninitialized) and must never be bitwise-compared\n"
       "    mi, evm, _ = trace_ieee(o, wv, ptr, adj, w, fl, 1.5)\n")
got = src.count(OLD)
assert got == 1, ("anchor count", got)
src = src.replace(OLD, NEW)
ast.parse(src)
open(P, "w").write(src)
print("e5 mirror comparison repaired")
