"""Probe patch 2: unpack the engine's (weights, pred) tuple at both call
sites. Regex-anchored with count asserts; ast.parse must pass after."""
import ast
import re

P = ("campaigns/una_10x/evidence/proofs/S10_sparse_projection/"
     "probe_sparse_projection_tiny.py")
src = open(P).read()

n_sites = len(re.findall(r"compact_vector_node_view_scope\(", src))
assert n_sites == 2, n_sites  # import line is not a call

src, n1 = re.subn(
    r"(compact_vector_node_view_scope\(.*?else 0\))\n",
    r"\1[0]\n", src, count=1, flags=re.S)
assert n1 == 1, n1

src, n2 = re.subn(
    r"(compact_vector_node_view_scope\(.*?1\.5, 0\))\n",
    r"\1[0]\n", src, count=1, flags=re.S)
assert n2 == 1, n2

ast.parse(src)
open(P, "w").write(src)
print("engine tuple unpacked at both call sites")
