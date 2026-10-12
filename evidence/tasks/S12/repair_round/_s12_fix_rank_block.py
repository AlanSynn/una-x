"""Rewrite the rank-closure block using the def's exact parameter names
(from AST — no retyped identifiers)."""
import ast
import re

PATH = "src/urban_network_analysis/Engines/_sparse_access_scratch.py"
src = open(PATH).read()
tree = ast.parse(src, PATH)
params = None
for node in ast.walk(tree):
    if isinstance(node, ast.FunctionDef) and node.name == "_sp_route_admits":
        params = [a.arg for a in node.args.args]
assert params and len(params) == 10, params
print("def params:", params)

# positional mapping: 0 ptr, 1 vec, 2 vec_weights, 3 network_node,
# 4 o_idxs, 5 o_weights, 6 d_idxs, 7 d_weights_tbl, 8 d_weights_vec, 9 cutoff
ptr, vec, wts, node_, o_idx, o_w, d_idx, d_w_tbl, d_w_vec, _ = params
lines = [
    "    # Rank closure (S12 review F1/F2/F3): the engine gates read "
    "terminal",
    "    # shape[0]/[1] and the fold indexes the destination weight "
    "vector as",
    "    # 1-D, so a degenerate rank either crashes the kernel while "
    "the baseline",
    "    # runs (F1: (D,1) d_weights) or raises IndexError from THIS "
    "gate instead",
    "    # of the baseline's TypingError (F2: 0-d d_weights, F3: 1-D "
    "d_terminal_weights).",
    "    # Refuse the whole class here — before any shape element "
    "access, so this",
    "    # gate itself still raises nothing — and keep every such "
    "input on the",
    "    # unchanged fallback, which is bitwise or exception-parity "
    "exact by",
    "    # construction.",
    "    if (d_weights_vec.ndim != 1",
    "            or adjacency_pointer.ndim != 1",
    "            or adjacency_vector.ndim != 1",
    "            or adjacency_vector_weights.ndim != 1",
    "            or adjacency_vector_network_node.ndim != 1",
    "            or o_terminal_idxs.ndim != 2",
    "            or o_terminal_weights.ndim != 2",
    "            or d_terminal_idxs.ndim != 2",
    "            or d_terminal_weights.ndim != 2):",
    '        return (False, "shape", 0)',
]
block = "\n".join(lines)
subs = [
    ("d_weights_vec.ndim", d_w_vec + ".ndim"),
    ("adjacency_pointer.ndim", ptr + ".ndim"),
    ("adjacency_vector.ndim", vec + ".ndim"),
    ("adjacency_vector_weights.ndim", wts + ".ndim"),
    ("adjacency_vector_network_node.ndim", node_ + ".ndim"),
    ("o_terminal_idxs.ndim", o_idx + ".ndim"),
    ("o_terminal_weights.ndim", o_w + ".ndim"),
    ("d_terminal_idxs.ndim", d_idx + ".ndim"),
    ("d_terminal_weights.ndim", d_w_tbl + ".ndim"),
]
for placeholder, real in subs:
    block = block.replace(placeholder, real)

# splice: replace everything from the "# Rank closure" comment through
# the first 'return (False, "shape", 0)' after it
pat = re.compile(
    r"    # Rank closure \(S12 review.*?return \(False, \"shape\", 0\)",
    re.S)
found = pat.findall(src)
assert len(found) == 1, ("rank block not found exactly once", len(found))
src = pat.sub(lambda m: block, src, count=1)
ast.parse(src, PATH)
open(PATH, "w").write(src)
print("block rewritten; params used:", [r for _, r in subs])
