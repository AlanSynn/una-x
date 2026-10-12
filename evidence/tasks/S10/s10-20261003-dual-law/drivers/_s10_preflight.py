"""S10 preflight: (1) engine sources byte-identical to the pinned
baseline; (2) true signature of the compact kernel via AST (render-safe);
(3) the fast-comparison law helpers exist in the oracle."""
import ast
import subprocess

DIFF = subprocess.run(
    ["git", "diff", "395cdc5f683894b6f2ba460f7dbcefee99981ba3", "HEAD",
     "--stat", "--", "src/urban_network_analysis/"],
    capture_output=True, text=True)
print("engine diff vs pin: <empty means unchanged>")
print(DIFF.stdout if DIFF.stdout.strip() else "(unchanged)")
assert not DIFF.stdout.strip(), "engine sources drifted from the pin"

SRC = "src/urban_network_analysis/Engines/AccessibilityWElevation.py"
tree = ast.parse(open(SRC).read())
for node in ast.walk(tree):
    if isinstance(node, ast.FunctionDef) and "scope" in node.name:
        args = [a.arg for a in node.args.args]
        print("engine fn:", node.name, "(", len(args), "args )")
        for a in args:
            print("   ", a)

ORACLE = "src/urban_network_analysis/kernels/scope_access.py"
otree = ast.parse(open(ORACLE).read())
names = {n.name for n in ast.walk(otree)
         if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
for want in ("fast_lt", "fast_le", "_tuple_lt", "_heap_push", "_heap_pop",
             "scope_search_schedule", "scope_route_admits", "tail_admits",
             "integrated_scope_access_oracle"):
    assert want in names, want
print("oracle surface OK")
