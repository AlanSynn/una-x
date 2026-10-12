"""C10 live engine facts: AST-parse the actual fold and its call sites.

Everything the metric dependency table and its tests assert about the
engine is derived HERE, at run time, from the engine source — no
dependency name, branch literal or binding is retyped anywhere else.
All names/expressions returned are exact AST-derived strings.
"""
import ast
import hashlib
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
ENGINE_REL = os.path.join("src", "urban_network_analysis", "Engines",
                          "AccessibilityWElevation.py")
ENGINE = REPO / ENGINE_REL
WRAPPER = "o_access"


def _fold_fn(tree, fold_name):
    fn = [n for n in ast.walk(tree)
          if isinstance(n, ast.FunctionDef) and n.name == fold_name]
    assert len(fn) == 1, (fold_name, len(fn))
    return fn[0]


def engine_sha256():
    h = hashlib.sha256()
    with open(ENGINE, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def scan_fn(fn, pset):
    """assignments: name -> [(value, enclosing If tests)] (all writes kept)."""
    assigns = {}

    def scan(node, path):
        if isinstance(node, ast.If):
            for child in node.body:
                scan(child, path + [(node.test, True)])
            for child in node.orelse:
                scan(child, path + [(node.test, False)])
            return
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    assigns.setdefault(t.id, []).append((node.value,
                                                         tuple(path)))
            return
        for child in ast.iter_child_nodes(node):
            scan(child, path)

    scan(fn, [])
    return assigns


def collect(root, root_path, assigns, pset):
    """Fixpoint closure: params read, polar branch conditions, calls."""
    names = set()
    conds = list(root_path)
    calls = set()
    stack = [root]
    seen = set()
    while stack:
        node = stack.pop()
        if id(node) in seen:
            continue
        seen.add(id(node))
        if isinstance(node, ast.Name):
            if node.id in pset:
                names.add(node.id)
            for value, path in assigns.get(node.id, ()):
                stack.append(value)
                stack.extend(t for t, _pos in path)
                continue
        if isinstance(node, ast.If):
            conds.append((node.test, True))
        f = node.func if isinstance(node, ast.Call) else None
        if f is not None:
            calls.add(f.id if isinstance(f, ast.Name) else
                      (f.attr if isinstance(f, ast.Attribute) else "?"))
        stack.extend(ast.iter_child_nodes(node))
    return names, conds, calls


def fold_facts(fold_name):
    """Per-output dependency facts of the compiled fold, from the AST.

    Each write to an output carries its enclosing conditions with
    polarity: facts[out]['branches'] lists one entry per write with the
    params that write reads; facts[out]['params_read'] is the union
    (identity-level, includes condition params).
    """
    tree = ast.parse(ENGINE.read_text())
    fn = _fold_fn(tree, fold_name)
    params = [a.arg for a in fn.args.args]
    pset = set(params)
    assigns = scan_fn(fn, pset)
    ret = fn.body[-1]
    assert isinstance(ret, ast.Return), type(ret)
    outs = [e.id for e in ret.value.elts]
    assert all(isinstance(e, ast.Name) for e in ret.value.elts), outs
    facts = {}
    used = set()
    for out in outs:
        entries = assigns.get(out) or [(ret.value, ())]
        names, conds, calls = set(), set(), set()
        branches = []
        for value, path in entries:
            n, c, k = collect(value, path, assigns, pset)
            names |= n
            calls |= k
            cond_names = set()
            plain = []
            for t, pos in c:
                text = ast.unparse(t)
                plain.append((text, pos))
                conds.add(text)
                for node in ast.walk(t):
                    if isinstance(node, ast.Name) and node.id in pset:
                        cond_names.add(node.id)
            branches.append({"when": [list(e) for e in plain],
                             "params_read": sorted(n | cond_names),
                             "calls": sorted(k)})
            used |= n | cond_names
        for b in branches:
            names |= set(b["params_read"])
        facts[out] = {"params_read": sorted(names),
                      "branch_conditions": sorted(conds),
                      "branches": branches,
                      "calls": sorted(calls)}
    return {
        "fold_name": fold_name,
        "engine_file": ENGINE_REL,
        "engine_sha256": engine_sha256(),
        "params": params,
        "outputs": outs,
        "per_output": facts,
        "params_dead_in_fold": sorted(pset - used),
        "decorators": [ast.unparse(d) for d in fn.decorator_list],
    }


def fold_call_bindings(fold_name, tree=None):
    """Every fold call site in the engine: kwarg -> source expression.

    Returns {lineno: {kwarg: unparse}}; assert-level consumers check all
    sites agree and that alias slots bind the SAME source expression as
    the corresponding shared slot.
    """
    tree = tree or ast.parse(ENGINE.read_text())
    sites = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        f = node.func
        if not (isinstance(f, ast.Name) and f.id == fold_name):
            continue
        kw = {}
        for k in node.keywords:
            assert k.arg is not None, (node.lineno, "positional in fold call")
            kw[k.arg] = ast.unparse(k.value)
        sites[node.lineno] = kw
    assert sites, fold_name
    return sites


def wrapper_derivation(wrapper=WRAPPER, kwarg_sub=None):
    """The derived-metric kwarg expression in the wrapper (e.g. growth)."""
    tree = ast.parse(ENGINE.read_text())
    wrap = [n for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef) and n.name == wrapper]
    assert len(wrap) == 1, wrapper
    kws = [k for k in ast.walk(wrap[0])
           if isinstance(k, ast.keyword) and k.arg
           and (kwarg_sub is None or kwarg_sub in k.arg)]
    assert len(kws) == 1, [k.arg for k in kws]
    return {"wrapper": wrapper, "kwarg": kws[0].arg,
            "expression": ast.unparse(kws[0].value)}


if __name__ == "__main__":
    import json
    import sys
    name = sys.argv[1] if len(sys.argv) > 1 else "reach_gravity_knn_access"
    ff = fold_facts(name)
    print(json.dumps(ff, indent=2))
    print("call sites:", sorted(fold_call_bindings(name)))
    print("derivation:", wrapper_derivation(kwarg_sub="growth"))
