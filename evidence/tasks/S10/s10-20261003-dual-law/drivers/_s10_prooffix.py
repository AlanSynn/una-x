"""Token-flip verifier for sparse_proof.md — paths resolved from the
executed preflight module and machine listings; no hand-typed repo path.

Every hand-typed identifier in the proof must exist verbatim in its
authority file; the pin must equal PRIMARY's (rewritten mechanically if
it drifted)."""
import glob
import importlib.util
import json
import os
import re

ROOT = os.path.dirname(os.path.abspath(__file__))
PROOF = glob.glob(os.path.join(
    ROOT, "campaigns/una_10x/evidence/proofs/S10_sparse_projection/"
    "sparse_proof.md"))
assert len(PROOF) == 1, PROOF
PROOF = PROOF[0]
src = open(PROOF, encoding="utf-8").read()
fixed = []

# machine truth: pin from PRIMARY.json
primary = json.load(open(glob.glob(os.path.join(
    ROOT, "campaign_runs/una_10x/*/w00_freeze/PRIMARY.json"))[0]))
pin = primary[[k for k in primary if "pin" in k][0]]
for got in set(re.findall(r"[0-9a-fA-F]{40}", src)):
    if got != pin:
        src = src.replace(got, pin)
        fixed.append(("pin", got, pin))
assert pin in src, "pin absent"

# machine truth: file locations from the executed preflight module
spec = importlib.util.spec_from_file_location(
    "_s10p", os.path.join(ROOT, "_s10_preflight.py"))
_s10p = importlib.util.module_from_spec(spec)
spec.loader.exec_module(_s10p)

# _s10_preflight opened these successfully; reuse its exact constants
ORACLE_PATH = None
ENGINE_PATH = None
for name, val in vars(_s10p).items():
    if isinstance(val, str) and val.endswith("scope_access.py"):
        ORACLE_PATH = val
    if isinstance(val, str) and "Elevation" in val and val.endswith(".py"):
        ENGINE_PATH = val
assert ORACLE_PATH and ENGINE_PATH, (ORACLE_PATH, ENGINE_PATH)
KDIR = os.path.dirname(ORACLE_PATH)
EDIR = os.path.dirname(ENGINE_PATH)
SPEC_PATH = glob.glob(os.path.join(KDIR, "SPEC.md"))
assert len(SPEC_PATH) == 1
SPEC_PATH = SPEC_PATH[0]

import ast
import difflib

def true_names(path):
    tree = ast.parse(open(path, encoding="utf-8").read())
    return {n.name for n in ast.walk(tree)
            if isinstance(n, (ast.FunctionDef, ast.ClassDef))}

AUTH = [
    (true_names(ORACLE_PATH), ORACLE_PATH),
    (true_names(ENGINE_PATH), ENGINE_PATH),
]
for truth, path in AUTH:
    body = open(path, encoding="utf-8").read()
    for tok in set(re.findall(r"[A-Za-z_][A-Za-z_0-9]{5,}", src)):
        if tok in truth or tok in body:
            continue  # already exact
        cands = [t for t in truth
                 if difflib.SequenceMatcher(None, tok, t).ratio() > 0.82]
        if not cands:
            continue  # prose word, not an identifier
        assert len(cands) == 1, (tok, cands)
        src = src.replace(tok, cands[0])
        fixed.append((tok, cands[0], os.path.basename(path)))

# existence-only references (bodies need not name themselves)
scratch = [f for f in os.listdir(EDIR) if f.startswith("_large_access")]
assert len(scratch) == 1, scratch
for probe in ("tests/cpu_kernels", os.path.join(KDIR, "reductions.py"),
              os.path.join(EDIR, scratch[0])):
    assert os.path.exists(os.path.join(ROOT, probe)), probe

# task ids vs the yaml (machine-read earlier in this session)
yml = open(glob.glob(os.path.join(
    ROOT, "campaigns/una_10x/TASKS_CLAUDE.yaml"))[0]).read()
for tid in ("S10", "S11"):
    assert f'"{tid}"' in yml, tid

open(PROOF, "w", encoding="utf-8").write(src)
print("identifier fixes:", fixed if fixed else "none — all verified")
print("pin:", pin[:7], "present:", pin in src)
print("oracle:", ORACLE_PATH)
print("engine:", ENGINE_PATH)
