"""Independent delta verification for workflows.py facade (reviewer script).

Ground truth = difflib opcodes between facade body and upstream file.
Then a byte-level reconstruction: strip header, remove D4 block, restore
module-top imports, assert byte equality.
"""
import difflib
import hashlib
import sys

FACADE = "/storage/scratch1/1/dsynn6/una-x/src/urban_network_analysis/compat/madina/una/workflows.py"
UPSTREAM = "/storage/scratch1/1/dsynn6/una-x/.refs/madina_ref/src/madina/una/workflows.py"

fb = open(FACADE, "rb").read()
ub = open(UPSTREAM, "rb").read()
print("md5 facade  :", hashlib.md5(fb).hexdigest())
print("md5 upstream:", hashlib.md5(ub).hexdigest())
print("facade bytes:", len(fb), "upstream bytes:", len(ub))

flines = fb.decode().splitlines(keepends=True)
ulines = ub.decode().splitlines(keepends=True)

# --- Step A: split header from body. Header = module docstring. The body
# must start at the first top-level line equal to upstream line 1.
assert ulines[0] == "import os\n", repr(ulines[0])
body_start = None
for i, ln in enumerate(flines):
    if ln == "import os\n":
        body_start = i
        break
assert body_start is not None
header = "".join(flines[:body_start])
body = "".join(flines[body_start:])
print("\nheader lines (incl. closing triple-quote):", body_start)
assert header.startswith('"""') and header.rstrip().endswith('"""')
# header must be a pure docstring: compile check
import ast
mod = ast.parse(header + "\n" + "pass\n")
assert isinstance(mod.body[0], ast.Expr) and isinstance(mod.body[0].value, ast.Constant)
print("header parses as a bare string expression: OK")
assert len(mod.body) == 2, mod.body  # docstring + the `pass` we added

# --- Step B: difflib ground truth on body vs upstream
sm = difflib.SequenceMatcher(None, ulines, body.splitlines(keepends=True), autojunk=False)
ops = [o for o in sm.get_opcodes() if o[0] != "equal"]
print("\nnon-equal opcodes:", len(ops))
for tag, i1, i2, j1, j2 in ops:
    print(f"  {tag}: upstream[{i1+1}:{i2+1}] -> facade_body[{j1+1}:{j2+1}]")
    for ln in ulines[i1:i2]:
        print("    UP -", ln.rstrip("\n"))
    for ln in body.splitlines(keepends=True)[j1:j2]:
        print("    FA +", ln.rstrip("\n"))

# expected: exactly 2 hunks. Hunk 1: upstream lines 13 and 19 deleted (the two
# module-top pydeck imports). Hunk 2: the D4 block inserted.
# SequenceMatcher may merge them; normalize: collect deleted upstream lines and
# inserted facade lines.
deleted_up = []
inserted_fa = []
for tag, i1, i2, j1, j2 in ops:
    deleted_up += [(i1 + k + 1, ulines[i1 + k]) for k in range(i2 - i1)]
    inserted_fa += [(j1 + k + 1, body.splitlines(keepends=True)[j1 + k]) for k in range(j2 - j1)]

print("\ndeleted upstream lines:", [n for n, _ in deleted_up])
for n, ln in deleted_up:
    print(f"   {n}: {ln.rstrip()}")
print("inserted facade-body lines:", len(inserted_fa))

assert [n for n, _ in deleted_up] == [13, 19], "unexpected deletions"
assert deleted_up[0][1] == "import pydeck as pdk\n", repr(deleted_up[0][1])
assert deleted_up[1][1] == "from pydeck.types import String\n", repr(deleted_up[1][1])
print("\ndeletions are exactly upstream lines 13 and 19: OK  (diff shape -2/+%d)" % len(inserted_fa))

# --- Step C: byte-level reconstruction. Remove the D4 block from the body and
# re-insert the two import lines at their exact upstream positions.
blines = body.splitlines(keepends=True)
# locate D4 block in body coordinates
d4_start = None
for i, ln in enumerate(blines):
    if ln.startswith("        # D4: pydeck is an optional dependency"):
        d4_start = i
        break
assert d4_start is not None
# block ends at the line ") from exc"
d4_end = None
for j in range(d4_start, len(blines)):
    if blines[j] == "            ) from exc\n":
        d4_end = j
        break
assert d4_end is not None
# difflib says the inserted region is 12 lines: the 11 D4 lines + one blank
# separator line (upstream has no blank between '):' and 'predicted_flow_gdf').
assert blines[d4_end + 1] == "\n", repr(blines[d4_end + 1])
print("\nD4 block: facade lines %d-%d + 1 blank = 12 lines (file coords %d-%d)" % (
    d4_start + 1, d4_end + 1, body_start + d4_start + 1, body_start + d4_end + 1))
recon = blines[:d4_start] + blines[d4_end + 2:]

# find indices in recon of the two insertion anchors
idx_pd = recon.index("import pandas as pd\n")
idx_pt = recon.index("from pathlib import Path\n")
recon = recon[:idx_pd] + ["import pydeck as pdk\n"] + recon[idx_pd:]
idx_pt = recon.index("from pathlib import Path\n")
recon = recon[:idx_pt] + ["from pydeck.types import String\n"] + recon[idx_pt:]

recon_b = "".join(recon).encode()
print("reconstructed md5:", hashlib.md5(recon_b).hexdigest())
if recon_b == ub:
    print("BYTE EQUALITY with upstream: PASS")
else:
    print("BYTE EQUALITY: FAIL")
    # show where
    rl = recon_b.decode().splitlines(keepends=True)
    sm2 = difflib.SequenceMatcher(None, ulines, rl, autojunk=False)
    for tag, i1, i2, j1, j2 in sm2.get_opcodes():
        if tag != "equal":
            print(f"  {tag}: up[{i1+1}:{i2+1}] recon[{j1+1}:{j2+1}]")
            for ln in ulines[i1:i2]:
                print("    UP -", ln.rstrip())
            for ln in rl[j1:j2]:
                print("    RC +", ln.rstrip())
    sys.exit(1)

# --- Step D: header line-count claim + verify header claims only D4
print("\nheader total lines (file coords 1..%d)" % body_start)
