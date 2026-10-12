"""S10 L2 evidence copier: raw mirror of drivers, run logs, receipts
and the independent review verdicts into evidence/tasks/S10/
(append-only, assert-not-exists, sha256 manifest)."""
import glob
import hashlib
import json
import os
import shutil

HERE = os.path.dirname(os.path.abspath(__file__))
PROOF = os.path.join(HERE, "campaigns", "una_10x", "evidence", "proofs",
                     "S10_sparse_projection")
DST = os.path.join(HERE, "evidence", "tasks", "S10",
                   "s10-20261003-dual-law")
assert not os.path.exists(DST), "destination exists: " + DST
os.makedirs(DST)

def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

# 1. drivers (scratch scripts, globbed — no hand-typed names)
drivers = sorted(glob.glob(os.path.join(HERE, "_s10_*.py")))
assert len(drivers) >= 3, drivers
# 2. proof tree (probe, proof, admission, run logs, findings, verdicts)
tree = sorted(glob.glob(os.path.join(PROOF, "**", "*"), recursive=True))
manifest = {"task": "S10", "run_tag": "s10-20261003-dual-law",
            "files": {}}
copied = 0
for src in drivers:
    rel = os.path.basename(src)
    dst = os.path.join(DST, "drivers", rel)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copyfile(src, dst)
    assert sha(dst) == sha(src)
    manifest["files"]["drivers/" + rel] = sha(src)
    copied += 1
for src in tree:
    if not os.path.isfile(src):
        continue
    if "__pycache__" in src:
        continue
    rel = os.path.relpath(src, PROOF)
    dst = os.path.join(DST, "proof_tree", rel)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copyfile(src, dst)
    assert sha(dst) == sha(src)
    manifest["files"]["proof_tree/" + rel] = sha(src)
    copied += 1
review = os.path.join(PROOF, "review_indep")
assert os.path.isdir(review), "review_indep missing — run after verdict"
assert any(f.startswith("verdict") for f in os.listdir(review)), \
    "no verdict files in review_indep"
mpath = os.path.join(DST, "manifest.json")
with open(mpath, "w") as f:
    json.dump(manifest, f, indent=2, sort_keys=True)
    f.write("\n")
print("copied", copied, "files ->", DST)
print("manifest:", mpath)
