"""Preserve the S10 NaN findings: copy the counterexample probes +
fresh run outputs + manifest into the S10 evidence tree (append-only,
assert-not-exists)."""
import glob
import hashlib
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
EV = os.path.join(HERE, "campaigns", "una_10x", "evidence", "proofs",
                  "S10_sparse_projection")
FDIR = os.path.join(EV, "findings")
assert not os.path.exists(FDIR), "findings dir already exists: " + FDIR
os.makedirs(FDIR)

ALAN = "/storage/home/hcoda1/1/dsynn6/micromamba/envs/alan/bin/python"
CACHE = "/storage/scratch1/1/dsynn6/campaign_data/numba_cache_h01"

def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()

manifest = {"task": "S10 sparse projection NaN findings",
            "files": {}}
hits = sorted(glob.glob(os.path.join(HERE, "_s10_nan_probe*.py")))
assert len(hits) == 2, ("expected the two nan probes", hits)
for src in hits:
    dst = os.path.join(FDIR, os.path.basename(src))
    with open(src, "rb") as a, open(dst, "wb") as b:
        b.write(a.read())
    assert sha(dst) == sha(src)
    env = dict(os.environ, NUMBA_CACHE_DIR=CACHE)
    r = subprocess.run([ALAN, src], capture_output=True, text=True,
                       env=env, timeout=600)
    out = os.path.join(FDIR, os.path.basename(src) + ".out")
    with open(out, "w") as f:
        f.write(r.stdout)
        if r.returncode != 0:
            f.write("\n[exit %d]\n%s" % (r.returncode, r.stderr))
    assert r.stdout.strip(), "empty output for " + src
    assert r.returncode == 0, (src, r.returncode, r.stderr[-400:])
    manifest["files"][os.path.basename(dst)] = {
        "sha256": sha(dst), "output_sha256": sha(out),
        "command": ALAN + " " + os.path.basename(src)}
    print("captured:", os.path.basename(dst))

mpath = os.path.join(FDIR, "manifest.json")
with open(mpath, "w") as f:
    json.dump(manifest, f, indent=2, sort_keys=True)
    f.write("\n")
print("findings dir:", FDIR)
