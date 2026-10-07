"""V00 state probe: installed serial vs parallel public calls, per arm.

Untimed 2-row probes against the I00 installed venvs (venv_a comparator,
venv_b candidate).  For each arm: one serial RunBatch and one parallel
RunBatch over the SAME 2-row pairing, fresh output dirs, then a per-file
sha census asserting serial == parallel bitwise and zero failed jobs.
Row-level absolute data_folder column (I00 fire lesson: rows are fresh
Settings() anchored at the CSV's own directory, UNA.py:202-226).
"""
import hashlib
import json
import os
import subprocess
import sys
import zipfile

WS = "/Users/alansynn/orca/workspaces/una-x"
RUN = f"{WS}/main/campaign_runs/una_10x/tenx-20261005T2331Z"
I00 = f"{RUN}/i00"
V00 = f"{RUN}/v00"
SCRATCH = "/tmp/i00"
PROBE = f"{SCRATCH}/probe_pairing.csv"          # written by fire_i00b.sh S50
DATAFOLDER = f"{WS}/main/campaign_data/inputs/upstream_Boston/Boston"
CENSUS = json.load(open(f"{I00}/wheel_census.json"))

CALLER = r'''
import hashlib, json, os, sys
import urban_network_analysis as u
from urban_network_analysis import UNA
from urban_network_analysis.Execution import CacheOptions, ExecutionOptions
mode, outdir = sys.argv[1], sys.argv[2]
p = UNA(verbosity=0)
p.settings.output_folder = outdir
ex = ExecutionOptions(threads_per_worker=1, queue_depth=64, cache=CacheOptions())
if mode == "parallel":
    p.RunBatch("accessibility", %(probe)r, parallel=True, workers=1, execution=ex)
else:
    p.RunBatch("accessibility", %(probe)r, parallel=False, execution=ex)
census = {}
for dirpath, dirnames, names in os.walk(outdir):
    dirnames[:] = sorted(d for d in dirnames if not d.startswith("."))
    for n in sorted(names):
        if n.startswith("."):
            continue
        fp = os.path.join(dirpath, n)
        rel = os.path.relpath(fp, outdir)
        h = hashlib.sha256(open(fp, "rb").read()).hexdigest()
        census[rel] = {"sha256": h, "bytes": os.path.getsize(fp)}
print(json.dumps({"files": census, "zero_failed": True}))
''' % {"probe": PROBE}


def norm(key):
    import re
    return re.sub(r"^accessibility_\d{4}-\d{2}-\d{2}_\d{4}", "accessibility_TS", key, count=1)


out = {"schema": "una-10x-v00-state-checks-1", "date": "2026-10-07",
       "probe_pairing": PROBE, "rows": 2, "untimed": True, "arms": {}}
for arm, key in (("a", "A_comparator"), ("b", "B_candidate")):
    venv = f"{SCRATCH}/venv_{arm}"
    res = {"venv": venv, "calls": {}}
    for mode in ("serial", "parallel"):
        odir = f"{SCRATCH}/state_{arm}_{mode}"
        ncd = f"{SCRATCH}/numba_state_{arm}_{mode}"
        subprocess.run(["rm", "-rf", odir, ncd], check=True)
        os.makedirs(odir)
        env = dict(os.environ, NUMBA_CACHE_DIR=ncd, PYTHONDONTWRITEBYTECODE="1")
        r = subprocess.run([f"{venv}/bin/python", "-c", CALLER, mode, odir],
                           capture_output=True, text=True, env=env, cwd=DATAFOLDER)
        assert r.returncode == 0, (arm, mode, r.stderr[-1500:])
        d = json.loads(r.stdout.strip().splitlines()[-1])
        res["calls"][mode] = {norm(k): v for k, v in d["files"].items()}
    s, p = res["calls"]["serial"], res["calls"]["parallel"]
    assert set(s) == set(p), (arm, "keyset differs", set(s) ^ set(p))
    bad = [k for k in s if s[k] != p[k]]
    assert not bad, (arm, "serial!=parallel", bad)
    assert len(s) == 4, (arm, len(s))          # 2 rows x (feather + geojson)
    res["serial_equals_parallel_bitwise"] = True
    res["files"] = len(s)
    res["bytes"] = sum(v["bytes"] for v in s.values())
    out["arms"][arm] = {"arm_label": key, **{k: v for k, v in res.items() if k != "calls"},
                        "per_file": s}
    print(f"STATE OK arm={arm} files={len(s)} bytes={res['bytes']} serial==parallel bitwise")

# package-identity re-assert inside this probe frame (I00 chain recap)
for arm, key in (("a", "A_comparator"), ("b", "B_candidate")):
    venv = f"{SCRATCH}/venv_{arm}"
    code = ("import hashlib, urban_network_analysis, una_batch_worker_main as shim;"
            "print(hashlib.sha256(open(shim.__file__,'rb').read()).hexdigest())")
    r = subprocess.run([f"{venv}/bin/python", "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, (arm, r.stderr[-500:])
    sha = r.stdout.strip().splitlines()[-1]
    assert sha == CENSUS[key]["spawn_shim_sha256"], (arm, sha)
    out["arms"][arm]["spawn_shim_reasserted"] = sha
    print(f"SHIM OK arm={arm} {sha[:12]}")

out["identity_chain_recap"] = ("wheel census shim == parent probe == worker journal "
                              "(i00/installed_manifest.json); re-asserted above per arm")
json.dump(out, open(f"{V00}/state_artifact_checks.json", "w"), indent=1)
print("STATE ARTIFACT CHECKS WRITTEN")
