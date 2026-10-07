#!/bin/zsh
# I00 fire (una_10x, integrator scope): freeze integrated source and installed
# package identities.  Stages (fail-closed, unique exit codes):
#   S10 declared state + A/B wheel builds from git-archive exports   (rc 10x)
#   S20 A/B wheel census (per-entry source parity, shim shas)        (rc 20x)
#   S30 fresh install venvs + parent identity probes                 (rc 30x)
#   S40 loadavg gate + installed A leg (comparator, frozen policy)   (rc 40x)
#   S41 loadavg gate + installed B leg (candidate, frozen policy)    (rc 41x)
#   S50 per-arm worker identity probe (2-row checkpoint journal)     (rc 50x)
#       + research-import refusal scan in a real batch interpreter
#   S60 installed_manifest.json + freeze_candidate.json              (rc 60x)
set -u
cd /Users/alansynn/orca/workspaces/una-x/main || exit 1

WS=/Users/alansynn/orca/workspaces/una-x
RUN=campaign_runs/una_10x/tenx-20261005T2331Z
I00=$RUN/i00
SCRATCH=/tmp/i00
CONSOLE=$SCRATCH/console_fire1.log
mkdir -p "$I00/legs" "$SCRATCH"
: > "$CONSOLE"
exec > "$CONSOLE" 2>&1

say() { print -r -- "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"; }
sha() { shasum -a 256 "$1" | awk '{print $1}'; }

A_REV=395cdc5f683894b6f2ba460f7dbcefee99981ba3
HEAD_AT_FIRE=$(git rev-parse HEAD)
say "S10 begin (HEAD_AT_FIRE=$HEAD_AT_FIRE)"
git show "$HEAD_AT_FIRE" --stat --format="%H %s" | head -3

# ---------- S10 exports + builds ----------
rm -rf "$SCRATCH/comp" "$SCRATCH/head" "$SCRATCH/dist_a" "$SCRATCH/dist_b"
mkdir -p "$SCRATCH/comp" "$SCRATCH/head" "$SCRATCH/dist_a" "$SCRATCH/dist_b"
git archive "$A_REV" | tar -x -C "$SCRATCH/comp" || exit 10
git archive "$HEAD_AT_FIRE" | tar -x -C "$SCRATCH/head" || exit 11
python3 - <<'PYEOF' || exit 12
p = "/tmp/i00/comp/pyproject.toml"
s = open(p).read()
anchor = '[tool.hatch.build.targets.wheel]\npackages = ["src/urban_network_analysis"]\n'
assert anchor in s, "comparator pyproject anchor missing"
block = anchor + """
# I00 A-arm packaging parity backport (the ONLY tree delta vs 395cdc5):
# the top-level spawn shim must ship at the wheel root or installed
# parallel RunBatch crashes at pool.start (defect inherited from 0acbb11
# into both arms; same fix N10 landed for the candidate arm).
[tool.hatch.build.targets.wheel.force-include]
"src/una_batch_worker_main.py" = "una_batch_worker_main.py"
"""
open(p, "w").write(s.replace(anchor, block))
print("A pyproject patched (packaging-only delta)")
PYEOF
( cd "$SCRATCH/comp" && "$WS/venv-n10wheel/bin/python" -m hatchling build -t wheel >/dev/null 2>&1 && mv dist/*.whl "$SCRATCH/dist_a/" && rmdir dist ) || exit 13
( cd "$SCRATCH/head" && "$WS/venv-n10wheel/bin/python" -m hatchling build -t wheel >/dev/null 2>&1 && mv dist/*.whl "$SCRATCH/dist_b/" && rmdir dist ) || exit 14
WA=$(ls "$SCRATCH/dist_a"/*.whl); WB=$(ls "$SCRATCH/dist_b"/*.whl)
WA_SHA=$(sha "$WA"); WB_SHA=$(sha "$WB")
say "S10 ok A=$WA_SHA B=$WB_SHA"

# ---------- S20 census ----------
say "S20 begin"
"$WS/venv-p01m/bin/python" - "$WA" "$WB" "$A_REV" "$HEAD_AT_FIRE" "$SCRATCH" "$I00" <<'PYEOF' || exit 21
import hashlib, json, subprocess, sys, zipfile
WA, WB, A_REV, HEAD, SCRATCH, I00 = sys.argv[1:7]
def blob(rev, path):
    return subprocess.run(["git", "show", f"{rev}:{path}"], capture_output=True,
                          check=True, cwd="/Users/alansynn/orca/workspaces/una-x/main").stdout
def sha(b): return hashlib.sha256(b).hexdigest()

def census(whl, rev, shim_expected):
    z = zipfile.ZipFile(whl)
    names = z.namelist()
    pys = sorted(n for n in names if n.endswith(".py"))
    shim = "una_batch_worker_main.py"
    assert shim in names, f"{whl}: spawn shim MISSING"
    shim_sha = sha(z.read(shim))
    assert shim_sha == shim_expected, (shim_sha, shim_expected)
    mismatch, generated = [], []
    for n in pys:
        if n == shim:
            continue
        try:
            ref = blob(rev, "src/" + n)
        except subprocess.CalledProcessError:
            generated.append(n); continue
        if sha(z.read(n)) != sha(ref):
            mismatch.append(n)
    assert not mismatch, f"{whl}: packaged .py differs from {rev[:7]}: {mismatch}"
    assert generated == ["urban_network_analysis/_build_info.py"], generated
    native = [n for n in names if "/native/" in n or n.endswith(".so")
              or "sitecustomize" in n or "una_n00_trace_producer" in n]
    assert not native, f"{whl}: native/ext leaked: {native}"
    return {"path": whl, "sha256": sha(open(whl, 'rb').read()), "entries": len(names),
            "py_entries": len(pys), "spawn_shim_at_root": True,
            "spawn_shim_sha256": shim_sha, "generated_entries": generated,
            "native_or_ext_or_sitecustomize_entries": native}

shim_a = sha(blob(A_REV, "src/una_batch_worker_main.py"))
shim_b = sha(blob(HEAD, "src/una_batch_worker_main.py"))
out = {"schema": "una-10x-i00-wheel-census-1",
       "A_comparator": {"source_rev": A_REV, "tree_delta": "pyproject.toml ONLY (force-include packaging parity backport)", **census(WA, A_REV, shim_a)},
       "B_candidate": {"source_rev": HEAD, "tree_delta": "none (git archive of HEAD)", **census(WB, HEAD, shim_b)}}
json.dump(out, open(f"{I00}/wheel_census.json", "w"), indent=1)
print("CENSUS OK A_shim", shim_a[:12], "B_shim", shim_b[:12])
PYEOF
say "S20 ok"

# ---------- S30 installs + parent probes ----------
say "S30 begin"
for ARM in a b; do
  rm -rf "$SCRATCH/venv_$ARM"
  cp -a "$WS/venv-p01m" "$SCRATCH/venv_$ARM" || exit 30
  "$SCRATCH/venv_$ARM/bin/python" -m pip uninstall -q -y urban-network-analysis || exit 31
  "$SCRATCH/venv_$ARM/bin/python" -m pip install -q --no-deps "$SCRATCH/dist_$ARM"/*.whl || exit 32
done
"$WS/venv-p01m/bin/python" - "$SCRATCH" "$I00" <<'PYEOF' || exit 33
import hashlib, json, sys
import subprocess
SCRATCH, I00 = sys.argv[1], sys.argv[2]
census = json.load(open(f"{I00}/wheel_census.json"))
out = {}
for arm in ("a", "b"):
    r = subprocess.run([f"{SCRATCH}/venv_{arm}/bin/python", "-c",
        "import hashlib, os, sys;"
        "import urban_network_analysis as u;"
        "import una_batch_worker_main as shim;"
        "assert '/venv_%s/' in u.__file__.replace('/private/var','/var') or '/venv_%s/' in u.__file__;"
        "print(u.__file__); print(shim.__file__);"
        "print(hashlib.sha256(open(shim.__file__,'rb').read()).hexdigest())" % (arm, arm)],
        capture_output=True, text=True)
    assert r.returncode == 0, (arm, r.stderr[-800:])
    pkg_f, shim_f, shim_sha = r.stdout.strip().splitlines()
    assert f"/venv_{arm}/" in pkg_f and f"/venv_{arm}/" in shim_f, (pkg_f, shim_f)
    assert shim_sha == census[f"{'A' if arm=='a' else 'B'}_{'comparator' if arm=='a' else 'candidate'}"]["spawn_shim_sha256"], shim_sha
    from urban_network_analysis import UNA  # constructible in THIS interpreter is not the probe; probe did import above
    out[arm] = {"package_file": pkg_f, "spawn_shim_file": shim_f,
                "spawn_shim_sha256_parent_probe": shim_sha}
json.dump(out, open(f"{I00}/install_identity.json", "w"), indent=1)
print("PARENT PROBES OK")
PYEOF
say "S30 ok"

# ---------- S40/S41 legs ----------
gate() {
  local tries=0 la
  while true; do
    la=$(sysctl -n vm.loadavg | awk '{print $2}')
    [ "$(python3 -c "print(1 if float('$la') < 8.0 else 0)")" = "1" ] && break
    tries=$((tries+1)); [ "$tries" -gt 90 ] && { say "GATE TIMEOUT loadavg=$la"; return 61; }
    say "gate hold #$tries loadavg=$la"; sleep 60
  done
  say "gate open pre_leg_loadavg=$(sysctl -n vm.loadavg | awk '{print $2}')"
}

PAIRING=/Users/alansynn/orca/workspaces/una-x/main/campaign_data/inputs/upstream_Boston/Boston/local_audit_manifest.csv
DATAFOLDER=/Users/alansynn/orca/workspaces/una-x/main/campaign_data/inputs/upstream_Boston/Boston

say "S40 begin (A leg: installed comparator)"
rm -rf "$SCRATCH/numba_a" "$SCRATCH/out_a"; mkdir -p "$SCRATCH/numba_a" "$SCRATCH/out_a"
gate || exit 61
PRE_LA_A=$(sysctl -n vm.loadavg | awk '{print $2}')
NUMBA_CACHE_DIR=$SCRATCH/numba_a PYTHONDONTWRITEBYTECODE=1 \
UNA_BATCH_PHASES= UNA_NATIVE_TRACE= \
"$SCRATCH/venv_a/bin/python" "$RUN/p05s/run_screen_leg.py" \
  --tree /Users/alansynn/orca/workspaces/una-x/main \
  --pairing "$PAIRING" --data-folder "$DATAFOLDER" \
  --outdir "$SCRATCH/out_a" --tag i00_installed_comparator \
  --out "$I00/legs/a_installed_comparator.json" \
  --parallel --workers 4 --threads 1 --queue-depth 64 --cache-mode off \
  || exit 40
say "S40 ok"

say "S41 begin (B leg: installed candidate)"
rm -rf "$SCRATCH/numba_b" "$SCRATCH/out_b"; mkdir -p "$SCRATCH/numba_b" "$SCRATCH/out_b"
gate || exit 61
PRE_LA_B=$(sysctl -n vm.loadavg | awk '{print $2}')
NUMBA_CACHE_DIR=$SCRATCH/numba_b PYTHONDONTWRITEBYTECODE=1 \
UNA_BATCH_PHASES= UNA_NATIVE_TRACE= \
"$SCRATCH/venv_b/bin/python" "$RUN/p05s/run_screen_leg.py" \
  --tree /Users/alansynn/orca/workspaces/una-x/main \
  --pairing "$PAIRING" --data-folder "$DATAFOLDER" \
  --outdir "$SCRATCH/out_b" --tag i00_installed_candidate \
  --out "$I00/legs/b_installed_candidate.json" \
  --parallel --workers 4 --threads 1 --queue-depth 64 --cache-mode off \
  || exit 41
say "S41 ok"

# ---------- S50 worker identity probes + research-import scan ----------
say "S50 begin"
"$WS/venv-p01m/bin/python" - "$SCRATCH" "$I00" "$PAIRING" "$PRE_LA_A" "$PRE_LA_B" <<'PYEOF' || exit 51
import json, re, sys, subprocess, csv, os
SCRATCH, I00, PAIRING, PRE_LA_A, PRE_LA_B = sys.argv[1:6]
census = json.load(open(f"{I00}/wheel_census.json"))

# 2-row sliced pairing for the untimed identity probe (derived file in /tmp)
rows = list(csv.reader(open(PAIRING)))
probe = f"{SCRATCH}/probe_pairing.csv"
with open(probe, "w", newline="") as f:
    w = csv.writer(f); w.writerow(rows[0]); w.writerow(rows[1]); w.writerow(rows[2])

DATAFOLDER = "/Users/alansynn/orca/workspaces/una-x/main/campaign_data/inputs/upstream_Boston/Boston"
out = {"probe_pairing": probe, "arms": {}}
for arm in ("a", "b"):
    key = f"{'A_comparator' if arm == 'a' else 'B_candidate'}"
    jdir = f"{SCRATCH}/journal_{arm}"; pout = f"{SCRATCH}/probe_out_{arm}"
    subprocess.run(["rm", "-rf", jdir, pout], check=True)
    code = f"""
import json, os, sys
import urban_network_analysis as u
from urban_network_analysis import UNA
from urban_network_analysis.Execution import CacheOptions, ExecutionOptions
bad = sorted(m for m in sys.modules if m.split('.')[0] in ('campaigns','campaign_runs','tests'))
p = UNA(verbosity=0)
p.settings.output_folder = {pout!r}
p.settings.data_folder = {DATAFOLDER!r}
ex = ExecutionOptions(threads_per_worker=1, queue_depth=64,
                      cache=CacheOptions(), checkpoint={jdir!r})
p.RunBatch('accessibility', {probe!r}, parallel=True, workers=1, execution=ex)
bad_after = sorted(m for m in sys.modules if m.split('.')[0] in ('campaigns','campaign_runs','tests'))
doc = json.load(open(os.path.join({jdir!r}, 'journal.json')))
print(json.dumps({{'research_imports_before': bad, 'research_imports_after': bad_after,
                   'spawn_shim': doc['identity_detail']['transport']['spawn_shim'],
                   'rows': len(doc.get('rows', {{}}))}}))
"""
    r = subprocess.run([f"{SCRATCH}/venv_{arm}/bin/python", "-c", code],
                       capture_output=True, text=True)
    assert r.returncode == 0, (arm, r.stderr[-1500:])
    d = json.loads(r.stdout.strip().splitlines()[-1])
    assert d["spawn_shim"] == census[key]["spawn_shim_sha256"], (arm, d["spawn_shim"])
    assert not d["research_imports_after"], (arm, d["research_imports_after"])
    out["arms"][arm] = d
out["pre_leg_loadavg"] = {"a": float(PRE_LA_A), "b": float(PRE_LA_B)}
json.dump(out, open(f"{I00}/worker_identity_probes.json", "w"), indent=1)
print("WORKER PROBES OK", {k: v['spawn_shim'][:12] for k, v in out['arms'].items()})
PYEOF
say "S50 ok"

# ---------- S60 manifests ----------
say "S60 begin"
"$WS/venv-p01m/bin/python" - "$I00" "$SCRATCH" "$HEAD_AT_FIRE" <<'PYEOF' || exit 60
import hashlib, json, re, sys
I00, SCRATCH, HEAD = sys.argv[1], sys.argv[2], sys.argv[3]
def sha(p): return hashlib.sha256(open(p, "rb").read()).hexdigest()
census = json.load(open(f"{I00}/wheel_census.json"))
ident = json.load(open(f"{I00}/install_identity.json"))
probes = json.load(open(f"{I00}/worker_identity_probes.json"))

ts = re.compile(r"accessibility_\d{4}-\d{2}-\d{2}_\d{4}")
def outcensus(path):
    return {ts.sub("accessibility_TS", f["rel"]): f["sha256"]
            for f in json.load(open(path))["output_files"]
            if not f["rel"].split("/")[0].startswith(".")}
leg_a = f"{I00}/legs/a_installed_comparator.json"
leg_b = f"{I00}/legs/b_installed_candidate.json"
RUN_DIR = I00.rsplit("/", 1)[0]
ref_c = outcensus(f"{RUN_DIR}/n00/legs/L2A1_numba.json")

reqs = {}
for tag, lp in (("a", leg_a), ("b", leg_b)):
    leg = json.load(open(lp))
    req = [f for f in leg["output_files"] if not f["rel"].split("/")[0].startswith(".")]
    assert leg["zero_failed_jobs"], (tag, leg["rows"])
    assert (len(req), sum(f["bytes"] for f in req)) == (64, 147829892), (tag, len(req))
    reqs[tag] = {"files": len(req), "bytes": sum(f["bytes"] for f in req),
                 "bitwise_equal_to_n00_L2A1_reference": outcensus(lp) == ref_c,
                 "wall_s": leg["timing"]["public_call_wall_s"],
                 "executable": leg["pins"]["executable"]}

manifest = {
 "schema": "una-10x-i00-installed-manifest-1",
 "task": "I00 freeze integrated source and installed package identities",
 "date": "2026-10-07",
 "head_at_fire": HEAD,
 "arms": {
  "A_comparator": {
    "source_rev": census["A_comparator"]["source_rev"],
    "declared_tree_delta": census["A_comparator"]["tree_delta"],
    "wheel": census["A_comparator"],
    "install_venv": f"{SCRATCH}/venv_a",
    "parent_probe": ident["a"], "worker_probe": probes["arms"]["a"],
    "leg": {"file": "legs/a_installed_comparator.json", **reqs["a"]}},
  "B_candidate": {
    "source_rev": census["B_candidate"]["source_rev"],
    "declared_tree_delta": census["B_candidate"]["tree_delta"],
    "wheel": census["B_candidate"],
    "install_venv": f"{SCRATCH}/venv_b",
    "parent_probe": ident["b"], "worker_probe": probes["arms"]["b"],
    "leg": {"file": "legs/b_installed_candidate.json", **reqs["b"]}},
 },
 "identity_chain": "wheel census shim sha == parent probe sha == runtime journal transport.spawn_shim (real spawn worker), per arm",
 "worker_identity_probe": {"untimed": True, "checkpointed_2row": True,
                           "policy_note": "timed cohort legs stay frozen-policy checkpoint-free; identity captured by separate disposable probe"},
 "research_import_scan": "empty in a real batch interpreter, both arms",
 "in_flight_exclusions": {
    "tests/batch/runtime/test_runtime_cancel_wedge.py": "author-owned UNCOMMITTED healthy-path control (RES-2 of DEF-24R intake 1e7237b); not part of the wheel by construction (tests unpackaged), not frozen"},
 "capability_gaps_retained": [
    "GPU region: explicit_unavailable (G11); Delta checklist gates any device claim",
    "native trace producer: retained capability, NOT in default gate stack (N10)",
    "comparator wedge: DEF-24 fix is candidate-only; baseline cancel semantics as-shipped (A2)",
    "RES-1 stranded-feeder interpreter-exit seam: recorded, no defect"],
}
json.dump(manifest, open(f"{I00}/installed_manifest.json", "w"), indent=1)

freeze = {
 "schema": "una-10x-i00-freeze-candidate-1",
 "date": "2026-10-07",
 "freeze_source": {"commit": HEAD, "branch": "perf/una-10x"},
 "frozen_identities": {
   "candidate_wheel_sha256": census["B_candidate"]["sha256"],
   "comparator_wheel_sha256": census["A_comparator"]["sha256"],
   "comparator_source_rev": census["A_comparator"]["source_rev"]},
 "invalidation_clause": "source/harness/environment changes after this freeze invalidate affected dependent tests/timings (dossier 07:53, dossier 09); prior files are kept, never edited",
 "next": "V00 gauntlet -> Q00 freeze records -> Q10 seven-pair AB/BA gate (gate_10x.py) -> R00 -> Z00",
}
json.dump(freeze, open(f"{I00}/freeze_candidate.json", "w"), indent=1)
print("MANIFESTS OK")
print("bitwise A:", reqs["a"]["bitwise_equal_to_n00_L2A1_reference"],
      "bitwise B:", reqs["b"]["bitwise_equal_to_n00_L2A1_reference"])
PYEOF
say "S60 ok"
say "I00 FIRE COMPLETE rc=0"
