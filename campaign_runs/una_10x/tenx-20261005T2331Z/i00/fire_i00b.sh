#!/bin/zsh
# I00 continuation fire (fire-1 failed at S50 only): legs S40/S41 already
# succeeded and are reused from disk.  Stages:
#   S50 worker identity probes, FIXED: probe rows carry an absolute
#       data_folder column.  RunBatch builds FRESH Settings() per row and
#       anchors relative inputs at the CSV's own directory unless the row
#       itself sets data_folder (UNA.py:202-226 docstring is explicit:
#       "regardless of the instance's data_folder") — fire-1 and the first
#       fire-2 attempt both died resolving /tmp/i00/*.geojson.
#   S60 installed_manifest.json + freeze_candidate.json              (rc 60x)
set -u
cd /Users/alansynn/orca/workspaces/una-x/main || exit 1

WS=/Users/alansynn/orca/workspaces/una-x
RUN=campaign_runs/una_10x/tenx-20261005T2331Z
I00=$RUN/i00
SCRATCH=/tmp/i00
CONSOLE=$SCRATCH/console_fire2.log
: > "$CONSOLE"
exec > "$CONSOLE" 2>&1

say() { print -r -- "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"; }
PAIRING=/Users/alansynn/orca/workspaces/una-x/main/campaign_data/inputs/upstream_Boston/Boston/local_audit_manifest.csv
DATAFOLDER=/Users/alansynn/orca/workspaces/una-x/main/campaign_data/inputs/upstream_Boston/Boston
PRE_LA_A=$(python3 -c "import json;print(json.load(open('$I00/legs/a_installed_comparator.json'))['pins']['host_ncpu'])" >/dev/null 2>&1; sysctl -n vm.loadavg | awk '{print $2}')

say "S50 begin (cwd-fix probes; legs reused from fire-1)"
# freshness gate: both legs must predate this continuation and be intact
for f in "$I00/legs/a_installed_comparator.json" "$I00/legs/b_installed_candidate.json"; do
  [ -s "$f" ] || exit 50
done
"$WS/venv-p01m/bin/python" - "$SCRATCH" "$I00" "$PAIRING" "$DATAFOLDER" <<'PYEOF' || exit 51
import json, sys, subprocess, csv
SCRATCH, I00, PAIRING, DATAFOLDER = sys.argv[1:5]
census = json.load(open(f"{I00}/wheel_census.json"))

rows = list(csv.reader(open(PAIRING)))
probe = f"{SCRATCH}/probe_pairing.csv"
with open(probe, "w", newline="") as f:
    w = csv.writer(f)
    hdr = rows[0] + ["data_folder"]          # row-level data_folder: fresh
    w.writerow(hdr)                           # Settings() per row keeps it
    for r in (rows[1], rows[2]):
        w.writerow(r + [DATAFOLDER])          # (UNA.py:225 guard then skips)

out = {"probe_pairing": probe, "arms": {},
       "cwd_fix": "probe rows carry absolute data_folder (Settings field, ApplyRow-applied); RunBatch would otherwise anchor relative inputs at the CSV directory (UNA.py:202-226) — fire-1 + fire-2-attempt-1 lesson; subprocess also runs with cwd=DATAFOLDER"}
for arm in ("a", "b"):
    key = "A_comparator" if arm == "a" else "B_candidate"
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
                       capture_output=True, text=True, cwd=DATAFOLDER)
    assert r.returncode == 0, (arm, r.stderr[-1500:])
    d = json.loads(r.stdout.strip().splitlines()[-1])
    assert d["spawn_shim"] == census[key]["spawn_shim_sha256"], (arm, d["spawn_shim"])
    assert d["rows"] == 2, (arm, d["rows"])
    assert not d["research_imports_after"], (arm, d["research_imports_after"])
    out["arms"][arm] = d
json.dump(out, open(f"{I00}/worker_identity_probes.json", "w"), indent=1)
print("WORKER PROBES OK", {k: v['spawn_shim'][:12] for k, v in out['arms'].items()})
PYEOF
say "S50 ok"

say "S60 begin"
"$WS/venv-p01m/bin/python" - "$I00" "$SCRATCH" <<'PYEOF' || exit 60
import hashlib, json, re, subprocess, sys
I00, SCRATCH, = sys.argv[1], sys.argv[2]
HEAD = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                      cwd="/Users/alansynn/orca/workspaces/una-x/main").stdout.strip()
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
 "head_at_wheel_build": census["B_candidate"]["source_rev"],
 "head_at_manifest_generation": HEAD,
 "fire_history": "fire-1 (console_fire1.log): S10-S41 complete incl. both legs; S50 died anchoring probe inputs at /tmp/i00 (RunBatch builds fresh Settings per row, CSV-dir anchoring, UNA.py:202-226). fire-2 attempt 1 (cwd=DATAFOLDER only) failed identically (console_fire2.log). fire-2 attempt 2 (this console): probe rows carry absolute data_folder column; S50+S60 rerun, legs reused from fire-1",
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
    "leg": {"file": "legs/b_installed_candidate.json", **reqs["b"]},
    "hidden_outputs_note": "43 hidden entries (.una-batch-prepared x38, .una-batch-cohort x5) = candidate preparation-stage artifacts, correctly outside the required census; comparator has none (pre-preparation implementation)"},
 },
 "identity_chain": "wheel census shim sha == parent probe sha == runtime journal transport.spawn_shim (real spawn worker), per arm",
 "worker_identity_probe": {"untimed": True, "checkpointed_2row": True,
                           "policy_note": "timed cohort legs stay frozen-policy checkpoint-free; identity captured by separate disposable probe"},
 "research_import_scan": "empty in a real batch interpreter, both arms",
 "co_tenancy": "verification legs ran under external host load (rustc build + enterprise pmd/log sweep) with loadavg gate <8 and in-run walls recorded; NOT gate figures (Q10 measures exclusively)",
 "in_flight_exclusions": {
    "tests/batch/runtime/test_runtime_cancel_wedge.py": "committed at 5e2a0ef AFTER fire-1 exported HEAD 1e7237b; tests are unpackaged and outside the frozen wheels; freeze_source stays 1e7237b"},
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
 "freeze_source": {"commit": census["B_candidate"]["source_rev"], "branch": "perf/una-10x"},
 "frozen_identities": {
   "candidate_wheel_sha256": census["B_candidate"]["sha256"],
   "comparator_wheel_sha256": census["A_comparator"]["sha256"],
   "comparator_source_rev": census["A_comparator"]["source_rev"]},
 "invalidation_clause": "source/harness/environment changes after this freeze invalidate affected dependent tests/timings (dossier 07:53, dossier 09); prior files are kept, never edited",
 "next": "V00 gauntlet -> Q00 freeze records -> Q10 seven-pair AB/BA gate (gate_10x.py) -> R00 -> Z00",
}
json.dump(freeze, open(f"{I00}/freeze_candidate.json", "w"), indent=1)
print("MANIFESTS OK wheel_source", census["B_candidate"]["source_rev"][:12],
      "generation_head", HEAD[:12])
print("bitwise A:", reqs["a"]["bitwise_equal_to_n00_L2A1_reference"],
      "bitwise B:", reqs["b"]["bitwise_equal_to_n00_L2A1_reference"])
print("walls: A", reqs["a"]["wall_s"], "B", reqs["b"]["wall_s"])
PYEOF
say "S60 ok"
say "I00 FIRE-2 COMPLETE rc=0"
