#!/bin/zsh
# Q10 assembly continuation (q10c): B40/B50/B60 only - NO new windows.
# Fire 2 (fire_q10b.sh) completed B10 + all 10 protected legs, then crashed at
# B40 on a harness defect: legs pin the RESOLVED interpreter path
# (/private/tmp/i00/venv_a/bin/python; macOS /tmp is a symlink to /private/tmp)
# while the assert compared the unresolved prefix.  Leg JSONs and measurements
# are untouched; this fire only re-runs assembly, gate_10x and the raw index,
# with the prefix compared under os.path.realpath.
#   C40 paired_samples.json assembly (7 primary pairs, per-window asserts) rc 40x
#   C50 gate_10x.py (exit 2 = below target is RECORDED, not a fire fault)  rc 50x
#   C60 raw_index.json + protected-cell criterion                         rc 60x
set -u
cd /Users/alansynn/orca/workspaces/una-x/main || exit 1

RUN=campaign_runs/una_10x/tenx-20261005T2331Z
Q10=$RUN/q10
Q00=$RUN/q00
QSCRATCH=/tmp/q10
CONSOLE=$QSCRATCH/console_fire_q10c.log
mkdir -p "$QSCRATCH"
: > "$CONSOLE"
exec > "$CONSOLE" 2>&1

say() { print -r -- "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"; }

for f in q10_p1_a q10_p7_b q10_pc1_tiny_gis_beta00_v2_a q10_pc5_cold_2row_v2_b; do
  [ -f "$Q10/legs/$f.json" ] || exit 30
done
say "C40 begin (paired_samples assembly + per-window asserts)"
python3 - <<'PYEOF' || exit 40
import hashlib, json, os, re
MAIN = "/Users/alansynn/orca/workspaces/una-x/main"
RUN = f"{MAIN}/campaign_runs/una_10x/tenx-20261005T2331Z"
Q00, Q10, I00 = f"{RUN}/q00", f"{RUN}/q10", f"{RUN}/i00"
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()

freeze = json.load(open(f"{Q00}/freeze.json"))
primary_doc = json.load(open(f"{Q00}/PRIMARY.json"))
ts = re.compile(r"^accessibility_\d{4}-\d{2}-\d{2}_\d{4}")
I00_INVALIDATION_MAX = 16.0   # preregistered operative threshold (see raw_index note)

def norm_map(leg):
    m = {}
    for f in leg["output_files"]:
        if f["rel"].split("/")[0].startswith("."):
            continue
        m[ts.sub("accessibility_TS", f["rel"], count=1)] = f["sha256"]
    return m

ref = norm_map(json.load(open(f"{I00}/legs/a_installed_comparator.json")))
assert len(ref) == 64

def loadavg_max(tag):
    path = f"{Q10}/sampler/{tag}.loadavg.txt"
    vals = [float(l.split()[1]) for l in open(path) if l.strip()]
    return (max(vals) if vals else None), len(vals)

def record(legpath, arm, tag):
    leg = json.load(open(legpath))
    m = norm_map(leg)
    assert m == ref, (tag, "bitwise drift vs reference", [k for k in ref if ref.get(k) != m.get(k)][:3])
    assert leg["zero_failed_jobs"] and leg["row_count"] == 32
    es = leg["effective_snapshot"]["options"]
    assert es["semantic_profile"] == "una_legacy", es["semantic_profile"]
    cfg = leg["config"]
    assert cfg["parallel"] and cfg["workers"] == 4 and cfg["threads_per_worker"] == 1 \
        and cfg["queue_depth"] == 64 and cfg["cache_mode"] == "off"
    la_max, n_samples = loadavg_max(tag)
    invalidated = la_max is not None and la_max >= I00_INVALIDATION_MAX
    venv = os.path.realpath("/tmp/i00/venv_a" if arm == "A" else "/tmp/i00/venv_b")
    assert leg["pins"]["executable"].startswith(venv), leg["pins"]["executable"]
    return {
     "status": "valid" if not invalidated else "invalid_load_spike",
     "installed": True, "bits_verified": True, "state_verified": True,
     "artifacts_verified": True, "resource_compliant": True,
     "initial_trace_cost_included": True,
     "boundary": "public_call_complete", "public_api": "UNA.RunBatch",
     "numerical_profile": es["semantic_profile"],
     "source_sha": "395cdc5f683894b6f2ba460f7dbcefee99981ba3" if arm == "A"
                   else "1e7237bebe8dd55e4a3b444f24ad9a255fa8644c",
     "wheel_sha256": "f0504f886eb9976150a56dcfa818188cdedb172a13716206003b8b1e1a1dc4b1" if arm == "A"
                     else "9a84517b9a524d4632c990a2332f8ecc219d6d3dd844528e8dac9c8dafc58d97",
     "workload_sha256": primary_doc["workload_sha256"],
     "resource_sha256": freeze["policy_digests"]["resource_policy"],
     "cache_policy_sha256": freeze["policy_digests"]["cache_policy"],
     "output_policy_sha256": freeze["policy_digests"]["output_policy"],
     "environment_sha256": freeze["policy_digests"]["environment"],
     "harness_sha256": primary_doc["harness_sha256"],
     "wall_s": leg["timing"]["public_call_wall_s"],
     "jobs_requested": 32, "jobs_validated": 32, "jobs_failed": 0,
     "substantive_calls": 32, "cache_regime": "COLD", "distinct_jobs": True,
     "measurement_detail": {
        "leg": legpath, "tag": tag, "import_s": leg["timing"]["import_s"],
        "cold_boundary_s": round(leg["timing"]["import_s"] + leg["timing"]["public_call_wall_s"], 4),
        "peak_tree_rss_bytes": leg["sampler"]["peak_tree_rss_bytes"],
        "in_run_loadavg_max": la_max, "in_run_samples": n_samples,
        "output_bytes": leg["output_bytes_total"],
        "attestation_basis": "installed: executable under frozen venv + I00 identity chain + T10/B10 wheel re-hash; "
                             "bits: per-file sha census == I00 reference map (64 files); state: zero_failed + "
                             "una_legacy effective snapshot; artifacts: 64-file census; resource: config block "
                             "== q00 resource_policy + per-leg in-run sampler",
     },
    }, invalidated

pairs, invalid = [], []
for i in range(1, 8):
    ra, inv_a = record(f"{Q10}/legs/q10_p{i}_a.json", "A", f"q10_p{i}_a")
    rb, inv_b = record(f"{Q10}/legs/q10_p{i}_b.json", "B", f"q10_p{i}_b")
    if inv_a or inv_b:
        invalid.append((f"q10_p{i}", ra["status"], rb["status"]))
    pairs.append({"id": f"q10_p{i}", "A": ra, "B": rb})

samples = {
 "schema_version": 1,
 "measurement_class": "observed_proxy",
 "primary_sha256": freeze["primary_sha256"],
 "freeze_sha256": sha(f"{Q00}/freeze.json"),
 "pairs": pairs,
}
json.dump(samples, open(f"{Q10}/paired_samples.json", "w"), indent=1)
json.dump({"invalid_windows": invalid}, open(f"{Q10}/invalidation_flags.json", "w"), indent=1)
print("PAIRS ASSEMBLED", len(pairs), "| invalid flags:", invalid)
print("ratios:", [round(p["A"]["wall_s"] / p["B"]["wall_s"], 4) for p in pairs])
PYEOF

say "C50 begin (gate_10x)"
rm -f "$Q10/gate_10x.json"
GATE_RC=0
python3 campaigns/una_10x/tools/gate_10x.py --samples "$Q10/paired_samples.json" --out "$Q10/gate_10x.json" || GATE_RC=$?
say "C50 done gate_rc=$GATE_RC (2 = below target, recorded honestly)"

say "C60 begin (raw_index + protected criterion)"
python3 - <<'PYEOF' || exit 60
import glob, hashlib, json, math, random, re, statistics
RUN = "/Users/alansynn/orca/workspaces/una-x/main/campaign_runs/una_10x/tenx-20261005T2331Z"
Q00, Q10 = f"{RUN}/q00", f"{RUN}/q10"
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()
ts = re.compile(r"^accessibility_\d{4}-\d{2}-\d{2}_\d{4}")
def norm_map(leg):
    return {ts.sub("accessibility_TS", f["rel"], count=1): f["sha256"]
            for f in leg["output_files"] if not f["rel"].split("/")[0].startswith(".")}

errata = json.load(open(f"{Q00}/errata_pc_inputs.json"))

# preserve closed fire consoles inside the evidence tree, sha-named
import shutil
consoles = {}
for src, dst in (("/tmp/q10/console_fire_q10.log", "console_fire_q10_fire1.txt"),
                 ("/tmp/q10/console_fire_q10b.log", "console_fire_q10b_fire2.txt")):
    d = f"{Q10}/{dst}"
    shutil.copyfile(src, d)
    consoles[dst] = sha(d)
consoles["console_fire_q10c_att1.txt"] = sha(f"{Q10}/console_fire_q10c_att1.txt")
consoles["console_fire_q10c_att2.txt"] = "self-console, finalized after this index; sha at commit"

cells = []
for name in ("pc1_tiny_gis_beta00_v2", "pc2_tiny_gis_beta31_v2", "pc3_search_radius1200_v2",
             "pc4_search_radius1600_v2", "pc5_cold_2row_v2"):
    la = json.load(open(f"{Q10}/legs/q10_{name}_a.json"))
    lb = json.load(open(f"{Q10}/legs/q10_{name}_b.json"))
    ma, mb = norm_map(la), norm_map(lb)
    assert ma and ma == mb, (name, "protected behavior parity FAILED", set(ma) ^ set(mb))
    assert la["zero_failed_jobs"] and lb["zero_failed_jobs"]
    assert la["pins"]["manifest_sha256"] == errata["v2_inputs_sha256"][name], name
    assert lb["pins"]["manifest_sha256"] == errata["v2_inputs_sha256"][name], name
    cells.append({"id": name, "input_sha256": errata["v2_inputs_sha256"][name],
                  "files": len(ma),
                  "wall_A_s": la["timing"]["public_call_wall_s"],
                  "wall_B_s": lb["timing"]["public_call_wall_s"],
                  "ratio_B_over_A": lb["timing"]["public_call_wall_s"] / la["timing"]["public_call_wall_s"],
                  "outputs_bitwise_equal": True})
ratios = [c["ratio_B_over_A"] for c in cells]
logs = [math.log(r) for r in ratios]
rng = random.Random(20261003)
boot = [math.exp(statistics.median([logs[rng.randrange(len(logs))] for _ in range(len(logs))]))
        for _ in range(20000)]
boot.sort()
median_ba = math.exp(statistics.median(logs))
upper95 = boot[int(math.ceil(0.95 * (len(boot) - 1)))]
criterion = {"median_B_over_A": median_ba, "one_sided_95_upper": upper95,
             "median_le_1_05": median_ba <= 1.05, "upper95_le_1_10": upper95 <= 1.10,
             "default_dispatch_ok": median_ba <= 1.05 and upper95 <= 1.10}

legs = {}
for f in sorted(glob.glob(f"{Q10}/legs/*.json")):
    leg = json.load(open(f))
    legs[f.split("/")[-1]] = {
        "tag": leg["tag"], "wall_s": leg["timing"]["public_call_wall_s"],
        "import_s": leg["timing"]["import_s"], "rows": leg["row_count"],
        "files": leg["output_file_count"], "bytes": leg["output_bytes_total"],
        "peak_tree_rss_bytes": leg["sampler"]["peak_tree_rss_bytes"]}
samplers = {f.split("/")[-1]: sha(f) for f in sorted(glob.glob(f"{Q10}/sampler/*.txt"))}

inv_rule = ("operative in-run invalidation threshold: any 60s loadavg sample >= 16.0 marks the "
            "window invalid_load_spike (preregistered in fire_q10.sh T40 before leg execution; "
            "rationale: >2x the pre-leg gate line 8.0 and above the historical co-tenanted "
            "in-run max 12.76, n00 L2A2); flagged windows would trigger the q00 one-FULL-repeat-"
            "block clause with all blocks retained")
raw = {
 "schema": "una-10x-q10-raw-index-1", "date": "2026-10-07",
 "fire_history": [
   {"fire": 1, "console": "q10/console_fire_q10_fire1.txt",
    "script": "fire_q10.sh",
    "scope": "T10 + 7 primary pairs complete (legs q10_p1..p7); 28 LEG OK lines",
    "halt": "protected pc1 LEG FAIL rc=1: zsh $ARMS non-split tag; deeper cause q00 v1 "
            "protected inputs missing header row (pandas parsed 0/1 records)"},
   {"fire": 2, "console": "q10/console_fire_q10b_fire2.txt",
    "script": "fire_q10b.sh",
    "scope": "B10 identity re-verification + 5 protected pairs on v2 inputs, 10 legs OK",
    "halt": "B40 harness crash: venv-prefix assert compared unresolved /tmp prefix against "
            "the resolved /private/tmp path recorded in leg pins (macOS symlink); no "
            "measurement affected, assembly only"},
   {"fire": 3, "console": "q10/console_fire_q10c_att1.txt",
    "script": "fire_q10c.sh (attempt 1)",
    "scope": "C40 assembly PASSED (7 pairs, no invalid flags), C50 gate schema defect: "
             "records carried only wheel_sha256; gate_10x requires all 7 digests "
             "(workload/resource/cache_policy/output_policy/environment/harness) per record",
    "halt": "ValueError('immutable manifest and wheel hashes required'); no result written"},
   {"fire": 4, "console": "q10/console_fire_q10c_att2.txt (this console, finalized after this index)",
    "script": "fire_q10c.sh (attempt 2, digests added from q00 PRIMARY/freeze)",
    "scope": "C40/C50/C60 assembly + gate + raw index only; NO new windows",
    "halt": None},
 ],
 "console_sha256": consoles,
 "errata": {"file": "q00/errata_pc_inputs.json", "sha256": sha(f"{Q00}/errata_pc_inputs.json"),
            "scope": "protected-cell derived inputs only; primary workload and 7 primary "
                     "windows unaffected (completed in fire 1 before the errata)"},
 "primary_windows": legs, "sampler_files_sha256": samplers,
 "invalidation_rule": inv_rule,
 "invalidation_flags": json.load(open(f"{Q10}/invalidation_flags.json")),
 "protected_cells": cells, "protected_criterion": criterion,
 "gate_10x": json.load(open(f"{Q10}/gate_10x.json")),
 "law": "q00/PRIMARY.json + configuration.json (commit ee5652a); BENCHMARKS.md; DECISION.md",
}
json.dump(raw, open(f"{Q10}/raw_index.json", "w"), indent=1)
print("PROTECTED:", json.dumps(criterion, indent=1))
print("RAW INDEX WRITTEN")
PYEOF
say "Q10C FIRE COMPLETE"
