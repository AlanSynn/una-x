#!/bin/zsh
# Q10 continuation fire (q10b): 5 protected pairs on v2 inputs + T40/T50/T60.
# Primary windows already completed in fire #1 (legs/q10_p{1..7}_{a,b}.json);
# fire #1 died at protected pc1 on two defects, both fixed here:
#   (1) q00 v1 protected inputs lacked the header row -> errata_pc_inputs.json,
#       inputs regenerated as *_v2.csv (v1 bytes kept, never edited)
#   (2) zsh `for ARM in $ARMS` does not word-split -> explicit arm lists below
# Stages (fail-closed, unique exit codes):
#   B10 frozen identities + q00 digests + v2 input shas                  rc 10x
#   B3x protected pairs pc1..pc5, alternating arm order                  rc 31x
#   B40 paired_samples.json assembly (7 primary pairs, per-window asserts) rc 40x
#   B50 gate_10x.py (exit 2 = below target is RECORDED, not a fire fault) rc 50x
#   B60 raw_index.json + protected-cell criterion                        rc 60x
set -u
cd /Users/alansynn/orca/workspaces/una-x/main || exit 1

WS=/Users/alansynn/orca/workspaces/una-x
RUN=campaign_runs/una_10x/tenx-20261005T2331Z
Q10=$RUN/q10
Q00=$RUN/q00
SCRATCH=/tmp/i00
QSCRATCH=/tmp/q10
CONSOLE=$QSCRATCH/console_fire_q10b.log
mkdir -p "$Q10/legs" "$Q10/sampler" "$QSCRATCH"
: > "$CONSOLE"
exec > "$CONSOLE" 2>&1

say() { print -r -- "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"; }
sha() { shasum -a 256 "$1" | awk '{print $1}'; }

DATAFOLDER=/Users/alansynn/orca/workspaces/una-x/main/campaign_data/inputs/upstream_Boston/Boston
PAIRING=/Users/alansynn/orca/workspaces/una-x/main/campaign_data/inputs/upstream_Boston/Boston/local_audit_manifest.csv
HARNESS=$RUN/p05s/run_screen_leg.py

# ---------- B10 frozen identities + q00 digests + v2 input shas ----------
say "B10 begin (frozen-identity verification)"
[ "$(sha "$SCRATCH/dist_a/urban_network_analysis-2.6.0-py3-none-any.whl")" = "f0504f886eb9976150a56dcfa818188cdedb172a13716206003b8b1e1a1dc4b1" ] || exit 10
[ "$(sha "$SCRATCH/dist_b/urban_network_analysis-2.6.0-py3-none-any.whl")" = "9a84517b9a524d4632c990a2332f8ecc219d6d3dd844528e8dac9c8dafc58d97" ] || exit 11
[ -x "$SCRATCH/venv_a/bin/python" ] || exit 12
[ -x "$SCRATCH/venv_b/bin/python" ] || exit 13
[ "$(sha "$HARNESS")" = "$(python3 -c "import json;print(json.load(open('$Q00/PRIMARY.json'))['harness_sha256'])")" ] || exit 14
[ "$(sha "$PAIRING")" = "$(python3 -c "import json;print(json.load(open('$Q00/PRIMARY.json'))['workload_sha256'])")" ] || exit 15
python3 - <<'EOF' || exit 16
import hashlib, json
Q = "campaign_runs/una_10x/tenx-20261005T2331Z/q00"
def sha(p): return hashlib.sha256(open(p, "rb").read()).hexdigest()
f = json.load(open(f"{Q}/freeze.json"))
assert f["primary_sha256"] == sha(f"{Q}/PRIMARY.json"), "PRIMARY bytes drifted since freeze"
assert f["configuration_sha256"] == sha(f"{Q}/configuration.json")
errata = json.load(open(f"{Q}/errata_pc_inputs.json"))
for name, s in errata["v2_inputs_sha256"].items():
    got = sha(f"{Q}/inputs/{name}.csv")
    assert got == s, (name, got, s)
print("Q00 digests intact; v2 protected inputs match errata")
EOF
for t in q10_p1_a q10_p7_b; do [ -f "$Q10/legs/$t.json" ] || exit 17; done
say "B10 ok (wheels/venvs/harness/manifest/Q00 digests + primary legs present)"

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

run_leg() {
  local tag=$1 venv=$2 pairing=$3 legjson=$4
  local outdir="/tmp/q10/out_$tag" ncd="/tmp/q10/nc_$tag" sampler="$Q10/sampler/$tag.loadavg.txt"
  rm -rf "$outdir" "$ncd" "$sampler"; mkdir -p "$outdir" "$ncd"
  gate || return 61
  ( while true; do print -r -- "$(date -u +%s) $(sysctl -n vm.loadavg | awk '{print $2}')" >> "$sampler"; sleep 60; done ) &
  local SPID=$!
  NUMBA_CACHE_DIR=$ncd PYTHONDONTWRITEBYTECODE=1 UNA_BATCH_PHASES= UNA_NATIVE_TRACE= \
    "$venv/bin/python" "$HARNESS" \
    --tree /Users/alansynn/orca/workspaces/una-x/main \
    --pairing "$pairing" --data-folder "$DATAFOLDER" \
    --outdir "$outdir" --tag "$tag" --out "$legjson" \
    --parallel --workers 4 --threads 1 --queue-depth 64 --cache-mode off
  local rc=$?
  kill $SPID 2>/dev/null; wait $SPID 2>/dev/null
  [ $rc -eq 0 ] || { say "LEG FAIL tag=$tag rc=$rc"; return 62; }
  say "LEG OK tag=$tag wall=$(python3 -c "import json;print(json.load(open('$legjson'))['timing']['public_call_wall_s'])")s"
}

# ---------- B3x protected pairs (explicit arm lists; NO $ARMS indirection) ----------
say "B30 begin (protected pairs, v2 inputs)"
pc=0
for spec in "pc1_tiny_gis_beta00_v2" "pc2_tiny_gis_beta31_v2" \
            "pc3_search_radius1200_v2" "pc4_search_radius1600_v2" "pc5_cold_2row_v2"; do
  INPUT=$Q00/inputs/$spec.csv
  if [ $((pc % 2)) -eq 0 ]; then
    for ARM in a b; do
      if [ "$ARM" = "a" ]; then VENV=$SCRATCH/venv_a; else VENV=$SCRATCH/venv_b; fi
      run_leg "q10_${spec}_${ARM}" "$VENV" "$INPUT" "$Q10/legs/q10_${spec}_${ARM}.json" || exit 310
    done
  else
    for ARM in b a; do
      if [ "$ARM" = "a" ]; then VENV=$SCRATCH/venv_a; else VENV=$SCRATCH/venv_b; fi
      run_leg "q10_${spec}_${ARM}" "$VENV" "$INPUT" "$Q10/legs/q10_${spec}_${ARM}.json" || exit 310
    done
  fi
  say "B3x pair $spec complete"
  pc=$((pc+1))
done
say "B30 ok (5 protected pairs complete)"

# ---------- B40 paired_samples assembly (7 primary pairs) ----------
say "B40 begin (paired_samples assembly + per-window asserts)"
python3 - <<'PYEOF' || exit 40
import hashlib, json, re
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
    venv = "/tmp/i00/venv_a" if arm == "A" else "/tmp/i00/venv_b"
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

say "B50 begin (gate_10x)"
rm -f "$Q10/gate_10x.json"
GATE_RC=0
python3 campaigns/una_10x/tools/gate_10x.py --samples "$Q10/paired_samples.json" --out "$Q10/gate_10x.json" || GATE_RC=$?
say "B50 done gate_rc=$GATE_RC (2 = below target, recorded honestly)"

# ---------- B60 raw_index + protected criterion ----------
say "B60 begin (raw_index + protected criterion)"
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
cells = []
for name in ("pc1_tiny_gis_beta00_v2", "pc2_tiny_gis_beta31_v2", "pc3_search_radius1200_v2",
             "pc4_search_radius1600_v2", "pc5_cold_2row_v2"):
    la = json.load(open(f"{Q10}/legs/q10_{name}_a.json"))
    lb = json.load(open(f"{Q10}/legs/q10_{name}_b.json"))
    ma, mb = norm_map(la), norm_map(lb)
    assert ma and ma == mb, (name, "protected behavior parity FAILED", set(ma) ^ set(mb))
    assert la["zero_failed_jobs"] and lb["zero_failed_jobs"]
    assert la["pins"]["manifest_sha256"] == errata["v2_inputs_sha256"][name], name
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
   {"fire": 1, "console": "q10/console_fire_q10_fire1_<sha8>.txt (committed alongside)",
    "scope": "T10 + 7 primary pairs complete (legs q10_p1..p7); 28 LEG OK lines",
    "halt": "protected pc1 LEG FAIL rc=1: zsh $ARMS non-split tag; deeper cause q00 v1 "
            "protected inputs missing header row (pandas parsed 0/1 records)"},
   {"fire": 2, "console": "this console (console_fire_q10b.log)",
    "scope": "B10 identity re-verification + 5 protected pairs on v2 inputs + B40/B50/B60",
    "halt": None},
 ],
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
say "Q10B FIRE COMPLETE"
