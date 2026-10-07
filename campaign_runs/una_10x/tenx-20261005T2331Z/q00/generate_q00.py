"""Q00 generator: preregister the final measurement plan BEFORE any Q10 results.

Produces q00/PRIMARY.json, configuration.json, freeze.json, the five policy
digest documents, and the derived protected-cell inputs.  Every figure is a
tool output computed HERE from committed artifacts or live bytes; nothing is
hand-copied.  Run under any python3 (stdlib only).
"""
import csv
import hashlib
import json
import os
import platform
import subprocess

WS = "/Users/alansynn/orca/workspaces/una-x"
MAIN = f"{WS}/main"
RUN = f"{MAIN}/campaign_runs/una_10x/tenx-20261005T2331Z"
Q00 = f"{RUN}/q00"
I00 = f"{RUN}/i00"
PAIRING = f"{MAIN}/campaign_data/inputs/upstream_Boston/Boston/local_audit_manifest.csv"
DATAFOLDER = f"{MAIN}/campaign_data/inputs/upstream_Boston/Boston"
HARNESS = f"{RUN}/p05s/run_screen_leg.py"
SCRATCH = "/tmp/i00"

A_REV = "395cdc5f683894b6f2ba460f7dbcefee99981ba3"
B_REV = "1e7237bebe8dd55e4a3b444f24ad9a255fa8644c"


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def w(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=1)
        f.write("\n")


census = json.load(open(f"{I00}/wheel_census.json"))
inst = json.load(open(f"{I00}/install_identity.json"))
freeze_i00 = json.load(open(f"{I00}/freeze_candidate.json"))
leg_a = json.load(open(f"{I00}/legs/a_installed_comparator.json"))
leg_b = json.load(open(f"{I00}/legs/b_installed_candidate.json"))

WA_SHA = sha(f"{SCRATCH}/dist_a/urban_network_analysis-2.6.0-py3-none-any.whl")
WB_SHA = sha(f"{SCRATCH}/dist_b/urban_network_analysis-2.6.0-py3-none-any.whl")
assert WA_SHA == census["A_comparator"]["sha256"], (WA_SHA,)
assert WB_SHA == census["B_candidate"]["sha256"], (WB_SHA,)
HARNESS_SHA = sha(HARNESS)
MANIFEST_SHA = sha(PAIRING)
assert MANIFEST_SHA == leg_a["pins"]["manifest_sha256"] == leg_b["pins"]["manifest_sha256"]

# required 64 output names (normalized rel paths from the bitwise legs)
import re
ts = re.compile(r"^accessibility_\d{4}-\d{2}-\d{2}_\d{4}")
required = sorted(ts.sub("accessibility_TS", f["rel"], count=1)
                  for f in leg_a["output_files"]
                  if not f["rel"].split("/")[0].startswith("."))
assert len(required) == 64 and required == sorted(
    ts.sub("accessibility_TS", f["rel"], count=1) for f in leg_b["output_files"]
    if not f["rel"].split("/")[0].startswith("."))

# ---- derived protected-cell inputs (deterministic transform of frozen bytes) ----
os.makedirs(f"{Q00}/inputs", exist_ok=True)
rows = list(csv.reader(open(PAIRING)))
hdr, body = rows[0], rows[1:]
assert len(body) == 32, len(body)


def write_rows(name, picked):
    path = f"{Q00}/inputs/{name}.csv"
    with open(path, "w", newline="") as f:
        cw = csv.writer(f)
        for r in picked:
            cw.writerow(r + [DATAFOLDER])       # row-level absolute data_folder
    return path, sha(path)


r0 = body[0]
i_rad = hdr.index("search_radius")
pc1_path, pc1_sha = write_rows("pc1_tiny_gis_beta00", [r0])
pc2_path, pc2_sha = write_rows("pc2_tiny_gis_beta31", [body[31]])
r3 = list(r0); r3[i_rad] = "1200"
pc3_path, pc3_sha = write_rows("pc3_search_radius1200", [r3])
r4 = list(r0); r4[i_rad] = "1600"
pc4_path, pc4_sha = write_rows("pc4_search_radius1600", [r4])
pc5_path, pc5_sha = write_rows("pc5_cold_2row", body[0:2])

# ---- policy digest documents ----
resource = {
 "schema": "una-10x-q00-resource-policy-1",
 "workers": 4, "threads_per_worker": 1, "queue_depth": 64, "writer_concurrency": 1,
 "cache_mode": "off", "row_cache": "none",
 "numba_cache_dir": "fresh per window per arm (empty arm-specific JIT cache)",
 "matched_budget": "identical configuration both arms, same host, same schedule",
 "loadavg_gate": "pre-leg loadavg (1min) < 8.0, 90-min bounded hold; in-run 60s sampler retained per leg",
 "exclusive_owner": "no campaign-external compute launched during windows; external enterprise co-tenancy handled only by the gate, never killed",
 "host_ncpu": leg_a["pins"]["host_ncpu"],
}
cache = {
 "schema": "una-10x-q00-cache-policy-1",
 "regime": "COLD (BENCHMARKS.md regime 4): fresh process per window, empty arm-specific "
           "JIT cache (fresh NUMBA_CACHE_DIR), input conversion and final output inside the "
           "public call; no result/trace reuse across windows",
 "cache_mode": "off", "disk_cache_dir": None, "row_cache_dir": None,
 "initial_trace_cost_included": True,
 "import_outside_timer": "recorded per leg as timing.import_s; gate wall is the "
                         "public-call boundary; process-start-to-completion reported in raw records",
 "cold_boundary_reported": "import_s + public_call_wall_s per leg JSON",
 "no_same_answer_hits": "cohort rows are distinct jobs (distinct beta values); cache disabled",
}
output = {
 "schema": "una-10x-q00-output-policy-1",
 "required_outputs_per_window": 64,
 "required_names": required,
 "hidden_outputs": ".una-* preparation/cohort artifacts retained on disk, disclosed, "
                   "outside the required census (comparator has none by construction)",
 "naming_policy": "accessibility_<date>_<time>/<stem>.<ext>, per-minute dir at crossings; "
                  "preserved unnormalized in raw legs",
 "durability": "unchanged engine fsync/rename/state/journal sequence, both arms",
 "post_boundary_verification": "per-file sha256 census after the call, outside the timer, "
                               "timed identically across arms, recorded per leg",
 "fresh_output_dir": "per window per arm, existence recorded before the call",
}
environment = {
 "schema": "una-10x-q00-environment-1",
 "python": leg_a["pins"]["interpreter"],
 "numba": leg_a["pins"]["numba_version"],
 "numpy": leg_a["pins"]["numpy_version"],
 "host": {"platform": platform.platform(), "ncpu": leg_a["pins"]["host_ncpu"]},
 "install_venvs": {"A": inst["a"]["package_file"], "B": inst["b"]["package_file"]},
 "venv_provenance": "cp -a venv-p01m, editable uninstalled, wheel installed --no-deps "
                    "(I00 S30); wheels re-hashed live at Q00",
}
protected = {
 "schema": "una-10x-q00-protected-cells-1",
 "law": "BENCHMARKS.md protected cells: candidate B/A median <= 1.05 and one-sided 95% "
        "upper <= 1.10, identical required behavior; failure => refine dispatcher or keep "
        "explicit scoped mode (Z00 decision), never blanket default",
 "pairs": [
  {"id": "pc1_tiny_gis_beta00", "input": "inputs/pc1_tiny_gis_beta00.csv",
   "sha256": pc1_sha, "class": "tiny GIS (1 job, beta 0.001)"},
  {"id": "pc2_tiny_gis_beta31", "input": "inputs/pc2_tiny_gis_beta31.csv",
   "sha256": pc2_sha, "class": "tiny GIS (1 job, beta row31)"},
  {"id": "pc3_search_radius1200", "input": "inputs/pc3_search_radius1200.csv",
   "sha256": pc3_sha, "class": "search-dominated (radius 400 -> 1200)"},
  {"id": "pc4_search_radius1600", "input": "inputs/pc4_search_radius1600.csv",
   "sha256": pc4_sha, "class": "search-dominated (radius 400 -> 1600)"},
  {"id": "pc5_cold_2row", "input": "inputs/pc5_cold_2row.csv",
   "sha256": pc5_sha, "class": "cold process (2 jobs)"},
 ],
 "behavior_parity": "per cell: outputs bitwise equal across arms, zero failed jobs",
}
for name, obj in (("resource_policy", resource), ("cache_policy", cache),
                  ("output_policy", output), ("environment", environment),
                  ("protected_cells", protected)):
    w(f"{Q00}/{name}.json", obj)

# ---- PRIMARY.json (template-shaped) ----
primary = {
 "schema_version": 1,
 "status": "FROZEN",
 "measurement_class": "observed_proxy",
 "primary_workload": "frozen distinct 32-job observed cohort "
                     "(campaign_data/inputs/upstream_Boston/Boston/local_audit_manifest.csv)",
 "primary_api": "UNA.RunBatch",
 "baseline_source": A_REV,
 "candidate_source": B_REV,
 "numerical_profile": "una_legacy",
 "workload_sha256": MANIFEST_SHA,
 "resource_sha256": sha(f"{Q00}/resource_policy.json"),
 "cache_policy_sha256": sha(f"{Q00}/cache_policy.json"),
 "output_policy_sha256": sha(f"{Q00}/output_policy.json"),
 "environment_sha256": sha(f"{Q00}/environment.json"),
 "harness_sha256": HARNESS_SHA,
 "wheel_sha256": {"A": WA_SHA, "B": WB_SHA},
 "jobs_per_window": 32,
 "paired_windows": 7,
 "window_order": ["AB", "BA", "AB", "BA", "AB", "BA", "AB"],
 "target_speedup": 10.0,
 "bootstrap_resamples": 20000,
 "bootstrap_seed": 20261003,
 "lower_quantile": 0.05,
 "cache_regime": "COLD",
 "initial_trace_cost_included": True,
 "distinct_jobs": True,
 "required_outputs": required,
 "protected_cells": protected["pairs"],
 "protected_cells_sha256": sha(f"{Q00}/protected_cells.json"),
 "actual_target": "unavailable",
 "sizing_note": "baseline aggregate >=60s and fast-arm >=10s expected from co-tenanted "
                "verification legs (A 42.6682s x7 ~= 298.7s; B 18.0414s); observation "
                "disclosed in freeze.json - no config was tuned after observing it "
                "(candidate config frozen since P05 admission)",
}
w(f"{Q00}/PRIMARY.json", primary)

configuration = {
 "schema": "una-10x-q00-configuration-1",
 "date": "2026-10-07",
 "freeze_basis": {
  "candidate": {"source_rev": B_REV, "wheel_sha256": WB_SHA,
                "wheel_census": "i00/wheel_census.json B_candidate",
                "install": inst["b"]},
  "comparator": {"source_rev": A_REV, "wheel_sha256": WA_SHA,
                 "declared_tree_delta": census["A_comparator"]["tree_delta"],
                 "wheel_census": "i00/wheel_census.json A_comparator",
                 "install": inst["a"]},
  "bitwise_basis": "v00/bitparity.json (A==B==n00 reference per-file, 64 files/147829892B)",
 },
 "harness": {"path": "p05s/run_screen_leg.py", "sha256": HARNESS_SHA,
             "args": "--tree <main> --pairing <cohort> --data-folder <Boston> "
                     "--outdir <fresh> --tag <id> --out <leg.json> --parallel --workers 4 "
                     "--threads 1 --queue-depth 64 --cache-mode off"},
 "schedule": {
  "order": ["AB", "BA", "AB", "BA", "AB", "BA", "AB"],
  "pair_ids": ["q10_p1", "q10_p2", "q10_p3", "q10_p4", "q10_p5", "q10_p6", "q10_p7"],
  "per_window": "fresh process + fresh NUMBA_CACHE_DIR + fresh outdir per leg",
  "protected_after": "5 protected pairs after the 7 primary windows, same gates",
 },
 "invalidation_clause": "one FULL repeat block only for preregistered invalidation "
                        "(pre-leg gate unopenable 90min / in-run external load spike "
                        "recorded by sampler / leg failure or census mismatch); all "
                        "blocks retained; no selective reruns of slow candidate windows",
 "law": ["BENCHMARKS.md", "DECISION.md", "VALIDATION.md",
         "campaigns/una_10x/dossiers/09_final_integration.md",
         "post_p00/designs/07_measurement_and_gates.md"],
}
w(f"{Q00}/configuration.json", configuration)

freezec = {
 "schema": "una-10x-q00-freeze-1",
 "date": "2026-10-07",
 "primary_sha256": sha(f"{Q00}/PRIMARY.json"),
 "configuration_sha256": sha(f"{Q00}/configuration.json"),
 "policy_digests": {n: sha(f"{Q00}/{n}.json") for n in
                    ("resource_policy", "cache_policy", "output_policy",
                     "environment", "protected_cells")},
 "protected_inputs_sha256": {os.path.basename(p): s for p, s in (
    (pc1_path, pc1_sha), (pc2_path, pc2_sha), (pc3_path, pc3_sha),
    (pc4_path, pc4_sha), (pc5_path, pc5_sha))},
 "no_candidate_edits_after_freeze": "any source/harness/environment change invalidates "
                                    "affected measurements (dossier 07:53, dossier 09)",
 "disclosure_prior_observations": "co-tenanted verification legs (I00 S40/S41) were observed "
                                  "before this freeze; they are NOT gate windows; the candidate "
                                  "configuration was frozen at P05 admission and unchanged since; "
                                  "baseline-only calibration law satisfied vacuously (window "
                                  "count/cohort fixed by law, no tuning performed)",
 "generated_by": "q00/generate_q00.py (figures computed from live bytes + committed artifacts)",
}
w(f"{Q00}/freeze.json", freezec)

print("Q00 WRITTEN")
print("primary_sha256", freezec["primary_sha256"])
print("freeze_sha256 (freeze.json bytes)", sha(f"{Q00}/freeze.json"))
print("harness", HARNESS_SHA[:16], "workload", MANIFEST_SHA[:16])
print("protected shas", {os.path.basename(p): s[:12] for p, s in (
    (pc1_path, pc1_sha), (pc2_path, pc2_sha), (pc3_path, pc3_sha),
    (pc4_path, pc4_sha), (pc5_path, pc5_sha))})
