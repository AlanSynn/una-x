"""I00 supervisor: leased L1/L2 reprofile of the composed five-track line
vs B0, plus the per-track ablation suite re-runs on the composed tree.

Authored by h05 (implementation/performance owner) cloned from the H05
supervisor frame (evidence/H05/h05_run_pilots.py); NEW instrument — per
h04's I00 role ruling it requires h04 review BEFORE its first fire, and
every fire runs under standing R1/R2/R3 two-key authorization.

Stages (run one at a time; h05 is the sole authorized firer):
  --stage init                    chain this lease log to the A3R lease
                                  log (parent sha pin + opening cumulative);
                                  refuses if this log already exists
  --stage reprofile --arm composed|b0 --spec <name> | --all
                                  one instrumented h05_profile.py session
                                  (same 8 SPECS as H05, observed frozen
                                  workloads)
  --stage ablations               R2 custody copies of the F1I/F2I/F3I
                                  evidence tails, then the five per-track
                                  pytest suites on the composed tree
  --stage status                  print lease log + heavy-wall accounting
                                  (read-only, never writes)

Arms:
  composed  venvs/campaign/bin/python + --diagnostic-source-root src
            (the campaign venv has NO installed urban_network_analysis;
            records are measurement_class diagnostic_profile_source and
            NEVER qualify — they are the I00 reprofile diagnostics)
  b0        campaign_data/venvs/b0_wheel/bin/python +
            --expected-site-packages (the B0 wheel install)

Every heavy child: bounded timeout, pre-launch pressure guard (available
vs pressure stop), lease window recording, cumulative heavy-wall check
against the 14400 s campaign budget. Children get PYTHONDONTWRITEBYTECODE=1
and a per-arm NUMBA_CACHE_DIR (fresh per-arm roots, disjoint from
nbc_composition and from each other) so no import writes bytecode into
the repo and no cache root is silently reused.
"""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time

CAMPAIGN = "/Users/alansynn/orca/workspaces/una-x"
REPO = f"{CAMPAIGN}/wt-large-e2e"
DATA = f"{CAMPAIGN}/campaign_data"
SRC = f"{REPO}/src"
PY_COMPOSED = f"{CAMPAIGN}/venvs/campaign/bin/python"
PY_B0 = f"{DATA}/venvs/b0_wheel/bin/python"
SITE_B0 = f"{DATA}/venvs/b0_wheel/lib/python3.11/site-packages"
NBC_COMPOSED = f"{DATA}/nbc_i00_composed"
NBC_B0 = f"{DATA}/nbc_i00_b0"
SESSIONS = f"{DATA}/i00_sessions"
CAPTURES = f"{DATA}/i00_captures"
OUTS = f"{DATA}/i00_out"
LEASE_LOG = f"{DATA}/i00_lease_log.json"
CUSTODY = DATA

PARENT_LOG = f"{DATA}/a3r_lease_log.json"
PARENT_SHA256 = ("30190534ba307f63ab6ba30c2db68684c"
                 "2010d5a6f3cc6f79fda47adf5fd433b")
# chain accounting: the parent's own ledger field, NOT a wall_s re-sum
# (a3r schema: cumulative_charged_s = opening_balance_s + in-log wall_s)
EXPECTED_PARENT_CHARGED_S = 1416.0
PARENT_CHARGED_TOL_S = 0.05

HEAVY_WALL_BUDGET_S = 14400.0
PHYS_BYTES = 17179869184
PRESSURE_STOP = max(1 << 30, int(0.10 * PHYS_BYTES))  # 1717986918
LAUNCH_TIMEOUT_S = 7200
ABLATION_TIMEOUT_S = 3600

MANIFESTS = {
    "O2": f"{REPO}/tests/large_e2e/inputs/O2.manifest.json",
    "O3_ACCESS": f"{REPO}/tests/large_e2e/inputs/O3_ACCESS.manifest.json",
    "O3_FLOW": f"{REPO}/tests/large_e2e/inputs/O3_FLOW.manifest.json",
    "O3_HOLDOUT": f"{REPO}/tests/large_e2e/inputs/O3_HOLDOUT.manifest.json",
}
FLOW_DECISION = f"{DATA}/h05_flow_decision.json"  # frozen: chosen_variant

# cell -> run name -> h05_profile.py invocation knobs (same 8 as H05)
SPECS = {
    "O2_cold":        {"cell": "O2", "mode": "clean", "fresh_cache": True},
    "O2_warm":        {"cell": "O2", "mode": "clean"},
    "ACCESS_od":      {"cell": "O3_ACCESS", "mode": "od_level", "capture": True},
    "ACCESS_clean":   {"cell": "O3_ACCESS", "mode": "clean"},
    "FLOW_sel1024_od": {"cell": "O3_FLOW", "mode": "od_level", "capture": True,
                        "origins": "sel1024"},
    "FLOW_chosen_od": {"cell": "O3_FLOW", "mode": "od_level", "capture": True,
                       "origins": "CHOSEN"},
    "HOLDOUT_clean":  {"cell": "O3_HOLDOUT", "mode": "clean"},
    "HOLDOUT_elev":   {"cell": "O3_HOLDOUT", "mode": "clean",
                       "extra": {"elevation": True, "elevation_penalty": 4,
                                 "network_file": "20260703_PercLenNetwork_InnerCore_3D.geojson"},
                       "verify_3d": True},
}
HOLDOUT_3D_SHA = "6f72bcb2aedecb72b36298645b45ca5551ec32514ab017c8113cb54f3530f9f5"
HOLDOUT_3D_PATH = (f"{CAMPAIGN}/campaign_data/inputs/"
                   "20260703_PercLenNetwork_InnerCore_3D.geojson")

ARMS = {
    "composed": {"py": PY_COMPOSED,
                 "site_args": ["--diagnostic-source-root", SRC],
                 "nbc": NBC_COMPOSED},
    "b0":       {"py": PY_B0,
                 "site_args": ["--expected-site-packages", SITE_B0],
                 "nbc": NBC_B0},
}
SPEC_ORDER = ["O2_cold", "O2_warm", "ACCESS_od", "ACCESS_clean",
              "FLOW_sel1024_od", "FLOW_chosen_od",
              "HOLDOUT_clean", "HOLDOUT_elev"]

# per-track ablation suites (suite dir under tests/large_e2e/)
ABLATIONS = ["A1", "A3", "F1", "F2", "F3"]
# evidence tails the F-suite harnesses OVERWRITE on sessionfinish
# (harness_f1/harness_f3 hard-code ARTIFACT_PATH; harness_f2 defaults
# there with the UNA_F2_ARTIFACTS override unused here)
ABLATION_TAILS = {
    "F1": f"{REPO}/campaigns/una_large_e2e/evidence/F1I/test_artifacts.json",
    "F2": f"{REPO}/campaigns/una_large_e2e/evidence/F2I/test_artifacts.json",
    "F3": f"{REPO}/campaigns/una_large_e2e/evidence/F3I/test_artifacts.json",
}


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_lease_log():
    if os.path.exists(LEASE_LOG):
        with open(LEASE_LOG) as f:
            return json.load(f)
    return None


def save_lease_log(log):
    with open(LEASE_LOG, "w") as f:
        json.dump(log, f, indent=1)


def stage_init(lease_id):
    if load_lease_log() is not None:
        print("[i00-run] REFUSED: lease log already exists (no double init)")
        return 2
    if not os.path.exists(PARENT_LOG):
        print("[i00-run] REFUSED: parent A3R lease log missing")
        return 2
    got = sha256_file(PARENT_LOG)
    if got != PARENT_SHA256:
        print(f"[i00-run] REFUSED: parent log sha {got} != pinned")
        return 2
    with open(PARENT_LOG) as f:
        parent = json.load(f)
    opening = round(float(parent["cumulative_charged_s"]), 1)
    if abs(opening - EXPECTED_PARENT_CHARGED_S) > PARENT_CHARGED_TOL_S:
        print(f"[i00-run] REFUSED: parent cumulative_charged_s {opening} != "
              f"{EXPECTED_PARENT_CHARGED_S}")
        return 2
    log = {
        "lease_id": lease_id,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "parent_log": PARENT_LOG,
        "parent_log_sha256": got,
        "opening_balance_s": opening,
        "opening_balance_source": ("cumulative_charged_s of "
                                   f"{PARENT_LOG} read fresh at init"),
        "budget_s": HEAVY_WALL_BUDGET_S,
        "pressure_stop_bytes": PRESSURE_STOP,
        "windows": [],
        "cumulative_charged_s": opening,
        "remaining_s": round(HEAVY_WALL_BUDGET_S - opening, 1),
    }
    save_lease_log(log)
    print(f"[i00-run] init: chained to {os.path.basename(PARENT_LOG)} "
          f"(opening {opening}s of {HEAVY_WALL_BUDGET_S}s)")
    return 0


def guard_and_launch(cmd, env, lease_id, what, timeout_s=LAUNCH_TIMEOUT_S,
                     window_class="heavy", cwd=None):
    import psutil
    log = load_lease_log()
    if log is None:
        print("[i00-run] REFUSED: lease not initialized (run --stage init)")
        return 2
    avail = psutil.virtual_memory().available
    ceiling = min(10650 * (1 << 20), int(0.80 * avail))
    entry = {
        "what": what, "class": window_class, "lease_id": lease_id,
        "start_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "available_at_admission_bytes": avail,
        "ceiling_bytes": ceiling, "pressure_stop_bytes": PRESSURE_STOP,
        "loadavg_at_admission": os.getloadavg(),
        "cmd": cmd,
        "env": {k: v for k, v in env.items()
                if k.startswith(("NUMBA", "L1", "PYTHONDONT", "UNA_"))},
    }
    if avail < PRESSURE_STOP:
        entry["refused"] = "available < pressure stop before launch (RESOURCES.md)"
        log["windows"].append(entry)
        save_lease_log(log)
        print(f"[i00-run] REFUSED: available {avail} < pressure stop {PRESSURE_STOP}")
        return 2
    if log["cumulative_charged_s"] >= HEAVY_WALL_BUDGET_S:
        entry["refused"] = "heavy wall budget exhausted"
        log["windows"].append(entry)
        save_lease_log(log)
        print("[i00-run] REFUSED: heavy wall exhausted")
        return 2
    log["windows"].append(entry)
    save_lease_log(log)

    t0 = time.monotonic()
    try:
        p = subprocess.run(cmd, env={**os.environ, **env}, timeout=timeout_s,
                           capture_output=True, text=True, cwd=cwd)
        rc, tail = p.returncode, (p.stdout[-3000:] + p.stderr[-3000:])
    except subprocess.TimeoutExpired:
        rc, tail = -9, f"TIMEOUT after {timeout_s}s"
    wall = time.monotonic() - t0

    log = load_lease_log()
    entry = log["windows"][-1]
    breach_file = None
    if what.startswith("pilot:"):
        _, arm, spec_name = what.split(":", 2)
        # driver writes dirname(--out)/pressure_breach_<run_id>.json
        # (h05_profile.py :1004-1007)
        cand = f"{SESSIONS}/pressure_breach_{arm}_{spec_name}.json"
        if os.path.exists(cand):
            breach_file = cand
    entry.update({"end_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                  "wall_s": round(wall, 1), "rc": rc,
                  "pressure_breach_file": breach_file,
                  "watchdog_note": ("pressure stop crossed mid-run; watchdog SIGTERM'd "
                                    "the child per RESOURCES.md" if breach_file else None),
                  "tail": tail[-2000:],
                  "loadavg_at_completion": os.getloadavg()})
    log["cumulative_charged_s"] = round(
        log["opening_balance_s"]
        + sum(w.get("wall_s", 0.0) or 0.0 for w in log["windows"]), 1)
    log["remaining_s"] = round(HEAVY_WALL_BUDGET_S - log["cumulative_charged_s"], 1)
    log["updated_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    save_lease_log(log)
    print(f"[i00-run] rc={rc} wall={wall:.1f}s "
          f"charged={log['cumulative_charged_s']}s "
          f"remaining={log['remaining_s']}s")
    return rc


def pilot_env(arm, spec):
    nbc = ARMS[arm]["nbc"]
    env = {"NUMBA_CACHE_DIR": nbc, "PYTHONDONTWRITEBYTECODE": "1"}
    if spec.get("fresh_cache"):
        if os.path.exists(nbc):
            shutil.rmtree(nbc)
        os.makedirs(nbc)
    return env


def run_pilot(arm, spec_name, lease_id):
    spec = SPECS[spec_name]
    if spec.get("verify_3d"):
        h = sha256_file(HOLDOUT_3D_PATH)
        if h != HOLDOUT_3D_SHA:
            print(f"[i00-run] REFUSED: 3D network hash {h} != pinned")
            return 2
    a = ARMS[arm]
    cell = spec["cell"]
    out_json = f"{SESSIONS}/{arm}_{spec_name}.json"
    out_root = f"{OUTS}/{arm}_{spec_name}"
    cap = f"{CAPTURES}/{arm}_{cell}" if spec.get("capture") else None
    cmd = [a["py"], f"{REPO}/benchmarks/large_e2e/h05_profile.py",
           "--manifest", MANIFESTS[cell], "--cell", cell,
           "--mode", spec["mode"], "--out", out_json, "--output-root", out_root,
           *a["site_args"],
           "--pressure-stop-bytes", str(PRESSURE_STOP),
           "--lease-id", lease_id, "--run-id", f"{arm}_{spec_name}"]
    if spec.get("origins"):
        choice = spec["origins"]
        if choice == "CHOSEN":
            with open(FLOW_DECISION) as f:
                choice = json.load(f)["chosen_variant"]
        cmd += ["--variant-id", choice]
    if spec.get("extra"):
        cmd += ["--extra-settings-json", json.dumps(spec["extra"])]
    if cap:
        cmd += ["--capture-dir", cap]
    os.makedirs(SESSIONS, exist_ok=True)
    os.makedirs(OUTS, exist_ok=True)
    return guard_and_launch(cmd, pilot_env(arm, spec), lease_id,
                            f"pilot:{arm}:{spec_name}")


def stage_ablations(lease_id):
    rc_total = 0
    # R2 custody: copy each live F-suite evidence tail BEFORE any suite
    # can overwrite it (harness sessionfinish flushes even on failure).
    custody_made = []
    for suite, path in ABLATION_TAILS.items():
        if os.path.exists(path):
            digest = sha256_file(path)
            dst = f"{CUSTODY}/interim_{digest[:8]}_i00_{suite}I_test_artifacts.json"
            shutil.copy2(path, dst)
            custody_made.append({"suite": suite, "src": path,
                                 "sha256": digest, "copy": dst})
    log = load_lease_log()
    if log is None:
        print("[i00-run] REFUSED: lease not initialized (run --stage init)")
        return 2
    log["windows"].append({
        "what": "custody:f_suite_tails", "class": "custody",
        "lease_id": lease_id,
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "copies": custody_made, "wall_s": 0.0, "rc": 0,
    })
    save_lease_log(log)
    for c in custody_made:
        print(f"[i00-run] custody {c['suite']}: {c['sha256'][:8]} -> "
              f"{os.path.basename(c['copy'])}")

    for suite in ABLATIONS:
        suite_dir = f"{REPO}/tests/large_e2e/{suite}"
        cmd = [PY_COMPOSED, "-m", "pytest",
               "-p", "no:cacheprovider", "-q"]
        env = {"PYTHONDONTWRITEBYTECODE": "1"}
        rc = guard_and_launch(cmd, env, lease_id, f"ablation:{suite}",
                              timeout_s=ABLATION_TIMEOUT_S,
                              window_class="ablation", cwd=suite_dir)
        if rc != 0:
            rc_total = rc if rc_total == 0 else rc_total
    return rc_total


def stage_status():
    log = load_lease_log()
    if log is None:
        print("[i00-run] no lease log (not initialized)")
        return 0
    print(json.dumps({k: log[k] for k in
                      ("lease_id", "parent_log_sha256", "opening_balance_s",
                       "cumulative_charged_s", "remaining_s", "budget_s")},
                     indent=1))
    print(f"[i00-run] {len(log['windows'])} windows")
    for w in log["windows"]:
        print(f"  {w.get('what')}: rc={w.get('rc')} wall={w.get('wall_s')}s"
              f"{' REFUSED: ' + w['refused'] if w.get('refused') else ''}")
    return 0
    for w in log["windows"]:
        print(f"  {w.get('what')}: rc={w.get('rc')} wall={w.get('wall_s')}s"
              f"{' REFUSED: ' + w['refused'] if w.get('refused') else ''}")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=["init", "reprofile", "ablations", "status"])
    ap.add_argument("--arm", choices=sorted(ARMS), default=None)
    ap.add_argument("--spec", default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--lease-id", default="I00")
    args = ap.parse_args()

    if args.stage != "status":
        if not sys.prefix.startswith(f"{CAMPAIGN}/venvs/campaign"):
            print("[i00-run] REFUSED: supervisor must run under the "
                  "campaign venv interpreter")
            return 2

    if args.stage == "init":
        return stage_init(args.lease_id)
    if args.stage == "reprofile":
        if not args.arm:
            print("[i00-run] REFUSED: --arm required")
            return 2
        names = SPEC_ORDER if args.all else [args.spec]
        for n in names:
            if n not in SPECS:
                print(f"[i00-run] REFUSED: unknown spec {n}")
                return 2
        rc_total = 0
        for n in names:
            rc = run_pilot(args.arm, n, args.lease_id)
            if rc != 0:
                rc_total = rc if rc_total == 0 else rc_total
        return rc_total
    if args.stage == "ablations":
        return stage_ablations(args.lease_id)
    return stage_status()


if __name__ == "__main__":
    sys.exit(main())
