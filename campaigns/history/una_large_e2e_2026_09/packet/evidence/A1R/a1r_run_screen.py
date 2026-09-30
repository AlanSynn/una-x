"""A1R screen supervisor: lease windows + paired-run matrix for the A1
marginal screen (O3_ACCESS, O3_HOLDOUT; B0 vs A1 candidate @ 7a27866).

Stages (one lease window per invocation, H05 supervisor discipline):
  --stage build    candidate wheel + fresh venv (a1r_build_candidate.py)
  --stage run      next pending matrix slot in fixed order (AB/BA alternation,
                   A = B0); cold slots wipe their arm's NUMBA_CACHE_DIR first
  --stage matrix   descriptive aggregate of the raw records (pair timings,
                   cold ratios, array-hash visibility). NO verdicts here —
                   disposition is h04-reviewer's.

Matrix (per cell): pair0 cold AB (b0_cold, cand_cold; per-arm cache root wiped),
pairs 1..3 warm: AB, BA, AB. 8 runs per cell, 16 total.
Fleet finding enforced: each arm gets its OWN NUMBA_CACHE_DIR root (cache
entries pickle the loading module name — arms must never share a cache).
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time

CAMPAIGN = "/Users/alansynn/orca/workspaces/una-x"
REPO = f"{CAMPAIGN}/wt-large-e2e"
EV = f"{REPO}/campaigns/una_large_e2e/evidence/A1R"
DATA = f"{CAMPAIGN}/campaign_data"
RAW = f"{EV}/raw"
LEASE_LOG = f"{DATA}/a1r_lease_log.json"
H05_LEASE_LOG = f"{DATA}/h05_lease_log.json"
MATRIX = f"{EV}/screen_matrix.json"
PHYS_BYTES = 17179869184
PRESSURE_STOP = max(1 << 30, int(0.10 * PHYS_BYTES))  # 1717986918
HEAVY_WALL_BUDGET_S = 14400.0
OPENING_BALANCE_S = 426.4  # consumed by H05 (h05_lease_log cumulative)
RUN_TIMEOUT_S = 2400
LEASE_ID = "A1R-lease-20260926"
CAND_COMMIT = "7a27866670ae2b7f9ed6a4d61e01d95c18e17f29"

MANIFESTS = {
    "O3_ACCESS": f"{REPO}/tests/large_e2e/inputs/O3_ACCESS.manifest.json",
    "O3_HOLDOUT": f"{REPO}/tests/large_e2e/inputs/O3_HOLDOUT.manifest.json",
}
ARMS = {
    "b0": {"python": f"{DATA}/venvs/b0_wheel/bin/python",
           "site_packages": f"{DATA}/venvs/b0_wheel/lib/python3.11/site-packages",
           "cache_root": f"{DATA}/nbc_a1r_b0",
           "wheel_record": f"{DATA}/h05_wheel_and_venv.json"},
    "cand": {"python": f"{DATA}/venvs/a1r_cand_wheel/bin/python",
             "site_packages": f"{DATA}/venvs/a1r_cand_wheel/lib/python3.11/site-packages",
             "cache_root": f"{DATA}/nbc_a1r_cand",
             "wheel_record": f"{DATA}/a1r_wheel_and_venv.json"},
}
# fixed execution order; (cell, run_id, arm, pair_index, order_label, cold)
SLOTS = []
for _cell in ("O3_ACCESS", "O3_HOLDOUT"):
    SLOTS.append((_cell, f"a1r_{_cell}_b0_cold", "b0", 0, "AB", True))
    SLOTS.append((_cell, f"a1r_{_cell}_cand_cold", "cand", 0, "AB", True))
    SLOTS.append((_cell, f"a1r_{_cell}_b0_w1", "b0", 1, "AB", False))
    SLOTS.append((_cell, f"a1r_{_cell}_cand_w1", "cand", 1, "AB", False))
    SLOTS.append((_cell, f"a1r_{_cell}_cand_w2", "cand", 2, "BA", False))
    SLOTS.append((_cell, f"a1r_{_cell}_b0_w2", "b0", 2, "BA", False))
    SLOTS.append((_cell, f"a1r_{_cell}_b0_w3", "b0", 3, "AB", False))
    SLOTS.append((_cell, f"a1r_{_cell}_cand_w3", "cand", 3, "AB", False))


def load_lease():
    if os.path.exists(LEASE_LOG):
        with open(LEASE_LOG) as f:
            return json.load(f)
    with open(H05_LEASE_LOG) as f:
        h05 = json.load(f)
    return {"lease_id": LEASE_ID,
            "opening_balance_s": OPENING_BALANCE_S,
            "opening_balance_source": H05_LEASE_LOG,
            "opening_balance_matches_h05_cumulative":
                abs(h05.get("cumulative_heavy_s", 0.0) - OPENING_BALANCE_S) < 0.05,
            "windows": [],
            "cumulative_heavy_s": 0.0}


def save_lease(log):
    with open(LEASE_LOG, "w") as f:
        json.dump(log, f, indent=1)


def guard_and_launch(cmd, env, what, timeout_s, run_id=None):
    import psutil
    log = load_lease()
    avail = psutil.virtual_memory().available
    ceiling = min(10650 * (1 << 20), int(0.80 * avail))
    entry = {"what": what, "lease_id": LEASE_ID,
             "start_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
             "available_at_admission_bytes": avail,
             "ceiling_bytes": ceiling, "pressure_stop_bytes": PRESSURE_STOP,
             "loadavg_at_admission": os.getloadavg(),
             "cmd": cmd,
             "env": {k: v for k, v in env.items() if k.startswith(("NUMBA", "L1"))}}
    used_before = log["cumulative_heavy_s"]
    if avail < PRESSURE_STOP:
        entry["refused"] = "available < pressure stop before launch (RESOURCES.md)"
        log["windows"].append(entry)
        save_lease(log)
        print(f"[a1r] REFUSED: available {avail} < pressure stop {PRESSURE_STOP}")
        return 2
    if OPENING_BALANCE_S + used_before >= HEAVY_WALL_BUDGET_S:
        entry["refused"] = "heavy wall budget exhausted (incl. H05 opening balance)"
        log["windows"].append(entry)
        save_lease(log)
        print("[a1r] REFUSED: heavy wall exhausted")
        return 2
    log["windows"].append(entry)
    save_lease(log)

    t0 = time.monotonic()
    try:
        p = subprocess.run(cmd, env={**os.environ, **env}, timeout=timeout_s,
                           capture_output=True, text=True)
        rc = p.returncode
        tail = (p.stdout[-3000:] + p.stderr[-3000:])
    except subprocess.TimeoutExpired:
        rc, tail = -9, f"TIMEOUT after {timeout_s}s"
    wall = time.monotonic() - t0

    log = load_lease()
    entry = log["windows"][-1]
    breach_file = None
    if run_id:
        cand = os.path.join(RAW, f"pressure_breach_{run_id}.json")
        if os.path.exists(cand):
            breach_file = cand
    entry.update({"end_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                  "wall_s": round(wall, 1), "rc": rc,
                  "pressure_breach_file": breach_file,
                  "watchdog_note": ("pressure stop crossed mid-run; watchdog SIGTERM'd "
                                    "the child per RESOURCES.md" if breach_file else None),
                  "tail": tail[-2000:],
                  "loadavg_at_completion": os.getloadavg()})
    log["cumulative_heavy_s"] = round(
        OPENING_BALANCE_S + sum(w.get("wall_s", 0.0) for w in log["windows"]), 1)
    save_lease(log)
    print(f"[a1r] rc={rc} wall={wall:.1f}s "
          f"cumulative_incl_opening={log['cumulative_heavy_s']}s")
    return rc


def cache_state(root):
    n = 0
    for _, _, files in os.walk(root):
        n += len(files)
    return {"path": root, "exists": os.path.isdir(root), "file_count": n}


def slot_done(cell, run_id):
    return os.path.exists(f"{RAW}/{run_id}.json") and \
        os.path.exists(f"{RAW}/{run_id}.a1r.json")


def stage_run(only_run_id=None):
    os.makedirs(RAW, exist_ok=True)
    pending = [s for s in SLOTS
               if not slot_done(s[0], s[1])
               and (only_run_id is None or s[1] == only_run_id)]
    if not pending:
        print("[a1r] no pending slots (all records + sidecars present)")
        return 0
    cell, run_id, arm, pair_idx, order, cold = pending[0]
    a = ARMS[arm]
    # prerequisites before any wipe or launch
    if not os.path.exists(a["python"]):
        print(f"[a1r] REFUSED: arm venv python missing: {a['python']} "
              f"(run --stage build first for the candidate arm)")
        return 2
    if not os.path.exists(a["wheel_record"]):
        print(f"[a1r] REFUSED: arm wheel record missing: {a['wheel_record']}")
        return 2
    # per-arm cache root, wiped exactly at the arm's cold slot (fleet finding:
    # arms never share a cache root; cold = first run on the wiped root)
    wiped = False
    if cold:
        if os.path.exists(a["cache_root"]):
            shutil.rmtree(a["cache_root"])
        os.makedirs(a["cache_root"])
        wiped = True
    env = {"NUMBA_CACHE_DIR": a["cache_root"]}
    cmd = [a["python"], f"{REPO}/benchmarks/large_e2e/h05_profile.py",
           "--manifest", MANIFESTS[cell], "--cell", cell, "--mode", "clean",
           "--out", f"{RAW}/{run_id}.json",
           "--output-root", f"{DATA}/a1r_out/{run_id}",
           "--expected-site-packages", a["site_packages"],
           "--lease-id", LEASE_ID, "--run-id", run_id]
    rc = guard_and_launch(cmd, env, f"run:{run_id}", RUN_TIMEOUT_S, run_id=run_id)

    # supervisor sidecar: A1R context around the untouched instrument record
    wheel_sha = None
    with open(a["wheel_record"]) as f:
        wheel_sha = json.load(f)["wheel"]["sha256"]
    sidecar = {
        "task": "A1R", "record": "run_sidecar", "role": "screen-executor",
        "run_id": run_id, "cell": cell, "arm": arm,
        "candidate_commit": CAND_COMMIT if arm == "cand" else None,
        "arm_python": a["python"], "arm_site_packages": a["site_packages"],
        "numba_cache_root": a["cache_root"],
        "cache_root_wiped_before_run": wiped,
        "cache_root_state_after_run": cache_state(a["cache_root"]),
        "wheel_sha256": wheel_sha,
        "pair_index": pair_idx, "pair_order": order,
        "slot_is_cold": cold,
        "screening_note": "raw instrument record is byte-authored by h05_profile.py "
                          "(record.task stays 'H05' by instrument design); this "
                          "sidecar carries the A1R context",
        "supervisor_rc": rc,
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with open(f"{RAW}/{run_id}.a1r.json", "w") as f:
        json.dump(sidecar, f, indent=1)
    # failed evidence is retained: even on rc != 0 the instrument record (if any)
    # and this sidecar stay; the slot does NOT auto-retry
    return rc


def stage_matrix():
    rows = []
    problems = []
    a_sp = {k: v["site_packages"] for k, v in ARMS.items()}
    for cell, run_id, arm, pair_idx, order, cold in SLOTS:
        rp = f"{RAW}/{run_id}.json"
        sp = f"{RAW}/{run_id}.a1r.json"
        if not (os.path.exists(rp) and os.path.exists(sp)):
            problems.append(f"{run_id}: missing record/sidecar")
            continue
        with open(rp) as f:
            r = json.load(f)
        with open(sp) as f:
            s = json.load(f)
        win_s = r.get("window", {}).get("application_window_ns", 0) / 1e9
        arrays = {k: v.get("sha256_bytes") for k, v in
                  (r.get("accessibility_arrays") or {}).items()
                  if isinstance(v, dict)}
        imp = r.get("import") or {}
        id_ok = (str(imp.get("identity_assertion", "")).startswith("ok")
                 and imp.get("expected_site_packages") == a_sp[arm])
        rows.append({
            "run_id": run_id, "cell": cell, "arm": arm, "pair_index": pair_idx,
            "pair_order": order, "cold": cold,
            "window_s": round(win_s, 4),
            "peak_rss_tree_mib": r.get("peak_rss_tree_sampled_mib"),
            "pressure_breach": bool(r.get("pressure_breach")),
            "rc": s.get("supervisor_rc"),
            "numba_cache_root": s.get("numba_cache_root"),
            "cache_root_wiped_before_run": s.get("cache_root_wiped_before_run"),
            "cache_files_after": (s.get("cache_root_state_after_run") or {}).get("file_count"),
            "numba_cache_dir_env": (r.get("env") or {}).get("NUMBA_CACHE_DIR"),
            "import_identity_ok": id_ok,
            "imported_from": imp.get("urban_network_analysis_file"),
            "counts": r.get("counts"),
            "accessibility_array_sha256": arrays,
            "verbatim_warnings": r.get("verbatim_warnings"),
        })
    summary = {
        "task": "A1R", "record": "screen_matrix", "role": "screen-executor",
        "descriptive_only": "timings/hashes aggregated for h04-reviewer's "
                            "disposition; NO gate verdicts are issued here",
        "candidate_commit": CAND_COMMIT,
        "lease_id": LEASE_ID,
        "lease_log": LEASE_LOG,
        "heavy_wall": {"opening_balance_s": OPENING_BALANCE_S,
                       "cumulative_incl_opening_s": load_lease()["cumulative_heavy_s"],
                       "budget_s": HEAVY_WALL_BUDGET_S},
        "arms": {k: {"python": v["python"], "cache_root": v["cache_root"],
                     "wheel_sha256": (json.load(open(v["wheel_record"]))
                                      .get("wheel", {}).get("sha256")
                                      if os.path.exists(v["wheel_record"]) else None)}
                 for k, v in ARMS.items()},
        "fleet_finding_applied": "per-arm NUMBA_CACHE_DIR roots (cache entries "
                                 "pickle the loading module name)",
        "slots": rows, "problems": problems,
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with open(MATRIX, "w") as f:
        json.dump(summary, f, indent=1)
    print(f"[a1r] matrix -> {MATRIX} slots={len(rows)} problems={problems}")
    return 0 if not problems else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=["build", "run", "matrix", "status"])
    ap.add_argument("--run-id", default=None,
                    help="run a specific slot instead of next-pending")
    a = ap.parse_args()
    if a.stage == "build":
        return guard_and_launch(
            [f"{CAMPAIGN}/venvs/campaign/bin/python", f"{EV}/a1r_build_candidate.py"],
            {"NUMBA_CACHE_DIR": f"{DATA}/nbc_a1r_build"}, "build", 3600)
    if a.stage == "run":
        return stage_run(a.run_id)
    if a.stage == "matrix":
        return stage_matrix()
    log = load_lease()
    print(json.dumps({"cumulative_incl_opening_s": log["cumulative_heavy_s"],
                      "budget_s": HEAVY_WALL_BUDGET_S,
                      "remaining_s": round(HEAVY_WALL_BUDGET_S - OPENING_BALANCE_S
                                           - log["cumulative_heavy_s"], 1),
                      "windows": len(log["windows"])}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
