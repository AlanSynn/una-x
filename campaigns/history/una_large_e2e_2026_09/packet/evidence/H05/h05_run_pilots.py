"""H05 supervisor: lease windows, bounded children, pilot/probe orchestration.

Stages (run one at a time under an explicit team-lead lease):
  --stage build         B0 wheel + fresh venv (h05_build_wheel_and_venv.py)
  --stage pilot --spec <name>   one instrumented h05_profile.py session (see SPECS)
  --stage decide_flow   O3_FLOW variant decision from the sel1024 pilot record
  --stage probe --probe A1|A2|A3|F1|F2   bounded kernel probes
  --stage status        print lease log + heavy-wall accounting

Every heavy child: bounded timeout, pre-launch memory guard (available vs
pressure stop), lease window recording (start/end UTC, available RAM at
admission, ceiling, pressure stop, loadavg), cumulative heavy-wall check against
the 14400 s budget.
"""
import argparse
import json
import os
import subprocess
import sys
import time

CAMPAIGN = "/Users/alansynn/orca/workspaces/una-x"
REPO = f"{CAMPAIGN}/wt-large-e2e"
EV = f"{REPO}/campaigns/una_large_e2e/evidence/H05"
DATA = f"{CAMPAIGN}/campaign_data"
PY_B0 = f"{DATA}/venvs/b0_wheel/bin/python"
NBC = f"{DATA}/nbc_h05"
SESSIONS = f"{DATA}/h05_sessions"
CAPTURES = f"{DATA}/h05_captures"
OUTS = f"{DATA}/h05_out"
LEASE_LOG = f"{DATA}/h05_lease_log.json"
HEAVY_WALL_BUDGET_S = 14400.0
PHYS_BYTES = 17179869184
PRESSURE_STOP = max(1 << 30, int(0.10 * PHYS_BYTES))  # 1717986918
LAUNCH_TIMEOUT_S = 7200

MANIFESTS = {
    "O2": f"{REPO}/tests/large_e2e/inputs/O2.manifest.json",
    "O3_ACCESS": f"{REPO}/tests/large_e2e/inputs/O3_ACCESS.manifest.json",
    "O3_FLOW": f"{REPO}/tests/large_e2e/inputs/O3_FLOW.manifest.json",
    "O3_HOLDOUT": f"{REPO}/tests/large_e2e/inputs/O3_HOLDOUT.manifest.json",
}
ORIGINS_FILES = {}  # superseded: variants now pass --variant-id (manifest-anchored)

# cell -> run name -> profile.py invocation knobs
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


def load_lease_log():
    if os.path.exists(LEASE_LOG):
        with open(LEASE_LOG) as f:
            return json.load(f)
    return {"lease_id": None, "windows": [], "cumulative_heavy_s": 0.0}


def save_lease_log(log):
    with open(LEASE_LOG, "w") as f:
        json.dump(log, f, indent=1)


def guard_and_launch(cmd, env, lease_id, what, timeout_s=LAUNCH_TIMEOUT_S):
    import psutil
    log = load_lease_log()
    avail = psutil.virtual_memory().available
    ceiling = min(10650 * (1 << 20), int(0.80 * avail))
    entry = {
        "what": what, "lease_id": lease_id,
        "start_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "available_at_admission_bytes": avail,
        "ceiling_bytes": ceiling, "pressure_stop_bytes": PRESSURE_STOP,
        "loadavg_at_admission": os.getloadavg(),
        "cmd": cmd, "env": {k: v for k, v in env.items()
                            if k.startswith(("NUMBA", "L1"))},
    }
    if avail < PRESSURE_STOP:
        entry["refused"] = "available < pressure stop before launch (RESOURCES.md)"
        log["windows"].append(entry)
        save_lease_log(log)
        print(f"[h05-run] REFUSED: available {avail} < pressure stop {PRESSURE_STOP}")
        return 2
    if log["cumulative_heavy_s"] >= HEAVY_WALL_BUDGET_S:
        entry["refused"] = "heavy wall budget exhausted"
        log["windows"].append(entry)
        save_lease_log(log)
        print("[h05-run] REFUSED: heavy wall exhausted")
        return 2
    log["windows"].append(entry)
    log["cumulative_heavy_s"] += 0  # updated at completion
    save_lease_log(log)

    t0 = time.monotonic()
    try:
        p = subprocess.run(cmd, env={**os.environ, **env}, timeout=timeout_s,
                           capture_output=True, text=True)
        rc, tail = p.returncode, (p.stdout[-3000:] + p.stderr[-3000:])
    except subprocess.TimeoutExpired:
        rc, tail = -9, f"TIMEOUT after {timeout_s}s"
    wall = time.monotonic() - t0

    log = load_lease_log()
    entry = log["windows"][-1]
    breach_file = None
    if what.startswith("pilot:"):
        cand = f"{SESSIONS}/pressure_breach_{what.split(':', 1)[1]}.json"
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
        sum(w.get("wall_s", 0.0) for w in log["windows"]), 1)
    save_lease_log(log)
    print(f"[h05-run] rc={rc} wall={wall:.1f}s cumulative={log['cumulative_heavy_s']}s")
    return rc


def pilot_env(spec):
    env = {"NUMBA_CACHE_DIR": NBC}
    if spec.get("fresh_cache"):
        import shutil
        if os.path.exists(NBC):
            shutil.rmtree(NBC)
        os.makedirs(NBC)
    return env


def run_pilot(spec_name, lease_id):
    spec = SPECS[spec_name]
    if spec.get("verify_3d"):
        import hashlib
        h = hashlib.sha256()
        with open(HOLDOUT_3D_PATH, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        if h.hexdigest() != HOLDOUT_3D_SHA:
            print(f"[h05-run] REFUSED: 3D network hash {h.hexdigest()} != pinned")
            return 2
    cell = spec["cell"]
    out_json = f"{SESSIONS}/{spec_name}.json"
    out_root = f"{OUTS}/{spec_name}"
    cap = f"{CAPTURES}/{cell}" if spec.get("capture") else None
    cmd = [PY_B0, f"{REPO}/benchmarks/large_e2e/h05_profile.py",
           "--manifest", MANIFESTS[cell], "--cell", cell,
           "--mode", spec["mode"], "--out", out_json, "--output-root", out_root,
           "--expected-site-packages",
           f"{DATA}/venvs/b0_wheel/lib/python3.11/site-packages",
           "--lease-id", lease_id or "", "--run-id", spec_name]
    if spec.get("origins"):
        choice = spec["origins"]
        if choice == "CHOSEN":
            with open(f"{DATA}/h05_flow_decision.json") as f:
                choice = json.load(f)["chosen_variant"]
        cmd += ["--variant-id", choice]
    if spec.get("extra"):
        cmd += ["--extra-settings-json", json.dumps(spec["extra"])]
    if cap:
        cmd += ["--capture-dir", cap]
    os.makedirs(SESSIONS, exist_ok=True)
    os.makedirs(OUTS, exist_ok=True)
    return guard_and_launch(cmd, pilot_env(spec), lease_id, f"pilot:{spec_name}")


def run_probe(probe, lease_id):
    env = {"NUMBA_CACHE_DIR": NBC}
    if probe in ("A1", "F2"):
        env["NUMBA_NRT_STATS"] = "1"
    cmd = [PY_B0, f"{EV}/h05_probe_kernels.py", "--probe", probe,
           "--tag", time.strftime("%H%M%S")]
    return guard_and_launch(cmd, env, lease_id, f"probe:{probe}",
                            timeout_s=900)


def run_build(lease_id):
    cmd = [f"{CAMPAIGN}/venvs/campaign/bin/python", f"{EV}/h05_build_wheel_and_venv.py"]
    return guard_and_launch(cmd, {"NUMBA_CACHE_DIR": NBC}, lease_id, "build",
                            timeout_s=3600)


def decide_flow(lease_id):
    """O3_FLOW variant selection from the measured sel1024 pilot."""
    with open(f"{SESSIONS}/FLOW_sel1024_od.json") as f:
        raw = json.load(f)
    win = raw["window"]["application_window_ns"] / 1e9
    # inner stages are nested spans (under flow_engine_centrality): aggregate
    # the span tree by inclusive t1-t0, not top_level_exclusive
    stage_inc = {}
    for s in raw["spans"]:
        stage_inc[s["stage"]] = stage_inc.get(s["stage"], 0.0) + (s["t1_ns"] - s["t0_ns"])
    origin_loop_s = stage_inc.get("flow_origin_loop", 0.0) / 1e9
    grad_s = stage_inc.get("flow_gradient_precompute", 0.0) / 1e9
    n_orig = raw["counts"]["O_rows"]
    rate = origin_loop_s / n_orig
    peak_mib = raw["window"]["peak_rss_tree_sampled_mib"]
    vprime_1024 = raw["counts"]["V_nodes"] + raw["counts"]["D_rows"] + n_orig
    vprime_all = raw["counts"]["V_nodes"] + raw["counts"]["D_rows"] + 14751
    proj_origin_all = rate * 14751
    # gradient transient ~ chunk*n_total*12B (measured peak covers it; record both)
    import psutil
    log = load_lease_log()
    last = [w for w in log["windows"] if w.get("what") == "pilot:FLOW_sel1024_od"][-1]
    ceiling = last["ceiling_bytes"]
    decision = {
        "record": "flow_variant_decision", "task": "H05", "utc": time.strftime(
            "%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "measured_sel1024": {
            "window_s": win, "n_origins": n_orig,
            "origin_loop_inclusive_s": origin_loop_s,
            "gradient_stage_s": grad_s,
            "per_origin_rate_s": rate, "peak_rss_mib": peak_mib,
            "v_prime": int(vprime_1024),
            "partition_ok": raw["partition_check"]["partition_ok"],
        },
        "projection_sel_all": {
            "origin_loop_projected_s": proj_origin_all,
            "v_prime": int(vprime_all),
            "gradient_chunk": int(1e8 // vprime_all),
            "gradient_dense_transient_bytes_est": int(
                (1e8 // vprime_all) * vprime_all * 12),
            "note": "gradient transient ~1.2 GiB by the chunk rule (chunk*n_total "
                    "~1e8 elems x 12 B) — essentially origin-count independent; "
                    "per-stripe origin scratch grows by 12B*delta_V'*n_threads",
        },
        "policy": ("choose the largest variant in {256,1024,all} whose projected peak "
                   "RSS fits the lease ceiling with headroom and whose projected wall "
                   "fits the remaining heavy budget; otherwise record a capacity "
                   "refusal for 'all' and pick the largest feasible"),
        "ceiling_bytes": ceiling, "remaining_heavy_s": HEAVY_WALL_BUDGET_S - log["cumulative_heavy_s"],
    }
    projected_peak_bytes = peak_mib * (1 << 20) + 12 * (vprime_all - vprime_1024) * 9
    feasible = (projected_peak_bytes <= 0.9 * ceiling and
                proj_origin_all < decision["remaining_heavy_s"] * 0.5)
    decision["projected_peak_bytes_sel_all"] = int(projected_peak_bytes)
    decision["feasible_all"] = bool(feasible)
    decision["chosen_variant"] = "selall" if feasible else "sel1024"
    if not feasible:
        decision["capacity_refusal"] = (
            "sel-all projected peak %.2f GiB vs 0.9x ceiling %.2f GiB, or projected "
            "origin loop %.0fs vs remaining heavy budget %.0fs" %
            (projected_peak_bytes / (1 << 30), 0.9 * ceiling / (1 << 30),
             proj_origin_all, decision["remaining_heavy_s"]))
    with open(f"{DATA}/h05_flow_decision.json", "w") as f:
        json.dump(decision, f, indent=1)
    print(f"[h05-run] chosen_variant={decision['chosen_variant']} "
          f"feasible_all={feasible}")
    return 0


def status():
    log = load_lease_log()
    print(json.dumps({"lease_id": log["lease_id"],
                      "cumulative_heavy_s": log["cumulative_heavy_s"],
                      "budget_s": HEAVY_WALL_BUDGET_S,
                      "n_windows": len(log["windows"])}, indent=1))
    for w in log["windows"]:
        print(f"  {w.get('what')}: rc={w.get('rc')} wall={w.get('wall_s')}s "
              f"avail={w.get('available_at_admission_bytes')} "
              f"refused={w.get('refused')}")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=["build", "pilot", "decide_flow", "probe", "status"])
    ap.add_argument("--spec")
    ap.add_argument("--probe")
    ap.add_argument("--lease-id", default=None)
    a = ap.parse_args()

    log = load_lease_log()
    if a.lease_id:
        log["lease_id"] = a.lease_id
        save_lease_log(log)

    if a.stage == "build":
        sys.exit(run_build(log["lease_id"]))
    if a.stage == "pilot":
        if not a.spec or a.spec not in SPECS:
            print(f"--spec must be one of {sorted(SPECS)}")
            sys.exit(2)
        sys.exit(run_pilot(a.spec, log["lease_id"]))

    if a.stage == "decide_flow":
        sys.exit(decide_flow(log["lease_id"]))
    if a.stage == "probe":
        sys.exit(run_probe(a.probe, log["lease_id"]))
    if a.stage == "status":
        sys.exit(status())


if __name__ == "__main__":
    main()
