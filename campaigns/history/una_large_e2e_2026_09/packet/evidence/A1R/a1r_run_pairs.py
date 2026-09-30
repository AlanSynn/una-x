"""A1R paired-screen supervisor — h04-reviewer's screen spec, executed.

Schedule per cell: 3 paired complete-job blocks, arm order AB, BA, AB
(A = B0 control). Pair 1 runs on FRESH per-arm NUMBA_CACHE_DIRs (created at
pair 1 -> symmetric cold first-call JIT); pairs 2-3 reuse them (warm).
(W,H) = (1,8) frozen for the WHOLE screen via NUMBA_NUM_THREADS=8
(preregistered accessibility grid W x H <= 9); identical across arms.

Failure policy per h04: on failure/timeout preserve full logs and STOP the
screen — no retry beyond the schedule without h04's approval.

Raw records: evidence/A1R/raw/a1rp_<cell>_p<N>_<ORDER>_<arm>.json
Summary (descriptive only, h04 owns disposition): campaign_data/a1r_screen_summary.json
"""
import argparse
import json
import os
import subprocess
import sys
import time

CAMPAIGN = "/Users/alansynn/orca/workspaces/una-x"
REPO = f"{CAMPAIGN}/wt-large-e2e"
EV = f"{REPO}/campaigns/una_large_e2e/evidence/A1R"
DATA = f"{CAMPAIGN}/campaign_data"
RAW = f"{EV}/raw"
SUMMARY = f"{DATA}/a1r_screen_summary.json"
LEASE_LOG = f"{DATA}/a1r_lease_log.json"
PHYS_BYTES = 17179869184
PRESSURE_STOP = max(1 << 30, int(0.10 * PHYS_BYTES))  # 1717986918
CEILING_CEILING_B = 11274289152  # 10.5 GiB H00 campaign ceiling (peak RSS)
OPENING_BALANCE_S = 426.4
HEAVY_WALL_BUDGET_S = 14400.0
RUN_TIMEOUT_S = 1200
LEASE_ID = "A1R-lease-20260926"
W_WORKERS, H_THREADS = 1, 8
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
# per cell: (pair, order, arm) — AB/BA alternation, A = b0 runs FIRST in AB
SCHEDULE = [(1, "AB", "b0"), (1, "AB", "cand"),
            (2, "BA", "cand"), (2, "BA", "b0"),
            (3, "AB", "b0"), (3, "AB", "cand")]


def wheel_sha(arm):
    with open(ARMS[arm]["wheel_record"]) as f:
        return json.load(f)["wheel"]["sha256"]


def cache_files(root):
    n = 0
    for _, _, files in os.walk(root):
        n += len(files)
    return n


def lease_append(entry):
    with open(LEASE_LOG) as f:
        log = json.load(f)
    log["windows"].append(entry)
    log["cumulative_heavy_s"] = round(
        OPENING_BALANCE_S + sum(w.get("wall_s", 0.0) for w in log["windows"]), 1)
    with open(LEASE_LOG, "w") as f:
        json.dump(log, f, indent=1)
    return log["cumulative_heavy_s"]


def run_slot(cell, pair, order, arm):
    import psutil
    import shutil
    a = ARMS[arm]
    run_id = f"a1rp_{cell}_p{pair}_{order}_{arm}"
    out_json = f"{RAW}/{run_id}.json"
    if os.path.exists(out_json):
        return 0, run_id, "already-present (resumed; no rerun)"
    # pair 1 = fresh per-arm cache for BOTH arms (cold, symmetric); reused after
    cache_fresh = "created_fresh" if pair == 1 else "reused"
    if pair == 1 and arm == "b0":  # wipe both roots once, at cell pair-1 start
        for armi in ARMS.values():
            if os.path.exists(armi["cache_root"]):
                shutil.rmtree(armi["cache_root"])
            os.makedirs(armi["cache_root"])
    env = {"NUMBA_CACHE_DIR": a["cache_root"], "NUMBA_NUM_THREADS": str(H_THREADS)}
    env.pop("L1_REUSE_DIR", None)
    cmd = [a["python"], f"{EV}/a1r_bare_job.py",
           "--manifest", MANIFESTS[cell], "--cell", cell,
           "--expected-site-packages", a["site_packages"],
           "--out-json", out_json,
           "--output-root", f"{DATA}/a1r_out_pairs/{run_id}",
           "--run-id", run_id]
    avail = psutil.virtual_memory().available
    ceiling = min(10650 * (1 << 20), int(0.80 * avail))
    entry = {"what": f"pairs_run:{run_id}", "lease_id": LEASE_ID,
             "start_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
             "available_at_admission_bytes": avail,
             "ceiling_bytes": ceiling, "pressure_stop_bytes": PRESSURE_STOP,
             "loadavg_at_admission": os.getloadavg(), "cmd": cmd, "env": env}
    with open(LEASE_LOG) as f:
        used = json.load(f)["cumulative_heavy_s"]
    if avail < PRESSURE_STOP:
        entry["refused"] = "available < pressure stop before launch"
        lease_append(entry)
        return 2, run_id, "REFUSED: memory pressure guard"
    if OPENING_BALANCE_S + used >= HEAVY_WALL_BUDGET_S:
        entry["refused"] = "heavy wall exhausted"
        lease_append(entry)
        return 2, run_id, "REFUSED: heavy wall exhausted"

    t0 = time.monotonic()
    try:
        p = subprocess.run(cmd, env={**os.environ, **env}, timeout=RUN_TIMEOUT_S,
                           capture_output=True, text=True)
        rc, tail = p.returncode, (p.stdout[-2000:] + p.stderr[-2000:])
    except subprocess.TimeoutExpired:
        rc, tail = -9, f"TIMEOUT after {RUN_TIMEOUT_S}s"
    wall = time.monotonic() - t0
    entry.update({"end_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                  "wall_s": round(wall, 1), "rc": rc, "tail": tail,
                  "loadavg_at_completion": os.getloadavg()})
    cumulative = lease_append(entry)
    print(f"[a1r-pairs] {run_id} rc={rc} wall={wall:.1f}s "
          f"cumulative={cumulative}s", flush=True)
    if rc != 0:
        with open(f"{RAW}/{run_id}.supervisor.log", "w") as f:
            json.dump({"run_id": run_id, "rc": rc, "tail": tail, "env": env,
                       "cmd": cmd}, f, indent=1)
    return rc, run_id, ("ok" if rc == 0 else f"FAILED rc={rc} (screen STOPPED, "
                                           "logs preserved, no retry)")


def summary():
    rows = []
    for cell in MANIFESTS:
        for pair, order, arm in SCHEDULE:
            run_id = f"a1rp_{cell}_p{pair}_{order}_{arm}"
            p = f"{RAW}/{run_id}.json"
            if not os.path.exists(p):
                rows.append({"run_id": run_id, "missing": True})
                continue
            with open(p) as f:
                r = json.load(f)
            rows.append({
                "run_id": run_id, "cell": cell, "pair": pair, "order": order,
                "arm": arm, "cold_warm": ("cold (fresh per-arm cache)" if pair == 1
                                          else "warm (reused cache)"),
                "WH": [W_WORKERS, H_THREADS],
                "window_s": round(r["window"]["application_window_ns"] / 1e9, 4),
                "numba_num_threads": r["process"]["numba_get_num_threads"],
                "numba_threads_baked_at_compile": r["process"]["numba_num_threads_config"],
                "cache_dir": r["env"]["NUMBA_CACHE_DIR"],
                "cache_file_count_at_run": cache_files(r["env"]["NUMBA_CACHE_DIR"]),
                "peak_rss_mib": r.get("peak_rss_ru_maxrss_mib"),
                "identity_ok": r["import"]["identity_assertion"].startswith("ok"),
                "counts": r.get("counts"),
                "search_radius": r.get("settings_effective_search_radius"),
                "array_hashes": {k: (v or {}).get("sha256_bytes") for k, v in
                                 (r.get("accessibility_arrays") or {}).items()},
                "verbatim_warnings": r.get("verbatim_warnings"),
                "fatal": r.get("fatal"),
            })
    # descriptive aggregates (h04 computes verdicts)
    agg = {}
    for cell in MANIFESTS:
        cr = [r for r in rows if r.get("cell") == cell and not r.get("missing")]
        b = sorted(r["window_s"] for r in cr if r["arm"] == "b0")
        c = sorted(r["window_s"] for r in cr if r["arm"] == "cand")
        per_pair = []
        for pnum in (1, 2, 3):
            bb = next((r for r in cr if r["pair"] == pnum and r["arm"] == "b0"), None)
            cc = next((r for r in cr if r["pair"] == pnum and r["arm"] == "cand"), None)
            if bb and cc:
                per_pair.append({
                    "pair": pnum, "order": bb["order"], "cold_warm": bb["cold_warm"],
                    "b0_s": bb["window_s"], "cand_s": cc["window_s"],
                    "b0_over_cand": round(bb["window_s"] / cc["window_s"], 4)})
        med = lambda xs: (xs[len(xs) // 2] if len(xs) % 2 else
                          (xs[len(xs) // 2 - 1] + xs[len(xs) // 2]) / 2)
        agg[cell] = {
            "n_b0": len(b), "n_cand": len(c),
            "median_b0_s": med(b) if b else None,
            "median_cand_s": med(c) if c else None,
            "proxy_speedup_median_b0_over_cand":
                round(med(b) / med(c), 4) if b and c else None,
            "cold_pair1_b0_over_cand":
                next((pp["b0_over_cand"] for pp in per_pair if pp["pair"] == 1), None),
            "per_pair": per_pair,
            "hash_sets_across_all_runs": len({tuple(sorted(r["array_hashes"].items()))
                                              for r in cr}),
            "peak_rss_mib_max": max((r["peak_rss_mib"] for r in cr), default=None),
            "ceiling_10p5gib_mib": CEILING_CEILING_B / (1 << 20),
            "warnings_nonzero_runs": [r["run_id"] for r in cr
                                      if r.get("verbatim_warnings")],
        }
    with open(LEASE_LOG) as f:
        lease = json.load(f)
    out = {
        "task": "A1R", "record": "screen_summary", "role": "screen-executor",
        "spec_source": "h04-reviewer screen request (3 paired blocks AB/BA/AB per "
                       "cell; pair 1 cold on fresh per-arm caches; (W,H)=(1,8) "
                       "frozen; bare complete-job windows, no leaf profilers)",
        "descriptive_only": "h04-reviewer owns disposition/verdicts",
        "schedule": "per cell: p1 AB (cold), p2 BA, p3 AB; A=B0",
        "arms": {k: {"venv_python": v["python"],
                     "site_packages": v["site_packages"],
                     "cache_root": v["cache_root"],
                     "wheel_sha256": wheel_sha(k)} for k, v in ARMS.items()},
        "candidate_commit": CAND_COMMIT,
        "WH_frozen": [W_WORKERS, H_THREADS],
        "grid_note": "W x H = 8 <= 9 (preregistered accessibility grid)",
        "runs": rows, "aggregates": agg,
        "heavy_wall": {"opening_balance_s": OPENING_BALANCE_S,
                       "cumulative_incl_opening_s": lease["cumulative_heavy_s"],
                       "budget_s": HEAVY_WALL_BUDGET_S},
        "lease_log": LEASE_LOG,
        "raw_records_dir": RAW,
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with open(SUMMARY, "w") as f:
        json.dump(out, f, indent=1)
    print(f"[a1r-pairs] summary -> {SUMMARY}")
    for cell, a in agg.items():
        print(f"  {cell}: med b0={a['median_b0_s']}s cand={a['median_cand_s']}s "
              f"proxy={a['proxy_speedup_median_b0_over_cand']} "
              f"cold={a['cold_pair1_b0_over_cand']} "
              f"hash_sets={a['hash_sets_across_all_runs']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["run", "summary"], default="run")
    a = ap.parse_args()
    if a.stage == "summary":
        summary()
        return 0
    os.makedirs(RAW, exist_ok=True)
    for arm in ARMS.values():
        if not (os.path.exists(arm["python"]) and os.path.exists(arm["wheel_record"])):
            print(f"[a1r-pairs] REFUSED: arm venv/wheel record missing: {arm}")
            return 2
    for cell in MANIFESTS:
        for pair, order, arm in SCHEDULE:
            rc, run_id, msg = run_slot(cell, pair, order, arm)
            print(f"[a1r-pairs] slot {run_id}: {msg}", flush=True)
            if rc != 0:
                print("[a1r-pairs] STOPPING screen per failure policy "
                      "(no retry; preserve evidence; h04 approval needed to continue)")
                summary()
                return rc
    summary()
    return 0


if __name__ == "__main__":
    sys.exit(main())
