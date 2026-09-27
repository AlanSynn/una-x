"""F1R paired-screen supervisor — screen_spec rev 5 executed.

Schedule: 3 paired complete-job blocks, arm order AB, BA, AB (A = B0 control).
Block 1 runs on FRESH per-arm NUMBA_CACHE_DIRs (created at block 1 -> symmetric
cold first-compile); blocks 2-3 reuse them (warm). Frozen triple (W,K,H) =
(1,8,1): W=1 (one job per invocation), K=8 (--flow-stripes -> una.topology.
num_threads, wired by the driver pre-window), H=1 (NUMBA_NUM_THREADS).

Rev-4 additions vs the a1r_run_pairs.py pattern (envelope numbers REV 5):
  - MATRIX-ENTRY bar: refuse to enter when available < 4632084480 B
    (0.80*available < 3534 MiB) — binds at initial fire and any post-void
    re-entry; within a matrix the per-window guard below is the sole binding
    mid-matrix protection
  - per-window memory admission REFUSES when 0.80*available < 3354 MiB (P5);
    a mid-matrix refusal VOIDS the partial matrix (no disposition from it; a
    conformant matrix restarts as a whole, still 3 blocks, fresh roots)
  - supervisor-side live watchdog on the child process tree: 0.25 s ps-sample,
    budget 3304 MiB, breach file + SIGTERM->SIGKILL (the driver's ru_maxrss is
    corroborating, not the trip mechanism)
  - per-block byte-identity: candidate output artifact map == b0's, with the
    timestamped export dir flow_<YYYY-MM-DD_HHMM>/ normalized; ANY delta is a
    TRACK REJECTION -> immediate stop and report (dispatch term)
  - per-block engine_config_observed check: all three fields == 8 (one
    mismatch disqualifies the block)

Failure policy: on failure/timeout/watchdog breach/identity delta preserve all
logs and STOP — no retry beyond the schedule without h04's approval.

--stage canary (h04-confirmed diagnostics class, pre-block-1): one sel256 run
per arm on THROWAWAY per-arm cache roots; admission rule + watchdog ACTIVE;
records to campaign_data/f1r_canary_record.json (NEVER f1r_screen_summary.json);
cross-arm identity recorded descriptively, gates nothing.

Raw records: evidence/F1R/raw/f1r_b<N>_<ORDER>_<arm>.json (gate) and
evidence/F1R/raw/f1r_canary_<arm>.json (diagnostics, never ingested by summary)
Summary (descriptive only, h04 owns disposition):
campaign_data/f1r_screen_summary.json
"""
import argparse
import json
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time

CAMPAIGN = "/Users/alansynn/orca/workspaces/una-x"
REPO = f"{CAMPAIGN}/wt-large-e2e"
EV = f"{REPO}/campaigns/una_large_e2e/evidence/F1R"
DATA = f"{CAMPAIGN}/campaign_data"
RAW = f"{EV}/raw"
SUMMARY = f"{DATA}/f1r_screen_summary.json"
CANARY_RECORD = f"{DATA}/f1r_canary_record.json"
LEASE_LOG = f"{DATA}/f1r_lease_log.json"
PHYS_BYTES = 17179869184
PRESSURE_STOP = max(1 << 30, int(0.10 * PHYS_BYTES))  # 1717986918
WATCHDOG_MIB = 3304   # 1.10 x governing cold peak 3003.25 = 3303.575 -> ceil (P5 rev 5)
ADMISSION_MIN_AVAIL_BYTES = 3354 * (1 << 20)  # refuse when 0.80*avail < this
MATRIX_ENTRY_MIN_AVAIL_BYTES = 4632084480  # 0.80*avail >= 3534 MiB = avail >= 4417.5 MiB (P5 rev 5)
HEAVY_WALL_BUDGET_S = 14400.0
RUN_TIMEOUT_S = 1200
LEASE_ID = "F1R"
K_STRIPES, H_THREADS = 8, 1
CAND_COMMIT = "c1d7edc519f254c0e81475509d34b437e33abf09"
MANIFEST = f"{REPO}/tests/large_e2e/inputs/O3_FLOW.manifest.json"
VARIANT = "sel1024"
CANARY_VARIANT = "sel256"  # diagnostics-class canary only; never gate math
FLOW_TS_RE = re.compile(r"flow_\d{4}-\d{2}-\d{2}_\d{4}")
T_CRIT_2DOF = 4.302652729911275  # two-sided 95%, 2 dof
BOOTSTRAP_SEED = 20260926
BOOTSTRAP_B = 10000

ARMS = {
    "b0": {"python": f"{DATA}/venvs/b0p_wheel/bin/python",
           "site_packages": f"{DATA}/venvs/b0p_wheel/lib/python3.11/site-packages",
           "cache_root": f"{DATA}/nbc_f1r_b0",
           "wheel_record": f"{DATA}/f1r_b0p_wheel_and_venv.json"},
    "cand": {"python": f"{DATA}/venvs/f1r_cand_wheel/bin/python",
             "site_packages": f"{DATA}/venvs/f1r_cand_wheel/lib/python3.11/site-packages",
             "cache_root": f"{DATA}/nbc_f1r_cand",
             "wheel_record": f"{DATA}/f1r_wheel_and_venv.json"},
}
# (block, order, arm) — AB/BA alternation, A = b0 runs FIRST in AB
SCHEDULE = [(1, "AB", "b0"), (1, "AB", "cand"),
            (2, "BA", "cand"), (2, "BA", "b0"),
            (3, "AB", "b0"), (3, "AB", "cand")]


def wheel_sha(arm):
    with open(ARMS[arm]["wheel_record"]) as f:
        rec = json.load(f)
    if "wheel" in rec:  # h05_wheel_and_venv / f1r_wheel_and_venv schema
        return rec["wheel"]["sha256"]
    return rec["b0p_arm"]["wheel_sha256"]  # f1r_b0p_wheel_and_venv schema (rev 5 b0 re-point)


def cache_files(root):
    n = 0
    for _, _, files in os.walk(root):
        n += len(files)
    return n


def lease_cumulative():
    with open(LEASE_LOG) as f:
        log = json.load(f)
    charged = sum(w.get("charged_s", 0.0) for w in log["windows"])
    cum = round(log["opening_balance_s"] + charged, 1)
    if log.get("cumulative_charged_s") != cum:
        log["cumulative_charged_s"] = cum
        log["remaining_s"] = round(HEAVY_WALL_BUDGET_S - cum, 1)
        with open(LEASE_LOG, "w") as f:
            json.dump(log, f, indent=1)
    return cum


def lease_append(entry):
    with open(LEASE_LOG) as f:
        log = json.load(f)
    log["windows"].append(entry)
    charged = sum(w.get("charged_s", 0.0) for w in log["windows"])
    log["cumulative_charged_s"] = round(log["opening_balance_s"] + charged, 1)
    log["remaining_s"] = round(HEAVY_WALL_BUDGET_S - log["cumulative_charged_s"], 1)
    with open(LEASE_LOG, "w") as f:
        json.dump(log, f, indent=1)
    return log["cumulative_charged_s"]


def tree_rss_kb(root_pid, snap):
    """Sum RSS (KB) of root_pid and all descendants given a pid->(ppid,rss) map."""
    kids = {}
    for pid, (ppid, _rss) in snap.items():
        kids.setdefault(ppid, []).append(pid)
    total, stack = 0, [root_pid]
    while stack:
        pid = stack.pop()
        total += snap.get(pid, (0, 0))[1]
        stack.extend(kids.get(pid, []))
    return total


def ps_snapshot():
    snap = {}
    out = subprocess.run(["ps", "-axo", "pid=,ppid=,rss="], capture_output=True,
                         text=True, timeout=10).stdout
    for line in out.splitlines():
        parts = line.split()
        if len(parts) == 3:
            try:
                snap[int(parts[0])] = (int(parts[1]), int(parts[2]))
            except ValueError:
                pass
    return snap


def run_slot(block, order, arm):
    import psutil
    a = ARMS[arm]
    run_id = f"f1r_b{block}_{order}_{arm}"
    out_json = f"{RAW}/{run_id}.json"
    # main() guarantees the loop never starts with existing records (S3 guard),
    # so no resume branch exists here by design
    cache_fresh = "created_fresh" if block == 1 else "reused"
    if block == 1 and arm == "b0":  # wipe both roots once, at block-1 start
        for armi in ARMS.values():
            if os.path.exists(armi["cache_root"]):
                shutil.rmtree(armi["cache_root"])
            os.makedirs(armi["cache_root"])
    out_root = f"{DATA}/f1r_out/{run_id}"
    if os.path.exists(out_root):
        shutil.rmtree(out_root)
    env = {"NUMBA_CACHE_DIR": a["cache_root"],
           "NUMBA_NUM_THREADS": str(H_THREADS)}
    child_env = {**os.environ, **env}
    child_env.pop("L1_REUSE_DIR", None)
    cache_pre = cache_files(a["cache_root"])
    cmd = [a["python"], f"{EV}/f1r_bare_job.py",
           "--manifest", MANIFEST, "--cell", "O3_FLOW",
           "--variant-id", VARIANT, "--arm", arm,
           "--expected-site-packages", a["site_packages"],
           "--out-json", out_json,
           "--output-root", out_root,
           "--run-id", run_id,
           "--flow-stripes", str(K_STRIPES),
           "--wheel-sha256", wheel_sha(arm),
           "--numba-threads", str(H_THREADS),
           "--block", str(block), "--order", order]
    avail = psutil.virtual_memory().available
    ceiling = min(10650 * (1 << 20), int(0.80 * avail))
    entry = {"what": f"gate_run:{run_id}", "lease_id": LEASE_ID,
             "block": block, "order": order, "arm": arm,
             "cache_state": cache_fresh,
             "cache_file_count_before": cache_pre,
             "start_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
             "available_at_admission_bytes": avail,
             "admission_rule": "refuse when 0.80*available < 3354 MiB (spec P5 rev 5)",
             "admission_080avail_bytes": int(0.80 * avail),
             "ceiling_bytes": ceiling, "pressure_stop_bytes": PRESSURE_STOP,
             "watchdog_budget_mib": WATCHDOG_MIB,
             "supervisor_python": sys.executable,
             "loadavg_at_admission": os.getloadavg(),
             "cmd": cmd,
             "env_overrides": env,
             "child_env_dropped": ["L1_REUSE_DIR"],
             "child_NUMBA_DISABLE_JIT_set": "NUMBA_DISABLE_JIT" in child_env}
    cum = lease_cumulative()
    if int(0.80 * avail) < ADMISSION_MIN_AVAIL_BYTES:
        entry["refused"] = ("0.80*available < 3354 MiB at admission (spec P5 "
                            "rev 5); mid-matrix refusal voids the partial matrix")
        lease_append(entry)
        return 2, run_id, "REFUSED: P5 memory admission guard"
    if avail < PRESSURE_STOP:
        entry["refused"] = "available < pressure stop before launch"
        lease_append(entry)
        return 2, run_id, "REFUSED: memory pressure guard"
    if cum >= HEAVY_WALL_BUDGET_S:
        entry["refused"] = "heavy wall exhausted"
        lease_append(entry)
        return 2, run_id, "REFUSED: heavy wall exhausted"

    t0 = time.monotonic()
    p = subprocess.Popen(cmd, env=child_env, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True)
    breach = {"run_id": run_id, "budget_mib": WATCHDOG_MIB, "samples": []}
    stop_evt = threading.Event()

    def watchdog():
        while not stop_evt.is_set() and p.poll() is None:
            try:
                kb = tree_rss_kb(p.pid, ps_snapshot())
            except Exception:
                kb = -1
            if kb > 0:
                breach["samples"].append(
                    {"t_s": round(time.monotonic() - t0, 2), "tree_rss_kb": kb})
                if kb > WATCHDOG_MIB * 1024:
                    breach["breached"] = True
                    breach["breach_kb"] = kb
                    try:
                        p.send_signal(signal.SIGTERM)
                        breach["action"] = "SIGTERM sent"
                    except ProcessLookupError:
                        breach["action"] = "process already gone"
                        return
                    for _ in range(40):  # 10 s grace, then SIGKILL
                        if p.poll() is not None:
                            breach["action"] += " -> exited on SIGTERM"
                            return
                        stop_evt.wait(0.25)
                    try:
                        p.send_signal(signal.SIGKILL)
                        breach["action"] += " -> SIGKILL after 10 s grace"
                    except ProcessLookupError:
                        pass
                    return
            stop_evt.wait(0.25)

    th = threading.Thread(target=watchdog, daemon=True)
    th.start()
    try:
        out, _ = p.communicate(timeout=RUN_TIMEOUT_S)
        rc = p.returncode
        tail = (out or "")[-2000:]
    except subprocess.TimeoutExpired:
        rc = -9
        tail = f"TIMEOUT after {RUN_TIMEOUT_S}s"
        p.kill()
        out, _ = p.communicate()
    stop_evt.set()
    th.join(timeout=2)
    wall = time.monotonic() - t0
    entry.update({"end_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                  "wall_s": round(wall, 1), "rc": rc, "tail": tail,
                  "cache_file_count_after": cache_files(a["cache_root"]),
                  "watchdog_samples": len(breach["samples"]),
                  "watchdog_peak_tree_kb": max(
                      (s["tree_rss_kb"] for s in breach["samples"]), default=0),
                  "loadavg_at_completion": os.getloadavg()})
    if breach.get("breached"):
        entry["watchdog_breach"] = breach
        with open(f"{DATA}/f1r_watchdog_breach_{run_id}.json", "w") as f:
            json.dump(breach, f, indent=1)
    cumulative = lease_append(entry)
    print(f"[f1r-pairs] {run_id} rc={rc} wall={wall:.1f}s "
          f"cumulative={cumulative}s", flush=True)
    if rc != 0:
        with open(f"{RAW}/{run_id}.supervisor.log", "w") as f:
            json.dump({"run_id": run_id, "rc": rc, "tail": tail, "env": env,
                       "cmd": cmd, "watchdog": breach}, f, indent=1)
    if breach.get("breached"):
        return 3, run_id, ("WATCHDOG BREACH (3304 MiB) — screen STOPPED, logs "
                           "preserved, no retry")
    return rc, run_id, ("ok" if rc == 0 else f"FAILED rc={rc} (screen STOPPED, "
                                            "logs preserved, no retry)")


def canary_slot(arm):
    """diagnostics_class/memory_canary (h04 condition set): one sel256 run per
    arm on THROWAWAY per-arm cache roots. Admission refusal rule and watchdog
    are ACTIVE — mechanism capture is the point. run_slot above is untouched;
    this is a separate, additive path whose records can never enter the matrix
    (run id pattern f1r_canary_<arm>, never f1r_b<N>_<ORDER>_<arm>)."""
    import psutil
    a = ARMS[arm]
    run_id = f"f1r_canary_{arm}"
    out_json = f"{RAW}/{run_id}.json"
    if os.path.exists(out_json):
        return 2, run_id, ("REFUSED: canary record already present (explicit "
                           "delete required to re-canary)")
    canary_root = f"{DATA}/nbc_f1r_canary_{arm}"
    if os.path.exists(canary_root):
        shutil.rmtree(canary_root)
    os.makedirs(canary_root)
    out_root = f"{DATA}/f1r_canary_out/{run_id}"
    if os.path.exists(out_root):
        shutil.rmtree(out_root)
    screen_roots_before = {k: cache_files(v["cache_root"]) for k, v in ARMS.items()}
    env = {"NUMBA_CACHE_DIR": canary_root, "NUMBA_NUM_THREADS": str(H_THREADS)}
    child_env = {**os.environ, **env}
    child_env.pop("L1_REUSE_DIR", None)
    cmd = [a["python"], f"{EV}/f1r_bare_job.py",
           "--manifest", MANIFEST, "--cell", "O3_FLOW",
           "--variant-id", CANARY_VARIANT, "--arm", arm,
           "--expected-site-packages", a["site_packages"],
           "--out-json", out_json,
           "--output-root", out_root,
           "--run-id", run_id,
           "--flow-stripes", str(K_STRIPES),
           "--wheel-sha256", wheel_sha(arm),
           "--numba-threads", str(H_THREADS),
           "--block", "0", "--order", "CANARY"]
    avail = psutil.virtual_memory().available
    ceiling = min(10650 * (1 << 20), int(0.80 * avail))
    entry = {"what": f"diagnostic_run:{run_id}",
             "diagnostics_class": "memory_canary", "gate_eligible": False,
             "variant": CANARY_VARIANT,
             "lease_id": LEASE_ID, "arm": arm,
             "cache_state": "throwaway_root_created_fresh",
             "start_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
             "available_at_admission_bytes": avail,
             "admission_080avail_bytes": int(0.80 * avail),
             "ceiling_bytes": ceiling, "pressure_stop_bytes": PRESSURE_STOP,
             "watchdog_budget_mib": WATCHDOG_MIB,
             "supervisor_python": sys.executable,
             "loadavg_at_admission": os.getloadavg(),
             "cmd": cmd, "env_overrides": env}
    cum = lease_cumulative()
    if int(0.80 * avail) < ADMISSION_MIN_AVAIL_BYTES:
        entry["refused"] = "0.80*available < 3354 MiB at admission (canary)"
        lease_append(entry)
        return 2, run_id, "REFUSED: P5 memory admission guard (canary)"
    if cum >= HEAVY_WALL_BUDGET_S:
        entry["refused"] = "heavy wall exhausted (canary)"
        lease_append(entry)
        return 2, run_id, "REFUSED: heavy wall exhausted (canary)"

    t0 = time.monotonic()
    p = subprocess.Popen(cmd, env=child_env, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True)
    breach = {"run_id": run_id, "budget_mib": WATCHDOG_MIB, "samples": []}
    stop_evt = threading.Event()

    def wd():  # same mechanism as run_slot's watchdog (SIGTERM -> 10s -> SIGKILL)
        while not stop_evt.is_set() and p.poll() is None:
            try:
                kb = tree_rss_kb(p.pid, ps_snapshot())
            except Exception:
                kb = -1
            if kb > 0:
                breach["samples"].append(
                    {"t_s": round(time.monotonic() - t0, 2), "tree_rss_kb": kb})
                if kb > WATCHDOG_MIB * 1024:
                    breach["breached"] = True
                    breach["breach_kb"] = kb
                    try:
                        p.send_signal(signal.SIGTERM)
                        breach["action"] = "SIGTERM sent"
                    except ProcessLookupError:
                        breach["action"] = "process already gone"
                        return
                    for _ in range(40):
                        if p.poll() is not None:
                            breach["action"] += " -> exited on SIGTERM"
                            return
                        stop_evt.wait(0.25)
                    try:
                        p.send_signal(signal.SIGKILL)
                        breach["action"] += " -> SIGKILL after 10 s grace"
                    except ProcessLookupError:
                        pass
                    return
            stop_evt.wait(0.25)

    th = threading.Thread(target=wd, daemon=True)
    th.start()
    try:
        out, _ = p.communicate(timeout=RUN_TIMEOUT_S)
        rc = p.returncode
        tail = (out or "")[-2000:]
    except subprocess.TimeoutExpired:
        rc = -9
        tail = f"TIMEOUT after {RUN_TIMEOUT_S}s"
        p.kill()
        out, _ = p.communicate()
    stop_evt.set()
    th.join(timeout=2)
    wall = time.monotonic() - t0
    entry.update({"end_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                  "wall_s": round(wall, 1), "rc": rc, "tail": tail,
                  "canary_cache_file_count_after": cache_files(canary_root),
                  "watchdog_samples": len(breach["samples"]),
                  "watchdog_peak_tree_kb": max(
                      (s["tree_rss_kb"] for s in breach["samples"]), default=0),
                  "loadavg_at_completion": os.getloadavg()})
    if breach.get("breached"):
        entry["watchdog_breach"] = breach
        with open(f"{DATA}/f1r_watchdog_breach_{run_id}.json", "w") as f:
            json.dump(breach, f, indent=1)
    lease_append(entry)
    print(f"[f1r-canary] {run_id} rc={rc} wall={wall:.1f}s", flush=True)
    if rc != 0:
        with open(f"{RAW}/{run_id}.supervisor.log", "w") as f:
            json.dump({"run_id": run_id, "rc": rc, "tail": tail, "env": env,
                       "cmd": cmd, "watchdog": breach}, f, indent=1)
    return rc, run_id, ("ok" if rc == 0 else
                        f"FAILED rc={rc} (canary stopped; mechanism defect -> "
                        "fix + reviewer delta re-verify + re-canary before block 1)")


def canary():
    """h04 conditions 4-6: own record file only, never f1r_screen_summary.json;
    cross-arm byte-identity recorded descriptively, gates nothing; honest
    framing recorded in the record itself."""
    # REV 5A DEFECT FIX (fired twice: rev-4 canary breach 2026-09-26 12:03Z and
    # rev-5 re-canary 2026-09-27 00:41Z, identical signature): the run stage
    # creates RAW/ in main(), but --stage canary returns before that line, so
    # on a virgin tree the DRIVER cannot write its --out-json under RAW/ (rc=1
    # after completing the whole job) and the rc!=0 branch then crashes the
    # supervisor writing {RAW}/{run_id}.supervisor.log. Create RAW here — the
    # canary is a legitimate first writer of the raw records dir.
    os.makedirs(RAW, exist_ok=True)
    # ordering: canary is pre-block-1 execution; refuse if gate records exist
    present = [f"f1r_b{b}_{o}_{a}" for b, o, a in SCHEDULE
               if os.path.exists(f"{RAW}/f1r_b{b}_{o}_{a}.json")]
    if present:
        print(f"[f1r-canary] REFUSED: gate records already present ({present}) "
              "— canary is a pre-block-1 diagnostic; ordering violated")
        return 2
    screen_roots_before = {k: cache_files(v["cache_root"]) for k, v in ARMS.items()}
    results, ok = {}, True
    for arm in ("b0", "cand"):
        rc, run_id, msg = canary_slot(arm)
        results[arm] = {"run_id": run_id, "rc": rc, "msg": msg}
        if rc != 0:
            ok = False
            print(f"[f1r-canary] {run_id}: {msg}")
            break
    recs = {arm: load_rec(r["run_id"]) for arm, r in results.items()
            if r["rc"] == 0}
    identity = None
    if len(recs) == 2:
        ab0, ac = array_hashes(recs["b0"]), array_hashes(recs["cand"])
        arr_delta = sorted(k for k in set(ab0) | set(ac)
                           if ab0.get(k) != ac.get(k))
        nb0, nc = norm_artifacts(recs["b0"]), norm_artifacts(recs["cand"])
        art_delta = [k for k in sorted(set(nb0) | set(nc))
                     if nb0.get(k) != nc.get(k)]
        fx = [((r.get("origin_selection") or {}).get("fixture") or {}).get("sha256")
              for r in recs.values()]
        identity = {
            "descriptive_only": True, "gates_nothing": True,
            "array_hashes_equal": not arr_delta,
            "artifact_identity_normalized": "IDENTICAL" if not art_delta else "DELTA",
            "artifact_delta_paths": art_delta[:10],
            "fixture_sha_equal": fx[0] == fx[1] and fx[0] is not None,
            "engine_config_observed": {arm: recs[arm].get("engine_config_observed")
                                       for arm in recs},
        }
    screen_roots_after = {k: cache_files(v["cache_root"]) for k, v in ARMS.items()}
    with open(LEASE_LOG) as f:
        lease = json.load(f)
    out = {
        "task": "F1R", "record": "memory_canary",
        "diagnostics_class": "memory_canary", "gate_eligible": False,
        "spec_basis": "h04-reviewer canary confirmation (6 conditions) on the "
                      "round-2 PASS; sanctioned by the reviewer authoring "
                      "checklist priming note (sel1024 peak ~2489 MiB vs A1R "
                      "~592 MiB)",
        "honest_framing": (
            "This canary validates mechanism liveness (admission rule, live "
            "watchdog tree sampling, K/H wiring, fixture path, record "
            "discipline); it is NOT a sizing run. MEASURED at rev-5 "
            "constants: the sel256 b0 cold-compile sampled tree peak was "
            "3225.671875 MiB over 60 samples (2026-09-27T00:41:42Z "
            "re-canary), only 78.328125 MiB (2.37%) under the 3304 MiB "
            "budget. The cold-compile transient is compile-dominated and "
            "variant-independent, with ~130 MiB run-to-run variance, so a "
            "canary false trip on variance alone is plausible. A canary "
            "trip is a diagnostics-class stop-and-preserve event: it costs "
            "one window and re-canaries on reviewer approval; it is NOT a "
            "matrix failure and NOT evidence against the envelope, and is "
            "default-adjudicated variance-class from the breach sample "
            "curve (variance-vs-defect is the reviewer's adjudication, not "
            "this record's). The rev-5 cold sel1024 sizing (b0p 2885.97 / "
            "cand 3003.25 MiB ru_maxrss, P5 rev 5) bounds the real matrix "
            "runs; the matrix's binding protection is the per-window guard "
            "at the same constants."),
        "variant": CANARY_VARIANT,
        "frozen_triple": {"W": 1, "K": K_STRIPES, "H": H_THREADS},
        "cache_discipline": {
            "canary_roots": {arm: f"{DATA}/nbc_f1r_canary_{arm}"
                             for arm in ARMS},
            "canary_roots_note": "throwaway, created fresh per arm, never the "
                                 "screen roots",
            "screen_roots_before_canary": screen_roots_before,
            "screen_roots_after_canary": screen_roots_after,
            "screen_roots_untouched": screen_roots_before == screen_roots_after},
        "arms": results,
        "cross_arm_identity_descriptive": identity,
        "watchdog": {arm: (recs or {}).get(arm, {}).get("peak_rss_ru_maxrss_mib")
                     for arm in results},
        "heavy_wall": {"cumulative_incl_opening_s": lease["cumulative_charged_s"],
                       "budget_s": HEAVY_WALL_BUDGET_S},
        "screen_summary_written": False,
        "note": "this stage NEVER writes f1r_screen_summary.json (h04 "
                "condition 4); the matrix has not started; ceiling intact",
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with open(CANARY_RECORD, "w") as f:
        json.dump(out, f, indent=1)
    print(f"[f1r-canary] record -> {CANARY_RECORD} ok={ok} "
          f"screen_roots_untouched={screen_roots_before == screen_roots_after}")
    return 0 if ok else 2


def norm_artifacts(rec):
    """Normalized {relpath: (sha, bytes)} with flow_<ts>/ dirs timestamped away."""
    out = {}
    for rel, meta in (rec.get("artifacts") or {}).items():
        out[FLOW_TS_RE.sub("flow_<TS>", rel)] = (meta["sha256"], meta["bytes"])
    return out


def array_hashes(rec):
    return {k: (v or {}).get("sha256_bytes") if isinstance(v, dict) else v
            for k, v in (rec.get("flow_arrays") or {}).items()}


def block_checks(rows):
    """engine_config + byte-identity per block. Returns (findings, track_ok).

    Called with the rows of COMPLETED blocks only (the caller decides when a
    block is complete); an empty rows list yields an empty findings list.
    """
    findings, track_ok = [], True
    for bnum in sorted({r["block"] for r in rows}):
        rb = next((r for r in rows if r.get("block") == bnum and r["arm"] == "b0"), None)
        rc = next((r for r in rows if r.get("block") == bnum and r["arm"] == "cand"), None)
        if not rb or not rc:
            continue
        failures = []
        for r, arm in ((rb, "b0"), (rc, "cand")):
            eco = r.get("engine_config_observed") or {}
            bad = {k: v for k, v in eco.items()
                   if k in ("flow_stripes_requested", "topology_num_threads",
                            "engine_num_threads") and v != K_STRIPES}
            if bad:
                failures.append({"mode": "engine_config_mismatch", "arm": arm,
                                 "bad": bad})
        ra_b0, ra_c = array_hashes(rb), array_hashes(rc)
        # node_flow may legitimately be None on both sides (compute_node_flow=false)
        arr_delta = sorted(k for k in set(ra_b0) | set(ra_c)
                           if ra_b0.get(k) != ra_c.get(k))
        if arr_delta:
            failures.append({"mode": "array_hash_delta", "keys": arr_delta})
        art_b0, art_c = norm_artifacts(rb), norm_artifacts(rc)
        art_delta = [k for k in sorted(set(art_b0) | set(art_c))
                     if art_b0.get(k) != art_c.get(k)]
        if art_delta:
            failures.append({"mode": "BYTE_IDENTITY_DELTA",
                             "paths": art_delta[:20]})
        fx_b0 = ((rb.get("origin_selection") or {}).get("fixture") or {})
        fx_c = ((rc.get("origin_selection") or {}).get("fixture") or {})
        fx_sha_b0 = fx_b0.get("sha256") if isinstance(fx_b0, dict) else None
        fx_sha_c = fx_c.get("sha256") if isinstance(fx_c, dict) else None
        if not (fx_sha_b0 and fx_sha_b0 == fx_sha_c):
            failures.append({"mode": "fixture_sha_mismatch",
                             "b0": fx_sha_b0, "cand": fx_sha_c})
        blk = {
            "block": bnum, "order": rb["order"],
            "engine_config_observed": {"b0": rb.get("engine_config_observed"),
                                       "cand": rc.get("engine_config_observed")},
            "array_hashes_equal": not arr_delta,
            "array_delta_keys": arr_delta,
            "array_hashes": {"b0": ra_b0, "cand": ra_c},
            "fixture_sha": fx_sha_b0,
            "artifact_count_normalized": len(art_b0),
            "byte_identity": "IDENTICAL" if not art_delta else "DELTA",
            "artifact_delta_paths": art_delta[:20],
            "verdict": "conformant" if not failures else "FAILED",
            "failures": failures,
        }
        if failures:
            track_ok = False
        findings.append(blk)
    return findings, track_ok


def load_rec(run_id):
    p = f"{RAW}/{run_id}.json"
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


def summary(refusal=None, stop_reason=None):
    rows = []
    for block, order, arm in SCHEDULE:
        run_id = f"f1r_b{block}_{order}_{arm}"
        r = load_rec(run_id)
        if r is None:
            rows.append({"run_id": run_id, "missing": True, "block": block,
                         "order": order, "arm": arm})
            continue
        rows.append({
            "run_id": run_id, "block": block, "order": order, "arm": arm,
            "cold_warm": ("cold (fresh per-arm cache)" if block == 1
                          else "warm (reused cache)"),
            "W_K_H": [1, K_STRIPES, H_THREADS],
            "window_s": round(r["window"]["application_window_ns"] / 1e9, 4),
            "engine_config_observed": r.get("engine_config_observed"),
            "resolved_gravity_cap": r.get("resolved_gravity_cap"),
            "numba_num_threads": r["process"]["numba_get_num_threads"],
            "cache_dir": r["env"]["NUMBA_CACHE_DIR"],
            "cache_file_count_at_run": cache_files(r["env"]["NUMBA_CACHE_DIR"]),
            "peak_rss_mib": r.get("peak_rss_ru_maxrss_mib"),
            "identity_ok": r["import"]["identity_assertion"].startswith("ok"),
            "counts": r.get("counts"),
            "origin_selection": r.get("origin_selection"),
            "artifacts": r.get("artifacts"),
            "array_hashes": array_hashes(r),
            "verbatim_warnings": r.get("verbatim_warnings"),
            "fatal": r.get("fatal"),
        })
    findings, track_ok = block_checks(rows)
    # descriptive statistics per spec conventions (h04 owns disposition)
    walls = {(r["block"], r["arm"]): r["window_s"] for r in rows
             if not r.get("missing") and "window_s" in r}
    stats = {"note": "unit = block (never the job); ratios b0/cand for speedup; "
                     "candidate-over-b0 for regressions"}
    if len(walls) == 6:
        b = [walls[(i, "b0")] for i in (1, 2, 3)]
        c = [walls[(i, "cand")] for i in (1, 2, 3)]
        med = lambda xs: sorted(xs)[1]  # 3 values
        r_warm = [walls[(i, "b0")] / walls[(i, "cand")] for i in (2, 3)]
        cold_reg = walls[(1, "cand")] / walls[(1, "b0")]
        logratios = [math.log(walls[(i, "b0")] / walls[(i, "cand")])
                     for i in (1, 2, 3)]
        m = sum(logratios) / 3
        se = math.sqrt(sum((x - m) ** 2 for x in logratios) / 2) / math.sqrt(3)
        import numpy as np
        rng = np.random.default_rng(BOOTSTRAP_SEED)
        lr = np.array(logratios)
        boots = rng.choice(lr, size=(BOOTSTRAP_B, 3), replace=True).mean(axis=1)
        b_lo, b_hi = np.percentile(boots, [2.5, 97.5])
        stats.update({
            "walls_b0_s": b, "walls_cand_s": c,
            "r_warm_b0_over_cand": [round(x, 4) for x in r_warm],
            "mean_warm_r": round(sum(r_warm) / 2, 4),
            "cold_regression_cand_over_b0_block1": round(cold_reg, 4),
            "proxy_speedup_median_b0_over_cand": round(med(b) / med(c), 4),
            "median_b0_s": med(b), "median_cand_s": med(c),
            "paired_log_ratio_diffs_ln_b0_minus_ln_cand": logratios,
            "t_interval_log_ratio_2dof": {
                "mean": round(m, 6), "se": round(se, 6),
                "t_crit": T_CRIT_2DOF,
                "ci_ln": [round(m - T_CRIT_2DOF * se, 6),
                          round(m + T_CRIT_2DOF * se, 6)],
                "ci_speedup": [round(math.exp(m - T_CRIT_2DOF * se), 4),
                               round(math.exp(m + T_CRIT_2DOF * se), 4)],
                "ci_lower_speedup_gt_1": math.exp(m - T_CRIT_2DOF * se) > 1.0},
            "bootstrap_descriptive": {
                "seed": BOOTSTRAP_SEED, "B": BOOTSTRAP_B,
                "ci_speedup": [round(float(math.exp(b_lo)), 4),
                               round(float(math.exp(b_hi)), 4)]},
            "investigate_trigger": any(
                walls[(i, "cand")] / walls[(i, "b0")] > 1.1 for i in (1, 2, 3)),
        })
    else:
        stats["incomplete"] = f"expected 6 runs, have {len(walls)}"
    with open(LEASE_LOG) as f:
        lease = json.load(f)
    out = {
        "task": "F1R", "record": "screen_summary", "role": "screen-executor",
        "spec_source": "evidence/F1R/screen_spec.json rev 5 (P4b bare-driver "
                       "vehicle; coordinator-committed before execution)",
        "descriptive_only": "h04-reviewer owns disposition/verdicts",
        "schedule": "3 blocks AB/BA/AB (A=b0); block 1 cold on fresh per-arm "
                    "cache roots, blocks 2-3 warm",
        "frozen_triple": {"W": 1, "K": K_STRIPES, "H": H_THREADS},
        "arms": {k: {"venv_python": v["python"],
                     "site_packages": v["site_packages"],
                     "cache_root": v["cache_root"],
                     "wheel_sha256": wheel_sha(k)} for k, v in ARMS.items()},
        "candidate_commit": CAND_COMMIT,
        "workload": {"cell": "O3_FLOW", "variant": VARIANT, "manifest": MANIFEST},
        "runs": rows,
        "block_findings": findings,
        "track_conformant": track_ok,
        "statistics": stats,
        "matrix_voided": bool(refusal),
        "matrix_voided_note": (refusal if refusal else "no mid-matrix refusal"),
        "matrix_complete": sum(1 for r in rows if not r.get("missing")) == 6,
        "runs_present": sum(1 for r in rows if not r.get("missing")),
        "restart_policy": "a voided matrix is not dispositional; a conformant "
                          "matrix restarts WHOLE (3 blocks, fresh roots, fresh "
                          "run ids) after reviewer ruling",
        "stop_reason": stop_reason,
        "heavy_wall": {"opening_balance_s": lease["opening_balance_s"],
                       "cumulative_incl_opening_s": lease["cumulative_charged_s"],
                       "budget_s": HEAVY_WALL_BUDGET_S},
        "lease_log": LEASE_LOG,
        "raw_records_dir": RAW,
        "exactness_anchor_note": "selection.json (h04) must cite F1I Phase-2 "
                                 "9.2 census artifacts as the numeric-identity "
                                 "anchor; screen owns speed only",
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with open(SUMMARY, "w") as f:
        json.dump(out, f, indent=1)
    print(f"[f1r-pairs] summary -> {SUMMARY}")
    s = stats.get("proxy_speedup_median_b0_over_cand")
    print(f"  proxy={s} cold_reg={stats.get('cold_regression_cand_over_b0_block1')} "
          f"track_conformant={track_ok}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["run", "summary", "canary"],
                    default="run")
    a = ap.parse_args()
    if a.stage == "summary":
        summary()
        return 0
    if a.stage == "canary":
        return canary()
    os.makedirs(RAW, exist_ok=True)
    for arm in ARMS.values():
        if not (os.path.exists(arm["python"]) and os.path.exists(arm["wheel_record"])
                and os.path.exists(arm["site_packages"])):
            print(f"[f1r-pairs] REFUSED: arm venv/wheel record missing: {arm}")
            return 2
    stop_reason = None
    # S3: a partial matrix must never silently resume — restart-whole requires
    # h04 approval and fresh run ids (spec P5; no-retry failure policy)
    run_ids = [f"f1r_b{b}_{o}_{a}" for b, o, a in SCHEDULE]
    present = [rid for rid in run_ids if os.path.exists(f"{RAW}/{rid}.json")]
    if len(present) == 6:
        print("[f1r-pairs] all 6 raw records present; no execution — use "
              "--stage summary to regenerate the summary")
        summary()
        return 0
    if present:
        print(f"[f1r-pairs] REFUSED: partial matrix present ({len(present)}/6: "
              f"{present}) — a voided matrix is not dispositional and never "
              "resumes; restart-whole requires h04 approval and fresh run ids")
        return 2
    # P5 rev 5 MATRIX-ENTRY bar: binds at initial fire and any post-void
    # re-entry (a re-entry is a fresh invocation; the all-6-present summary
    # regeneration above opens no window and is not an entry)
    import psutil
    avail_entry = psutil.virtual_memory().available
    entry_rec = {"what": "matrix_entry_check", "lease_id": LEASE_ID,
                 "start_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                 "available_at_admission_bytes": avail_entry,
                 "admission_080avail_bytes": int(0.80 * avail_entry),
                 "entry_bar_bytes": MATRIX_ENTRY_MIN_AVAIL_BYTES,
                 "entry_rule": ("0.80*available >= 3534 MiB at matrix entry "
                                "and any post-void re-entry (spec P5 rev 5)")}
    if avail_entry < MATRIX_ENTRY_MIN_AVAIL_BYTES:
        entry_rec["refused"] = ("matrix ENTRY bar not met: available < "
                                "4632084480 B (0.80*available < 3534 MiB); "
                                "no window opened, uncharged")
        lease_append(entry_rec)
        print(f"[f1r-pairs] REFUSED: matrix ENTRY bar (P5 rev 5): available "
              f"{avail_entry} B < {MATRIX_ENTRY_MIN_AVAIL_BYTES} B; no window "
              "opened, uncharged; re-invoke when the bar is met")
        return 2
    lease_append(entry_rec)
    for block, order, arm in SCHEDULE:
        rc, run_id, msg = run_slot(block, order, arm)
        print(f"[f1r-pairs] slot {run_id}: {msg}", flush=True)
        if rc != 0:
            stop_reason = msg
            print("[f1r-pairs] STOPPING screen per failure policy "
                  "(no retry; preserve evidence; h04 approval needed)")
            if msg.startswith("REFUSED"):
                summary(refusal=msg, stop_reason=stop_reason)
            else:
                summary(stop_reason=stop_reason)
            if rc == 3:
                return 3
            return rc
        # as soon as BOTH arms of this block exist: engine_config + byte-identity
        peer = "b0" if arm == "cand" else "cand"
        if load_rec(f"f1r_b{block}_{order}_{peer}") is not None:
            rows_now = []
            for a2 in ("b0", "cand"):
                r2 = load_rec(f"f1r_b{block}_{order}_{a2}")
                rows_now.append({**r2, "block": block, "order": order,
                                 "arm": a2, "run_id": f"f1r_b{block}_{order}_{a2}"})
            _, track_ok = block_checks(rows_now)
            if not track_ok:
                stop_reason = ("conformity failure (byte-identity delta / engine "
                               "config mismatch / fixture mismatch) in completed "
                               f"block {block} — STOP and report")
                print(f"[f1r-pairs] {stop_reason}")
                summary(refusal=None, stop_reason=stop_reason)
                return 3
    summary()
    return 0


if __name__ == "__main__":
    sys.exit(main())
