"""A3R paired-screen supervisor — campaigns/una_large_e2e/evidence/A3R/screen_spec.json executed.

Schedule: 3 paired complete-job blocks, arm order AB, BA, AB (A = b0 runs
first). Block 1 runs on FRESH per-arm NUMBA_CACHE_DIRs (wiped at block 1 ->
symmetric cold); blocks 2-3 reuse them (warm). Frozen pair (W,H) = (1,8)
(accessibility grid W x H <= 9; A1R precedent). PER-ARM BUILDS (arms-
mechanism rule, per-arm-build precedent — A3 has NO production toggle, so
the F2R single-binary launcher form is unavailable): TWO wheels from
`git archive` of the two PINNED commits (b0 = 9b1340e, cand = f17184b)
into TWO fresh venvs; ONE fresh interpreter process per (arm, block) run.

CLONE-ROT vs evidence/F2R/f2r_run_pairs.py (950 lines), stated per the P3
requirement:
  KEPT verbatim (machinery): REV-9 quiet-window gate with bounded foreground
    poll rounds; memory admission guard (0.80*avail < budget), pressure
    stop, heavy-wall guard; live watchdog (0.25 s ps tree samples, SIGTERM
    -> 10 s grace -> SIGKILL); lease charge-at-append (charged_s written
    ONLY in the append write; ledger recompute only there; refused /
    no-window paths uncharged); forward lease schema (crashed_child_start
    with EXPLICIT NULL PAIR); stage guard rename_aside on failure (out_root
    AND cache root); per-arm warm-cache monotonicity census +
    cross-arm family census after each block pair; G2 warning parity
    ENFORCED in the block comparator; matrix entry bar + 100x
    worst-case-with-one-restart headroom; refuse-on-partial-resume
    (restart-whole); supervisor campaign-venv guard (sys.prefix);
    descriptive-only summary with R2 interim custody of a prior summary;
    console transcript persisted at fire time.
  CHANGED: paths -> campaigns/una_large_e2e/evidence/A3R (campaign-tree
    convention; the repo-root evidence/F2R home was an anomaly — disclosed
    in the spec) + a3r_* names; ARMS = TWO wheel/venv pairs (a3r_b0_wheel
    from 9b1340e, a3r_cand_wheel from f17184b), per-arm git_archive_ref
    pin (refusal on either arm's mismatch); NO launcher (removed with the
    single-binary mechanism; driver takes --arm); MANIFEST -> O3_ACCESS
    (the A3I admission cell; P6 estimand comparability); frozen triple ->
    (W,H) = (1,8), NUMBA_NUM_THREADS only (no flow stripes on
    accessibility); LOCAL_FAMILY_MARKERS -> the two A3 symbols
    (_a3_scope_search_tailless, _a3_tail_admits); block comparator ->
    accessibility form (G1 binds on the FOUR in-memory accessibility
    array hashes; exported artifacts recorded + cross-compared
    DESCRIPTIVE with run-id normalization — disclosure D3); H-pin check
    reads process.numba_num_threads_config AND numba_get_num_threads == 8;
    G5 route receipts checked from records (cand: A3 symbols present,
    guard replay admits; b0: A3 symbols absent); leg-C share aggregation
    (P6 core) replaces the wall-delta G3 as the saving reading (ceiling +
    leg-C noise guard; disclosure D2); Q00 cold-peak basis = per-arm
    sizing governing (envelope record).
  ADDED:
    - --stage init: creates campaign_data/a3r_lease_log.json IF ABSENT,
      chained to the F2R log (opening = F2R cumulative read fresh at
      init; parent_log_sha256 = sha256 of the F2R log bytes at init;
      source_commit = cand commit; budget 14,400). The supervisor never
      auto-creates the log outside init; lease_guard refuses without it
      and verifies the pinned opening.
    - --stage sizing: per-arm COLD sizing runs per the recipe (item 11/13):
      direct frozen-driver invocations, throwaway per-arm cache roots
      (nbc_a3r_sizing_<arm>, deleted after census), run ids OUTSIDE the
      a3r_b<N> gate pattern, records to campaign_data (never evidence/),
      REV-9 preconditions IN-record, lease-charged diagnostics_class.
      Emits campaign_data/a3r_envelope_proposal.json (budget =
      max(1.10 x governing, governing x (1 + 0.077) ) per the Q00 rule,
      entry = budget/0.8) — the ENVELOPE ITSELF is pinned by spec
      amendment before matrix entry (F2R envelope pattern).
    - ENVELOPE GATE: --stage run REFUSES unless campaign_data/
      a3r_envelope.json EXISTS and matches the supervisor's memory/wall
      constants byte-for-value (draft constants are placeholders that
      cannot fire — the envelope cannot pre-exist sizing). Same gate
      form as the F1R v4-entry-bar lesson (gate ruled in prose but absent
      from the vehicle is a latent gap).

Failure policy: on failure/timeout/watchdog breach/identity delta preserve
all logs, rename aside, STOP — no retry beyond the schedule without h04's
approval. Refuse-on-partial-resume (restart-whole). Nothing fires before
the committed spec + vehicles clear review (R1/R2/R3 two-key fire).
"""
import argparse
import hashlib
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
CAMPAIGN_VENV = f"{CAMPAIGN}/venvs/campaign"
REPO = f"{CAMPAIGN}/wt-large-e2e"
EV = f"{REPO}/campaigns/una_large_e2e/evidence/A3R"
DATA = f"{CAMPAIGN}/campaign_data"
RAW = f"{EV}/raw"
SUMMARY = f"{DATA}/a3r_screen_summary.json"
LEASE_LOG = f"{DATA}/a3r_lease_log.json"
CONSOLE_DIR = f"{DATA}/a3r_console"
PARENT_LEASE_LOG = f"{DATA}/f2r_lease_log.json"
ENVELOPE = f"{DATA}/a3r_envelope.json"
ENVELOPE_PROPOSAL = f"{DATA}/a3r_envelope_proposal.json"
PHYS_BYTES = 17179869184
PRESSURE_STOP = max(1 << 30, int(0.10 * PHYS_BYTES))
# ---- memory/wall constants: PLACEHOLDER until sizing; --stage run REFUSES
# unless ENVELOPE exists and matches these values (see header ADDED gate).
WATCHDOG_MIB = 3436
WATCHDOG_BUDGET_BYTES = 3602907136
ADMISSION_MIN_AVAIL_BYTES = 4503633920
MATRIX_ENTRY_MIN_AVAIL_BYTES = 4556062720
WORST_CASE_ONE_RESTART_S = 118.0        # spec budget_arithmetic orientation
REV9_MIN_AVAIL_BYTES = 5368709120
REV9_MAX_LOADAVG = 8.0
REV9_ROUNDS, REV9_POLL_S = 6, 30
HEAVY_WALL_BUDGET_S = 14400.0
RUN_TIMEOUT_S = 1200
LEASE_ID = "A3R"
H_THREADS = 8
SOURCE_COMMIT = "f17184bf931ac5012bc3b85e91588502937906a7"
B0_COMMIT = "9b1340eb008accdff45fbdf57c6009dd5a6b5146"
MANIFEST = f"{REPO}/tests/large_e2e/inputs/O3_ACCESS.manifest.json"
T_CRIT_1DOF = 12.706204736174704         # two-sided 95%, 1 dof; tan(19*pi/40) = cot(pi/40)
T_CRIT_2DOF = 4.302652729749464          # two-sided 95%, 2 dof; sqrt(722/39)
# P6 preregistered constants (spec estimand_and_preregistration; decimals as
# printed in the A3I admission record / proof.md section 6 derivations):
P6_CEILING = 0.010638766512560852        # share_pre_T x (6661/55820)
P6_PRE_T_SHARE = 0.08915417305677027     # pinned T3 reading (H=10 frame, orientation)
P6_POST_T_PREDICTED = 0.07851540654420941
G4_BAND = [1.05, 1.25]                   # a-priori illustration; band fixed at prereg review from sizing cold walls
Q00_BAND_PERCENT = [7.5, 7.7]
HEADROOM_X = 100
LOCAL_FAMILY_MARKERS = ("_a3_scope_search_tailless", "_a3_tail_admits")
RUN_ID_TS_RE = re.compile(r"a3r_b\d+_[AB]{2}_[a-z0-9]+")

ARMS = {
    "b0": {"python": f"{DATA}/venvs/a3r_b0_wheel/bin/python",
           "site_packages": f"{DATA}/venvs/a3r_b0_wheel/lib/python3.11/site-packages",
           "wheel_record": f"{DATA}/a3r_wheel_and_venv_b0.json",
           "cache_root": f"{DATA}/nbc_a3r_b0",
           "git_archive_ref": B0_COMMIT},
    "cand": {"python": f"{DATA}/venvs/a3r_cand_wheel/bin/python",
             "site_packages": f"{DATA}/venvs/a3r_cand_wheel/lib/python3.11/site-packages",
             "wheel_record": f"{DATA}/a3r_wheel_and_venv_cand.json",
             "cache_root": f"{DATA}/nbc_a3r_cand",
             "git_archive_ref": SOURCE_COMMIT},
}
SCHEDULE = [(1, "AB", "b0"), (1, "AB", "cand"),
            (2, "BA", "cand"), (2, "BA", "b0"),
            (3, "AB", "b0"), (3, "AB", "cand")]
RUN_PREFIX = "a3r"


def wheel_record(arm):
    with open(ARMS[arm]["wheel_record"]) as f:
        return json.load(f)


def wheel_sha(arm):
    return wheel_record(arm)["wheel"]["sha256"]


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def cache_walk(root):
    out = {}
    for dirpath, _, files in os.walk(root):
        for fn in files:
            p = os.path.join(dirpath, fn)
            out[os.path.relpath(p, root)] = (sha256_file(p), os.path.getsize(p))
    return out


def cache_files(root):
    n = 0
    for _, _, fs in os.walk(root):
        n += len(fs)
    return n


def lease_cumulative():
    with open(LEASE_LOG) as f:
        log = json.load(f)
    charged = sum(w.get("charged_s", 0.0) for w in log["windows"])
    return round(log["opening_balance_s"] + charged, 1)


def lease_append(entry):
    if "refused" not in entry and entry.get("wall_s") is not None:
        entry["charged_s"] = entry["wall_s"]
    with open(LEASE_LOG) as f:
        log = json.load(f)
    log["windows"].append(entry)
    charged = sum(w.get("charged_s", 0.0) for w in log["windows"])
    log["cumulative_charged_s"] = round(log["opening_balance_s"] + charged, 1)
    log["remaining_s"] = round(HEAVY_WALL_BUDGET_S - log["cumulative_charged_s"], 1)
    log["updated_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    tmp = LEASE_LOG + ".tmp"
    with open(tmp, "w") as f:
        json.dump(log, f, indent=1)
    os.replace(tmp, LEASE_LOG)
    return log["cumulative_charged_s"]


def lease_init():
    """--stage init: create the A3R lease log chained to the F2R log, IF
    absent. Refuses when present (never rebuilds; append-only afterwards)."""
    if os.path.exists(LEASE_LOG):
        print(f"[a3r-pairs] REFUSED: lease log already exists ({LEASE_LOG}) "
              "— append-only; init never rebuilds")
        return 2
    if not os.path.exists(PARENT_LEASE_LOG):
        print(f"[a3r-pairs] REFUSED: parent F2R lease log missing "
              f"({PARENT_LEASE_LOG})")
        return 2
    parent_bytes = open(PARENT_LEASE_LOG, "rb").read()
    with open(PARENT_LEASE_LOG) as f:
        plog = json.load(f)
    stored = plog.get("cumulative_charged_s")
    recomputed = round(plog["opening_balance_s"]
                       + sum(w.get("charged_s", 0.0) for w in plog["windows"]), 1)
    if stored is None or abs(stored - recomputed) > 0.05:
        print(f"[a3r-pairs] REFUSED: parent ledger divergence stored="
              f"{stored} recomputed={recomputed} — stop-and-report, never patch")
        return 2
    log = {
        "lease_id": LEASE_ID,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "opening_balance_s": recomputed,
        "opening_balance_source": (f"cumulative_charged_s of {PARENT_LEASE_LOG} "
                                   "read fresh at init"),
        "parent_log_sha256": hashlib.sha256(parent_bytes).hexdigest(),
        "source_commit": SOURCE_COMMIT,
        "budget_s": HEAVY_WALL_BUDGET_S,
        "cumulative_charged_s": recomputed,
        "remaining_s": round(HEAVY_WALL_BUDGET_S - recomputed, 1),
        "windows": [],
    }
    tmp = LEASE_LOG + ".tmp"
    with open(tmp, "w") as f:
        json.dump(log, f, indent=1)
    os.replace(tmp, LEASE_LOG)
    print(f"[a3r-pairs] lease log created: opening={recomputed}s "
          f"parent_log_sha256={log['parent_log_sha256'][:8]} "
          f"source_commit={SOURCE_COMMIT[:8]}")
    return 0


def lease_guard():
    if not os.path.exists(LEASE_LOG):
        print(f"[a3r-pairs] REFUSED: lease log missing ({LEASE_LOG}) — run "
              "--stage init first (chained opening is an instrument-owned "
              "creation)")
        return None
    with open(LEASE_LOG) as f:
        log = json.load(f)
    if log.get("source_commit") != SOURCE_COMMIT:
        print("[a3r-pairs] REFUSED: lease source_commit != screened commit")
        return None
    return log


def rev9_gate(run_id):
    import psutil
    readings = []
    for i in range(REV9_ROUNDS):
        avail = psutil.virtual_memory().available
        la = os.getloadavg()
        stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        readings.append({"poll": i, "utc": stamp,
                         "available_bytes": int(avail), "loadavg": la})
        if avail >= REV9_MIN_AVAIL_BYTES and la[0] < REV9_MAX_LOADAVG:
            return True, {"run_id": run_id, "gate": "REV-9 quiet window",
                          "admitted": True, "rounds_used": i + 1,
                          "readings": readings,
                          "rev9_admitted_end": stamp}
        if i < REV9_ROUNDS - 1:
            time.sleep(REV9_POLL_S)
    return False, {"run_id": run_id, "gate": "REV-9 quiet window",
                   "admitted": False, "rounds_used": REV9_ROUNDS,
                   "readings": readings,
                   "not_fired_reason": ("quiet-window cap exhausted "
                                        f"({REV9_ROUNDS} x {REV9_POLL_S}s)")}


def tree_rss_kb(root_pid, snap):
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


def rename_aside(path):
    if path and os.path.exists(path):
        aside = f"{path}_incident"
        n = 1
        while os.path.exists(aside):
            n += 1
            aside = f"{path}_incident{n}"
        os.rename(path, aside)
        return os.path.basename(aside)
    return None


def envelope_check():
    """ENVELOPE GATE: memory/wall constants are placeholders until the
    sizing-derived envelope is pinned by spec amendment. Refuse unless the
    envelope record exists and matches, value for value."""
    if not os.path.exists(ENVELOPE):
        print(f"[a3r-pairs] REFUSED: envelope record missing ({ENVELOPE}) — "
              "run --stage sizing, then pin the envelope constants by spec "
              "amendment before any matrix fire")
        return False
    with open(ENVELOPE) as f:
        env_rec = json.load(f)
    want = {"watchdog_budget_mib": WATCHDOG_MIB,
            "watchdog_budget_bytes": WATCHDOG_BUDGET_BYTES,
            "admission_min_avail_bytes": ADMISSION_MIN_AVAIL_BYTES,
            "matrix_entry_min_avail_bytes": MATRIX_ENTRY_MIN_AVAIL_BYTES,
            "worst_case_one_restart_s": WORST_CASE_ONE_RESTART_S}
    bad = {k: (want[k], env_rec.get(k)) for k in want
           if env_rec.get(k) != want[k]}
    if bad:
        print(f"[a3r-pairs] REFUSED: envelope record does not match the "
              f"supervisor constants (spec amendment required): {bad}")
        return False
    return True


def driver_cmd(arm, run_id, out_json, out_root):
    a = ARMS[arm]
    return [a["python"], f"{EV}/a3r_bare_job.py",
            "--manifest", MANIFEST, "--cell", "O3_ACCESS",
            "--arm", arm,
            "--expected-site-packages", a["site_packages"],
            "--out-json", out_json,
            "--output-root", out_root,
            "--run-id", run_id]


def run_slot(block, order, arm):
    import psutil
    a = ARMS[arm]
    run_id = f"{RUN_PREFIX}_b{block}_{order}_{arm}"
    out_json = f"{RAW}/{run_id}.json"
    cache_fresh = "wiped_fresh" if block == 1 else "reused"
    if block == 1 and arm == "b0":  # wipe BOTH roots once, at block-1 start
        for armi in ARMS.values():
            if os.path.exists(armi["cache_root"]):
                shutil.rmtree(armi["cache_root"])
            os.makedirs(armi["cache_root"])
    out_root = f"{DATA}/a3r_out/{run_id}"
    if os.path.exists(out_root):
        shutil.rmtree(out_root)
    env = {"NUMBA_CACHE_DIR": a["cache_root"],
           "NUMBA_NUM_THREADS": str(H_THREADS),
           "PYTHONDONTWRITEBYTECODE": "1"}
    child_env = {**os.environ, **env}
    cmd = driver_cmd(arm, run_id, out_json, out_root)
    avail = psutil.virtual_memory().available
    entry = {"what": f"gate_run:{run_id}", "class": "gate_window",
             "lease_id": LEASE_ID, "gate_eligible": True,
             "block": block, "order": order, "arm": arm,
             "git_archive_ref": a["git_archive_ref"],
             "wheel_sha8": wheel_sha(arm)[:8],
             "cache_state": cache_fresh,
             "cache_file_count_before": cache_files(a["cache_root"]),
             "start_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
             "available_at_admission_bytes": int(avail),
             "admission_rule": (f"refuse when 0.80*available < budget "
                                f"{WATCHDOG_BUDGET_BYTES} B (envelope)"),
             "admission_080avail_bytes": int(0.80 * avail),
             "pressure_stop_bytes": PRESSURE_STOP,
             "watchdog_budget_mib": WATCHDOG_MIB,
             "supervisor_python": sys.executable,
             "loadavg_at_admission": os.getloadavg(),
             "cmd": cmd, "env_overrides": env}
    cum = lease_cumulative()
    if int(0.80 * avail) < WATCHDOG_BUDGET_BYTES:
        entry["refused"] = ("0.80*available < budget at admission (envelope); "
                            "mid-matrix refusal voids the matrix")
        entry["gate_eligible"] = False
        lease_append(entry)
        return 2, run_id, "REFUSED: memory admission guard"
    if avail < PRESSURE_STOP:
        entry["refused"] = "available < pressure stop before launch"
        entry["gate_eligible"] = False
        lease_append(entry)
        return 2, run_id, "REFUSED: memory pressure guard"
    if cum >= HEAVY_WALL_BUDGET_S:
        entry["refused"] = "heavy wall exhausted"
        entry["gate_eligible"] = False
        lease_append(entry)
        return 2, run_id, "REFUSED: heavy wall exhausted"
    admitted, rev9 = rev9_gate(run_id)
    entry["rev9_gate"] = rev9
    if not admitted:
        entry["class"] = "quiet_window_miss"
        entry["not_fired"] = True
        entry["gate_eligible"] = False
        entry["window_at_end_utc"] = None
        entry["rev9_admitted_end"] = None
        lease_append(entry)
        return 2, run_id, ("STOP: REV-9 quiet window not met within the poll "
                           "cap; nothing fired, uncharged")
    entry["rev9_admitted_end"] = rev9["rev9_admitted_end"]

    t0 = time.monotonic()
    with open(f"{CONSOLE_DIR}/{run_id}.log", "w") as con:
        p = subprocess.Popen(cmd, env={**child_env}, stdout=con,
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
                        {"t_s": round(time.monotonic() - t0, 2),
                         "tree_rss_kb": kb})
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
            rc = p.wait(timeout=RUN_TIMEOUT_S)
            timed_out = False
        except subprocess.TimeoutExpired:
            rc = -9
            timed_out = True
            p.kill()
            p.wait()
        stop_evt.set()
        th.join(timeout=2)
    with open(f"{CONSOLE_DIR}/{run_id}.log") as con:
        tail = con.read()[-2000:]
    wall = time.monotonic() - t0
    entry.update({"window_at_end_utc": time.strftime(
                      "%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                  "wall_s": round(wall, 1), "rc": rc, "tail": tail,
                  "console_log": f"{CONSOLE_DIR}/{run_id}.log",
                  "cache_file_count_after": cache_files(a["cache_root"]),
                  "watchdog_samples": len(breach["samples"]),
                  "watchdog_peak_tree_kb": max(
                      (s["tree_rss_kb"] for s in breach["samples"]), default=0),
                  "loadavg_at_completion": os.getloadavg()})
    if breach.get("breached"):
        entry["watchdog_breach"] = breach
        with open(f"{DATA}/a3r_watchdog_breach_{run_id}.json", "w") as f:
            json.dump(breach, f, indent=1)
    if rc != 0 and (rc < 0 or timed_out):
        entry["class"] = "crashed_child_start"  # explicit NULL PAIR, keys present
        entry["gate_eligible"] = False
        entry["window_at_end_utc"] = None
        entry["rev9_admitted_end"] = None
        entry["null_pair_note"] = ("crashed/killed child: window_at_end_utc and "
                                   "rev9_admitted_end are EXPLICIT NULLS, keys "
                                   "present (forward schema); wall still charged")
    cumulative = lease_append(entry)
    print(f"[a3r-pairs] {run_id} rc={rc} wall={wall:.1f}s "
          f"cumulative={cumulative}s", flush=True)
    if rc != 0:
        with open(f"{RAW}/{run_id}.supervisor.log", "w") as f:
            json.dump({"run_id": run_id, "rc": rc, "tail": tail, "env": env,
                       "cmd": cmd, "watchdog": breach}, f, indent=1)
        aside_out = rename_aside(out_root)
        aside_cache = rename_aside(a["cache_root"])
        entry["renamed_aside"] = {"out_root": aside_out,
                                  "cache_root": aside_cache,
                                  "note": "stage guard: failed run's state "
                                          "renamed aside before any re-fire"}
        with open(f"{RAW}/{run_id}.supervisor.log", "a") as f:
            f.write(json.dumps({"renamed_aside": entry["renamed_aside"]},
                               indent=1))
    if breach.get("breached"):
        return 3, run_id, (f"WATCHDOG BREACH ({WATCHDOG_MIB} MiB) — screen "
                           "STOPPED, state renamed aside, logs preserved")
    return rc, run_id, ("ok" if rc == 0 else f"FAILED rc={rc} (screen STOPPED, "
                        "state renamed aside, logs preserved, no retry)")


def norm_artifacts(rec):
    out = {}
    for rel, meta in (rec.get("artifacts") or {}).items():
        out[RUN_ID_TS_RE.sub("a3r_<RUN>", rel)] = (meta["sha256"], meta["bytes"])
    return out


def array_hashes(rec):
    return {k: (v or {}).get("sha256_bytes") if isinstance(v, dict) else v
            for k, v in (rec.get("accessibility_arrays") or {}).items()}


def kernel_census_check(prev_census, census, arm, block):
    fails = []
    if prev_census is None:
        return fails
    new = sorted(set(census) - set(prev_census))
    if new:
        fails.append({"mode": "warm_cache_entry_added", "arm": arm,
                      "block": block, "entries": new})
    changed = sorted(k for k in set(census) & set(prev_census)
                     if census[k] != prev_census[k])
    if changed:
        fails.append({"mode": "warm_cache_entry_content_changed", "arm": arm,
                      "block": block, "entries": changed[:20],
                      "note": "warm blocks never rewrite cache files; "
                              "same-name content/size change = recompile = "
                              "confound"})
    return fails


def cross_arm_census_check(census_b0, census_cand, block):
    fails = []
    cand_only = sorted(set(census_cand) - set(census_b0))
    b0_only = sorted(set(census_b0) - set(census_cand))
    unexpected = [n for n in cand_only
                  if not any(m in n for m in LOCAL_FAMILY_MARKERS)]
    if unexpected:
        fails.append({"mode": "UNPREDICTED_extra_cache_entry", "block": block,
                      "unexpected_cand_only_entries": unexpected[:20]})
    return fails, cand_only, b0_only


def legc_share(rec, kind):
    return (((rec.get("a3_legc") or {}).get(kind) or {})
            .get("share_serial_equiv"))


def block_checks(rows):
    """H-pin + module pins + G1 array identity + G2 warning parity + G5 route
    receipts per completed block. Artifacts are cross-compared DESCRIPTIVE
    (disclosure D3): a delta is RECORDED, not gating. Returns
    (findings, track_ok)."""
    findings, track_ok = [], True
    for bnum in sorted({r["block"] for r in rows}):
        rb = next((r for r in rows if r.get("block") == bnum and r["arm"] == "b0"), None)
        rc = next((r for r in rows if r.get("block") == bnum and r["arm"] == "cand"), None)
        if not rb or not rc:
            continue
        failures = []
        for r, arm in ((rb, "b0"), (rc, "cand")):
            proc = r.get("process") or {}
            hbad = {k: proc.get(k) for k in
                    ("numba_num_threads_config", "numba_get_num_threads")
                    if proc.get(k) != H_THREADS}
            if hbad:
                failures.append({"mode": "H_pin_mismatch", "arm": arm, "bad": hbad})
            if not (r.get("module_pins") or {}).get("all_match", False):
                failures.append({"mode": "module_pin_mismatch", "arm": arm})
        # G5 route receipts
        pres_b = ((rb.get("route_receipts") or {}).get("symbol_presence") or {})
        pres_c = ((rc.get("route_receipts") or {}).get("symbol_presence") or {})
        if not (pres_b.get("_a3_tail_admits") is False
                and pres_b.get("_a3_scope_search_tailless") is False
                and pres_b.get("_a1_scope_admits") is True):
            failures.append({"mode": "G5_b0_signature_missing", "b0": pres_b})
        if not (pres_c.get("_a3_tail_admits") is True
                and pres_c.get("_a3_scope_search_tailless") is True
                and pres_c.get("_a1_scope_admits") is True):
            failures.append({"mode": "G5_cand_signature_missing", "cand": pres_c})
        for r, arm in ((rb, "b0"), (rc, "cand")):
            rep = ((r.get("route_receipts") or {}).get("guard_conditions_replay")
                   or {})
            if rep.get("a3_guard_would_admit") is not True:
                failures.append({"mode": "G5_guard_replay_not_admitting",
                                 "arm": arm, "replay": rep})
        # G1 byte identity on the four in-memory arrays
        ra_b0, ra_c = array_hashes(rb), array_hashes(rc)
        arr_delta = sorted(k for k in set(ra_b0) | set(ra_c)
                           if ra_b0.get(k) != ra_c.get(k))
        if arr_delta:
            failures.append({"mode": "ARRAY_IDENTITY_DELTA", "keys": arr_delta})
        # artifacts: DESCRIPTIVE compare (run-id normalized), recorded not gating
        art_b0, art_c = norm_artifacts(rb), norm_artifacts(rc)
        art_delta = [k for k in sorted(set(art_b0) | set(art_c))
                     if art_b0.get(k) != art_c.get(k)]
        wb, wc = rb.get("verbatim_warnings") or [], rc.get("verbatim_warnings") or []
        if len(wb) != len(wc):
            failures.append({"mode": "G2_WARNING_PARITY_COUNT",
                             "b0": len(wb), "cand": len(wc)})
        blk = {
            "block": bnum, "order": rb["order"],
            "numba_threads": {"b0": rb.get("process", {}).get("numba_get_num_threads"),
                              "cand": rc.get("process", {}).get("numba_get_num_threads")},
            "module_pins": {"b0": rb.get("module_pins"),
                            "cand": rc.get("module_pins")},
            "route_receipts": {"b0": rb.get("route_receipts"),
                               "cand": rc.get("route_receipts")},
            "array_hashes_equal": not arr_delta,
            "array_delta_keys": arr_delta,
            "array_hashes": {"b0": ra_b0, "cand": ra_c},
            "warning_parity": {"b0_count": len(wb), "cand_count": len(wc),
                               "equal": len(wb) == len(wc),
                               "b0": wb, "cand": wc},
            "artifact_count_normalized": len(art_b0),
            "artifact_delta_descriptive": art_delta[:20],
            "artifact_compare": "DESCRIPTIVE ONLY — G1 binds on the four "
                                "in-memory accessibility arrays (spec D3)",
            "block_conformance": "conformant" if not failures else "FAILED",
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
        run_id = f"{RUN_PREFIX}_b{block}_{order}_{arm}"
        r = load_rec(run_id)
        if r is None:
            rows.append({"run_id": run_id, "missing": True, "block": block,
                         "order": order, "arm": arm})
            continue
        win_ns = (r.get("window") or {}).get("application_window_ns")
        ident = ((r.get("import") or {}).get("identity_assertion") or "")
        legc = r.get("a3_legc") or {}
        rows.append({
            "run_id": run_id, "block": block, "order": order, "arm": arm,
            "cold_warm": ("cold (fresh per-arm cache)" if block == 1
                          else "warm (reused cache)"),
            "W_H": [1, H_THREADS],
            "window_s": (round(win_ns / 1e9, 6) if win_ns else None),
            "numba_num_threads": (r.get("process") or {}).get("numba_get_num_threads"),
            "module_pins": r.get("module_pins"),
            "route_receipts": r.get("route_receipts"),
            "a3_legc": {"full": legc.get("full_nd_init"),
                        "tailfree": legc.get("tailfree_V_init"),
                        "production_expression": legc.get("production_expression"),
                        "cutoff": legc.get("cutoff"),
                        "n_full_nd": legc.get("n_full_nd"),
                        "n_tailfree_V": legc.get("n_tailfree_V")},
            "share_full": legc_share(r, "full_nd_init"),
            "share_tailfree": legc_share(r, "tailfree_V_init"),
            "cache_dir": (r.get("env") or {}).get("NUMBA_CACHE_DIR"),
            "peak_rss_mib": r.get("peak_rss_ru_maxrss_mib"),
            "identity_ok": ident.startswith("ok"),
            "counts": r.get("counts"),
            "artifacts": r.get("artifacts"),
            "array_hashes": array_hashes(r),
            "verbatim_warnings": r.get("verbatim_warnings"),
            "fatal": r.get("fatal"),
        })
    findings, track_ok = block_checks(rows)
    census = {}
    for arm in ("b0", "cand"):
        root = ARMS[arm]["cache_root"]
        census[arm] = cache_walk(root) if os.path.isdir(root) else {}
    census_fails, cand_only, b0_only = cross_arm_census_check(
        census["b0"], census["cand"], "final")
    census_rec = {
        "roots": {arm: ARMS[arm]["cache_root"] for arm in ARMS},
        "entry_counts": {arm: len(census[arm]) for arm in census},
        "cand_only_entries": cand_only,
        "b0_only_entries_recorded": b0_only,
        "cand_only_all_a3_family": not census_fails,
        "failures": census_fails,
        "note": ("cand-only entries must ALL match the A3 family markers "
                 "(compiled = called); an UNPREDICTED extra entry = "
                 "packaging-confound signature"),
    }
    stats = {"note": "unit = the PAIR (block); block_set_ledger: admission "
                     "reads ONLY warm blocks 2-3; block 1 feeds cold "
                     "regression (G4) and Q00 peaks; proxy prohibition: no "
                     "all-blocks aggregate is compared against any gate",
             "gate_quantities_are_data": "boolean comparisons below are "
             "recorded data points; h04-reviewer owns every gate evaluation "
             "and the disposition"}
    walls = {(r["block"], r["arm"]): r["window_s"] for r in rows
             if not r.get("missing") and r.get("window_s")}
    shares_full = {(r["block"], r["arm"]): r["share_full"] for r in rows
                   if not r.get("missing") and r.get("share_full") is not None}
    shares_tf = {(r["block"], r["arm"]): r["share_tailfree"] for r in rows
                 if not r.get("missing") and r.get("share_tailfree") is not None}
    if len(walls) == 6 and len(shares_full) == 6 and len(shares_tf) == 6:
        b = [walls[(i, "b0")] for i in (1, 2, 3)]
        c = [walls[(i, "cand")] for i in (1, 2, 3)]
        sf_b = [shares_full[(i, "b0")] for i in (1, 2, 3)]
        stf_c = [shares_tf[(i, "cand")] for i in (1, 2, 3)]
        # P6 core: warm-governing production-share midpoints (blocks 2-3)
        b0_share_mid = (sf_b[1] + sf_b[2]) / 2
        cand_share_mid = (stf_c[1] + stf_c[2]) / 2
        realized_saving = b0_share_mid - cand_share_mid
        spread_b0_share = abs(sf_b[1] - sf_b[2])
        spread_cand_share = abs(stf_c[1] - stf_c[2])
        m_e = cand_share_mid
        stats.update({
            "p6_estimand": {
                "form": "share = legC_mean_s x O_rows / application_window_s "
                        "(same estimand as the pinned T3 reading; per-run "
                        "in-record computation)",
                "pinned_pre_T_share_H10_orientation": P6_PRE_T_SHARE,
                "pinned_ceiling_serial_equiv": P6_CEILING,
                "post_T_predicted_orientation": P6_POST_T_PREDICTED,
                "b0_share_full_warm_mid": round(b0_share_mid, 12),
                "cand_share_tailfree_warm_mid": round(cand_share_mid, 12),
                "realized_saving_share": round(realized_saving, 12),
                "ceiling_data_point_saving_le_ceiling":
                    realized_saving <= P6_CEILING,
                "m_e_criterion_i_input": round(m_e, 12),
                "m_e_data_point_me_ge_0_05": m_e >= 0.05,
                "noise_guard_legc": {
                    "form": "gates.G3_analog: |realized_saving| must EXCEED "
                            "max(within-arm warm leg-C share spread)",
                    "spread_b0_share": round(spread_b0_share, 12),
                    "spread_cand_share": round(spread_cand_share, 12),
                    "max_spread_share": round(max(spread_b0_share,
                                                  spread_cand_share), 12),
                    "saving_exceeds_max_spread":
                        abs(realized_saving) > max(spread_b0_share,
                                                   spread_cand_share)},
                "frame_transfer_anchor": {
                    "b0_share_full_warm_mid_vs_pinned": round(b0_share_mid, 12),
                    "orientation_band_H8": [0.078675895, 0.090888226],
                    "note": "non-binding orientation (spec D1); material "
                            "departure = H-frame finding routed, not a gate "
                            "fail by itself"},
            },
            "legc_all_shares": {
                "b0_full": {f"b{i}": shares_full[(i, "b0")] for i in (1, 2, 3)},
                "cand_tailfree": {f"c{i}": shares_tf[(i, "cand")]
                                  for i in (1, 2, 3)},
                "b0_tailfree": {f"b{i}": shares_tf[(i, "b0")]
                                for i in (1, 2, 3)},
                "cand_full": {f"c{i}": shares_full[(i, "cand")]
                              for i in (1, 2, 3)}},
            "walls_b0_s": b, "walls_cand_s": c,
            "cold_regression_cand_over_b0_block1": round(c[0] / b[0], 6),
            "g4_band_illustration": {"band": G4_BAND,
                                     "cold_regression": round(c[0] / b[0], 6),
                                     "note": "a-priori illustration; band "
                                             "fixed at prereg review from "
                                             "sizing cold walls"},
            "warm_walls_descriptive": {
                "b0_midpoint_s": round((b[1] + b[2]) / 2, 6),
                "cand_midpoint_s": round((c[1] + c[2]) / 2, 6),
                "note": "wall deltas are DESCRIPTIVE (D2): the accounted "
                        "wall-level saving ~0.0011 of the window is far "
                        "below run-to-run wall spread"},
            "q00_block1_cold_peaks": {
                "per_arm": {arm: next(
                    (r.get("peak_rss_mib") for r in rows
                     if r.get("block") == 1 and r["arm"] == arm
                     and not r.get("missing") and r.get("peak_rss_mib")), None)
                    for arm in ("b0", "cand")},
                "preregistered_band_percent": Q00_BAND_PERCENT,
                "basis": "per-arm sizing governing (envelope record); "
                         "materially above the 7.5-7.7% precedent band = "
                         "Q00 blocking note"},
        })
    else:
        stats["incomplete"] = (f"expected 6 runs, walls={len(walls)} "
                               f"shares_full={len(shares_full)} "
                               f"shares_tailfree={len(shares_tf)}")
    if os.path.exists(LEASE_LOG):
        with open(LEASE_LOG) as f:
            lease = json.load(f)
        lease_block = {"opening_balance_s": lease["opening_balance_s"],
                       "parent_log_sha256": lease.get("parent_log_sha256"),
                       "cumulative_incl_opening_s": lease["cumulative_charged_s"],
                       "budget_s": HEAVY_WALL_BUDGET_S}
    else:
        lease_block = {"note": f"lease log not present at summary time "
                               f"({LEASE_LOG})", "budget_s": HEAVY_WALL_BUDGET_S}
    out = {
        "task": "A3R", "record": "screen_summary", "role": "screen-executor",
        "spec_source": ("campaigns/una_large_e2e/evidence/A3R/screen_spec.json "
                        "(bytes pinned at vehicle commit)"),
        "descriptive_only": "h04-reviewer owns disposition/verdicts; this "
                            "summary computes quantities and records data "
                            "points only",
        "schedule": "3 blocks AB/BA/AB (A=b0); block 1 cold on fresh per-arm "
                    "cache roots, blocks 2-3 warm; PER-ARM wheels/venvs, ONE "
                    "fresh interpreter per run",
        "frozen_pair": {"W": 1, "H": H_THREADS},
        "arms": {k: {"venv_python": v["python"],
                     "site_packages": v["site_packages"],
                     "cache_root": v["cache_root"],
                     "git_archive_ref": v["git_archive_ref"]}
                 for k, v in ARMS.items()},
        "candidate_commit": SOURCE_COMMIT,
        "b0_commit": B0_COMMIT,
        "workload": {"cell": "O3_ACCESS", "manifest": MANIFEST},
        "runs": rows,
        "block_findings": findings,
        "track_conformant": track_ok,
        "kernel_set_census": census_rec,
        "statistics": stats,
        "matrix_voided": bool(refusal or stop_reason),
        "matrix_voided_note": (refusal if refusal else
                               (stop_reason if stop_reason else
                                "no mid-matrix refusal or stop")),
        "matrix_complete": sum(1 for r in rows if not r.get("missing")) == 6,
        "runs_present": sum(1 for r in rows if not r.get("missing")),
        "restart_policy": "a voided matrix is not dispositional; a conformant "
                          "matrix restarts WHOLE (3 blocks, fresh roots, fresh "
                          "run ids) after reviewer ruling",
        "stop_reason": stop_reason,
        "heavy_wall": lease_block,
        "lease_log": LEASE_LOG,
        "raw_records_dir": RAW,
        "console_transcripts_dir": CONSOLE_DIR,
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    if os.path.exists(SUMMARY):
        prev_bytes = open(SUMMARY, "rb").read()
        prev_sha = hashlib.sha256(prev_bytes).hexdigest()
        interim = f"{DATA}/interim_{prev_sha[:8]}_summary.json"
        if not os.path.exists(interim):
            with open(interim, "wb") as g:
                g.write(prev_bytes)
        print(f"[a3r-pairs] R2 custody: prior summary "
              f"{prev_sha[:8]} -> {interim}")
    tmp = SUMMARY + ".tmp"
    with open(tmp, "w") as f:
        json.dump(out, f, indent=1)
    os.replace(tmp, SUMMARY)
    print(f"[a3r-pairs] summary -> {SUMMARY}")
    print(f"  track_conformant={track_ok} runs_present={out['runs_present']}")


def sizing_stage():
    """Per-arm COLD sizing runs (diagnostics_class): direct frozen-driver
    invocations, throwaway cache roots, REV-9 preconditions in-record,
    records to campaign_data, lease-charged; emits the envelope PROPOSAL.
    The envelope itself is pinned by spec amendment (never by this stage)."""
    import psutil
    if lease_guard() is None:
        return 2
    # [A3R-B7 fold, h04] attempt-1 crash fix (rc=1, FileNotFoundError at the
    # console open below): main() creates CONSOLE_DIR only in the run branch
    # (:1031-1032), after this stage's early return at :1005-1006. Ruled
    # single-site additive fold — dispatch-level refusal paths are untouched
    # and still write nothing.
    os.makedirs(CONSOLE_DIR, exist_ok=True)
    for arm in ("b0", "cand"):
        run_id = f"a3r_sizing_{arm}"
        rec_path = f"{DATA}/a3r_sizing_{arm}.json"
        sizing_root = f"{DATA}/nbc_a3r_sizing_{arm}"
        out_root = f"{DATA}/a3r_sizing_out/{run_id}"
        if os.path.exists(rec_path):
            print(f"[a3r-pairs] REFUSED: sizing record exists ({rec_path}) — "
                  "a rejected sizing record is RETAINED; re-run needs a NEW "
                  "run id + fresh throwaway root after reviewer ruling")
            return 2
        avail = psutil.virtual_memory().available
        la = os.getloadavg()
        precondition = {"available_at_start_bytes": int(avail),
                        "available_rule": ">= 5,368,709,120 B",
                        "loadavg_1m": la[0], "loadavg_rule": "< 8.0"}
        if avail < REV9_MIN_AVAIL_BYTES or la[0] >= REV9_MAX_LOADAVG:
            precondition["met"] = False
            with open(rec_path + ".rejected", "w") as f:
                json.dump({"run_id": run_id, "precondition": precondition,
                           "note": "rejected pre-fire; retained"}, f, indent=1)
            print(f"[a3r-pairs] sizing precondition NOT met for {arm}: "
                  f"{precondition}; rejected record retained")
            return 2
        precondition["met"] = True
        if os.path.exists(sizing_root):
            shutil.rmtree(sizing_root)
        os.makedirs(sizing_root)
        if os.path.exists(out_root):
            shutil.rmtree(out_root)
        env = {"NUMBA_CACHE_DIR": sizing_root,
               "NUMBA_NUM_THREADS": str(H_THREADS),
               "PYTHONDONTWRITEBYTECODE": "1"}
        cmd = driver_cmd(arm, run_id, rec_path, out_root)
        t0 = time.monotonic()
        timed_out = False
        with open(f"{CONSOLE_DIR}/{run_id}.log", "w") as con:
            try:
                p = subprocess.run(cmd, env={**os.environ, **env}, stdout=con,
                                   stderr=subprocess.STDOUT, text=True,
                                   timeout=RUN_TIMEOUT_S)
                rc = p.returncode
            except subprocess.TimeoutExpired:
                rc, timed_out = -9, True
        wall = round(time.monotonic() - t0, 1)
        peak = None
        if os.path.exists(rec_path):
            with open(rec_path) as f:
                peak = json.load(f).get("peak_rss_ru_maxrss_mib")
        lease_append({"what": f"sizing_run:{run_id}", "class": "diagnostics_window",
                      "lease_id": LEASE_ID, "gate_eligible": False,
                      "arm": arm, "start_utc": time.strftime(
                          "%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                      "wall_s": wall, "rc": rc,
                      "timed_out": timed_out,
                      "precondition": precondition,
                      "throwaway_root": sizing_root,
                      "cache_entry_count_after": cache_files(sizing_root),
                      "peak_rss_ru_maxrss_mib": peak,
                      "note": "throwaway root disjoint from matrix roots; "
                              "record in campaign_data, never evidence/"})
        shutil.rmtree(sizing_root)
        shutil.rmtree(out_root, ignore_errors=True)
        print(f"[a3r-pairs] sizing {arm}: rc={rc} wall={wall}s peak={peak} MiB")
        if rc != 0:
            print("[a3r-pairs] sizing run FAILED — stop-and-report")
            return rc
    # envelope proposal from the two realized sizing peaks
    peaks = {}
    for arm in ("b0", "cand"):
        with open(f"{DATA}/a3r_sizing_{arm}.json") as f:
            peaks[arm] = json.load(f).get("peak_rss_ru_maxrss_mib")
    if not all(isinstance(v, (int, float)) and v > 0 for v in peaks.values()):
        print(f"[a3r-pairs] REFUSED: sizing peaks unusable: {peaks}")
        return 2
    gov = max(peaks.values())
    budget_mib = math.ceil(max(1.10 * gov, gov * 1.077))
    proposal = {
        "task": "A3R", "record": "envelope_proposal",
        "sizing_peaks_mib": peaks,
        "governing_mib": gov,
        "proposed_watchdog_budget_mib": budget_mib,
        "proposed_watchdog_budget_bytes": int(budget_mib * (1 << 20)),
        "proposed_admission_min_avail_bytes":
            int(math.ceil(0.80 ** -1 * budget_mib * (1 << 20))),
        "proposed_matrix_entry_min_avail_bytes":
            int(math.ceil(0.80 ** -1 * budget_mib * (1 << 20))) + 50 * (1 << 20),
        "q00_note": "budget = max(1.10 x governing, governing x 1.077) per "
                    "the Q00 sizing rule; entry = budget/0.8 (+50 MiB margin)",
        "note": ("PROPOSAL ONLY — the envelope is pinned by spec amendment "
                 "(constants re-derived and frozen in the supervisor) before "
                 "any matrix fire"),
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    tmp = ENVELOPE_PROPOSAL + ".tmp"
    with open(tmp, "w") as f:
        json.dump(proposal, f, indent=1)
    os.replace(tmp, ENVELOPE_PROPOSAL)
    print(f"[a3r-pairs] envelope proposal -> {ENVELOPE_PROPOSAL}: {proposal}")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["init", "sizing", "run", "summary"],
                    default="run")
    ap.add_argument("--run-id-prefix", default="a3r")
    a = ap.parse_args()
    global RUN_PREFIX
    RUN_PREFIX = a.run_id_prefix
    if os.path.realpath(sys.prefix) != os.path.realpath(CAMPAIGN_VENV):
        print(f"[a3r-pairs] REFUSED: supervisor interpreter is not "
              f"the campaign venv: sys.executable {sys.executable}, "
              f"sys.prefix {sys.prefix} != pinned {CAMPAIGN_VENV}")
        return 2
    if a.stage == "init":
        return lease_init()
    if a.stage == "summary":
        summary()
        return 0
    if a.stage == "sizing":
        return sizing_stage()
    # ---- stage run ----
    if a.run_id_prefix == "a3r":
        print("[a3r-pairs] REFUSED: --stage run requires an explicit fresh "
              "--run-id-prefix (attempt-1 prefix 'a3r' is landed+committed; "
              "a default-prefix run would overwrite landed records)")
        return 2
    if lease_guard() is None:
        return 2
    for arm in ("b0", "cand"):
        rec = ARMS[arm]
        if not (os.path.exists(rec["python"]) and os.path.exists(rec["wheel_record"])
                and os.path.exists(rec["site_packages"])):
            print(f"[a3r-pairs] REFUSED: wheel/venv record missing for "
                  f"{arm}: {rec}")
            return 2
        arch_ref = wheel_record(arm).get("git_archive_ref")
        if arch_ref != rec["git_archive_ref"]:
            print(f"[a3r-pairs] REFUSED: {arm} wheel record git_archive_ref "
                  f"({arch_ref}) != pinned commit {rec['git_archive_ref']} — "
                  "packaging pin violated (wheel must be archived from the "
                  "pinned commit, never fire-time HEAD)")
            return 2
    if not envelope_check():
        return 2
    os.makedirs(RAW, exist_ok=True)
    os.makedirs(CONSOLE_DIR, exist_ok=True)
    stop_reason = None
    run_ids = [f"{RUN_PREFIX}_b{b}_{o}_{a2}" for b, o, a2 in SCHEDULE]
    present = [rid for rid in run_ids if os.path.exists(f"{RAW}/{rid}.json")]
    if len(present) == 6:
        print("[a3r-pairs] all 6 raw records present; no execution — use "
              "--stage summary to regenerate the summary")
        summary()
        return 0
    if present:
        print(f"[a3r-pairs] REFUSED: partial matrix present ({len(present)}/6: "
              f"{present}) — a voided matrix is not dispositional and never "
              "resumes; restart-whole requires h04 approval and fresh run ids")
        return 2
    import psutil
    avail_entry = psutil.virtual_memory().available
    cum_entry = lease_cumulative()
    headroom_ok = (WORST_CASE_ONE_RESTART_S * HEADROOM_X
                   <= HEAVY_WALL_BUDGET_S - cum_entry)
    entry_rec = {"what": "matrix_entry_check", "class": "matrix_entry",
                 "lease_id": LEASE_ID, "gate_eligible": False,
                 "start_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                 "available_at_admission_bytes": int(avail_entry),
                 "entry_bar_bytes": MATRIX_ENTRY_MIN_AVAIL_BYTES,
                 "entry_rule": (f"available >= {MATRIX_ENTRY_MIN_AVAIL_BYTES} B "
                                "at matrix entry and any post-void re-entry "
                                "(envelope: entry = budget/0.8 + 50 MiB)"),
                 "headroom_rule": (f"worst-case-with-one-restart "
                                   f"{WORST_CASE_ONE_RESTART_S}s x {HEADROOM_X} "
                                   "<= remaining"),
                 "remaining_s": round(HEAVY_WALL_BUDGET_S - cum_entry, 1),
                 "headroom_ok": headroom_ok}
    if avail_entry < MATRIX_ENTRY_MIN_AVAIL_BYTES:
        entry_rec["refused"] = ("matrix ENTRY bar not met; no window opened, "
                                "uncharged")
        lease_append(entry_rec)
        print(f"[a3r-pairs] REFUSED: matrix ENTRY bar: available {avail_entry} "
              f"B < {MATRIX_ENTRY_MIN_AVAIL_BYTES} B; re-invoke when met")
        return 2
    if not headroom_ok:
        entry_rec["refused"] = "100x headroom rule not met against remaining"
        lease_append(entry_rec)
        print("[a3r-pairs] REFUSED: 100x worst-case headroom rule")
        return 2
    lease_append(entry_rec)
    prev_census = {"b0": None, "cand": None}
    for block, order, arm in SCHEDULE:
        rc, run_id, msg = run_slot(block, order, arm)
        print(f"[a3r-pairs] slot {run_id}: {msg}", flush=True)
        if rc != 0:
            stop_reason = msg
            print("[a3r-pairs] STOPPING screen per failure policy "
                  "(no retry; preserve evidence; h04 approval needed)")
            summary(refusal=msg if msg.startswith("REFUSED") else None,
                    stop_reason=stop_reason)
            return 3 if rc == 3 else rc
        peer = "b0" if arm == "cand" else "cand"
        if load_rec(f"{RUN_PREFIX}_b{block}_{order}_{peer}") is not None:
            rows_now = []
            for a2 in ("b0", "cand"):
                r2 = load_rec(f"{RUN_PREFIX}_b{block}_{order}_{a2}")
                rows_now.append({**r2, "block": block, "order": order,
                                 "arm": a2,
                                 "run_id": f"{RUN_PREFIX}_b{block}_{order}_{a2}"})
            findings, track_ok = block_checks(rows_now)
            census_now = {a2: cache_walk(ARMS[a2]["cache_root"])
                          for a2 in ("b0", "cand")}
            cfails = []
            for a2 in ("b0", "cand"):
                cfails += kernel_census_check(prev_census[a2], census_now[a2],
                                              a2, block)
            xfails, _cand_only, _b0_only = cross_arm_census_check(
                census_now["b0"], census_now["cand"], block)
            cfails += xfails
            prev_census = census_now
            if not track_ok or cfails:
                stop_reason = ("conformity failure in completed block "
                               f"{block}: {findings} census={cfails} — STOP "
                               "and report")
                print(f"[a3r-pairs] {stop_reason}")
                summary(refusal=None, stop_reason=stop_reason)
                return 3
    summary()
    return 0


if __name__ == "__main__":
    sys.exit(main())
