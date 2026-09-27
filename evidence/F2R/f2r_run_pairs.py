"""F2R paired-screen supervisor — evidence/F2R/screen_spec.json (P1 bytes) executed.

Schedule: 3 paired complete-job blocks, arm order AB, BA, AB (A = b0 runs
first). Block 1 runs on FRESH per-arm NUMBA_CACHE_DIRs (wiped at block 1 ->
symmetric cold first-compile, condition 5); blocks 2-3 reuse them (warm).
Frozen triple (W,K,H) = (1,8,1). SAME-BINARY ARMS (central ruling): ONE wheel
from git archive of 61f3db6 in ONE fresh venv — b0 and cand share the venv
and differ ONLY in the launcher pin (b0) vs default import (cand); ONE fresh
interpreter process per (arm, block) run (condition 1).

CLONE-ROT vs campaigns/una_large_e2e/evidence/F1R/f1r_run_pairs.py (883
lines), stated per the P3 requirement:
  KEPT: SCHEDULE shape and S3 no-partial-resume guard; per-window memory
    admission + pressure stop; supervisor-side live watchdog (0.25 s ps tree
    samples, SIGTERM -> 10 s grace -> SIGKILL); per-block byte-identity with
    flow_<ts> normalization; engine_config three-field check; lease
    charge-at-append (charged_s written ONLY in the append write; ledger
    recompute only there); descriptive-only summary (h04 owns disposition).
  CHANGED: paths -> evidence/F2R (repo-root vehicle home) + f2r_* names;
    ARMS collapse to ONE shared wheel/venv (central ruling) — arms differ
    only in cache root and the launcher pin; watchdog budget 3304 ->
    3,436 MiB (spec envelope: 1.10 x F3R realized 3,123.203125, ceil);
    admission bar 0.80*avail < budget 3,602,907,136 B; matrix-entry bar
    avail < 4,556,062,720 B (entry 4,295.0 MiB + 50 MiB margin); lease log
    f2r (pre-existing, chained — REFUSES to auto-create; opening chain
    verified against the spec constants); launcher pin: b0 runs get
    --launcher-path/--launcher-sha256, supervisor refuses on launcher-byte
    drift from LAUNCHER_SHA256; run ids <RUN_PREFIX>_b<N>_<ORDER>_<arm>
    via --run-id-prefix (default "f2r" = attempt-1 ids; a fresh attempt
    passes a fresh prefix so landed records are never rewritten); 100x
    worst-case-with-one-restart headroom check at matrix entry.
  ADDED (F2R obligations absent from the skeleton):
    - REV-9 quiet-window gate EVERY admission: available >= 5,368,709,120 B
      AND loadavg[0] < 8.0; bounded foreground poll rounds (6 x 30 s), ONE
      firer per run id, no detached waiters; readings recorded per entry;
      cap exhaust = not_fired uncharged entry + STOP.
    - G2 WARNING PARITY enforced in the block comparator (equal verbatim
      warning counts per block per arm; both lists recorded).
    - KERNEL-SET CENSUS (G4 comparability, verdict item c): per-block
      cross-arm cache-root diff — cand_only entries must ALL match the
      local-kernel family (_accumulate_od_flow_local); ANY b0-only entry on
      a later block, or any per-arm size jump across blocks, = packaging-
      confound signature -> STOP before the next gate window.
    - FORWARD LEASE SCHEMA: class gate_window / crashed_child_start with
      EXPLICIT NULL PAIR (window_at_end_utc + rev9_admitted_end keys
      PRESENT, null) for crashed/killed children; charge rule per spec.
    - CONSOLE TRANSCRIPT persisted to campaign_data/f2r_console/<run_id>.log
      in the same call as the fire.
    - STAGE GUARD: failed run's out_root AND cache root renamed aside
      (_incident suffix) before returning — never a bare re-fire.
  REMOVED: the F1R canary stage (superseded: F2R mechanism priming is the
    diagnostics legs under f2r_lease_pass.py, diagnostics_class, throwaway
    roots, never gate records); the bootstrap CI block (superseded by the
    REV 4 statistics discipline: warm-pair admission CI, 1 dof).

Failure policy: on failure/timeout/watchdog breach/identity delta preserve
all logs, rename aside, STOP — no retry beyond the schedule without h04's
approval. Refuse-on-partial-resume (restart-whole). Nothing fires before P4b.
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
EV = f"{REPO}/evidence/F2R"
DATA = f"{CAMPAIGN}/campaign_data"
RAW = f"{EV}/raw"
SUMMARY = f"{DATA}/f2r_screen_summary.json"
LEASE_LOG = f"{DATA}/f2r_lease_log.json"
CONSOLE_DIR = f"{DATA}/f2r_console"
PHYS_BYTES = 17179869184
PRESSURE_STOP = max(1 << 30, int(0.10 * PHYS_BYTES))
WATCHDOG_MIB = 3436                      # ceil(1.10 x 3,123.203125) (spec envelope)
WATCHDOG_BUDGET_BYTES = 3602907136       # == 3,436 MiB
ADMISSION_MIN_AVAIL_BYTES = 4503633920   # entry bar: avail floor where 0.80*avail == budget
MATRIX_ENTRY_MIN_AVAIL_BYTES = 4556062720  # +50 MiB margin bar (spec)
REV9_MIN_AVAIL_BYTES = 5368709120
REV9_MAX_LOADAVG = 8.0
REV9_ROUNDS, REV9_POLL_S = 6, 30         # bounded foreground: ~3 min per call
HEAVY_WALL_BUDGET_S = 14400.0
RUN_TIMEOUT_S = 1200
LEASE_ID = "F2R"
K_STRIPES, H_THREADS = 8, 1
SOURCE_COMMIT = "61f3db64801018aea0c19d27a27c25506ed64e2d"
MANIFEST = f"{REPO}/tests/large_e2e/inputs/O3_FLOW.manifest.json"
VARIANT = "sel1024"
FLOW_TS_RE = re.compile(r"flow_\d{4}-\d{2}-\d{2}_\d{4}")
T_CRIT_1DOF = 12.706204736174704         # two-sided 95%, 1 dof; EXACT closed form tan(19*pi/40) = cot(pi/40) = 12.7062047361747046460...; prior constant 12.706204736432095 was a 2.6e-10 mis-precision (coordinator-caught pre-routing)
T_CRIT_2DOF = 4.302652729749464          # two-sided 95%, 2 dof (descriptive companion); EXACT closed form sqrt(722/39) = 4.3026527297494638523...; prior constant 4.302652729911275 was a 1.6e-10 mis-precision (h04-caught, same class as the df=1 fix)
G3_MINIMUM_DELTA = 0.02738813230584978   # preregistered (spec gates.G3)
G4_BOUND = 1.0985                        # derived cold bound (spec gates.G4)
WORST_CASE_ONE_RESTART_S = 94.27973534   # spec budget_arithmetic
HEADROOM_X = 100
LAUNCHER = f"{EV}/f2r_route_launcher.py"
LAUNCHER_SHA256 = "b65867cc731ac6b79025a065d3c95c36f46c35377243b4d68b6217a0e89707b3"
LOCAL_FAMILY_MARKERS = ("_accumulate_od_flow_local",)
Q00_BAND_BASIS_MIB = 2905.265625         # Q00 band denominator (two-bases rule)

WHEEL_VENV = {"python": f"{DATA}/venvs/f2r_wheel/bin/python",
              "site_packages": f"{DATA}/venvs/f2r_wheel/lib/python3.11/site-packages",
              "wheel_record": f"{DATA}/f2r_wheel_and_venv.json"}
ARMS = {
    "b0":   {**WHEEL_VENV, "cache_root": f"{DATA}/nbc_f2r_b0"},
    "cand": {**WHEEL_VENV, "cache_root": f"{DATA}/nbc_f2r_cand"},
}
SCHEDULE = [(1, "AB", "b0"), (1, "AB", "cand"),
            (2, "BA", "cand"), (2, "BA", "b0"),
            (3, "AB", "b0"), (3, "AB", "cand")]

# Run-id root; --run-id-prefix retargets a fresh attempt (restart-whole,
# h04 ruling (c)) so landed attempt-1 records are never rewritten.
RUN_PREFIX = "f2r"


def wheel_record():
    """The P5 record is the authority for BOTH the wheel sha and the ref it
    was archived from. Packaging pin (frozen spec): the wheel MUST come from
    `git archive 61f3db6...`, NEVER fire-time HEAD (P1 advanced HEAD to
    b3f8a15 evidence-only; archiving HEAD would embed the wrong ref even
    though AggregateFlow.py is byte-identical — the module-pin assertion
    alone would not catch it). Lease windows stamp the ACTUAL ref."""
    with open(WHEEL_VENV["wheel_record"]) as f:
        return json.load(f)


def wheel_sha():
    return wheel_record()["wheel"]["sha256"]


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def cache_walk(root):
    """relpath -> (sha256, bytes) for every file under a cache root."""
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
    """Read-only cumulative; ledger-field recomputes happen ONLY in
    lease_append, in the same write that appends the entry."""
    with open(LEASE_LOG) as f:
        log = json.load(f)
    charged = sum(w.get("charged_s", 0.0) for w in log["windows"])
    return round(log["opening_balance_s"] + charged, 1)


def lease_append(entry):
    """SOLE charging mechanism for supervisor-run windows (spec
    entry_schema_forward.charge_rule): charged_s stamped AT APPEND TIME
    (default wall_s; refused / no-window paths write none); ledger fields
    recompute ONLY inside this append write; append-only."""
    if "refused" not in entry and entry.get("wall_s") is not None:
        entry["charged_s"] = entry["wall_s"]
    with open(LEASE_LOG) as f:
        log = json.load(f)
    log["windows"].append(entry)
    charged = sum(w.get("charged_s", 0.0) for w in log["windows"])
    log["cumulative_charged_s"] = round(log["opening_balance_s"] + charged, 1)
    log["remaining_s"] = round(HEAVY_WALL_BUDGET_S - log["cumulative_charged_s"], 1)
    log["updated_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with open(LEASE_LOG, "w") as f:
        json.dump(log, f, indent=1)
    return log["cumulative_charged_s"]


def lease_guard():
    """The lease log pre-exists via f2r_lease_pass.py init (chained opening);
    the supervisor never creates it and refuses without it."""
    if not os.path.exists(LEASE_LOG):
        print(f"[f2r-pairs] REFUSED: lease log missing ({LEASE_LOG}) — run "
              "f2r_lease_pass.py init first (chained opening is an "
              "instrument-owned creation)")
        return None
    with open(LEASE_LOG) as f:
        log = json.load(f)
    if abs(log.get("opening_balance_s", -1) - 1182.3) > 1e-9:
        print("[f2r-pairs] REFUSED: lease opening_balance_s != 1182.3 (spec "
              "chain constant)")
        return None
    if log.get("source_commit") != SOURCE_COMMIT:
        print("[f2r-pairs] REFUSED: lease source_commit != screened commit")
        return None
    return log


def rev9_gate(run_id):
    """REV-9 quiet-window gate, EVERY admission: available_bytes >=
    5,368,709,120 AND loadavg[0] < 8.0. Bounded FOREGROUND poll rounds; fires
    in the same call the gate first reads satisfied; ONE firer per run id
    (the caller guarantees the run id is fresh). Returns (admitted, rec)."""
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
    """Stage guard: rename a failed run's state aside (_incident suffix)."""
    if path and os.path.exists(path):
        aside = f"{path}_incident"
        n = 1
        while os.path.exists(aside):
            n += 1
            aside = f"{path}_incident{n}"
        os.rename(path, aside)
        return os.path.basename(aside)
    return None


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
    out_root = f"{DATA}/f2r_out/{run_id}"
    if os.path.exists(out_root):
        shutil.rmtree(out_root)
    env = {"NUMBA_CACHE_DIR": a["cache_root"],
           "NUMBA_NUM_THREADS": str(H_THREADS)}
    child_env = {**os.environ, **env}
    cmd = [a["python"], f"{EV}/f2r_bare_job.py",
           "--manifest", MANIFEST, "--cell", "O3_FLOW",
           "--variant-id", VARIANT, "--arm", arm,
           "--expected-site-packages", a["site_packages"],
           "--out-json", out_json,
           "--output-root", out_root,
           "--run-id", run_id,
           "--flow-stripes", str(K_STRIPES),
           "--wheel-sha256", wheel_sha(),
           "--numba-threads", str(H_THREADS),
           "--block", str(block), "--order", order]
    if arm == "b0":
        if sha256_file(LAUNCHER) != LAUNCHER_SHA256:
            print("[f2r-pairs] REFUSED: launcher bytes drift from the pinned "
                  "sha (reviewed vehicle; re-review required)")
            return 2, run_id, "REFUSED: launcher sha drift"
        cmd += ["--launcher-path", LAUNCHER, "--launcher-sha256", LAUNCHER_SHA256]
    avail = psutil.virtual_memory().available
    entry = {"what": f"gate_run:{run_id}", "class": "gate_window",
             "lease_id": LEASE_ID, "gate_eligible": True,
             "block": block, "order": order, "arm": arm,
             "git_archive_ref": wheel_record().get("git_archive_ref"),
             "launcher_sha8": (LAUNCHER_SHA256[:8] if arm == "b0" else None),
             "cache_state": cache_fresh,
             "cache_file_count_before": cache_files(a["cache_root"]),
             "start_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
             "available_at_admission_bytes": int(avail),
             "admission_rule": ("refuse when 0.80*available < budget "
                                "3,602,907,136 B (spec envelope)"),
             "admission_080avail_bytes": int(0.80 * avail),
             "pressure_stop_bytes": PRESSURE_STOP,
             "watchdog_budget_mib": WATCHDOG_MIB,
             "supervisor_python": sys.executable,
             "loadavg_at_admission": os.getloadavg(),
             "cmd": cmd, "env_overrides": env}
    cum = lease_cumulative()
    if int(0.80 * avail) < WATCHDOG_BUDGET_BYTES:
        entry["refused"] = ("0.80*available < budget at admission (spec "
                            "envelope); mid-matrix refusal voids the matrix")
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
        entry["gate_eligible"] = False  # nothing fired, no window opened
        entry["window_at_end_utc"] = None
        entry["rev9_admitted_end"] = None
        lease_append(entry)  # no window opened -> charge_rule writes no charged_s
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
        with open(f"{DATA}/f2r_watchdog_breach_{run_id}.json", "w") as f:
            json.dump(breach, f, indent=1)
    if rc != 0 and (rc < 0 or timed_out):
        entry["class"] = "crashed_child_start"  # explicit NULL PAIR, keys present
        entry["gate_eligible"] = False  # a crashed window is not gate evidence
        entry["window_at_end_utc"] = None
        entry["rev9_admitted_end"] = None
        entry["null_pair_note"] = ("crashed/killed child: window_at_end_utc and "
                                   "rev9_admitted_end are EXPLICIT NULLS, keys "
                                   "present (forward schema); wall still charged")
    cumulative = lease_append(entry)
    print(f"[f2r-pairs] {run_id} rc={rc} wall={wall:.1f}s "
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
        out[FLOW_TS_RE.sub("flow_<TS>", rel)] = (meta["sha256"], meta["bytes"])
    return out


def array_hashes(rec):
    return {k: (v or {}).get("sha256_bytes") if isinstance(v, dict) else v
            for k, v in (rec.get("flow_arrays") or {}).items()}


def _fixture_content(path):
    """Canonical cross-arm fixture content (type + crs + features), EXCLUDING
    the layer "name" that embeds each arm's run_id (F1R rev-5B lesson)."""
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            d = json.load(f)
    except (OSError, ValueError):
        return None
    return json.dumps({k: d.get(k) for k in ("type", "crs", "features")},
                      sort_keys=True)


def kernel_census_check(prev_census, census, arm, block):
    """Per-arm across-block cache check: warm blocks add NOTHING (any new
    entry = red flag) and NEVER rewrite an existing cache file — common keys
    are compared on (sha256, bytes); a same-name content/size change =
    recompile = packaging-confound signature (spec mandatory census
    size-jump leg). Returns list of failures."""
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
    """Cand-only entries must ALL be the local-kernel family; b0-only entries
    are RECORDED (cand missing something b0 compiled would be a comparability
    signal) but the gated family rule governs cand_only only."""
    fails = []
    cand_only = sorted(set(census_cand) - set(census_b0))
    b0_only = sorted(set(census_b0) - set(census_cand))
    unexpected = [n for n in cand_only
                  if not any(m in n for m in LOCAL_FAMILY_MARKERS)]
    if unexpected:
        fails.append({"mode": "UNPREDICTED_extra_cache_entry", "block": block,
                      "unexpected_cand_only_entries": unexpected[:20]})
    return fails, cand_only, b0_only


def block_checks(rows):
    """engine_config + G1 byte identity + G2 warning parity + fixture content
    per completed block. Returns (findings, track_ok)."""
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
            if not (r.get("module_pin") or {}).get("match", False):
                failures.append({"mode": "module_pin_mismatch", "arm": arm})
            if not (r.get("f2_log_line") or {}).get("present", False):
                failures.append({"mode": "f2_log_line_missing", "arm": arm})
        ra_b0, ra_c = array_hashes(rb), array_hashes(rc)
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
        wb, wc = rb.get("verbatim_warnings") or [], rc.get("verbatim_warnings") or []
        if len(wb) != len(wc):
            failures.append({"mode": "G2_WARNING_PARITY_COUNT",
                             "b0": len(wb), "cand": len(wc)})
        fx_b0 = ((rb.get("origin_selection") or {}).get("fixture") or {})
        fx_c = ((rc.get("origin_selection") or {}).get("fixture") or {})
        fx_content_b0 = _fixture_content(fx_b0.get("path")
                                         if isinstance(fx_b0, dict) else None)
        fx_content_c = _fixture_content(fx_c.get("path")
                                        if isinstance(fx_c, dict) else None)
        if not (fx_content_b0 is not None and fx_content_b0 == fx_content_c):
            failures.append({"mode": "fixture_content_mismatch"})
        blk = {
            "block": bnum, "order": rb["order"],
            "engine_config_observed": {"b0": rb.get("engine_config_observed"),
                                       "cand": rc.get("engine_config_observed")},
            "module_pin": {"b0": (rb.get("module_pin") or {}).get("sha256"),
                           "cand": (rc.get("module_pin") or {}).get("sha256")},
            "route_state": {"b0": rb.get("route_state"),
                            "cand": rc.get("route_state")},
            "array_hashes_equal": not arr_delta,
            "array_delta_keys": arr_delta,
            "fixture_content_equal": fx_content_b0 is not None
                                     and fx_content_b0 == fx_content_c,
            "warning_parity": {"b0_count": len(wb), "cand_count": len(wc),
                               "equal": len(wb) == len(wc),
                               "b0": wb, "cand": wc},
            "artifact_count_normalized": len(art_b0),
            "byte_identity": "IDENTICAL" if not art_delta else "DELTA",
            "artifact_delta_paths": art_delta[:20],
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
        rows.append({
            "run_id": run_id, "block": block, "order": order, "arm": arm,
            "cold_warm": ("cold (fresh per-arm cache)" if block == 1
                          else "warm (reused cache)"),
            "W_K_H": [1, K_STRIPES, H_THREADS],
            "window_s": (round(win_ns / 1e9, 6) if win_ns else None),
            "engine_config_observed": r.get("engine_config_observed"),
            "module_pin": r.get("module_pin"),
            "route_state": r.get("route_state"),
            "f2_log_line": r.get("f2_log_line"),
            "resolved_gravity_cap": r.get("resolved_gravity_cap"),
            "numba_num_threads": ((r.get("process") or {})
                                  .get("numba_get_num_threads")),
            "cache_dir": (r.get("env") or {}).get("NUMBA_CACHE_DIR"),
            "numba_cache_census": r.get("numba_cache_census"),
            "peak_rss_mib": r.get("peak_rss_ru_maxrss_mib"),
            "identity_ok": ident.startswith("ok"),
            "counts": r.get("counts"),
            "origin_selection": r.get("origin_selection"),
            "artifacts": r.get("artifacts"),
            "array_hashes": array_hashes(r),
            "verbatim_warnings": r.get("verbatim_warnings"),
            "fatal": r.get("fatal"),
        })
    findings, track_ok = block_checks(rows)
    # kernel-set census (G4 comparability; verdict item c) — MANDATORY
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
        "cand_only_all_local_family": not census_fails,
        "failures": census_fails,
        "note": ("cand warm cache carries the local-kernel family entries the "
                 "b0 arm lacks; an UNPREDICTED extra entry = packaging-"
                 "confound signature, stop before any gate window"),
    }
    # REV 4 statistics — DESCRIPTIVE ONLY (h04 owns every gate evaluation)
    stats = {"note": "unit = the PAIR (block), never the job; ratios "
                     "b0_wall/cand_wall for speedup; block_set_ledger: "
                     "admission reads ONLY warm blocks 2-3; block 1 feeds "
                     "cold_regression (G4); the blocks 1-3 proxy is "
                     "DESCRIPTIVE ONLY and is NEVER compared against the G3 "
                     "minimum or the noise guard (proxy_prohibition)",
             "gate_quantities_are_data": "boolean comparisons below are "
             "recorded data points; h04-reviewer owns every gate evaluation "
             "and the disposition"}
    walls = {(r["block"], r["arm"]): r["window_s"] for r in rows
             if not r.get("missing") and r.get("window_s")}
    if len(walls) == 6:
        b = [walls[(i, "b0")] for i in (1, 2, 3)]
        c = [walls[(i, "cand")] for i in (1, 2, 3)]
        b_mid, c_mid = (b[1] + b[2]) / 2, (c[1] + c[2]) / 2
        delta_frac = (b_mid - c_mid) / b_mid
        midpoint_delta_s = b_mid - c_mid
        spread_b0, spread_c = abs(b[1] - b[2]), abs(c[1] - c[2])
        r_warm = [walls[(i, "b0")] / walls[(i, "cand")] for i in (2, 3)]
        lr_warm = [math.log(x) for x in r_warm]
        m1 = sum(lr_warm) / 2
        se1 = math.sqrt(sum((x - m1) ** 2 for x in lr_warm) / 1) / math.sqrt(2)
        lr3 = [math.log(walls[(i, "b0")] / walls[(i, "cand")]) for i in (1, 2, 3)]
        m3 = sum(lr3) / 3
        se3 = math.sqrt(sum((x - m3) ** 2 for x in lr3) / 2) / math.sqrt(3)
        q00_per_arm = {}
        for arm in ("b0", "cand"):
            pk = next((r.get("peak_rss_mib") for r in rows
                       if r.get("block") == 1 and r["arm"] == arm
                       and not r.get("missing") and r.get("peak_rss_mib")), None)
            q00_per_arm[arm] = {
                "peak_rss_mib": pk,
                "percent_over_basis": (round((pk / Q00_BAND_BASIS_MIB - 1) * 100, 4)
                                       if pk else None)}
        stats.update({
            "walls_b0_s": b, "walls_cand_s": c,
            "per_block_ratio_r_i_b0_over_cand": [round(x, 6) for x in
                                                 (b[0] / c[0], r_warm[0], r_warm[1])],
            "cold_regression_cand_over_b0_block1": round(c[0] / b[0], 6),
            "g4_band_and_bound": {"band_lo": 1.066, "band_hi": 1.0985,
                                  "frozen_bound": G4_BOUND,
                                  "cold_regression": round(c[0] / b[0], 6)},
            "warm_governing_blocks_2_3": {
                "b0_midpoint_s": round(b_mid, 6),
                "cand_midpoint_s": round(c_mid, 6),
                "midpoint_delta_frac": round(delta_frac, 8),
                "midpoint_delta_frac_note": "feeds ONLY the G3 point test "
                                           "(median_cand < median_b0 x (1 - "
                                           "minimum_delta))",
                "spread_b0_s": round(spread_b0, 6),
                "spread_cand_s": round(spread_c, 6),
                "noise_guard_preregistered": {
                    "form": "gates.G3_noise_guard: measured MIDPOINT delta "
                            "(b0_mid - cand_mid, SECONDS) must EXCEED "
                            "max(|b2-b3|, |c2-c3|) (SECONDS); delta <= spread "
                            "-> not_demonstrated regardless of the point test",
                    "midpoint_delta_s": round(midpoint_delta_s, 6),
                    "max_spread_s": round(max(spread_b0, spread_c), 6),
                    "delta_exceeds_max_spread":
                        midpoint_delta_s > max(spread_b0, spread_c)},
                "minimum_delta_preregistered": G3_MINIMUM_DELTA,
                "delta_meets_minimum": delta_frac >= G3_MINIMUM_DELTA,
                "point_test_data": c_mid < b_mid * (1 - G3_MINIMUM_DELTA)},
            "admission_ci_warm_pair_1dof": {
                "log_ratio_diffs_ln": lr_warm, "mean_ln": round(m1, 6),
                "se": round(se1, 6), "t_crit": T_CRIT_1DOF,
                "ci_ln": [round(m1 - T_CRIT_1DOF * se1, 6),
                          round(m1 + T_CRIT_1DOF * se1, 6)],
                "ci_speedup": [round(math.exp(m1 - T_CRIT_1DOF * se1), 6),
                               round(math.exp(m1 + T_CRIT_1DOF * se1), 6)],
                "note": "t-interval on the TWO paired warm log-ratio "
                        "differences (blocks 2-3, 1 dof); very limited "
                        "small-sample precision"},
            "descriptive_includes_cold": {
                "proxy_speedup_median_b0_over_cand_blocks_1_3":
                    round(sorted(b)[1] / sorted(c)[1], 6),
                "proxy_status": "DESCRIPTIVE ONLY — never an admission input; "
                                "no comparison against the G3 minimum or the "
                                "noise guard is computed or permitted "
                                "(proxy_prohibition)",
                "t_interval_3pair_2dof": {
                    "mean_ln": round(m3, 6), "se": round(se3, 6),
                    "t_crit": T_CRIT_2DOF,
                    "ci_speedup": [round(math.exp(m3 - T_CRIT_2DOF * se3), 6),
                                   round(math.exp(m3 + T_CRIT_2DOF * se3), 6)],
                    "label": "descriptive-includes-cold"}},
            "q00_block1_cold_peaks": {
                "band_basis_mib": Q00_BAND_BASIS_MIB,
                "per_arm": q00_per_arm,
                "preregistered_band_percent": [7.5, 7.7],
                "note": "band denominator = the F3R SIZING GOVERNING basis "
                        "(two-bases rule); watchdog stands on the REALIZED "
                        "basis 3,123.203125 MiB"},
        })
    else:
        stats["incomplete"] = f"expected 6 runs, have {len(walls)}"
    if os.path.exists(LEASE_LOG):
        with open(LEASE_LOG) as f:
            lease = json.load(f)
        lease_block = {"opening_balance_s": lease["opening_balance_s"],
                       "cumulative_incl_opening_s": lease["cumulative_charged_s"],
                       "budget_s": HEAVY_WALL_BUDGET_S}
    else:
        lease_block = {"note": f"lease log not present at summary time "
                               f"({LEASE_LOG})", "budget_s": HEAVY_WALL_BUDGET_S}
    out = {
        "task": "F2R", "record": "screen_summary", "role": "screen-executor",
        "spec_source": "evidence/F2R/screen_spec.json (REV 6 bytes b954cf8c, "
                       "commit ec97f34)",
        "descriptive_only": "h04-reviewer owns disposition/verdicts; this "
                            "summary computes quantities and records data "
                            "points only",
        "schedule": "3 blocks AB/BA/AB (A=b0); block 1 cold on fresh per-arm "
                    "cache roots, blocks 2-3 warm; ONE wheel/venv, ONE fresh "
                    "interpreter per run",
        "frozen_triple": {"W": 1, "K": K_STRIPES, "H": H_THREADS},
        "arms": {k: {"venv_python": v["python"],
                     "site_packages": v["site_packages"],
                     "cache_root": v["cache_root"]}
                 for k, v in ARMS.items()},
        "launcher": {"path": LAUNCHER, "sha256": LAUNCHER_SHA256,
                     "sha8_note": "sha8 rides every lease window entry"},
        "candidate_commit": SOURCE_COMMIT,
        "workload": {"cell": "O3_FLOW", "variant": VARIANT, "manifest": MANIFEST},
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
        print(f"[f2r-pairs] R2 custody: prior summary "
              f"{prev_sha[:8]} -> {interim}")
    with open(SUMMARY, "w") as f:
        json.dump(out, f, indent=1)
    print(f"[f2r-pairs] summary -> {SUMMARY}")
    print(f"  track_conformant={track_ok} runs_present={out['runs_present']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["run", "summary"], default="run")
    ap.add_argument("--run-id-prefix", default="f2r")
    a = ap.parse_args()
    global RUN_PREFIX
    RUN_PREFIX = a.run_id_prefix
    if os.path.realpath(sys.prefix) != os.path.realpath(CAMPAIGN_VENV):
        print(f"[f2r-pairs] REFUSED: supervisor interpreter is not "
              f"the campaign venv: sys.executable {sys.executable}, "
              f"sys.prefix {sys.prefix} != pinned {CAMPAIGN_VENV} "
              "(spec env; campaign venv per attempt-1 precedent - the "
              "window[8]-class deviation refuses)")
        return 2
    if a.stage == "run" and a.run_id_prefix == "f2r":
        print("[f2r-pairs] REFUSED: --stage run requires an explicit fresh "
              "--run-id-prefix (attempt-1 prefix 'f2r' is landed+committed; "
              "a default-prefix run would overwrite landed records)")
        return 2
    if a.stage == "summary":
        summary()
        return 0
    if lease_guard() is None:
        return 2
    if sha256_file(LAUNCHER) != LAUNCHER_SHA256:
        print("[f2r-pairs] REFUSED: launcher bytes drift from the pinned sha "
              "(reviewed vehicle; re-review required)")
        return 2
    for arm in ARMS.values():
        if not (os.path.exists(arm["python"]) and os.path.exists(arm["wheel_record"])
                and os.path.exists(arm["site_packages"])):
            print(f"[f2r-pairs] REFUSED: shared wheel/venv record missing: {arm}")
            return 2
    arch_ref = wheel_record().get("git_archive_ref")
    if arch_ref != SOURCE_COMMIT:
        print(f"[f2r-pairs] REFUSED: wheel record git_archive_ref ({arch_ref}) "
              f"!= pinned source commit {SOURCE_COMMIT} — packaging pin "
              "violated (wheel must be archived from 61f3db6, never "
              "fire-time HEAD)")
        return 2
    os.makedirs(RAW, exist_ok=True)
    os.makedirs(CONSOLE_DIR, exist_ok=True)
    stop_reason = None
    run_ids = [f"{RUN_PREFIX}_b{b}_{o}_{a}" for b, o, a in SCHEDULE]
    present = [rid for rid in run_ids if os.path.exists(f"{RAW}/{rid}.json")]
    if len(present) == 6:
        print("[f2r-pairs] all 6 raw records present; no execution — use "
              "--stage summary to regenerate the summary")
        summary()
        return 0
    if present:
        print(f"[f2r-pairs] REFUSED: partial matrix present ({len(present)}/6: "
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
                 "entry_rule": ("available >= 4,556,062,720 B at matrix entry "
                                "and any post-void re-entry (spec envelope: "
                                "entry 4,295.0 MiB + 50 MiB margin)"),
                 "headroom_rule": (f"worst-case-with-one-restart "
                                   f"{WORST_CASE_ONE_RESTART_S}s x {HEADROOM_X} "
                                   "<= remaining"),
                 "remaining_s": round(HEAVY_WALL_BUDGET_S - cum_entry, 1),
                 "headroom_ok": headroom_ok}
    if avail_entry < MATRIX_ENTRY_MIN_AVAIL_BYTES:
        entry_rec["refused"] = ("matrix ENTRY bar not met; no window opened, "
                                "uncharged")
        lease_append(entry_rec)
        print(f"[f2r-pairs] REFUSED: matrix ENTRY bar: available {avail_entry} "
              f"B < {MATRIX_ENTRY_MIN_AVAIL_BYTES} B; re-invoke when met")
        return 2
    if not headroom_ok:
        entry_rec["refused"] = "100x headroom rule not met against remaining"
        lease_append(entry_rec)
        print("[f2r-pairs] REFUSED: 100x worst-case headroom rule")
        return 2
    lease_append(entry_rec)
    prev_census = {"b0": None, "cand": None}
    for block, order, arm in SCHEDULE:
        rc, run_id, msg = run_slot(block, order, arm)
        print(f"[f2r-pairs] slot {run_id}: {msg}", flush=True)
        if rc != 0:
            stop_reason = msg
            print("[f2r-pairs] STOPPING screen per failure policy "
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
            # per-arm warm-cache monotonicity + cross-arm family check
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
                print(f"[f2r-pairs] {stop_reason}")
                summary(refusal=None, stop_reason=stop_reason)
                return 3
    summary()
    return 0


if __name__ == "__main__":
    sys.exit(main())
