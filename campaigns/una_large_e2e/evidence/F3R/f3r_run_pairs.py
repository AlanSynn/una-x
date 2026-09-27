"""F3R paired-screen supervisor — screen_spec executed (P1-P7 gated).

Schedule: 3 paired complete-job blocks, arm order AB, BA, AB (A = b0 control).
Block 1 runs on FRESH per-arm NUMBA_CACHE_DIRs (wiped at block-1 start ->
symmetric cold first-compile); blocks 2-3 reuse them (warm). Frozen triple
(W,K,H) = (1,8,1): W=1, K=8 (--flow-stripes -> una.topology.num_threads,
wired by the driver pre-window), H=1 (NUMBA_NUM_THREADS).

F3R deltas vs the f1r_run_pairs.py skeleton:
  - STAGES: sizing (P6 diagnostics-class UNCHARGED cold sel1024 run per arm
    under a PROVISIONAL watchdog), envelope (derive the fresh P6 constants
    from the sizing peaks), run (the 3-block matrix; REQUIRES the envelope
    file - P6 unmet = refuse; a PROVISIONAL envelope is ALSO refused — the
    provisional guards sizing runs only, pre-ruling (a)), summary. Sizing
    is sequenced AFTER the spec-commit + P4b chain (h04 sizing-routing
    ruling: sizing records will cite the committed spec-of-record sha, and
    the F3R closeout manifest v4 reconciles them EXPLICITLY alongside raw
    matrix records since they post-date the v3 filing). No sel256 canary
    stage: the F3R spec has none; the sizing run at the real workload
    subsumes the F1R canary's mechanism-liveness role.
  - CACHE ROOTS (h04's root condition, option (i)): sizing compiles into
    DISJOINT per-arm roots nbc_f3r_<arm>_sizing (created + wiped by the
    sizing stage); the matrix roots nbc_f3r_<arm> are created + wiped only
    by the run stage at block-1 start, so block 1's cold freshness is
    structural — sizing cannot warm the matrix roots (sizing on shared
    roots would populate them and silently corrupt the cold-block
    admission/watchdog accounting).
  - ENVELOPE NOT INHERITED (spec P6): F1R's 3304/3354/3534/4632084480 sized
    F1's arms. watchdog = ceil(1.10 x governing peak); per-window refusal
    when 0.80*available < watchdog + 50 MiB; matrix-ENTRY bar avail >=
    (watchdog + 230 MiB)/0.80. The sizing runs themselves are guarded by
    PROVISIONAL_WATCHDOG_MIB = 3304, justified in-file: block 1 is
    compile-dominated (JIT transient ~2.9-3.1 GiB +/-130 MiB band) and the
    F3 engine delta cannot raise the compile transient (b0's engine IS the
    F1R-cand engine minus nothing at compile time; cand only reduces the
    RUNTIME transient), so the F1R envelope is a conservative provisional
    for the sizing measurements that REPLACE it; every sizing record marks
    the watchdog PROVISIONAL and names its replacement semantics.
  - LEASE: f3r_lease_log.json created on first need with opening_balance_s =
    f1r_lease_log.json's cumulative_charged_s READ AT CREATION (never
    hardcoded), with the source log path + sha recorded. P6 sizing entries
    carry NO wall_s (uncharged by ruling; wall time lives in the sizing
    record). Matrix windows charge via lease_append (sole charging vehicle).
  - G7 (PD-E): driver exit 3 = fallback!=none / receipt mismatch STOP-AND-
    PRESERVE - the run's record is complete (the driver guarantees this);
    the supervisor stops the matrix immediately, no retry, and marks the
    stop reason G7 (distinct message from a watchdog breach — load-bearing
    for adjudication, pre-ruling (e)). Watchdog breach also stops (rc 3
    path preserved). G7's operative referent for 'predicted range' is the
    driver's re-simulation output (spec rev 6).
  - PRE-MATRIX REFUSALS: run stage requires evidence/F3R/screen_spec.json
    (P1 committed) and campaign_data/f3r_envelope.json (P6, non-provisional);
    S3 partial-matrix refusal inherited (a voided matrix never resumes).
  - SUMMARY: f3_analysis computes the ruled statistics descriptively (h04
    owns disposition): per-arm WARM GOVERNING peak = max(ru_maxrss over
    blocks 2-3); measured warm delta vs the PREREGISTERED minimum
    549,262,578 B = 0.5 x (b0_accounted 1,299,851,748 [chunk_b0 x V' x 13,
    the SAME measured V' in both factors, recomputed from the records and
    ASSERTED equal to the pinned value — pre-ruling (f)] - cand_accounted
    worst case 192 MiB); the observed-accounted variant recorded
    descriptively; G4 single-pair cold ratio cand/b0 with the ruled 5%
    bound (one cold block per arm - the 'median' is the single pair);
    cache-entry censuses per arm per block (kernel-set identity receipt
    context). ALL capacity_only (DECISION line 11): no throughput claim
    anywhere.

Failure policy: on failure/timeout/watchdog breach/identity delta/G7 preserve
all logs and STOP - no retry beyond the schedule without h04's approval.

Raw records: evidence/F3R/raw/f3r_b<N>_<ORDER>_<arm>.json (gate);
campaign_data/f3r_sizing_raw_<arm>.json (P6 diagnostics, uncharged).
Summary (descriptive only, h04 owns disposition):
campaign_data/f3r_screen_summary.json
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
EV = f"{REPO}/campaigns/una_large_e2e/evidence/F3R"
SPEC = f"{EV}/screen_spec.json"
DATA = f"{CAMPAIGN}/campaign_data"
RAW = f"{EV}/raw"
SUMMARY = f"{DATA}/f3r_screen_summary.json"
SIZING_RAW = {arm: f"{DATA}/f3r_sizing_raw_{arm}.json" for arm in ("b0", "cand")}
ENVELOPE = f"{DATA}/f3r_envelope.json"
LEASE_LOG = f"{DATA}/f3r_lease_log.json"
PRIOR_LEASE_LOG = f"{DATA}/f1r_lease_log.json"
PHYS_BYTES = 17179869184
PRESSURE_STOP = max(1 << 30, int(0.10 * PHYS_BYTES))  # 1717986918
PROVISIONAL_WATCHDOG_MIB = 3304  # sizing-stage guard ONLY (justified above)
RUN_TIMEOUT_S = 1200
HEAVY_WALL_BUDGET_S = 14400.0
LEASE_ID = "F3R"
K_STRIPES, H_THREADS = 8, 1
B0_COMMIT = "c37cf224afe678e0500749531a43c6be181dabef"
CAND_COMMIT = "b387f901caa1b529138452b084d016b9b1bead50"
MANIFEST = f"{REPO}/tests/large_e2e/inputs/O3_FLOW.manifest.json"
VARIANT = "sel1024"
FLOW_TS_RE = re.compile(r"flow_\d{4}-\d{2}-\d{2}_\d{4}")
G3_PREREGISTERED_MIN_DELTA_BYTES = 549262578  # spec PD-A: 0.5 x accounted reduction
G3_B0_ACCOUNTED_BYTES = 1299851748            # pinned formula value at V'=56844
G3_CAND_WORST_CASE_PAYLOAD_MIB = 192.0        # cap 256 MiB - margin 64 MiB
G4_COLD_RATIO_BOUND = 1.05                    # PD-C single-pair 5% bound

ARMS = {
    "b0": {"python": f"{DATA}/venvs/f3r_b0_wheel/bin/python",
           "site_packages": f"{DATA}/venvs/f3r_b0_wheel/lib/python3.11/site-packages",
           "cache_root": f"{DATA}/nbc_f3r_b0",
           "sizing_cache_root": f"{DATA}/nbc_f3r_b0_sizing",
           "commit": B0_COMMIT},
    "cand": {"python": f"{DATA}/venvs/f3r_cand_wheel/bin/python",
             "site_packages": f"{DATA}/venvs/f3r_cand_wheel/lib/python3.11/site-packages",
             "cache_root": f"{DATA}/nbc_f3r_cand",
             "sizing_cache_root": f"{DATA}/nbc_f3r_cand_sizing",
             "commit": CAND_COMMIT},
}
WHEEL_RECORD = f"{DATA}/f3r_wheel_and_venv.json"
# (block, order, arm) — AB/BA alternation, A = b0 runs FIRST in AB
SCHEDULE = [(1, "AB", "b0"), (1, "AB", "cand"),
            (2, "BA", "cand"), (2, "BA", "b0"),
            (3, "AB", "b0"), (3, "AB", "cand")]


def sha256_file(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def wheel_sha(arm):
    with open(WHEEL_RECORD) as f:
        rec = json.load(f)
    return rec["arms"][arm]["wheel"]["sha256"]  # f3r_wheel_and_venv schema


def cache_files(root):
    n = 0
    for _, _, files in os.walk(root):
        n += len(files)
    return n


def load_envelope(required):
    """P6 envelope constants. required=True (matrix) refuses when absent."""
    if not os.path.exists(ENVELOPE):
        if required:
            return None
        return {"watchdog_budget_mib": PROVISIONAL_WATCHDOG_MIB,
                "provisional": True,
                "provisional_justification": PROVISIONAL_JUSTIFICATION}
    with open(ENVELOPE) as f:
        env = json.load(f)
    env.setdefault("provisional", False)
    return env


PROVISIONAL_JUSTIFICATION = (
    "PROVISIONAL sizing-stage guard = the F1R envelope watchdog 3304 MiB: "
    "block 1 is compile-dominated (JIT transient ~2.9-3.1 GiB, +/-130 MiB "
    "band) and the F3 engine delta cannot raise the compile transient (cand "
    "only reduces the RUNTIME transient), so the F1R envelope is a "
    "conservative provisional for the P6 sizing measurements that REPLACE "
    "it; the matrix refuses to enter without the derived f3r_envelope.json")


def ensure_lease_log():
    """Create f3r_lease_log.json on first need: opening = f1r cumulative
    READ AT CREATION, with source provenance. Never hardcodes the figure."""
    if os.path.exists(LEASE_LOG):
        return
    with open(PRIOR_LEASE_LOG) as f:
        prior = json.load(f)
    opening = prior["cumulative_charged_s"]
    log = {
        "task": "F3R", "lease_id": LEASE_ID,
        "opening_balance_s": opening,
        "opening_source_log": PRIOR_LEASE_LOG,
        "opening_source_log_sha256_at_creation": sha256_file(PRIOR_LEASE_LOG),
        "opening_source_cumulative_s": opening,
        "budget_s": HEAVY_WALL_BUDGET_S,
        "windows": [],
        "cumulative_charged_s": opening,
        "remaining_s": round(HEAVY_WALL_BUDGET_S - opening, 1),
        "updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with open(LEASE_LOG, "w") as f:
        json.dump(log, f, indent=1)


def lease_append(entry):
    # Sole charging mechanism for supervisor-run windows (ruling carried
    # from F1R rev 5C): charged_s written AT APPEND TIME whenever wall_s is
    # present and the entry is not a refusal. P6 sizing entries deliberately
    # carry NO wall_s (uncharged diagnostics class; wall time lives in the
    # sizing record). Ledger-field recomputes happen ONLY here.
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


def launch_and_watch(cmd, child_env, entry, stop_evt, breach):
    """Shared child launch + live watchdog (SIGTERM -> 10 s -> SIGKILL).
    Returns (rc, tail, wall_s). Watchdog budget comes from entry's
    watchdog_budget_mib (envelope-driven, not a module constant). The caller
    stamps admission fields + cmd/env on the entry BEFORE calling this."""
    budget_mib = entry["watchdog_budget_mib"]
    t0 = time.monotonic()
    p = subprocess.Popen(cmd, env=child_env, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True)
    breach["budget_mib"] = budget_mib
    stop_evt_local = stop_evt

    def watchdog():
        while not stop_evt_local.is_set() and p.poll() is None:
            try:
                kb = tree_rss_kb(p.pid, ps_snapshot())
            except Exception:
                kb = -1
            if kb > 0:
                breach["samples"].append(
                    {"t_s": round(time.monotonic() - t0, 2), "tree_rss_kb": kb})
                if kb > budget_mib * 1024:
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
                        stop_evt_local.wait(0.25)
                    try:
                        p.send_signal(signal.SIGKILL)
                        breach["action"] += " -> SIGKILL after 10 s grace"
                    except ProcessLookupError:
                        pass
                    return
            stop_evt_local.wait(0.25)

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
    stop_evt_local.set()
    th.join(timeout=2)
    wall = time.monotonic() - t0
    entry["watchdog_samples"] = len(breach["samples"])
    entry["watchdog_peak_tree_kb"] = max(
        (s["tree_rss_kb"] for s in breach["samples"]), default=0)
    entry["loadavg_at_completion"] = os.getloadavg()
    return rc, tail, wall


def admission_refusal(entry, min_080avail_bytes, cum):
    """Shared pre-launch guards; True (and entry marked refused) on refusal.
    Stamps the memory-admission fields on the entry either way, so a refused
    entry carries the same admission evidence as a launched one."""
    import psutil
    avail = psutil.virtual_memory().available
    entry.update({"available_at_admission_bytes": avail,
                  "admission_080avail_bytes": int(0.80 * avail),
                  "admission_min_080avail_bytes": min_080avail_bytes,
                  "ceiling_bytes": min(10650 * (1 << 20), int(0.80 * avail)),
                  "pressure_stop_bytes": PRESSURE_STOP,
                  "supervisor_python": sys.executable,
                  "loadavg_at_admission": os.getloadavg()})
    if int(0.80 * avail) < min_080avail_bytes:
        entry["refused"] = (f"0.80*available < {min_080avail_bytes} B at "
                            "admission; mid-matrix refusal voids the partial "
                            "matrix")
        lease_append(entry)
        return True
    if avail < PRESSURE_STOP:
        entry["refused"] = "available < pressure stop before launch"
        lease_append(entry)
        return True
    if cum >= HEAVY_WALL_BUDGET_S:
        entry["refused"] = "heavy wall exhausted"
        lease_append(entry)
        return True
    return False


def sizing_slot(arm):
    """P6 diagnostics-class UNCHARGED cold sel1024 run per arm. Watchdog =
    provisional 3304 MiB (justified in-file); admission guards ACTIVE; the
    record lands in campaign_data (never evidence/raw - it is not a gate
    record); the lease entry carries NO wall_s (uncharged by ruling).
    Cache roots: the sizing run compiles into its own DISJOINT roots
    (nbc_f3r_<arm>_sizing) — the matrix roots nbc_f3r_<arm> are never
    touched by sizing, so matrix block-1 freshness stays structural
    (h04's root condition, option (i))."""
    a = ARMS[arm]
    run_id = f"f3r_sizing_{arm}"
    out_json = SIZING_RAW[arm]
    out_root = f"{DATA}/f3r_sizing_out/{run_id}"
    # S2 (h04 amendment round): a FAILED prior sizing (rc != 0) retains its
    # raw record + supervisor.log with NO derived sizing record, so the
    # stage guard passes on an approved re-fire and the SAME run_id would
    # overwrite retained failure evidence - the item-16 masked-evidence
    # class. Rename any prior sizing evidence aside, _incident-suffixed,
    # before any deletion or overwrite (item-13 retain-and-reference); the
    # rename list rides the lease entry and the sizing record.
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    renamed_aside = []
    for prior in (out_json, f"{out_json}.supervisor.log", out_root):
        if os.path.exists(prior):
            aside = f"{prior}.incident_{stamp}"
            n = 0
            while os.path.exists(aside):
                n += 1
                aside = f"{prior}.incident_{stamp}_{n}"
            os.rename(prior, aside)
            renamed_aside.append(aside)
    if os.path.exists(out_root):
        shutil.rmtree(out_root)
    envl = load_envelope(required=False)
    child_env = {**os.environ,
                 "NUMBA_CACHE_DIR": a["sizing_cache_root"],
                 "NUMBA_NUM_THREADS": str(H_THREADS)}
    child_env.pop("L1_REUSE_DIR", None)
    cmd = [a["python"], f"{EV}/f3r_bare_job.py",
           "--manifest", MANIFEST, "--cell", "O3_FLOW",
           "--variant-id", VARIANT, "--arm", arm,
           "--expected-site-packages", a["site_packages"],
           "--out-json", out_json,
           "--output-root", out_root,
           "--run-id", run_id,
           "--flow-stripes", str(K_STRIPES),
           "--wheel-sha256", wheel_sha(arm),
           "--numba-threads", str(H_THREADS),
           "--block", "0", "--order", "CANARY"]
    entry = {"what": f"diagnostic_run:{run_id}", "diagnostics_class": "p6_sizing",
             "gate_eligible": False, "uncharged": "spec P6: sizing is "
             "diagnostics class and UNCHARGED (no wall_s on this entry; wall "
             "in the sizing record)", "lease_id": LEASE_ID, "arm": arm,
             "cache_state": "sizing_root_wiped_fresh_cold (DISJOINT from the "
                            "matrix roots nbc_f3r_b0/nbc_f3r_cand — sizing "
                            "compiles into its own roots only; matrix block-1 "
                            "freshness stays structural)",
             "start_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
             "prior_evidence_renamed_aside": renamed_aside,
             "watchdog_budget_mib": envl["watchdog_budget_mib"],
             "watchdog_provisional": envl.get("provisional", False),
             "admission_rule": "shared guards (0.80*avail vs provisional "
                               "watchdog+50 MiB; pressure stop; heavy wall)",
             "cmd": cmd,
             "env_overrides": {"NUMBA_CACHE_DIR": child_env["NUMBA_CACHE_DIR"],
                               "NUMBA_NUM_THREADS": child_env["NUMBA_NUM_THREADS"]},
             "child_NUMBA_DISABLE_JIT_set": "NUMBA_DISABLE_JIT" in child_env}
    if admission_refusal(entry, (envl["watchdog_budget_mib"] + 50) * (1 << 20),
                         0.0):
        return 2, run_id, f"REFUSED (sizing): {entry['refused']}"
    if os.path.exists(a["sizing_cache_root"]):
        shutil.rmtree(a["sizing_cache_root"])
    os.makedirs(a["sizing_cache_root"])
    breach = {"run_id": run_id, "samples": []}
    rc, tail, wall = launch_and_watch(cmd, child_env, entry,
                                      threading.Event(), breach)
    entry.update({"end_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                  "rc": rc, "tail": tail,
                  "cache_file_count_after": cache_files(a["sizing_cache_root"])})
    if breach.get("breached"):
        entry["watchdog_breach"] = breach
        with open(f"{DATA}/f3r_watchdog_breach_{run_id}.json", "w") as f:
            json.dump(breach, f, indent=1)
    lease_append(entry)  # no wall_s on purpose -> uncharged
    if rc != 0:
        with open(f"{out_json}.supervisor.log", "w") as f:
            json.dump({"run_id": run_id, "rc": rc, "tail": tail,
                       "watchdog": breach}, f, indent=1)
        return rc, run_id, f"FAILED rc={rc} (sizing stopped, logs preserved)"
    rec = json.load(open(out_json))
    sizing = {
        "task": "F3R", "record": "p6_sizing", "arm": arm, "run_id": run_id,
        "diagnostics_class": "p6_sizing", "gate_eligible": False,
        "uncharged": "spec P6",
        "spec_of_record": {
            "path": SPEC,
            "sha256": sha256_file(SPEC) if os.path.exists(SPEC) else None,
            "binding": "this sizing record is interpretable only against the "
                       "COMMITTED spec of record hashed above (created after "
                       "the v3 manifest filing, so the F3R closeout manifest "
                       "v4 reconciles sizing records EXPLICITLY alongside "
                       "raw matrix records — closeout obligation, h04 "
                       "sizing-routing ruling)",
        },
        "peak_rss_ru_maxrss_mib": rec.get("peak_rss_ru_maxrss_mib"),
        "wall_s": round(wall, 1),
        "watchdog": {"budget_mib": envl["watchdog_budget_mib"],
                     "provisional": envl.get("provisional", False),
                     "provisional_marking": "PROVISIONAL — valid for THIS "
                                            "sizing run only; replacement "
                                            "semantics per spec P6: the "
                                            "matrix budget is "
                                            "ceil(1.10 x governing peak) "
                                            "derived from THIS record pair "
                                            "into f3r_envelope.json, entry "
                                            "bar = refusal + 180 MiB "
                                            "spread; F1R constants are NOT "
                                            "inherited and the run stage "
                                            "REFUSES a provisional "
                                            "envelope",
                     "peak_tree_kb": entry["watchdog_peak_tree_kb"],
                     "breached": bool(breach.get("breached"))},
        "cache_roots": {"sizing_root_used": a["sizing_cache_root"],
                        "matrix_roots_untouched": [ARMS[x]["cache_root"]
                                                   for x in ("b0", "cand")]},
        "engine_shape_measured": rec.get("engine_shape_measured"),
        "mechanism_receipt_inputs_measured": rec.get(
            "mechanism_receipt_inputs_measured"),
        "f3_tail_capture": {k: v for k, v in
                            (rec.get("f3_tail_capture") or {}).items()
                            if k != "parsed"},
        "record_sha256": sha256_file(out_json),
        "prior_evidence_renamed_aside": renamed_aside,
        "capacity_only": "DECISION line 11",
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with open(f"{DATA}/f3r_sizing_{arm}.json", "w") as f:
        json.dump(sizing, f, indent=1)
    print(f"[f3r-pairs] sizing {arm}: peak={sizing['peak_rss_ru_maxrss_mib']} "
          f"MiB wall={wall:.1f}s (uncharged)", flush=True)
    return 0, run_id, "ok"


def envelope_stage():
    """P6 derivation: governing = max(arm peaks); watchdog = ceil(1.10 x g);
    refusal when 0.80*avail < watchdog+50 MiB; entry bar avail >=
    (watchdog+230 MiB)/0.80. Refuses unless BOTH sizing records exist."""
    peaks = {}
    for arm in ("b0", "cand"):
        p = f"{DATA}/f3r_sizing_{arm}.json"
        if not os.path.exists(p):
            print(f"[f3r-envelope] REFUSED: sizing record missing: {p} — run "
                  "--stage sizing first")
            return 2
        rec = json.load(open(p))
        peaks[arm] = rec["peak_rss_ru_maxrss_mib"]
    governing = max(peaks.values())
    watchdog = math.ceil(1.10 * governing)
    admission_min = (watchdog + 50) * (1 << 20)
    entry_min = int(math.ceil((watchdog + 230) * (1 << 20) / 0.80))
    env = {
        "task": "F3R", "record": "p6_envelope",
        "derivation": "spec P6 (F1R rev-5 one-direction recipe): governing = "
                      "max(arm cold sel1024 peaks); watchdog = "
                      "ceil(1.10*governing); per-window refusal when "
                      "0.80*available < watchdog + 50 MiB; matrix-ENTRY bar "
                      "available >= (watchdog + 230 MiB)/0.80",
        "per_arm_cold_peaks_mib": peaks,
        "governing_peak_mib": governing,
        "watchdog_budget_mib": watchdog,
        "admission_min_080avail_bytes": admission_min,
        "matrix_entry_min_avail_bytes": entry_min,
        "basis": {arm: {"record": f"{DATA}/f3r_sizing_{arm}.json",
                        "record_sha256": json.load(
                            open(f"{DATA}/f3r_sizing_{arm}.json"))[
                                "record_sha256"]}
                  for arm in ("b0", "cand")},
        "not_inherited": "F1R constants 3304/3354/3534/4632084480 sized F1's "
                         "arms and are NOT inherited (spec P6)",
        "capacity_only": "DECISION line 11",
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with open(ENVELOPE, "w") as f:
        json.dump(env, f, indent=1)
    print(f"[f3r-envelope] governing={governing} watchdog={watchdog} "
          f"admission_min_080={admission_min} entry_min_avail={entry_min} "
          f"-> {ENVELOPE}")
    return 0


def run_slot(block, order, arm, envl):
    import psutil
    a = ARMS[arm]
    run_id = f"f3r_b{block}_{order}_{arm}"
    out_json = f"{RAW}/{run_id}.json"
    cache_fresh = "created_fresh" if block == 1 else "reused"
    if block == 1 and arm == "b0":  # wipe both roots once, at block-1 start
        for armi in ARMS.values():
            if os.path.exists(armi["cache_root"]):
                shutil.rmtree(armi["cache_root"])
            os.makedirs(armi["cache_root"])
    out_root = f"{DATA}/f3r_out/{run_id}"
    if os.path.exists(out_root):
        shutil.rmtree(out_root)
    budget_mib = envl["watchdog_budget_mib"]
    child_env = {**os.environ,
                 "NUMBA_CACHE_DIR": a["cache_root"],
                 "NUMBA_NUM_THREADS": str(H_THREADS)}
    child_env.pop("L1_REUSE_DIR", None)
    cmd = [a["python"], f"{EV}/f3r_bare_job.py",
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
    entry = {"what": f"gate_run:{run_id}", "lease_id": LEASE_ID,
             "block": block, "order": order, "arm": arm,
             "cache_state": cache_fresh,
             "cache_file_count_before": cache_files(a["cache_root"]),
             "start_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
             "watchdog_budget_mib": budget_mib,
             "envelope_provisional": bool(envl.get("provisional", False)),
             "admission_rule": ("refuse when 0.80*available < "
                                f"{(budget_mib + 50) * (1 << 20)} B "
                                "(watchdog + 50 MiB, spec P6)"),
             "cmd": cmd,
             "env_overrides": {"NUMBA_CACHE_DIR": child_env["NUMBA_CACHE_DIR"],
                               "NUMBA_NUM_THREADS": child_env["NUMBA_NUM_THREADS"]},
             "child_NUMBA_DISABLE_JIT_set": "NUMBA_DISABLE_JIT" in child_env}
    cum = lease_cumulative()
    if admission_refusal(entry, (budget_mib + 50) * (1 << 20), cum):
        return 2, run_id, f"REFUSED: {entry['refused']}"

    breach = {"run_id": run_id, "samples": []}
    rc, tail, wall = launch_and_watch(cmd, child_env, entry,
                                      threading.Event(), breach)
    entry.update({"end_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                  "wall_s": round(wall, 1), "rc": rc, "tail": tail,
                  "cache_file_count_after": cache_files(a["cache_root"])})
    if breach.get("breached"):
        entry["watchdog_breach"] = breach
        with open(f"{DATA}/f3r_watchdog_breach_{run_id}.json", "w") as f:
            json.dump(breach, f, indent=1)
    cumulative = lease_append(entry)
    print(f"[f3r-pairs] {run_id} rc={rc} wall={wall:.1f}s "
          f"cumulative={cumulative}s", flush=True)
    if rc != 0:
        with open(f"{RAW}/{run_id}.supervisor.log", "w") as f:
            json.dump({"run_id": run_id, "rc": rc, "tail": tail,
                       "watchdog": breach}, f, indent=1)
    if breach.get("breached"):
        return 3, run_id, (f"WATCHDOG BREACH ({budget_mib} MiB) — screen "
                           "STOPPED, logs preserved, no retry")
    if rc == 3:
        return 3, run_id, ("G7 STOP-AND-PRESERVE (driver exit 3: fallback!=none "
                           "or binding receipt mismatch) — run COMPLETED, "
                           "record preserved, matrix STOPPED per PD-E")
    return rc, run_id, ("ok" if rc == 0 else f"FAILED rc={rc} (screen STOPPED, "
                                             "logs preserved, no retry)")


def lease_cumulative():
    with open(LEASE_LOG) as f:
        log = json.load(f)
    charged = sum(w.get("charged_s", 0.0) for w in log["windows"])
    return round(log["opening_balance_s"] + charged, 1)


def norm_artifacts(rec):
    out = {}
    for rel, meta in (rec.get("artifacts") or {}).items():
        out[FLOW_TS_RE.sub("flow_<TS>", rel)] = (meta["sha256"], meta["bytes"])
    return out


def array_hashes(rec):
    return {k: (v or {}).get("sha256_bytes") if isinstance(v, dict) else v
            for k, v in (rec.get("flow_arrays") or {}).items()}


def _fixture_content(path):
    """Canonical cross-arm fixture content (type + crs + features, excluding
    the layer-name metadata); None when absent/unreadable -> check fails
    closed. Inherited unchanged from the F1R skeleton (REV 5B lesson)."""
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            d = json.load(f)
    except (OSError, ValueError):
        return None
    return json.dumps({k: d.get(k) for k in ("type", "crs", "features")},
                      sort_keys=True)


def block_checks(rows):
    """Per-block: engine_config + array/byte/fixture-content identity (the
    G1/G2/G5/G6 inputs) + the F3 arm-identity echo (b0 zero [F3] lines /
    cand exactly one, enforced by the driver at run time; echoed here)."""
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
        fx_content_b0 = _fixture_content(fx_b0.get("path")
                                         if isinstance(fx_b0, dict) else None)
        fx_content_c = _fixture_content(fx_c.get("path")
                                        if isinstance(fx_c, dict) else None)
        fx_content_equal = (fx_content_b0 is not None
                            and fx_content_b0 == fx_content_c)
        if not fx_content_equal:
            failures.append({"mode": "fixture_content_mismatch",
                             "b0_sha": fx_sha_b0, "cand_sha": fx_sha_c})
        f3_echo = {}
        for r, arm in ((rb, "b0"), (rc, "cand")):
            cap = r.get("f3_tail_capture") or {}
            expected = 0 if arm == "b0" else 1
            f3_echo[arm] = {"n_f3_lines": cap.get("n_f3_lines"),
                            "expected": expected,
                            "ok": cap.get("n_f3_lines") == expected}
        blk = {
            "block": bnum, "order": rb["order"],
            "engine_config_observed": {"b0": rb.get("engine_config_observed"),
                                       "cand": rc.get("engine_config_observed")},
            "array_hashes_equal": not arr_delta,
            "array_delta_keys": arr_delta,
            "array_hashes": {"b0": ra_b0, "cand": ra_c},
            "fixture_sha": {"b0": fx_sha_b0, "cand": fx_sha_c},
            "fixture_content_equal": fx_content_equal,
            "f3_arm_identity_echo": f3_echo,
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


def f3_analysis(rows):
    """Ruled descriptive statistics (h04 owns disposition). ALL capacity_only:
    the G3 comparison is on ru_maxrss peaks, the G4 comparison is a wall-time
    CAPACITY bound - no throughput claim is made or admissible."""
    out = {"capacity_only": "DECISION line 11",
           "primary_ruled": "G3 on the WARM GOVERNING block per arm = "
                            "max(ru_maxrss over warm blocks 2-3)"}
    have = {(r["block"], r["arm"]): r for r in rows
            if not r.get("missing") and r.get("peak_rss_mib") is not None}
    need = [(b, a) for b in (1, 2, 3) for a in ("b0", "cand")]
    if not all(k in have for k in need):
        out["incomplete"] = f"have {len(have)}/6 runs"
        return out
    gov = {a: max(have[(b, a)]["peak_rss_mib"] for b in (2, 3))
           for a in ("b0", "cand")}
    cold = {a: have[(1, a)]["peak_rss_mib"] for a in ("b0", "cand")}
    delta_mib = gov["b0"] - gov["cand"]
    delta_bytes = delta_mib * (1 << 20)
    # b0_accounted recomputed from the b0 record's MEASURED V' (never
    # hardcoded factors): chunk_b0 = max(1, int(1e8 // V')), transient
    # slice = chunk_b0 rows x V' cols x 13 B (dist 8 + pred 4 + mask 1);
    # the SAME measured V' sits in both factors (ruling (f)); the recompute
    # must equal the pinned formula value exactly or the analysis refuses
    b0_rec = have[(2, "b0")]
    v_prime = ((b0_rec.get("engine_shape_measured") or {})
               .get("v_prime_row_width_n_total"))
    b0_accounted = (max(1, int(1e8 // v_prime)) * v_prime * 13
                    if v_prime else None)
    if b0_accounted != G3_B0_ACCOUNTED_BYTES:
        print(f"[f3r-analysis] REFUSED: b0_accounted recompute "
              f"{b0_accounted} != pinned {G3_B0_ACCOUNTED_BYTES} "
              f"(measured V' {v_prime}; formula factors must be identical)")
        out["b0_accounted_assertion_failed"] = {
            "recomputed": b0_accounted, "pinned": G3_B0_ACCOUNTED_BYTES,
            "measured_v_prime": v_prime}
        return out
    # cand_accounted OBSERVED (descriptive): worst per-instance theorem LHS
    # minus margin, from the cand warm governing block's receipt
    # (resim_perinstance_v1 form)
    cand_rec = have[(2, "cand")]
    cand_receipt = cand_rec.get("mechanism_receipt") or {}
    worst_inst = (((cand_receipt.get("binding_legs") or {})
                   .get("e_perinstance_reconstruction") or {})
                  .get("worst_instance") or {})
    margin_bytes = ((cand_receipt.get("binding_legs") or {})
                    .get("c_resim_elementwise") or {}
                    .get("resim_inputs") or {}).get("margin_bytes")
    cand_accounted_observed = (worst_inst.get("lhs_bytes") - margin_bytes
                               if worst_inst.get("lhs_bytes") is not None
                               and margin_bytes is not None else None)
    cand_accounted_worst = G3_CAND_WORST_CASE_PAYLOAD_MIB * (1 << 20)
    out.update({
        "warm_governing_peaks_mib": gov,
        "cold_peaks_mib": cold,
        "measured_warm_governing_delta_bytes": round(delta_bytes),
        "measured_warm_governing_delta_mib": round(delta_mib, 2),
        "preregistered_min_delta_bytes": G3_PREREGISTERED_MIN_DELTA_BYTES,
        "g3_measured_delta_at_least_preregistered_min":
            delta_bytes >= G3_PREREGISTERED_MIN_DELTA_BYTES,
        "b0_accounted_bytes_recomputed": b0_accounted,
        "b0_accounted_formula": "max(1, int(1e8 // V_measured)) * V_measured "
                                "* 13 — the SAME measured V' in both factors "
                                "(ruling (f)); asserted equal to the pinned "
                                f"{G3_B0_ACCOUNTED_BYTES} B",
        "b0_accounted_assertion": {
            "pinned_bytes": G3_B0_ACCOUNTED_BYTES,
            "recomputed_bytes": b0_accounted,
            "pass": b0_accounted == G3_B0_ACCOUNTED_BYTES},
        "cand_accounted_worst_case_bytes": cand_accounted_worst,
        "cand_accounted_worst_case_basis": "cap 256 MiB - margin 64 MiB = 192 MiB payload bound (spec PD-A)",
        "accounted_reduction_preregistered_basis_bytes":
            (b0_accounted - cand_accounted_worst) if b0_accounted else None,
        "cand_accounted_observed_bytes_descriptive": cand_accounted_observed,
        "measured_delta_vs_observed_accounted_descriptive":
            (round(delta_bytes / (b0_accounted - cand_accounted_observed), 4)
             if b0_accounted and cand_accounted_observed
             and b0_accounted > cand_accounted_observed else None),
        "g4_cold_single_pair": {
            "note": "PD-C: one cold block per arm - the 'median' IS the "
                    "single pair; wall-time capacity bound, capacity_only",
            "cold_wall_b0_s": have[(1, "b0")]["window_s"],
            "cold_wall_cand_s": have[(1, "cand")]["window_s"],
            "cold_ratio_cand_over_b0": round(
                have[(1, "cand")]["window_s"] / have[(1, "b0")]["window_s"], 4),
            "bound_le": G4_COLD_RATIO_BOUND,
            "within_bound": have[(1, "cand")]["window_s"]
                            <= G4_COLD_RATIO_BOUND * have[(1, "b0")]["window_s"]},
        "warm_pair_times_descriptive": {
            f"block{b}": {"b0_s": have[(b, "b0")]["window_s"],
                          "cand_s": have[(b, "cand")]["window_s"]}
            for b in (2, 3)},
    })
    return out


def summary(refusal=None, stop_reason=None):
    rows = []
    for block, order, arm in SCHEDULE:
        run_id = f"f3r_b{block}_{order}_{arm}"
        r = load_rec(run_id)
        if r is None:
            rows.append({"run_id": run_id, "missing": True, "block": block,
                         "order": order, "arm": arm})
            continue
        # S1 (h04 amendment round): driver fatal() can fire before the
        # window/process/env/import sections exist, and the summary path's
        # whole job is preserving evidence on stop - direct indexing here
        # KeyErrors exactly when preservation matters most. Safe access plus
        # an explicit record_complete flag; partial records stay visible.
        win = r.get("window") or {}
        proc = r.get("process") or {}
        env = r.get("env") or {}
        imp = r.get("import") or {}
        app_ns = win.get("application_window_ns")
        rows.append({
            "run_id": run_id, "block": block, "order": order, "arm": arm,
            "cold_warm": ("cold (fresh per-arm cache)" if block == 1
                          else "warm (reused cache)"),
            "W_K_H": [1, K_STRIPES, H_THREADS],
            "record_complete": all((win, proc, env, imp)),
            "window_s": round(app_ns / 1e9, 4) if app_ns is not None else None,
            "engine_config_observed": r.get("engine_config_observed"),
            "resolved_gravity_cap": r.get("resolved_gravity_cap"),
            "numba_num_threads": proc.get("numba_get_num_threads"),
            "cache_dir": env.get("NUMBA_CACHE_DIR"),
            "cache_file_count_at_run": (cache_files(env["NUMBA_CACHE_DIR"])
                                        if env.get("NUMBA_CACHE_DIR") else None),
            "peak_rss_mib": r.get("peak_rss_ru_maxrss_mib"),
            "identity_ok": (imp["identity_assertion"].startswith("ok")
                            if "identity_assertion" in imp else None),
            "f3_module_identity": r.get("f3_module_identity"),
            "f3_tail_capture": r.get("f3_tail_capture"),
            "mechanism_receipt": r.get("mechanism_receipt"),
            "engine_shape_measured": r.get("engine_shape_measured"),
            "mechanism_receipt_inputs_measured": r.get(
                "mechanism_receipt_inputs_measured"),
            "g7_stop_and_preserve": r.get("g7_stop_and_preserve"),
            "counts": r.get("counts"),
            "origin_selection": r.get("origin_selection"),
            "array_hashes": array_hashes(r),
            "verbatim_warnings": r.get("verbatim_warnings"),
            "fatal": r.get("fatal"),
        })
    findings, track_ok = block_checks(rows)
    with open(LEASE_LOG) as f:
        lease = json.load(f)
    envl = json.load(open(ENVELOPE)) if os.path.exists(ENVELOPE) else None
    out = {
        "task": "F3R", "record": "screen_summary", "role": "screen-executor",
        "spec_source": f"evidence/F3R/screen_spec.json (P1-committed; sha256 "
                       f"{sha256_file(SPEC) if os.path.exists(SPEC) else 'MISSING'})",
        "descriptive_only": "h04-reviewer owns disposition/verdicts",
        "capacity_only": "DECISION line 11: wall times are capacity_only; no "
                         "throughput or performance claim made or admissible",
        "schedule": "3 blocks AB/BA/AB (A=b0); block 1 cold on fresh per-arm "
                    "cache roots, blocks 2-3 warm",
        "frozen_triple": {"W": 1, "K": K_STRIPES, "H": H_THREADS},
        "arms": {k: {"venv_python": v["python"],
                     "site_packages": v["site_packages"],
                     "cache_root": v["cache_root"],
                     "commit": v["commit"],
                     "wheel_sha256": wheel_sha(k)} for k, v in ARMS.items()},
        "envelope": envl,
        "workload": {"cell": "O3_FLOW", "variant": VARIANT, "manifest": MANIFEST},
        "runs": rows,
        "block_findings": findings,
        "track_conformant": track_ok,
        "f3_analysis": f3_analysis(rows),
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
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with open(SUMMARY, "w") as f:
        json.dump(out, f, indent=1)
    print(f"[f3r-pairs] summary -> {SUMMARY}")
    print(f"  f3_analysis={json.dumps(out['f3_analysis'])[:400]}")
    print(f"  track_conformant={track_ok}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["sizing", "envelope", "run", "summary"],
                    default="run")
    a = ap.parse_args()
    ensure_lease_log()
    if a.stage == "summary":
        summary()
        return 0
    if a.stage == "envelope":
        return envelope_stage()
    for arm in ARMS.values():
        if not (os.path.exists(arm["python"]) and os.path.exists(WHEEL_RECORD)
                and os.path.exists(arm["site_packages"])):
            print(f"[f3r-pairs] REFUSED: arm venv/wheel record missing: {arm}")
            return 2
    if a.stage == "sizing":
        ok = True
        for arm in ("b0", "cand"):
            if os.path.exists(f"{DATA}/f3r_sizing_{arm}.json"):
                print(f"[f3r-pairs] sizing record present for {arm} — delete "
                      "explicitly to re-size (evidence discipline)")
                ok = False
                break
        if ok:
            for arm in ("b0", "cand"):
                rc, run_id, msg = sizing_slot(arm)
                print(f"[f3r-pairs] {run_id}: {msg}", flush=True)
                if rc != 0:
                    return rc
        return 0
    # ---- matrix entry (stage == run) ----
    if not os.path.exists(SPEC):
        print(f"[f3r-pairs] REFUSED: {SPEC} missing — P1 (committed spec) unmet")
        return 2
    envl = load_envelope(required=True)
    if envl is None:
        print(f"[f3r-pairs] REFUSED: {ENVELOPE} missing — P6 sizing/envelope "
              "unmet (run --stage sizing then --stage envelope)")
        return 2
    if envl.get("provisional", False):
        print("[f3r-pairs] REFUSED: envelope is PROVISIONAL — the provisional "
              "watchdog is valid for sizing runs ONLY; the matrix refuses to "
              "enter without the P6-derived f3r_envelope.json (pre-ruling (a))")
        return 2
    os.makedirs(RAW, exist_ok=True)
    stop_reason = None
    run_ids = [f"f3r_b{b}_{o}_{a}" for b, o, a in SCHEDULE]
    present = [rid for rid in run_ids if os.path.exists(f"{RAW}/{rid}.json")]
    if len(present) == 6:
        print("[f3r-pairs] all 6 raw records present; no execution — use "
              "--stage summary to regenerate the summary")
        summary()
        return 0
    if present:
        print(f"[f3r-pairs] REFUSED: partial matrix present ({len(present)}/6: "
              f"{present}) — a voided matrix is not dispositional and never "
              "resumes; restart-whole requires h04 approval and fresh run ids")
        return 2
    import psutil
    avail_entry = psutil.virtual_memory().available
    entry_min = envl["matrix_entry_min_avail_bytes"]
    entry_rec = {"what": "matrix_entry_check", "lease_id": LEASE_ID,
                 "start_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                 "available_at_admission_bytes": avail_entry,
                 "admission_080avail_bytes": int(0.80 * avail_entry),
                 "entry_bar_bytes": entry_min,
                 "envelope_source": ENVELOPE,
                 "entry_rule": ("available >= (watchdog + 230 MiB)/0.80 at "
                                "matrix entry and any post-void re-entry "
                                "(spec P6)")}
    if avail_entry < entry_min:
        entry_rec["refused"] = (f"matrix ENTRY bar not met: available < "
                                f"{entry_min} B; no window opened, uncharged")
        lease_append(entry_rec)
        print(f"[f3r-pairs] REFUSED: matrix ENTRY bar (P6): available "
              f"{avail_entry} B < {entry_min} B; no window opened, uncharged; "
              "re-invoke when the bar is met")
        return 2
    lease_append(entry_rec)
    for block, order, arm in SCHEDULE:
        rc, run_id, msg = run_slot(block, order, arm, envl)
        print(f"[f3r-pairs] slot {run_id}: {msg}", flush=True)
        if rc != 0:
            stop_reason = msg
            print("[f3r-pairs] STOPPING screen per failure policy "
                  "(no retry; preserve evidence; h04 approval needed)")
            if msg.startswith("REFUSED"):
                summary(refusal=msg, stop_reason=stop_reason)
            else:
                summary(stop_reason=stop_reason)
            return rc
        peer = "b0" if arm == "cand" else "cand"
        if load_rec(f"f3r_b{block}_{order}_{peer}") is not None:
            rows_now = []
            for a2 in ("b0", "cand"):
                r2 = load_rec(f"f3r_b{block}_{order}_{a2}")
                rows_now.append({**r2, "block": block, "order": order,
                                 "arm": a2, "run_id": f"f3r_b{block}_{order}_{a2}"})
            _, track_ok = block_checks(rows_now)
            if not track_ok:
                stop_reason = ("conformity failure (byte-identity delta / engine "
                               "config mismatch / fixture mismatch) in completed "
                               f"block {block} — STOP and report")
                print(f"[f3r-pairs] {stop_reason}")
                summary(refusal=None, stop_reason=stop_reason)
                return 3
    summary()
    return 0


if __name__ == "__main__":
    sys.exit(main())
