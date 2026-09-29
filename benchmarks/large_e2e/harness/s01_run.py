"""S01 supervisor (task #36) -- equal-resource CPU scheduling selection.

Chain-initializes the SINGLE campaign lease ledger from the frozen W00 final
log, then runs tranche-gated installed sweeps of the frozen W/H matrix
(tests/large_e2e/scheduling/matrix.py) through run.py in installed mode.

Design authority: proposal_s01_20260928T220026Z_h05_amendment2.md
(custody, sha 353f822f...) + h04 SS6 fold rulings.  Tranche gates ENFORCE the
recorded counts and budget bounds -- any deviation is a STOP, never an
extended matrix, never a stretched cap.  The independent reviewer's output
file is NEVER created or touched here: it appears on disk only when the
reviewer writes it at review close (amendment #2 SS4.4).

Stages (fail-fast, chained with && by the fire protocol):
  init     chain-init s01_lease_log.json (refuse-if-exists; W00 pin +
           control base + instrument pins verified live)
  p0       baseline-only pilots (3 cells x 2, B0 arm only); derives fixed
           K_jobs per family; evaluates the SS6.3 queue-2W criterion from
           recorded producer-wait spans (trigger = STOP, review re-route)
  p1       sweep: 6 accessibility + 1 flow configs x 2 batches per arm,
           create-exclusive out dirs, fresh per-batch JIT cache roots
  confirm  one batch per arm per family at the selected B0 config
  records  assemble evidence/S01/{configurations.json, memory.json}
  status   read-only ledger + progress report
"""
import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time

CAMPAIGN = "/Users/alansynn/orca/workspaces/una-x"
REPO = f"{CAMPAIGN}/wt-large-e2e"
DATA = f"{CAMPAIGN}/campaign_data"
CAMPAIGN_PY = f"{CAMPAIGN}/venvs/campaign/bin/python"

sys.path.insert(0, f"{REPO}/tests/large_e2e/scheduling")
import matrix  # noqa: E402  (frozen S01 matrix; pure data, no IO)

LEASE_LOG = f"{DATA}/s01_lease_log.json"
RUNROOT = f"{DATA}/s01_run"
NBCROOT = f"{DATA}"
EVIDENCE = f"{REPO}/campaigns/una_large_e2e/evidence/S01"
CONTROL = f"{REPO}/campaigns/una_large_e2e/evidence/control.json"
RUN_PY = f"{REPO}/benchmarks/large_e2e/run.py"
POOL_PY = f"{REPO}/benchmarks/large_e2e/harness/pool.py"

BUDGET_S = matrix.LEDGER["budget_total_s"]
PRESSURE_STOP = matrix.PRESSURE_STOP_BYTES

# Instrument pins verified live at init (fresh tool outputs, 2026-09-28).
# run.py is the unchanged W00 triad member; pool.py carries the S01
# producer-wait instrumentation the SS6.3(i) criterion consumes.
PINS = {
    RUN_PY: "c041c33f4ece70e381fb970d67793425221a694d9ed257c512ddb6a9d57f0f22",
    POOL_PY: "25e43bb707019a437be347731bda87bb4dcaf3f5b34a77b52d8716992167e5e1",
    matrix.RUN_MANIFESTS["O3_ACCESS"]:
        "a946658d7ebdeba8c07ae56f92abe586c607e7ad92aa3b58d1a55b4a1ecd5c25",
    matrix.RUN_MANIFESTS["O3_FLOW"]:
        "a02d5f6b3d1357640a55bbed9347920efcc86f599ac74912f729e6f736b966d5",
}
# O2 pilots reuse the W00 smoke manifest (frozen, sha-pinned by w00_run.py).
O2_MANIFEST = f"{REPO}/tests/large_e2e/installed/w00_O2_smoke.manifest.json"
O2_MANIFEST_SHA256 = (
    "84350bee33d414826371f38b13a37331d971502b79bbe463c0c7f98d828990c0")

CFGKEY_TIMEOUT_S = matrix.BATCH_TIMEOUT_S
PILOT_TIMEOUT_S = matrix.PILOT_TIMEOUT_S


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


# --- class-8 window schema (W00 parity) ------------------------------------

def _complete_refused(entry, window_class):
    entry["class"] = window_class
    entry["end_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    entry["wall_s"] = 0.0
    entry["rc"] = 2


def _open_window(what, lease_id):
    log = load_lease_log()
    if log is None:
        return None
    log["windows"].append({
        "what": what, "class": "crashed_child_start", "lease_id": lease_id,
        "start_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "end_utc": None, "wall_s": None, "rc": None})
    save_lease_log(log)
    return len(log["windows"]) - 1


def _close_window(index, cls, rc, extra=None):
    if index is None:
        return
    log = load_lease_log()
    entry = log["windows"][index]
    entry["class"] = cls
    entry["end_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    entry["wall_s"] = 0.0   # envelopes are wall-0: child windows carry wall
    entry["rc"] = rc
    if extra:
        entry.update(extra)
    log["updated_utc"] = entry["end_utc"]
    save_lease_log(log)


def _run_stage(what, fn, lease_id, *args):
    idx = _open_window(what, lease_id)
    rc = fn(*args)
    _close_window(idx, "stage", rc)
    return rc


def _log_has_pending_crash(log):
    return any(w.get("end_utc") is None for w in log["windows"])


def child_env(extra):
    """Clean-room child env (W00 parity): no PYTHONPATH, no
    NUMBA_DISABLE_JIT, no inherited NUMBA_CACHE_DIR."""
    env = {k: v for k, v in os.environ.items()
           if k not in ("PYTHONPATH", "NUMBA_DISABLE_JIT", "NUMBA_CACHE_DIR")}
    env.update(extra)
    return env


def memory_ceiling_mib():
    import psutil
    avail = psutil.virtual_memory().available
    return int(min(matrix.CEILING_MIB_CAP,
                   matrix.CEILING_FRACTION * avail) / (1 << 20)), avail


def guard_and_launch(cmd, env, lease_id, what, timeout_s,
                     cwd=None, window_class="heavy"):
    """W00-parity admission + class-8 window + charging.  Refusals happen
    BEFORE launch (pressure / budget); a refused window is a decided window
    (rc 2), never a silent skip."""
    import psutil
    log = load_lease_log()
    if log is None:
        print("[s01-run] REFUSED: lease not initialized (run --stage init)")
        return 2
    avail = psutil.virtual_memory().available
    ceiling = min(matrix.CEILING_MIB_CAP << 20,
                  int(matrix.CEILING_FRACTION * avail))
    entry = {
        "what": what, "class": "crashed_child_start",
        "window_class": window_class, "lease_id": lease_id,
        "start_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "end_utc": None, "wall_s": None, "rc": None,
        "available_at_admission_bytes": avail, "ceiling_bytes": ceiling,
        "pressure_stop_bytes": PRESSURE_STOP,
        "loadavg_at_admission": os.getloadavg(),
        "cmd": cmd, "cwd": cwd,
        "env": {k: v for k, v in env.items()
                if k.startswith(("NUMBA", "PYTHONDONT", "UNA_"))},
    }
    if avail < PRESSURE_STOP:
        entry["refused"] = ("available < pressure stop before launch "
                            "(RESOURCES.md)")
        _complete_refused(entry, window_class)
        log["windows"].append(entry)
        save_lease_log(log)
        print(f"[s01-run] REFUSED: available {avail} < pressure stop")
        return 2
    if log["cumulative_charged_s"] >= BUDGET_S:
        entry["refused"] = "heavy wall budget exhausted"
        _complete_refused(entry, window_class)
        log["windows"].append(entry)
        save_lease_log(log)
        print("[s01-run] REFUSED: heavy wall exhausted")
        return 2
    log["windows"].append(entry)
    save_lease_log(log)

    t0 = time.monotonic()
    try:
        p = subprocess.run(cmd, env=env, timeout=timeout_s, cwd=cwd,
                           capture_output=True, text=True)
        rc, tail = p.returncode, (p.stdout[-3000:] + p.stderr[-3000:])
    except subprocess.TimeoutExpired:
        rc, tail = -9, f"TIMEOUT after {timeout_s}s"
    wall = time.monotonic() - t0

    log = load_lease_log()
    entry = log["windows"][-1]
    entry.update({"class": window_class,
                  "end_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                           time.gmtime()),
                  "wall_s": round(wall, 1), "rc": rc, "tail": tail[-2000:],
                  "loadavg_at_completion": os.getloadavg()})
    log["cumulative_charged_s"] = round(
        log["opening_balance_s"]
        + sum(w.get("wall_s", 0.0) or 0.0 for w in log["windows"]), 1)
    log["remaining_s"] = round(BUDGET_S - log["cumulative_charged_s"], 1)
    log["updated_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    save_lease_log(log)
    print(f"[s01-run] rc={rc} wall={wall:.1f}s "
          f"charged={log['cumulative_charged_s']}s "
          f"remaining={log['remaining_s']}s")
    return rc


# --- child argv builders ----------------------------------------------------

def _common_run_args(manifest, arm, w, h, jobs, queue_depth, cache_root,
                     out_dir, ceiling_mib):
    return [
        "--manifest", manifest,
        "--arm", f"s01_{arm}",
        "--identity", matrix.ARM_IDENTITIES[arm],
        "--mode", "batch",
        "--jobs", str(jobs),
        "--workers", str(w),
        "--numba-threads", str(h),
        "--flow-stripes", "default",
        "--queue-depth", str(queue_depth),
        "--writer-limit", str(w),          # writer limit = W everywhere
        "--cpu-budget", str(matrix.C_SLOTS),
        "--memory-budget-mib", str(ceiling_mib),
        "--timeout-s", str(CFGKEY_TIMEOUT_S),
        "--cache-root", cache_root,
        "--out", out_dir,
    ]


def _run_child(args_tail, lease_id, what, timeout_s, env_extra=None):
    cmd = [sys.executable, RUN_PY] + args_tail
    env = child_env(env_extra or {})
    return guard_and_launch(cmd, env, lease_id, what, timeout_s,
                            cwd=RUNROOT, window_class="heavy")


def _refuse_existing(*paths):
    """[path_collision] preconditions: every listed path must be ABSENT
    pre-launch (create-exclusive; residue is never auto-disposed)."""
    present = [p for p in paths if os.path.exists(p)]
    if present:
        for p in present:
            print(f"[s01-run] REFUSED: path exists (fresh-path guard): {p}")
        return False
    return True


def _read_session(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def _session_valid(session):
    return bool(session) and session.get("status") == "valid" \
        and (session.get("counts", {}).get("validated", 0) or 0) > 0


def _producer_wait_share(session):
    """SS6.3(i): producer-wait share from RECORDED harness spans only."""
    dispatch = session.get("dispatch") or {}
    total_ns = dispatch.get("producer_wait_total_ns") or 0
    wall_s = (session.get("times_ns", {}) or {}).get(
        "warm_batch_application_wall_s")
    if not wall_s:
        return None
    return total_ns / 1e9 / wall_s


# --- stages -----------------------------------------------------------------

def stage_init(lease_id):
    if load_lease_log() is not None:
        print("[s01-run] REFUSED: s01 lease log already exists "
              "(no double init)")
        return 2
    if not matrix.matrix_valid():
        print("[s01-run] REFUSED: frozen matrix failed invariant check")
        return 2
    parent_log = matrix.LEDGER["w00_final_log"]
    if not os.path.exists(parent_log):
        print("[s01-run] REFUSED: W00 final lease log missing")
        return 2
    got = sha256_file(parent_log)
    if got != matrix.W00_CHAIN_PIN:
        print(f"[s01-run] REFUSED: W00 log sha {got} != pinned")
        return 2
    with open(parent_log) as f:
        parent = json.load(f)
    opening = round(float(parent["cumulative_charged_s"]), 1)
    if abs(opening - matrix.LEDGER["w00_charged_s"]) > 0.1:
        print(f"[s01-run] REFUSED: W00 cumulative_charged_s {opening} != "
              f"{matrix.LEDGER['w00_charged_s']}")
        return 2
    with open(CONTROL) as f:
        control = json.load(f)
    got_base = control["selected_source"]["commit"]
    if got_base != matrix.BASE_COMMIT:
        print(f"[s01-run] REFUSED: control selected_source.commit "
              f"{got_base} != base {matrix.BASE_COMMIT}")
        return 2
    for path, pin in list(PINS.items()) + [(O2_MANIFEST, O2_MANIFEST_SHA256)]:
        got = sha256_file(path)
        if got != pin:
            print(f"[s01-run] REFUSED: pinned instrument sha mismatch "
                  f"{path}: {got} != {pin}")
            return 2
    if os.path.exists(RUNROOT):
        print(f"[s01-run] REFUSED: run root exists (fresh-path guard): "
              f"{RUNROOT}")
        return 2
    os.makedirs(RUNROOT)
    save_lease_log({
        "lease_id": lease_id,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "task": "S01",
        "parent_log": parent_log,
        "parent_log_sha256": got,
        "base_commit": matrix.BASE_COMMIT,
        "base_src_tree": matrix.BASE_SRC_TREE,
        "matrix_module": ("tests/large_e2e/scheduling/matrix.py sha256 "
                          + sha256_file(f"{REPO}/tests/large_e2e/"
                                        f"scheduling/matrix.py")),
        "pins": {p: s for p, s in PINS.items()},
        "o2_pilot_manifest": O2_MANIFEST,
        "o2_pilot_manifest_sha256": O2_MANIFEST_SHA256,
        "opening_balance_s": opening,
        "opening_balance_source": ("cumulative_charged_s of the frozen W00 "
                                   "final log, read fresh at init"),
        "budget_s": BUDGET_S,
        "s01_cap_s": matrix.LEDGER["s01_cap_s"],
        "reserve_floor_s": matrix.LEDGER["reserve_floor_s"],
        "pressure_stop_bytes": PRESSURE_STOP,
        "k_fit_branch": matrix.K_FIT["branch"],
        "windows": [],
        "cumulative_charged_s": opening,
        "remaining_s": round(BUDGET_S - opening, 1),
    })
    print(f"[s01-run] init: chained to W00 final log (opening {opening}s "
          f"of {BUDGET_S}s; cap {matrix.LEDGER['s01_cap_s']}s; reserve "
          f"floor {matrix.LEDGER['reserve_floor_s']}s)")
    return 0


def _pilot_cell_manifest(cell):
    if cell == "O2":
        return O2_MANIFEST
    return matrix.RUN_MANIFESTS[cell]


def stage_p0(lease_id, arm):
    if arm != "b0":
        print("[s01-run] REFUSED: P0 pilots are baseline-only (--arm b0)")
        return 2
    log = load_lease_log()
    if log is None or _log_has_pending_crash(log):
        print("[s01-run] REFUSED: lease not initialized or has pending "
              "crashed windows")
        return 2
    ceiling_mib, _avail = memory_ceiling_mib()
    pilot_records = []
    for cell in matrix.PILOT_CELLS:
        ref_w, ref_h = matrix.PILOT_REFERENCE_CONFIG[
            "flow" if cell == "O3_FLOW" else "accessibility"]
        for n in (1, 2):
            out_dir = f"{RUNROOT}/pilots/{cell}_pilot_{n}"
            cache_root = f"{NBCROOT}/nbc_s01_pilot_{cell}_{n}"
            if not _refuse_existing(out_dir, cache_root):
                return 2
            tail = _common_run_args(
                _pilot_cell_manifest(cell), "b0", ref_w, ref_h, jobs=1,
                queue_depth=1, cache_root=cache_root, out_dir=out_dir,
                ceiling_mib=ceiling_mib)
            rc = _run_child(tail, lease_id, f"pilot:{cell}:{n}",
                            PILOT_TIMEOUT_S)
            if rc != 0:
                print(f"[s01-run] STOP: pilot {cell}#{n} rc={rc} "
                      "(fail-fast; residue retained)")
                return rc if rc != 0 else 1
            session = _read_session(f"{out_dir}/session.json")
            wall = (session.get("times_ns", {})
                    .get("warm_batch_application_wall_s")) if session else None
            pilot_records.append({
                "cell": cell, "pilot": n, "ref_config": [ref_w, ref_h],
                "warm_batch_application_wall_s": wall,
                "producer_wait_share": _producer_wait_share(session),
                "out_dir": out_dir,
            })

    # Fixed K_jobs per SWEPT family (BENCHMARKS: divisible by largest
    # compared W; >=60 s warm-window target; never truncate a job).  The O2
    # pilot is recorded for cold/warm context; O2 is not a swept workload.
    k_jobs = {}
    for family, cell in matrix.SWEEP_WORKLOADS.items():
        walls = [r["warm_batch_application_wall_s"] for r in pilot_records
                 if r["cell"] == cell and r["warm_batch_application_wall_s"]]
        if not walls:
            print(f"[s01-run] STOP: no warm wall recorded for {cell}")
            return 2
        wall = max(walls)
        lw = matrix.largest_w(family)
        k = max(lw, int(math.ceil(60.0 / wall / lw)) * lw)
        k_jobs[family] = {"k_jobs": k, "basis_cell": cell,
                          "warm_job_wall_s_basis": wall,
                          "largest_w": lw}

    # SS6.3 queue-2W criterion on the B0 pilot leg, PER FAMILY; every
    # application is recorded.  A trigger is a matrix change -> STOP and
    # route through review; the cap never stretches.
    queue_evaluations = {}
    trigger = False
    for family, cell in matrix.SWEEP_WORKLOADS.items():
        shares = [r["producer_wait_share"] for r in pilot_records
                  if r["cell"] == cell]
        shares = [s for s in shares if s is not None]
        peak_frac = _max_live_memory_frac(pilot_records)
        mem_headroom = (peak_frac is not None
                        and peak_frac <= matrix.QUEUE_POLICY[
                            "live_memory_frac_of_ceiling_max"])
        triggered = bool(shares) and max(shares) > \
            matrix.QUEUE_POLICY["producer_wait_share_max"] and mem_headroom
        queue_evaluations[family] = {
            "producer_wait_shares": shares,
            "live_memory_frac_of_ceiling": peak_frac,
            "triggered": triggered,
            "thresholds": {
                "producer_wait_share_max":
                    matrix.QUEUE_POLICY["producer_wait_share_max"],
                "live_memory_frac_of_ceiling_max":
                    matrix.QUEUE_POLICY["live_memory_frac_of_ceiling_max"]},
        }
        if triggered:
            trigger = True

    os.makedirs(EVIDENCE, exist_ok=True)
    configurations_path = f"{EVIDENCE}/configurations.json"
    if os.path.exists(configurations_path):
        # never silent: interim custody pin BEFORE any rewrite (census
        # item 6); a ruled re-route re-opens P0 without losing bytes
        prior_sha = sha256_file(configurations_path)
        shutil.copy2(configurations_path,
                     f"{DATA}/custody/interim_{prior_sha[:8]}_"
                     f"s01_configurations.json")
    configurations = {
        "schema_version": 1,
        "task": "S01",
        "record": "configurations",
        "stage": "p0_close",
        "base_commit": matrix.BASE_COMMIT,
        "frozen_matrix": {
            "accessibility_configs": matrix.ACCESSIBILITY_CONFIGS,
            "flow_configs": matrix.FLOW_CONFIGS,
            "batches_per_cell": matrix.BATCHES_PER_CELL,
            "arms": matrix.ARMS,
            "total_batches": matrix.total_batches(),
            "sweep_workloads": matrix.SWEEP_WORKLOADS,
            "k_fit": matrix.K_FIT,
        },
        # h04 Ruling 2: bounded-fixture provenance travels with the record
        "access_origins": matrix.ACCESS_ORIGINS,
        "p0": {"pilots": pilot_records, "k_jobs": k_jobs,
               "queue_2w_criterion": queue_evaluations,
               "writer_policy": matrix.WRITER_POLICY},
    }
    with open(configurations_path, "w") as f:
        json.dump(configurations, f, indent=1)
    if trigger:
        print("[s01-run] STOP: queue-2W criterion TRIGGERED on the B0 pilot "
              "leg (see configurations.json p0.queue_2w_criterion) -- a 2W "
              "config is a matrix change; route through review before P1")
        return 2
    print(f"[s01-run] p0 complete: k_jobs={ {f: v['k_jobs'] for f, v in k_jobs.items()} }; "
          f"queue criterion not triggered")
    return 0


def _max_live_memory_frac(pilot_records):
    """Peak live process-tree memory as a fraction of the ceiling, from the
    recorded sampler peaks of the pilot sessions (tree_rss_bytes_sum)."""
    fracs = []
    for r in pilot_records:
        session = _read_session(f"{r['out_dir']}/session.json")
        if not session:
            continue
        peak = (session.get("process_tree_memory", {}).get("peak") or {})
        peak_bytes = peak.get("tree_rss_bytes_sum")
        ceiling = session.get("requested_config", {}).get(
            "memory_budget_mib")
        if peak_bytes and ceiling:
            fracs.append(peak_bytes / (ceiling * (1 << 20)))
    return max(fracs) if fracs else None


def _expected_batch_dirs(arm):
    """The frozen P1 batch plan for one arm, in execution order.  Out dirs
    are ARM-SCOPED: a shared dir would make the second arm's pass see the
    first arm's valid sessions as its own 'already done' and never launch
    (arm-collision defect found by the C1 coverage extension)."""
    plan = []
    for family, cell in matrix.SWEEP_WORKLOADS.items():
        configs = (matrix.ACCESSIBILITY_CONFIGS if family == "accessibility"
                   else matrix.FLOW_CONFIGS)
        for w, h in configs:
            for b in (1, 2):
                plan.append({
                    "family": family, "cell": cell, "w": w, "h": h,
                    "batch": b,
                    "cfgkey": f"W{w}H{h}",
                    "out_dir": f"{RUNROOT}/{cell}/W{w}H{h}/{arm}/batch_{b}",
                    "cache_root": f"{NBCROOT}/nbc_s01_{arm}_{cell}_W{w}H{h}_b{b}",
                })
    return plan


def stage_p1(lease_id, arm):
    log = load_lease_log()
    if log is None or _log_has_pending_crash(log):
        print("[s01-run] REFUSED: lease not initialized or has pending "
              "crashed windows")
        return 2
    if not os.path.exists(f"{EVIDENCE}/configurations.json"):
        print("[s01-run] REFUSED: no P0 record (run --stage p0 first)")
        return 2
    with open(f"{EVIDENCE}/configurations.json") as f:
        configurations = json.load(f)
    k_jobs = configurations["p0"]["k_jobs"]
    if configurations.get("p0", {}).get("queue_2w_criterion") and any(
            v.get("triggered")
            for v in configurations["p0"]["queue_2w_criterion"].values()):
        print("[s01-run] REFUSED: queue-2W trigger recorded at P0; review "
              "re-route required before P1")
        return 2

    plan = _expected_batch_dirs(arm)
    done, remaining = [], []
    for row in plan:
        session = _read_session(f"{row['out_dir']}/session.json")
        if session is None:
            if os.path.exists(row["out_dir"]) or os.path.exists(
                    row["cache_root"]):
                print(f"[s01-run] REFUSED: residue without valid session at "
                      f"{row['out_dir']} (routed ruling required; never "
                      f"auto-disposed)")
                return 2
            remaining.append(row)
        elif _session_valid(session):
            done.append(row)
        else:
            print(f"[s01-run] REFUSED: invalid/failed session retained at "
                  f"{row['out_dir']} (STOP; route through ruling)")
            return 2
    if len(done) + len(remaining) != matrix.TOTAL_BATCHES // len(matrix.ARMS):
        print("[s01-run] REFUSED: batch-count deviation from the frozen "
              "matrix (STOP, never an extended matrix)")
        return 2

    # Tranche gate: budget projection from recorded pilot walls, margin x1.4
    # (amendment SS3); binding reserve floor at the gate.
    projected_s = 0.0
    for row in remaining:
        fam = row["family"]
        wall = k_jobs[fam]["warm_job_wall_s_basis"]
        projected_s += k_jobs[fam]["k_jobs"] * wall / row["w"]
    projected_s = round(projected_s, 1)
    cap = matrix.LEDGER["s01_cap_s"]
    floor = matrix.LEDGER["reserve_floor_s"]
    charged_now = log["cumulative_charged_s"]
    if charged_now + 1.4 * projected_s > cap:
        print(f"[s01-run] REFUSED: tranche gate -- charged {charged_now}s + "
              f"1.4 x projected {projected_s}s exceeds S01 cap {cap}s")
        return 2
    if BUDGET_S - (charged_now + projected_s) < floor:
        print(f"[s01-run] REFUSED: tranche gate -- projected remaining "
              f"{round(BUDGET_S - charged_now - projected_s, 1)}s breaches "
              f"the binding {floor}s reserve floor")
        return 2

    ceiling_mib, _avail = memory_ceiling_mib()
    for row in remaining:
        if not _refuse_existing(row["out_dir"], row["cache_root"]):
            return 2
        tail = _common_run_args(
            matrix.RUN_MANIFESTS[row["cell"]], arm, row["w"], row["h"],
            jobs=k_jobs[row["family"]]["k_jobs"],
            queue_depth=row["w"], cache_root=row["cache_root"],
            out_dir=row["out_dir"], ceiling_mib=ceiling_mib)
        rc = _run_child(
            tail, lease_id,
            f"p1:{arm}:{row['cell']}:{row['cfgkey']}:b{row['batch']}",
            CFGKEY_TIMEOUT_S)
        if rc != 0:
            print(f"[s01-run] STOP: p1 batch {row['cell']}/{row['cfgkey']} "
                  f"b{row['batch']} rc={rc} (fail-fast; residue retained; "
                  f"route through ruling)")
            return rc if rc != 0 else 1
    print(f"[s01-run] p1 arm {arm} complete: {len(remaining)} batches this "
          f"pass, {len(done)} already valid")
    return 0


def _collect_results():
    """Per (family, arm, config): the valid batch sessions."""
    results = []
    for arm in matrix.ARMS:
        for row in _expected_batch_dirs(arm):
            session = _read_session(f"{row['out_dir']}/session.json")
            if session is None:
                continue
            results.append(dict(row, arm=arm, session=session))
    return results


def _median(values):
    vs = sorted(values)
    n = len(vs)
    if n == 0:
        return None
    return vs[n // 2] if n % 2 else (vs[n // 2 - 1] + vs[n // 2]) / 2


def _family_selections():
    """Strongest feasible B0 config per family + ranked selected candidates.
    Feasible = every batch validated; rank by median primary throughput."""
    per = {}
    for r in _collect_results():
        key = (r["family"], r["arm"], r["cfgkey"])
        tp = (r["session"].get("throughput") or {}).get(
            "primary_jobs_per_s")
        valid = _session_valid(r["session"])
        per.setdefault(key, []).append(
            {"throughput": tp, "valid": valid, "row": r})
    selections = {}
    for family in matrix.SWEEP_WORKLOADS:
        table = []
        for (fam, arm, cfgkey), entries in per.items():
            if fam != family:
                continue
            tps = [e["throughput"] for e in entries if e["throughput"]]
            feasible = all(e["valid"] for e in entries) and len(tps) == \
                matrix.BATCHES_PER_CELL
            table.append({
                "arm": arm, "cfgkey": cfgkey,
                "median_primary_jobs_per_s": _median(tps),
                "feasible": feasible,
            })
        b0 = [t for t in table if t["arm"] == "b0" and t["feasible"]
              and t["median_primary_jobs_per_s"]]
        sel = [t for t in table if t["arm"] == "selected"
               and t["median_primary_jobs_per_s"]]
        b0_sorted = sorted(b0,
                           key=lambda t: t["median_primary_jobs_per_s"],
                           reverse=True)
        sel_sorted = sorted(sel,
                            key=lambda t: t["median_primary_jobs_per_s"],
                            reverse=True)
        selections[family] = {
            "strongest_feasible_b0": b0_sorted[0] if b0_sorted else None,
            "b0_ranking": b0_sorted,
            "selected_ranking": sel_sorted,
        }
    return selections


def stage_confirm(lease_id, arm):
    log = load_lease_log()
    if log is None or _log_has_pending_crash(log):
        print("[s01-run] REFUSED: lease not initialized or has pending "
              "crashed windows")
        return 2
    results = _collect_results()
    valid_p1 = [r for r in results if _session_valid(r["session"])]
    if len(valid_p1) != matrix.TOTAL_BATCHES:
        print(f"[s01-run] REFUSED: confirmation requires all "
              f"{matrix.TOTAL_BATCHES} P1 batches valid (have "
              f"{len(valid_p1)})")
        return 2
    selections = _family_selections()
    with open(f"{EVIDENCE}/configurations.json") as f:
        configurations = json.load(f)
    k_jobs = configurations["p0"]["k_jobs"]
    ceiling_mib, _avail = memory_ceiling_mib()
    launched = 0
    for family, cell in matrix.SWEEP_WORKLOADS.items():
        strongest = selections[family]["strongest_feasible_b0"]
        if not strongest:
            print(f"[s01-run] STOP: no feasible B0 config for {family}")
            return 2
        w, h = (int(t) for t in strongest["cfgkey"][1:].split("H"))
        out_dir = f"{RUNROOT}/{cell}/confirm_{strongest['cfgkey']}/{arm}"
        cache_root = f"{NBCROOT}/nbc_s01_confirm_{arm}_{cell}"
        if not _refuse_existing(out_dir, cache_root):
            return 2
        tail = _common_run_args(
            matrix.RUN_MANIFESTS[cell], arm, w, h,
            jobs=k_jobs[family]["k_jobs"], queue_depth=w,
            cache_root=cache_root, out_dir=out_dir, ceiling_mib=ceiling_mib)
        rc = _run_child(tail, lease_id,
                        f"confirm:{arm}:{cell}:{strongest['cfgkey']}",
                        CFGKEY_TIMEOUT_S)
        if rc != 0:
            print(f"[s01-run] STOP: confirmation {family}/{arm} rc={rc}")
            return rc if rc != 0 else 1
        launched += 1
    print(f"[s01-run] confirm arm {arm} complete: {launched} batches")
    return 0


def _confirm_matches(selections):
    """Confirm sessions re-derived from the SAME strongest_feasible_b0
    selections stage_confirm consumed.  _collect_results() emits only P1
    plan rows (cell/W*H/batch_*), so confirm dirs must be derived here
    from the selections -- a "/confirm_" filter over plan rows can never
    match (review D1)."""
    confirm = []
    for arm in matrix.ARMS:
        for family, cell in matrix.SWEEP_WORKLOADS.items():
            strongest = selections[family]["strongest_feasible_b0"]
            if not strongest:
                return None
            out_dir = f"{RUNROOT}/{cell}/confirm_{strongest['cfgkey']}/{arm}"
            session = _read_session(f"{out_dir}/session.json")
            if session is None:
                continue
            confirm.append({"family": family, "cell": cell, "arm": arm,
                            "cfgkey": strongest["cfgkey"],
                            "out_dir": out_dir, "session": session})
    return confirm


def stage_records(lease_id):
    log = load_lease_log()
    if log is None or _log_has_pending_crash(log):
        print("[s01-run] REFUSED: lease not initialized or has pending "
              "crashed windows")
        return 2
    results = _collect_results()
    valid = [r for r in results if _session_valid(r["session"])]
    if len(valid) != matrix.TOTAL_BATCHES:
        print(f"[s01-run] REFUSED: records requires all "
              f"{matrix.TOTAL_BATCHES} P1 batches valid (have {len(valid)})")
        return 2
    selections = _family_selections()
    confirm = _confirm_matches(selections)
    if (confirm is None
            or len(confirm) != 2 * len(matrix.SWEEP_WORKLOADS)
            or not all(_session_valid(c["session"]) for c in confirm)):
        print("[s01-run] REFUSED: confirmation passes incomplete")
        return 2
    configurations_path = f"{EVIDENCE}/configurations.json"
    if os.path.exists(configurations_path):
        # interim custody pin BEFORE the overwrite (census item 6)
        prior_sha = sha256_file(configurations_path)
        shutil.copy2(configurations_path,
                     f"{DATA}/custody/interim_{prior_sha[:8]}_"
                     f"s01_configurations.json")
        with open(configurations_path) as f:
            configurations = json.load(f)
    else:
        return 2
    configurations["record"] = "configurations_final"
    configurations["stage"] = "records"
    configurations["p1_batches"] = [
        {
            "family": r["family"], "cell": r["cell"], "arm": r["arm"],
            "cfgkey": r["cfgkey"], "batch": r["batch"],
            "out_dir": r["out_dir"],
            "status": r["session"].get("status"),
            "throughput_primary_jobs_per_s": (r["session"].get("throughput")
                                              or {}).get(
                "primary_jobs_per_s"),
            "warm_batch_application_wall_s": (
                r["session"].get("times_ns", {}).get(
                    "warm_batch_application_wall_s")),
            "validated_batch_wall_s": (r["session"].get("times_ns", {})
                                       .get("validated_batch_wall_s")),
            "producer_wait_total_ns": (r["session"].get("dispatch") or {})
            .get("producer_wait_total_ns"),
            "peak_process_tree": (r["session"].get("process_tree_memory", {})
                                  .get("peak")),
        }
        for r in results if "/confirm_" not in r["out_dir"]
    ]
    configurations["confirmation_batches"] = [
        {"family": r["family"], "cell": r["cell"], "arm": r["arm"],
         "cfgkey": r["cfgkey"], "out_dir": r["out_dir"],
         "throughput_primary_jobs_per_s": (r["session"].get("throughput")
                                           or {}).get(
             "primary_jobs_per_s")}
        for r in confirm]
    configurations["selections"] = selections
    with open(configurations_path, "w") as f:
        json.dump(configurations, f, indent=1)

    # memory.json: per-cell process-tree budgets (TASKS output contract)
    memory = {"schema_version": 1, "task": "S01",
              "record": "per_cell_process_tree_budgets", "cells": {}}
    for r in results:
        peak = (r["session"].get("process_tree_memory", {}).get("peak"))
        ceiling_mib = (r["session"].get("requested_config", {})
                       .get("memory_budget_mib"))
        key = f"{r['cell']}/{r['arm']}/{r['cfgkey']}"
        memory["cells"][key] = {
            "peak": peak, "ceiling_mib": ceiling_mib,
            "fits": bool(peak and ceiling_mib)}
    with open(f"{EVIDENCE}/memory.json", "w") as f:
        json.dump(memory, f, indent=1)
    print("[s01-run] records complete: configurations.json + memory.json "
          "written (the independent reviewer's output file is intentionally "
          "absent from every write site in this supervisor)")
    return 0


def stage_status():
    log = load_lease_log()
    if log is None:
        print("[s01-run] STATUS: lease not initialized")
        return 0
    pending = [w for w in log["windows"] if w.get("end_utc") is None]
    print(f"[s01-run] STATUS: lease {log['lease_id']} windows="
          f"{len(log['windows'])} pending_crash={len(pending)} "
          f"charged={log['cumulative_charged_s']}s "
          f"remaining={log['remaining_s']}s "
          f"(cap {log['s01_cap_s']}s, floor {log['reserve_floor_s']}s)")
    for cell in sorted(os.listdir(RUNROOT)) if os.path.exists(RUNROOT) else []:
        print(f"  {cell}/")
    return 0


def main(argv=None):
    if sys.prefix != os.path.dirname(os.path.dirname(CAMPAIGN_PY)):
        print(f"[s01-run] REFUSED: run under the campaign venv "
              f"({CAMPAIGN_PY}), got prefix {sys.prefix}")
        return 2
    parser = argparse.ArgumentParser(
        description="S01 supervisor (task #36)")
    parser.add_argument("--stage", required=True,
                        choices=["init", "p0", "p1", "confirm", "records",
                                 "status"])
    parser.add_argument("--arm", choices=["b0", "selected", "all"],
                        default=None)
    parser.add_argument("--lease-id", default=None)
    args = parser.parse_args(argv)

    lease_id = args.lease_id or f"S01-lease-{time.strftime('%Y%m%d')}"
    if args.stage in ("p0", "p1", "confirm") and args.arm not in (
            "b0", "selected"):
        print("[s01-run] REFUSED: --arm b0|selected is required for armed "
              "stages")
        return 2

    if args.stage == "init":
        return _run_stage("stage:init", stage_init, lease_id, lease_id)
    if args.stage == "p0":
        return _run_stage("stage:p0", stage_p0, lease_id, lease_id, args.arm)
    if args.stage == "p1":
        return _run_stage("stage:p1", stage_p1, lease_id, lease_id, args.arm)
    if args.stage == "confirm":
        return _run_stage("stage:confirm", stage_confirm, lease_id,
                          lease_id, args.arm)
    if args.stage == "records":
        return _run_stage("stage:records", stage_records, lease_id)
    return stage_status()


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
