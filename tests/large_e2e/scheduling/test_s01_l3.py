"""S01 L3 scheduling tests (task #36) -- sandboxed, no heavy execution.

Every test is module-patched into tmp directories (reviewer probe posture):
no subprocess launch of run.py children, no JIT compilation, no writes
outside tmp_path.  The suite freezes the matrix invariants, the supervisor's
refusal gates, the SS6.3(i) producer-wait span mechanics, and the
review.json independence guarantee (amendment #2 SS4.4).
"""
from __future__ import annotations

import ast
import hashlib
import json
import os
import sys
import threading
import time

import pytest

REPO = "/Users/alansynn/orca/workspaces/una-x/wt-large-e2e"
SCHED = f"{REPO}/tests/large_e2e/scheduling"
HARNESS = f"{REPO}/benchmarks/large_e2e/harness"
for p in (SCHED, HARNESS, f"{REPO}/benchmarks/large_e2e"):
    if p not in sys.path:
        sys.path.insert(0, p)

import matrix  # noqa: E402
import s01_run  # noqa: E402


def _sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_json(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f)


# ---------------------------------------------------------------------------
# Frozen matrix invariants
# ---------------------------------------------------------------------------

def test_matrix_invariants():
    assert matrix.matrix_valid() is True
    assert matrix.accessibility_batches() == 24
    assert matrix.flow_batches() == 4
    assert matrix.total_batches() == 28
    assert (1, 1) in matrix.ACCESSIBILITY_CONFIGS
    assert len(matrix.ACCESSIBILITY_CONFIGS) == 6
    assert len(matrix.FLOW_CONFIGS) == 1
    for w, h in matrix.ACCESSIBILITY_CONFIGS + matrix.FLOW_CONFIGS:
        assert matrix.config_within_c(w, h), (w, h)
    for w, h in matrix.FLOW_CONFIGS:
        assert h >= matrix.K_FIT["default_k"]      # H >= K, no schedule proof
    assert matrix.HOLDOUT_NEVER_SWEPT is True
    assert "O3_HOLDOUT" not in matrix.SWEEP_WORKLOADS.values()
    assert matrix.K_FIT["branch"] == "production_default_fits"
    assert matrix.K_FIT["labelled_profile_used"] is False
    assert matrix.K_FIT["flow_stripes_cli"] == "default"


def test_matrix_pins_shapes():
    assert len(matrix.W00_CHAIN_PIN) == 64
    int(matrix.W00_CHAIN_PIN, 16)
    assert matrix.LEDGER["budget_total_s"] == 14400.0
    assert matrix.LEDGER["w00_charged_s"] == 1991.9
    assert matrix.LEDGER["s01_cap_s"] == 7000.0
    assert matrix.LEDGER["reserve_floor_s"] == 3500.0
    assert matrix.BASE_COMMIT == (
        "6b53a274e8a8d3e7360ab5e1b38f741107ce3358")
    # live W00 log bytes must still match the pin (re-checked, not assumed)
    assert _sha(matrix.LEDGER["w00_final_log"]) == matrix.W00_CHAIN_PIN


def test_k_jobs_divisibility_helper():
    assert matrix.largest_w("accessibility") == 9
    assert matrix.largest_w("flow") == 1
    assert matrix.k_jobs_divisible(9, "accessibility") is True
    assert matrix.k_jobs_divisible(18, "accessibility") is True
    assert matrix.k_jobs_divisible(8, "accessibility") is False
    assert matrix.k_jobs_divisible(7, "flow") is True


def test_run_manifests_exist_and_sha_pinned():
    for cell, path in matrix.RUN_MANIFESTS.items():
        assert os.path.exists(path), path
        data = json.load(open(path))
        assert data["analysis"] in ("accessibility", "flow")
        assert data["settings"]["search_radius"] == 500
        for entry in data["input_files"]:
            assert os.path.exists(entry["path"]), entry["path"]
            assert _sha(entry["path"]) == entry["sha256"]
            assert os.path.getsize(entry["path"]) == entry["bytes"]
    flow = json.load(open(matrix.RUN_MANIFESTS["O3_FLOW"]))
    assert flow["settings"]["origins_file"] == "O3_FLOW_origins_sel1024.geojson"
    assert (flow["settings"]["origins_file"] and
            matrix.FLOW_VARIANT["origins_fixture_sha256"] ==
            flow["input_files"][1]["sha256"])


def test_access_fixture_identity():
    """h04 Ruling 2: bounded 1,024-origin fixture, provenance pinned, and the
    superseded sel_all selection must not survive anywhere workload-facing."""
    acc = json.load(open(matrix.RUN_MANIFESTS["O3_ACCESS"]))
    fixture = matrix.ACCESS_ORIGINS["origins_fixture"]
    pin = matrix.ACCESS_ORIGINS["origins_fixture_sha256"]
    assert os.path.exists(fixture), fixture
    assert _sha(fixture) == pin                       # live bytes == matrix pin
    assert acc["settings"]["origins_file"] == os.path.basename(fixture)
    assert pin == acc["input_files"][1]["sha256"]     # manifest consumes the fixture
    assert os.path.getsize(fixture) == acc["input_files"][1]["bytes"]
    # provenance pins re-verified against live bytes, not assumed
    prov = matrix.ACCESS_ORIGINS["provenance"]
    assert _sha(prov["parent_manifest"]) == prov["parent_manifest_sha256"]
    assert _sha(prov["derivation_record"]) == prov["derivation_record_sha256"]
    custody_pin = os.path.join(os.path.dirname(fixture), "custody",
                               "interim_70a1083f_s01_O3_ACCESS_origins_"
                               "sel1024.geojson")
    assert os.path.exists(custody_pin), custody_pin
    assert _sha(custody_pin) == pin                   # custody == live fixture
    # supersession: sel_all must not survive as the selection or origins_file
    assert "bounded fixture" in matrix.ACCESS_ORIGINS["selection"]
    assert "sel_all" not in acc["settings"]["origins_file"]
    assert matrix.ACCESS_ORIGINS["provenance"]["q10_transfer_note"]


# ---------------------------------------------------------------------------
# review.json independence (amendment #2 SS4.4)
# ---------------------------------------------------------------------------

def test_no_review_json_write_site_in_supervisor():
    tree = ast.parse(open(f"{HARNESS}/s01_run.py").read())
    offenders = [node.value for node in ast.walk(tree)
                 if isinstance(node, ast.Constant)
                 and isinstance(node.value, str)
                 and "review.json" in node.value.lower()]
    assert offenders == []
    assert "review.json" not in open(f"{HARNESS}/s01_run.py").read()


# ---------------------------------------------------------------------------
# init gates (sandboxed)
# ---------------------------------------------------------------------------

@pytest.fixture()
def sandbox(tmp_path, monkeypatch):
    """Patch every absolute surface of s01_run into tmp_path."""
    parent = tmp_path / "w00_lease_log.json"
    parent.write_bytes(
        open(matrix.LEDGER["w00_final_log"], "rb").read())
    control = tmp_path / "control.json"
    _write_json(str(control), {"selected_source": {"commit":
                                                   matrix.BASE_COMMIT}})
    monkeypatch.setattr(matrix, "LEDGER",
                        dict(matrix.LEDGER, w00_final_log=str(parent)))
    monkeypatch.setattr(s01_run, "LEASE_LOG", str(tmp_path / "s01_lease_log.json"))
    monkeypatch.setattr(s01_run, "RUNROOT", str(tmp_path / "s01_run"))
    monkeypatch.setattr(s01_run, "EVIDENCE", str(tmp_path / "evidence"))
    monkeypatch.setattr(s01_run, "DATA", str(tmp_path))   # custody pins tmp-only
    monkeypatch.setattr(s01_run, "NBCROOT", str(tmp_path / "nbc"))
    (tmp_path / "custody").mkdir(exist_ok=True)   # pin target, W00 parity
    monkeypatch.setattr(s01_run, "CONTROL", str(control))
    monkeypatch.setattr(s01_run, "PINS", {})
    monkeypatch.setattr(s01_run, "O2_MANIFEST", str(parent))
    # the sandboxed parent IS a byte-copy of the real W00 log, so the O2
    # pilot-manifest pin must follow the copy's true sha
    monkeypatch.setattr(s01_run, "O2_MANIFEST_SHA256",
                        _sha(str(parent)))
    return tmp_path


def test_init_happy_path_chains_w00(sandbox):
    rc = s01_run.stage_init("S01-lease-test")
    assert rc == 0
    log = json.load(open(sandbox / "s01_lease_log.json"))
    assert log["opening_balance_s"] == 1991.9
    assert log["cumulative_charged_s"] == 1991.9
    assert log["remaining_s"] == 12408.1
    assert log["parent_log_sha256"] == matrix.W00_CHAIN_PIN
    assert log["windows"] == []
    assert os.path.isdir(sandbox / "s01_run")


def test_init_refuses_double_init(sandbox):
    (sandbox / "s01_lease_log.json").write_text("{}")
    assert s01_run.stage_init("S01-lease-test") == 2


def test_init_refuses_w00_pin_mismatch(sandbox):
    parent = sandbox / "w00_lease_log.json"
    payload = json.loads(parent.read_text())
    payload["cumulative_charged_s"] = 1992.0
    parent.write_text(json.dumps(payload))
    assert s01_run.stage_init("S01-lease-test") == 2
    assert not os.path.exists(sandbox / "s01_lease_log.json")


def test_init_refuses_wrong_charged(sandbox):
    # right bytes are impossible to fake cheaply here; instead right sha via
    # untouched copy but patched pin value
    monkey_sha = "0" * 64
    import matrix as m
    saved = m.W00_CHAIN_PIN
    try:
        m.W00_CHAIN_PIN = monkey_sha
        assert s01_run.stage_init("S01-lease-test") == 2
    finally:
        m.W00_CHAIN_PIN = saved


def test_init_refuses_wrong_control_base(sandbox):
    control = sandbox / "control.json"
    _write_json(str(control), {"selected_source": {"commit": "0" * 40}})
    assert s01_run.stage_init("S01-lease-test") == 2


def test_init_refuses_pin_mismatch(sandbox):
    real = f"{HARNESS}/pool.py"
    saved = dict(s01_run.PINS)
    try:
        s01_run.PINS = {real: "f" * 64}
        assert s01_run.stage_init("S01-lease-test") == 2
    finally:
        s01_run.PINS.clear()
        s01_run.PINS.update(saved)


def test_init_refuses_existing_runroot(sandbox):
    (sandbox / "s01_run").mkdir()
    assert s01_run.stage_init("S01-lease-test") == 2


# ---------------------------------------------------------------------------
# P1 tranche gates (sandboxed; no child ever launches)
# ---------------------------------------------------------------------------

def _primed_sandbox(sandbox, monkeypatch, charged=2000.0):
    log = {
        "lease_id": "S01-lease-test",
        "opening_balance_s": 1991.9,
        "cumulative_charged_s": charged,
        "remaining_s": round(14400.0 - charged, 1),
        "s01_cap_s": 7000.0,
        "reserve_floor_s": 3500.0,
        "windows": [
            {"what": "stage:init", "class": "stage", "end_utc": "x",
             "wall_s": 0.0, "rc": 0},
        ],
    }
    _write_json(sandbox / "s01_lease_log.json", log)
    evidence = sandbox / "evidence"
    k_jobs = {
        "accessibility": {"k_jobs": 9, "basis_cell": "O3_ACCESS",
                          "warm_job_wall_s_basis": 20.0, "largest_w": 9},
        "flow": {"k_jobs": 9, "basis_cell": "O3_FLOW",
                 "warm_job_wall_s_basis": 20.0, "largest_w": 1},
    }
    _write_json(str(evidence / "configurations.json"), {
        "schema_version": 1, "task": "S01", "stage": "p0_close",
        "p0": {"k_jobs": k_jobs, "queue_2w_criterion": {
            "accessibility": {"triggered": False},
            "flow": {"triggered": False}}}})
    return k_jobs


def test_p1_refuses_without_p0_record(sandbox, monkeypatch):
    _primed_sandbox(sandbox, monkeypatch)
    os.remove(sandbox / "evidence" / "configurations.json")
    assert s01_run.stage_p1("S01-lease-test", "b0") == 2


def test_p1_refuses_queue_trigger(sandbox, monkeypatch):
    primed = _primed_sandbox(sandbox, monkeypatch)
    path = sandbox / "evidence" / "configurations.json"
    cfg = json.load(open(path))
    cfg["p0"]["queue_2w_criterion"]["accessibility"]["triggered"] = True
    _write_json(str(path), cfg)
    assert s01_run.stage_p1("S01-lease-test", "b0") == 2


def test_p1_refuses_residue_without_session(sandbox, monkeypatch):
    _primed_sandbox(sandbox, monkeypatch)
    row = s01_run._expected_batch_dirs("b0")[0]
    os.makedirs(row["out_dir"])
    assert s01_run.stage_p1("S01-lease-test", "b0") == 2


def test_p1_refuses_invalid_retained_session(sandbox, monkeypatch):
    _primed_sandbox(sandbox, monkeypatch)
    row = s01_run._expected_batch_dirs("b0")[0]
    _write_json(f"{row['out_dir']}/session.json",
                {"status": "failed", "counts": {"validated": 0}})
    assert s01_run.stage_p1("S01-lease-test", "b0") == 2


def test_p1_tranche_cap_gate(sandbox, monkeypatch):
    # charged such that charged + 1.4*projected > 7,000 cap
    # projected (b0): access 9*20*(1/.5+.333...) computed from plan; force
    # with charged near cap
    _primed_sandbox(sandbox, monkeypatch, charged=6900.0)
    assert s01_run.stage_p1("S01-lease-test", "b0") == 2


def test_p1_budget_gates_refuse(sandbox, monkeypatch):
    # At the frozen constants the 7,000 s cap gate shadows the 3,500 s
    # reserve-floor gate (cap fires at charged+1.4*proj > 7000 before the
    # floor's charged+proj > 10,900); both checks are enforced in
    # stage_p1 and this asserts the tranche gate refuses either way.
    _primed_sandbox(sandbox, monkeypatch, charged=11000.0)
    assert s01_run.stage_p1("S01-lease-test", "b0") == 2


def test_p1_all_done_is_noop(sandbox, monkeypatch):
    _primed_sandbox(sandbox, monkeypatch)
    for row in s01_run._expected_batch_dirs("b0"):
        _write_json(f"{row['out_dir']}/session.json",
                    {"status": "valid", "counts": {"validated": 9}})
    before = (sandbox / "s01_lease_log.json").read_bytes()
    assert s01_run.stage_p1("S01-lease-test", "b0") == 0
    after = (sandbox / "s01_lease_log.json").read_bytes()
    assert before == after          # no child launched, no window appended


def test_p1_count_deviation_refusal(sandbox, monkeypatch):
    _primed_sandbox(sandbox, monkeypatch)
    saved = matrix.TOTAL_BATCHES
    try:
        matrix.TOTAL_BATCHES = 26
        assert s01_run.stage_p1("S01-lease-test", "b0") == 2
    finally:
        matrix.TOTAL_BATCHES = saved


# ---------------------------------------------------------------------------
# Admission guard (pressure / budget) refuses BEFORE launch
# ---------------------------------------------------------------------------

class _FakeVM:
    def __init__(self, available):
        self.available = available


def test_pressure_guard_refuses_before_launch(sandbox, monkeypatch):
    log = {
        "lease_id": "S01-lease-test", "opening_balance_s": 1991.9,
        "cumulative_charged_s": 2000.0, "remaining_s": 12400.0,
        "windows": [],
    }
    _write_json(sandbox / "s01_lease_log.json", log)
    import psutil
    monkeypatch.setattr(psutil, "virtual_memory",
                        lambda: _FakeVM(matrix.PRESSURE_STOP_BYTES - 1))
    rc = s01_run.guard_and_launch(
        ["true"], {}, "S01-lease-test", "probe", 5)
    assert rc == 2
    stored = json.load(open(sandbox / "s01_lease_log.json"))
    entry = stored["windows"][-1]
    assert entry["rc"] == 2 and entry["end_utc"] is not None
    assert "pressure stop" in entry["refused"]


def test_budget_guard_refuses_when_exhausted(sandbox, monkeypatch):
    log = {
        "lease_id": "S01-lease-test", "opening_balance_s": 1991.9,
        "cumulative_charged_s": 14400.0, "remaining_s": 0.0,
        "windows": [],
    }
    _write_json(sandbox / "s01_lease_log.json", log)
    rc = s01_run.guard_and_launch(
        ["true"], {}, "S01-lease-test", "probe", 5)
    assert rc == 2


# ---------------------------------------------------------------------------
# SS6.3(i) producer-wait spans (real queue plumbing, no processes)
# ---------------------------------------------------------------------------

class _StubSpec:
    def __init__(self, queue_depth, timeout_s):
        self.workers = 1
        self.queue_depth = queue_depth
        self.timeout_s = timeout_s

    def result_queue_maxsize(self):
        return 4


def test_producer_wait_span_recorded_per_job():
    from harness.pool import WorkerPool
    pool = WorkerPool(_StubSpec(1, 30.0), {}, {})

    def release():
        time.sleep(0.4)
        pool.job_queue.get_nowait()          # unblock the producer put
        pool.result_queue.put({"type": "job_result", "job_id": 0})

    pool.job_queue.put({"warmup_dummy": True})
    thread = threading.Thread(target=release)
    thread.start()
    try:
        summary = pool.dispatch_and_drain([{"job_id": 0}])
    finally:
        thread.join(timeout=10)
    waits = summary["producer_wait_ns_by_job"]
    assert waits[0] > 2 * 10**8              # the induced ~0.4 s block
    assert summary["producer_wait_total_ns"] == waits[0]
    assert summary["producer_wait_nonzero_jobs"] == 1
    assert "never wall deltas" in summary["producer_wait_semantics"]
    assert pool.retained_results[0]["job_id"] == 0


def test_producer_wait_zero_when_queue_free():
    from harness.pool import WorkerPool
    pool = WorkerPool(_StubSpec(4, 30.0), {}, {})
    pool.result_queue.put({"type": "job_result", "job_id": 0})
    summary = pool.dispatch_and_drain([{"job_id": 0}])
    # an unblocked put is sub-millisecond-ish; the measured span may still
    # be nonzero from first-put queue/feeder setup, so bound the magnitude
    # rather than assert exact zero
    assert summary["producer_wait_ns_by_job"][0] < 50 * 10**6
    assert summary["producer_wait_total_ns"] == \
        summary["producer_wait_ns_by_job"][0]


# ---------------------------------------------------------------------------
# Child argv: exact flag surface accepted by the frozen spec parser
# ---------------------------------------------------------------------------

def test_child_flags_parse_against_spec_parser():
    from harness.spec import build_parser
    for arm in matrix.ARMS:
        for row in s01_run._expected_batch_dirs(arm):
            argv = s01_run._common_run_args(
                matrix.RUN_MANIFESTS[row["cell"]], arm, row["w"], row["h"],
                jobs=9, queue_depth=row["w"],
                cache_root=row["cache_root"], out_dir=row["out_dir"],
                ceiling_mib=8000)
            assert len(argv) % 2 == 0
            args = build_parser().parse_args(argv)
            assert args.workers == row["w"]
            assert args.numba_threads == row["h"]
            assert args.queue_depth == row["w"]          # queue starts W
            assert args.writer_limit == row["w"]         # writer limit = W
            assert args.cpu_budget == matrix.C_SLOTS
            assert args.flow_stripes is None             # production default
            assert args.mode == "batch"
            assert args.arm == f"s01_{arm}"


def test_status_stage_is_readonly(sandbox):
    log = {
        "lease_id": "S01-lease-test", "opening_balance_s": 1991.9,
        "cumulative_charged_s": 2000.0, "remaining_s": 12400.0,
        "s01_cap_s": 7000.0, "reserve_floor_s": 3500.0, "windows": [],
    }
    _write_json(sandbox / "s01_lease_log.json", log)
    before = (sandbox / "s01_lease_log.json").read_bytes()
    assert s01_run.stage_status() == 0
    assert (sandbox / "s01_lease_log.json").read_bytes() == before


def test_arm_required_for_armed_stages(capsys):
    rc = s01_run.main(["--stage", "p1"])
    assert rc == 2
    assert "--arm" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# confirm / records / P0-derivation coverage (review C1; sandboxed, no
# child ever launches -- _run_child and memory_ceiling_mib are stubbed)
# ---------------------------------------------------------------------------

def _session(tp, wall, wait_ns=10**7, valid=True, validated=9):
    return {
        "status": "valid" if valid else "failed",
        "counts": {"validated": validated},
        "throughput": {"primary_jobs_per_s": tp},
        "times_ns": {"warm_batch_application_wall_s": wall,
                     "validated_batch_wall_s": wall},
        "process_tree_memory": {"peak": {"tree_rss_bytes_sum": 10**9}},
        "requested_config": {"memory_budget_mib": 8000},
        "dispatch": {"producer_wait_total_ns": wait_ns},
    }


TP_B0 = {"W1H1": [10.0, 12.0], "W2H2": [20.0, 24.0], "W3H3": [30.0, 33.0],
         "W4H2": [5.0, 5.0], "W2H4": [3.0, 3.0], "W9H1": [1.0, 1.0],
         "W1H9": [7.0, 9.0]}
TP_SEL = {"W1H1": [11.0, 13.0], "W2H2": [21.0, 25.0], "W3H3": [31.0, 34.0],
          "W4H2": [6.0, 6.0], "W2H4": [4.0, 4.0], "W9H1": [2.0, 2.0],
          "W1H9": [8.0, 10.0]}


def _populate_p1(sandbox, tp_b0=TP_B0, tp_sel=TP_SEL, invalid=()):
    for arm, tps in (("b0", tp_b0), ("selected", tp_sel)):
        for row in s01_run._expected_batch_dirs(arm):
            pair = tps[row["cfgkey"]]
            tp = pair[(row["batch"] - 1) % len(pair)]
            payload = _session(tp, 20.0)
            if (arm, row["cfgkey"], row["batch"]) in invalid:
                payload = _session(tp, 20.0, valid=False)
            _write_json(f"{row['out_dir']}/session.json", payload)


def _seed_confirm_dirs(sandbox, cfg_by_cell, arms=matrix.ARMS, drop=()):
    for arm in arms:
        for cell, cfg in cfg_by_cell.items():
            out = f"{s01_run.RUNROOT}/{cell}/confirm_{cfg}/{arm}"
            if (arm, cell) in drop:
                continue
            _write_json(f"{out}/session.json", _session(40.0, 20.0))


def test_family_selections_median_and_feasibility(sandbox):
    _primed_sandbox(sandbox, None)
    _populate_p1(sandbox, invalid={("selected", "W2H2", 1)})
    sel = s01_run._family_selections()
    b0 = sel["accessibility"]
    assert b0["strongest_feasible_b0"]["cfgkey"] == "W3H3"
    assert b0["strongest_feasible_b0"]["median_primary_jobs_per_s"] == 31.5
    w2h2 = [t for t in sel["accessibility"]["selected_ranking"]
            if t["cfgkey"] == "W2H2"][0]
    assert w2h2["feasible"] is False            # one invalid batch
    assert w2h2["median_primary_jobs_per_s"] == 23.0
    ranks = [t["median_primary_jobs_per_s"]
             for t in sel["accessibility"]["b0_ranking"]]
    assert ranks == sorted(ranks, reverse=True)
    assert sel["flow"]["strongest_feasible_b0"]["cfgkey"] == "W1H9"


def test_records_success_with_confirm_dirs(sandbox, monkeypatch):
    _primed_sandbox(sandbox, monkeypatch)
    _populate_p1(sandbox)
    _seed_confirm_dirs(sandbox, {"O3_ACCESS": "W3H3", "O3_FLOW": "W1H9"})
    prior_sha = _sha(f"{s01_run.EVIDENCE}/configurations.json")
    assert s01_run.stage_records("S01-lease-test") == 0
    cfg = json.load(open(f"{s01_run.EVIDENCE}/configurations.json"))
    assert cfg["record"] == "configurations_final"
    assert cfg["stage"] == "records"
    assert len(cfg["p1_batches"]) == 28
    assert len(cfg["confirmation_batches"]) == 4
    assert {b["cfgkey"] for b in cfg["confirmation_batches"]} == \
        {"W3H3", "W1H9"}
    assert {b["arm"] for b in cfg["confirmation_batches"]} == set(matrix.ARMS)
    assert "selections" in cfg
    mem = json.load(open(f"{s01_run.EVIDENCE}/memory.json"))
    # per-CONFIG table: 7 configs x 2 arms; the 2 batches per config share
    # one key by design (cell/arm/cfgkey)
    assert len(mem["cells"]) == 14
    # custody: interim pin of the PRIOR p0_close bytes before the rewrite
    pins = os.listdir(os.path.join(s01_run.DATA, "custody"))
    assert any(p.startswith(f"interim_{prior_sha[:8]}_s01_configurations")
               for p in pins), pins


def test_records_refuses_missing_confirm_session(sandbox, monkeypatch):
    _primed_sandbox(sandbox, monkeypatch)
    _populate_p1(sandbox)
    _seed_confirm_dirs(sandbox, {"O3_ACCESS": "W3H3", "O3_FLOW": "W1H9"},
                       drop={("selected", "O3_FLOW")})
    before = open(f"{s01_run.EVIDENCE}/configurations.json", "rb").read()
    assert s01_run.stage_records("S01-lease-test") == 2
    # refusal leaves the P0 record byte-identical (no rewrite on refuse)
    assert open(f"{s01_run.EVIDENCE}/configurations.json", "rb").read() == \
        before


def test_records_refuses_invalid_confirm_session(sandbox, monkeypatch):
    _primed_sandbox(sandbox, monkeypatch)
    _populate_p1(sandbox)
    _seed_confirm_dirs(sandbox, {"O3_ACCESS": "W3H3", "O3_FLOW": "W1H9"})
    bad = (f"{s01_run.RUNROOT}/O3_FLOW/confirm_W1H9/selected/session.json")
    _write_json(bad, _session(40.0, 20.0, valid=False))
    assert s01_run.stage_records("S01-lease-test") == 2


def test_records_refuses_pending_crash(sandbox, monkeypatch):
    _primed_sandbox(sandbox, monkeypatch)
    _populate_p1(sandbox)
    _seed_confirm_dirs(sandbox, {"O3_ACCESS": "W3H3", "O3_FLOW": "W1H9"})
    path = sandbox / "s01_lease_log.json"
    log = json.load(open(path))
    log["windows"].append({"what": "p1:b0:O3_ACCESS:W1H1:b1",
                           "class": "crashed_child_start",
                           "end_utc": None, "wall_s": None, "rc": None})
    _write_json(str(path), log)
    assert s01_run.stage_records("S01-lease-test") == 2


def test_confirm_selects_strongest_b0_and_names_dirs(sandbox, monkeypatch):
    _primed_sandbox(sandbox, monkeypatch)
    _populate_p1(sandbox)
    calls = []

    def stub(args_tail, lease_id, what, timeout_s, env_extra=None):
        out = args_tail[args_tail.index("--out") + 1]
        _write_json(f"{out}/session.json", _session(40.0, 20.0))
        calls.append({"what": what, "tail": list(args_tail)})
        return 0

    monkeypatch.setattr(s01_run, "_run_child", stub)
    monkeypatch.setattr(s01_run, "memory_ceiling_mib",
                        lambda: (8000, 8 * 2**30))
    assert s01_run.stage_confirm("S01-lease-test", "b0") == 0
    assert [c["what"] for c in calls] == [
        "confirm:b0:O3_ACCESS:W3H3", "confirm:b0:O3_FLOW:W1H9"]
    for c in calls:
        cell = c["what"].split(":")[2]
        t = c["tail"]
        out = t[t.index("--out") + 1]
        assert "/confirm_" in out and out.endswith("/b0")
        assert t[t.index("--cache-root") + 1] == \
            f"{s01_run.NBCROOT}/nbc_s01_confirm_b0_{cell}"
        assert t[t.index("--jobs") + 1] == "9"   # k_jobs from the P0 record
    acc = [c for c in calls if "O3_ACCESS" in c["what"]][0]["tail"]
    assert acc[acc.index("--workers") + 1] == "3"
    assert acc[acc.index("--numba-threads") + 1] == "3"
    assert acc[acc.index("--queue-depth") + 1] == "3"
    flow = [c for c in calls if "O3_FLOW" in c["what"]][0]["tail"]
    assert flow[flow.index("--workers") + 1] == "1"
    assert flow[flow.index("--numba-threads") + 1] == "9"
    assert flow[flow.index("--queue-depth") + 1] == "1"


def test_confirm_requires_all_p1_valid(sandbox, monkeypatch):
    _primed_sandbox(sandbox, monkeypatch)
    _populate_p1(sandbox, invalid={("b0", "W1H1", 1)})
    assert s01_run.stage_confirm("S01-lease-test", "b0") == 2


def _p0_pilot_stub(calls, wait_ns_access=10**7):
    walls = {"pilot:O3_ACCESS:1": 3.0, "pilot:O3_ACCESS:2": 3.6,
             "pilot:O3_FLOW:1": 5.0, "pilot:O3_FLOW:2": 5.0,
             "pilot:O2:1": 30.0, "pilot:O2:2": 30.0}

    def stub(args_tail, lease_id, what, timeout_s, env_extra=None):
        out = args_tail[args_tail.index("--out") + 1]
        wait = wait_ns_access if "O3_ACCESS" in what else 10**7
        _write_json(f"{out}/session.json",
                    _session(1.0, walls[what], wait_ns=wait))
        calls.append(what)
        return 0

    return stub


def test_p0_k_jobs_derivation_and_criterion(sandbox, monkeypatch):
    assert s01_run.stage_init("S01-lease-test") == 0
    calls = []
    monkeypatch.setattr(s01_run, "_run_child", _p0_pilot_stub(calls))
    monkeypatch.setattr(s01_run, "memory_ceiling_mib",
                        lambda: (8000, 8 * 2**30))
    assert s01_run.stage_p0("S01-lease-test", "b0") == 0
    assert len(calls) == 6
    cfg = json.load(open(f"{s01_run.EVIDENCE}/configurations.json"))
    assert cfg["stage"] == "p0_close"
    kj = cfg["p0"]["k_jobs"]
    # basis = max(3.0, 3.6) = 3.6 s: ceil(60/3.6/9)=2 -> 18, % largest W == 0
    assert kj["accessibility"]["k_jobs"] == 18
    assert kj["accessibility"]["k_jobs"] % 9 == 0
    assert kj["accessibility"]["warm_job_wall_s_basis"] == 3.6
    assert kj["flow"]["k_jobs"] == 12            # ceil(60/5/1) = 12
    assert kj["flow"]["k_jobs"] % matrix.largest_w("flow") == 0
    assert len(cfg["p0"]["pilots"]) == 6
    crit = cfg["p0"]["queue_2w_criterion"]
    assert crit["accessibility"]["triggered"] is False
    assert crit["flow"]["triggered"] is False
    assert crit["accessibility"]["thresholds"][
        "producer_wait_share_max"] == 0.10
    assert cfg["access_origins"]["origins_fixture_sha256"] == \
        matrix.ACCESS_ORIGINS["origins_fixture_sha256"]


def test_p0_queue_trigger_records_then_stops(sandbox, monkeypatch):
    assert s01_run.stage_init("S01-lease-test") == 0
    # pre-seed a prior P0 record: the rewrite must pin its bytes first
    prior_path = f"{s01_run.EVIDENCE}/configurations.json"
    _write_json(prior_path, {"record": "configurations", "prior": True})
    prior_sha = _sha(prior_path)
    calls = []
    monkeypatch.setattr(s01_run, "_run_child", _p0_pilot_stub(
        calls, wait_ns_access=10**9))     # share 1.0 s / 3.6 s ~ 0.28 > 0.10
    monkeypatch.setattr(s01_run, "memory_ceiling_mib",
                        lambda: (8000, 8 * 2**30))
    assert s01_run.stage_p0("S01-lease-test", "b0") == 2
    cfg = json.load(open(prior_path))
    assert cfg["p0"]["queue_2w_criterion"]["accessibility"]["triggered"] \
        is True
    assert cfg["p0"]["queue_2w_criterion"]["flow"]["triggered"] is False
    pins = os.listdir(os.path.join(s01_run.DATA, "custody"))
    assert any(p.startswith(f"interim_{prior_sha[:8]}_s01_configurations")
               for p in pins), pins
    # and P1 must now refuse on the recorded trigger
    assert s01_run.stage_p1("S01-lease-test", "b0") == 2


def test_p0_refuses_non_b0_arm(sandbox):
    assert s01_run.stage_init("S01-lease-test") == 0
    assert s01_run.stage_p0("S01-lease-test", "selected") == 2


# ---------------------------------------------------------------------------
# D4 regression: memory_ceiling_mib unit collision (review D4, 2026-09-29).
# The bug shipped int(min(CAP_MiB, FRAC * avail_bytes) / 2^20): the MiB
# constant always won the min() against a bytes-scale fraction, so the real
# function returned 0 MiB at every realistic avail and the SS6.3 queue-2W
# memory criterion ran fail-open (_max_live_memory_frac skips falsy
# ceilings).  All three tests invoke the REAL function -- only its
# environment (psutil.virtual_memory) is patched, never the function.
# ---------------------------------------------------------------------------

def test_memory_ceiling_live_self_consistency():
    """T1: at live avail, the real function equals the intended formula over
    the avail it returned, and is at least 1 MiB."""
    mib, avail = s01_run.memory_ceiling_mib()
    assert mib == min(matrix.CEILING_MIB_CAP << 20,
                      int(matrix.CEILING_FRACTION * avail)) // (1 << 20)
    assert mib >= 1


def test_memory_ceiling_agrees_with_guard_expression(monkeypatch):
    """T2: the supervisor ceiling and the guard_and_launch ceiling
    (s01_run.py :169-170) must agree BY TEST at every avail, including the
    cap/fraction crossover, so the two expressions cannot drift apart."""
    import psutil
    cross = (matrix.CEILING_MIB_CAP << 20) / matrix.CEILING_FRACTION
    avail_list = sorted({10 ** e for e in range(3, 13)}
                        | {int(cross) + d for d in range(-50, 51, 7)}
                        | {int(cross) - 1, int(cross), int(cross) + 1}
                        | {0, 1 << 20, matrix.PRESSURE_STOP_BYTES,
                           32 * 2 ** 30, 1 << 40})
    assert {int(cross) - 1, int(cross), int(cross) + 1} <= set(avail_list)
    for avail in avail_list:
        monkeypatch.setattr(psutil, "virtual_memory",
                            lambda a=avail: _FakeVM(a))
        mib, got = s01_run.memory_ceiling_mib()
        guard = min(matrix.CEILING_MIB_CAP << 20,
                    int(matrix.CEILING_FRACTION * got))
        assert got == avail, (avail, got)
        assert mib == guard >> 20, (avail, mib, guard)


def test_memory_ceiling_cap_branch_returns_cap(monkeypatch):
    """T3: exact-bug signature -- at a LARGE avail the function must return
    the converted cap branch EXACTLY (the bug returned 0 precisely there;
    T1's live avail exercises only the fraction branch)."""
    import psutil
    monkeypatch.setattr(psutil, "virtual_memory",
                        lambda: _FakeVM(32 * 2 ** 30))
    mib, avail = s01_run.memory_ceiling_mib()
    assert avail == 32 * 2 ** 30
    assert mib == matrix.CEILING_MIB_CAP
    assert mib == 10650
