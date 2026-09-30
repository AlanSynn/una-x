"""The ten required harness mutants (HARNESS.md "Required harness mutants").

Each control runs an intentionally broken candidate and asserts the harness
check FAILS it with the specific typed violation.  Controls 1 and 3 share
one traced engagement-leg run; the rest are fast in-process or bounded
subprocess checks.
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from benchmarks.platform import records as records_mod
from benchmarks.platform.errors import (BatchRunFailed, EngagementError,
                                        IdentityError, ManifestError,
                                        ModelDispatchError,
                                        OutputDirReuseError,
                                        OutputValidationError,
                                        ProvenanceError,
                                        RecordAuditError,
                                        ResourceLedgerError,
                                        SafetyViolation)
from benchmarks.platform.guards import OutputDirGuard, SourceWriteGuard
from benchmarks.platform.identity import resolve_identity, assert_installed
from benchmarks.platform.manifest import load_manifest
from benchmarks.platform.provenance import check_oracle_provenance
from benchmarks.platform.records import validate_record
from benchmarks.platform.runner import (_assert_dispatch,
                                        _assert_engagement,
                                        assert_window_validated,
                                        enforce_ledger, run_bounded_batch,
                                        run_engagement_leg)
from benchmarks.platform.validation import (compare_dirs_bitwise,
                                            validate_output_obligations)

PY = sys.executable


# ---- control 1: --analysis flow invokes accessibility -----------------------
def test_mutant1_wrong_dispatch_fails(legacy_python, base_plan,
                                      frozen_constants, tmp_path):
    plan = dict(base_plan, mutant="wrong_dispatch",
                model=dict(base_plan["model"], analysis="flow"))
    markers = run_engagement_leg(plan, tmp_path / "m1",
                                 frozen_constants["repo"], legacy_python,
                                 timeout_s=540.0)
    with pytest.raises(ModelDispatchError, match="expects UNA.RunFlow"):
        _assert_dispatch({"model": {"analysis": "flow"}}, markers)


# ---- control 2: source checkout shadows installed wheel --------------------
def test_mutant2_source_shadowing_fails(frozen_constants):
    info = resolve_identity(
        PY, extra_env={"PYTHONPATH": frozen_constants["src_root"]})
    with pytest.raises(IdentityError, match="site-packages"):
        assert_installed(info, "urban_network_analysis",
                         forbidden_roots=[frozen_constants["src_root"]])


# ---- control 3: CPU fallback reported native/GPU ---------------------------
def test_mutant3_fake_backend_engagement_fails(legacy_python, base_plan,
                                               frozen_constants, tmp_path):
    plan = dict(base_plan, backend_reported="native")
    markers = run_engagement_leg(plan, tmp_path / "m3",
                                 frozen_constants["repo"], legacy_python,
                                 timeout_s=540.0)
    manifest = {"backend_requested": "native"}
    with pytest.raises(EngagementError,
                       match="no native module loaded"):
        _assert_engagement(manifest, markers)


# ---- control 4: missing output / empty comparison counted as success -------
def test_mutant4_missing_output_and_empty_comparison(tmp_path):
    out = tmp_path / "outputs"
    out.mkdir()
    obligations = {"files": {
        "Results.geojson": {"format": "geojson", "expect_features": 2},
        "Results.feather": {"format": "feather"},
    }}
    with pytest.raises(OutputValidationError, match="missing"):
        validate_output_obligations(out, obligations)

    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(Exception, match="empty comparison"):
        compare_dirs_bitwise(empty, empty, label="mutant4")

    # a failed job counted as success:
    window = {"status": "failed", "failures": ["job 1 exited -9"],
              "validated_successes": 1, "submitted": 2}
    with pytest.raises(BatchRunFailed):
        assert_window_validated(window)


# ---- control 5: wrong workload hash / changed profile before timing --------
def test_mutant5_workload_drift_fails_before_timing(tmp_path, smoke_root):
    fixture = tmp_path / "fixture5"
    from benchmarks.platform.fixtures import build_smoke_fixture
    build_smoke_fixture(fixture)
    man = tmp_path / "m5.json"
    from benchmarks.platform.fixtures import smoke_manifest
    smoke_manifest(man, fixture)
    # freeze, then tamper the workload
    net = fixture / "network.geojson"
    doc = json.loads(net.read_text())
    doc["features"][0]["properties"]["Geometric"] = 12345.0
    net.write_text(json.dumps(doc))
    with pytest.raises(ManifestError, match="changed since freeze"):
        load_manifest(man, verify_inputs=True)


# ---- control 6: child dies with full queues -> bounded fail + cleanup ------
def test_mutant6_child_death_full_queues_bounded(tmp_path):
    work = tmp_path / "m6"
    ok_job = [PY, "-c", "import time; time.sleep(0.6)"]
    killer = [PY, "-c",
              "import os, signal, time; time.sleep(0.3); "
              "os.kill(os.getpid(), signal.SIGKILL)"]
    # queue fills (2 running = queue_depth), then a child dies mid-flight
    factories = [
        lambda i: ok_job,      # 0 slow-ok
        lambda i: killer,      # 1 dies with queue full
        lambda i: ok_job,      # 2 waiting in bounded queue
        lambda i: ok_job,      # 3 never submitted after failure
    ]
    t0 = time.perf_counter()
    window = run_bounded_batch(
        factories, {"max_workers": 2, "queue_depth": 2}, work, str(work),
        deadline_s=30.0)
    elapsed = time.perf_counter() - t0
    assert elapsed < 25.0, "bounded fail took too long (deadlock?)"
    assert window["status"] == "failed"
    assert window["results"][1]["exitcode"] == -signal.SIGKILL
    assert any("exited" in f for f in window["failures"])
    with pytest.raises(BatchRunFailed):
        assert_window_validated(window)
    # cleanup: no orphaned children from this window remain
    assert not _popen_children_alive(work)


def _popen_children_alive(work: Path) -> bool:
    out = subprocess.run([PY, "-c",
                          "import os; print('\\n'.join("
                          "os.listdir('/proc')))"], capture_output=True)
    pids = {int(x) for x in out.stdout.split() if x.isdigit()}
    mine = os.getpid()
    for pid in pids:
        try:
            stat = Path(f"/proc/{pid}/stat").read_bytes()
            ppid = int(stat[stat.rfind(b")") + 2:].split()[1])
        except (OSError, IndexError, ValueError):
            continue
        if ppid == mine:
            # only our own direct children from this test process count;
            # pytest workers/collectors are excluded by command match
            try:
                cmd = Path(f"/proc/{pid}/cmdline").read_bytes()
            except OSError:
                continue
            if b"-c" in cmd and b"sleep" in cmd or b"SIGKILL" in cmd:
                return True
    return False


# ---- control 7: reused output dir / test writes committed evidence --------
def test_mutant7_output_reuse_and_committed_write_guard(tmp_path,
                                                        frozen_constants):
    out = tmp_path / "m7_out"
    guard_a = OutputDirGuard(out, "run-aaa", frozen_constants["repo"])
    guard_a.prepare()
    guard_a.finalize({"status": "complete"})
    guard_b = OutputDirGuard(out, "run-bbb", frozen_constants["repo"])
    with pytest.raises(OutputDirReuseError, match="run-aaa"):
        guard_b.prepare()

    # a run writing a 'committed' file: the guard denies before the write
    tracked = tmp_path / "committed_evidence.json"
    tracked.write_text('{"original": true}\n')
    with SourceWriteGuard(tracked={str(tracked)}):
        with pytest.raises(SafetyViolation, match="committed file"):
            with open(tracked, "a", encoding="utf-8") as f:
                f.write("contamination")
    assert tracked.read_text() == '{"original": true}\n'


# ---- control 8: parent-memory-only / unsynchronized result ----------------
def _complete_record() -> dict:
    return {
        "run_id": "r-x", "harness_version": "h1", "campaign": "c",
        "profile": "una_legacy", "model": {"analysis": "accessibility"},
        "workload_class": "L2", "requested_backend": "reference",
        "effective_backend": "reference", "backend_counters": {},
        "dispatch_called": ["urban_network_analysis.UNA.RunAccessibility"],
        "timings": {"child_wall_s": 1.0, "child_import_s": 0.1,
                    "child_compute_s": 0.8, "engagement_wall_s": 1.1,
                    "launch_to_exit_s": 1.2},
        "cache_state": {"numba_cache": "cold"}, "export_synced": True,
        "outputs": ["Results.geojson"],
        "child_identity": {"python": "/p", "module_file": "/m",
                           "installed_tree_sha256": "abc"},
        "input_hashes": {}, "settings_fingerprint": {},
        "resource_policy": {"max_workers": 1},
        "memory": {"peak_is_simultaneous_tree_sum": True,
                   "samples_taken": 5, "per_sample": [{"t": 0}]},
        "counts": {"completed": 1, "failed": 0, "cancelled": 0},
        "fault_status": "none",
    }


def test_mutant8_thin_measurement_records_fail():
    base = _complete_record()
    validate_record(base)  # sanity: complete record passes

    parent_only = dict(base, memory={"peak_is_simultaneous_tree_sum": False,
                                     "samples_taken": 1,
                                     "per_sample": [{"t": 0}]})
    with pytest.raises(RecordAuditError,
                       match="simultaneous whole-tree sum"):
        validate_record(parent_only)

    no_memory = dict(base)
    no_memory.pop("memory")
    with pytest.raises(RecordAuditError, match="process-tree memory"):
        validate_record(no_memory)

    unsynced = dict(base, export_synced=False)
    with pytest.raises(RecordAuditError, match="synchronization"):
        validate_record(unsynced)

    thin_timings = dict(base, timings={"child_wall_s": 1.0})
    with pytest.raises(RecordAuditError, match="boundaries"):
        validate_record(thin_timings)

    fake_accel = dict(base, requested_backend="native",
                      backend_counters={"native_module_loaded": False})
    with pytest.raises(RecordAuditError, match="without"):
        validate_record(fake_accel)

    dropped_job = dict(base, counts={"completed": 1})
    with pytest.raises(RecordAuditError, match="counts missing"):
        validate_record(dropped_job)


# ---- control 9: oracle generated by candidate / filename-only reuse --------
def test_mutant9_oracle_provenance_fails(frozen_constants):
    cand = {"installed_tree_sha256": "cand-tree", "module_file": "/cand/m",
            "profile": "candidate"}
    ref = {"wheel_tree_sha256": frozen_constants["wheel_tree_sha256"]}

    with pytest.raises(ProvenanceError, match="filename-only"):
        check_oracle_provenance({}, cand, ref)
    with pytest.raises(ProvenanceError, match="missing fields"):
        check_oracle_provenance({"profile": "una_legacy"}, cand, ref)
    with pytest.raises(ProvenanceError, match="different reference build"):
        check_oracle_provenance({
            "producer_python": "/p", "reference_wheel_sha256": "other",
            "reference_source_sha256": "s", "profile": "una_legacy",
            "run_id": "r1", "harness_version": "h1"}, cand, ref)
    # self-oracle: candidate's own tree produced the 'reference'
    with pytest.raises(ProvenanceError, match="candidate"):
        check_oracle_provenance({
            "producer_python": "/p", "reference_wheel_sha256": "cand-tree",
            "reference_source_sha256": "s", "profile": "candidate",
            "run_id": "r1", "harness_version": "h1",
            "producer_run_equals_candidate": True,
            "producer_module_file": "/cand/m"},
            cand, {"wheel_tree_sha256": "cand-tree"})


# ---- control 10: resource swell while claiming the same policy -------------
def test_mutant10_resource_ledger_fails(tmp_path):
    declared = {"max_workers": 1, "queue_depth": 2,
                "memory_limit_bytes": None, "threads_per_worker": 1}
    with pytest.raises(ResourceLedgerError, match="declared"):
        run_bounded_batch([lambda i: [PY, "-c", "pass"]],
                          {"max_workers": 2, "queue_depth": 2}, tmp_path,
                          str(tmp_path), deadline_s=30.0,
                          enforce_policy=True, declared_policy=declared)

    window = {"wall_s": 1, "deadline_s": 30, "submitted": 2,
              "results": [{"job": 0, "exitcode": 0, "t": 0.1},
                          {"job": 1, "exitcode": 0, "t": 0.2}],
              "validated_successes": 2, "failures": [], "peak_inflight": 4,
              "bounded": True, "status": "complete"}
    with pytest.raises(ResourceLedgerError, match="concurrent workers"):
        enforce_ledger(window, declared)

    window["peak_inflight"] = 1
    with pytest.raises(ResourceLedgerError, match="memory_limit"):
        enforce_ledger(window,
                       dict(declared, memory_limit_bytes=1024),
                       sampler_stats={"peak_simultaneous_rss_bytes": 4096})
