"""Pool + fault-injection negative tests (HARNESS.md behaviors 4, 5, 10).

Every case drives the REAL run.py as a subprocess with --test-fault and the
stub engine.  Death faults must fail fast (bounded by --timeout-s), cancel
outstanding owned jobs, and leave ZERO surviving owned processes.  Fault
injection forces measurement_class="harness_selftest", so even the
non-invalidating fault can never produce qualification data.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import stubkit
from stubkit import read_session, run_harness


def run_fault(tmp_path: Path, out_dir: Path, stub_identity, fault: str,
              jobs: int = 2, workers: int = 1, **kwargs):
    manifest = stubkit.make_manifest(tmp_path / "w.json", tmp_path / "inputs")
    kwargs.setdefault("timeout_s", 30.0)
    kwargs.setdefault("timeout", 180.0)
    return run_harness(out_dir, manifest, stub_identity, jobs=jobs,
                       workers=workers, test_fault=fault, **kwargs)


def raw_records(out_dir: Path) -> list[dict]:
    raw = out_dir / "raw"
    records = []
    for path in sorted(raw.glob("job_*.json")):
        records.append(json.loads(path.read_text()))
    return records


# ----------------------------------------------------------------------
# Dispatch/verification invalidity (run executes, then is INVALID)
# ----------------------------------------------------------------------
def test_worker_raise_records_failure_and_invalidates(tmp_path, stub_identity):
    out = tmp_path / "out"
    started = time.monotonic()
    completed = run_fault(tmp_path, out, stub_identity, "worker_raise", jobs=2)
    wall = time.monotonic() - started
    assert completed.returncode == 1, completed.stderr
    session = read_session(out)
    assert session["status"] == "invalid"
    assert session["measurement_class"] == "harness_selftest"
    assert session["qualification_valid"] is False
    assert session["failure"]["reason"] == "job_failures_present"
    # The pool kept draining: BOTH jobs were attempted and each got a
    # rich failure record on disk (behavior 10: records on failure too).
    failed = [r for r in raw_records(out) if r["type"] == "job_failed"]
    assert len(failed) == 2
    assert all("injected fault: worker_raise" in r["error"]["message"]
               for r in failed)
    assert wall < 120  # no hang: the pool survived and shut down cleanly


def test_zero_size_output_rejected_by_worker(tmp_path, stub_identity):
    """A zero-byte placeholder is refused before the worker ever reports it
    as an artifact (and the coordinator re-checks independently)."""
    out = tmp_path / "out"
    completed = run_fault(tmp_path, out, stub_identity, "zero_size_output")
    assert completed.returncode == 1, completed.stderr
    session = read_session(out)
    assert session["status"] == "invalid"
    failed = [r for r in raw_records(out) if r["type"] == "job_failed"]
    assert failed
    assert "zero-size" in failed[0]["error"]["message"]


def test_missing_companion_file_rejected(tmp_path, stub_identity):
    out = tmp_path / "out"
    completed = run_fault(tmp_path, out, stub_identity,
                          "missing_companion_file")
    assert completed.returncode == 1, completed.stderr
    session = read_session(out)
    assert session["status"] == "invalid"
    failed = [r for r in raw_records(out) if r["type"] == "job_failed"]
    assert failed
    assert "missing required companion" in failed[0]["error"]["message"]


def test_altered_bytes_caught_by_coordinator_rehash(tmp_path, stub_identity):
    """The worker hashed honestly, then the bytes changed; the coordinator's
    INDEPENDENT re-hash at validation must catch it."""
    out = tmp_path / "out"
    completed = run_fault(tmp_path, out, stub_identity, "altered_bytes")
    assert completed.returncode == 1, completed.stderr
    session = read_session(out)
    assert session["status"] == "invalid"
    assert session["failure"]["reason"] == "artifact_signature_mismatch"
    # The success record was retained on disk (records on success AND the
    # run is still invalid).
    assert any(r["type"] == "job_result" for r in raw_records(out))


def test_duplicate_job_id_rejected(tmp_path, stub_identity):
    out = tmp_path / "out"
    completed = run_fault(tmp_path, out, stub_identity, "duplicate_job_id",
                          jobs=2)
    assert completed.returncode == 1, completed.stderr
    session = read_session(out)
    assert session["status"] == "invalid"
    assert session["failure"]["reason"] == "duplicate_job_id"
    assert session["failure"]["detail"]["duplicates"] == {"0": 2}


def test_wrong_engine_signature_rejected(tmp_path, stub_identity):
    out = tmp_path / "out"
    completed = run_fault(tmp_path, out, stub_identity,
                          "wrong_engine_signature")
    assert completed.returncode == 1, completed.stderr
    session = read_session(out)
    assert session["status"] == "invalid"
    assert session["failure"]["reason"] == "wrong_engine_signature"
    detail = session["failure"]["detail"]
    assert detail["analysis"] == "accessibility"
    assert detail["reported_engine_attr"] == "flow"


# ----------------------------------------------------------------------
# Liveness: abrupt worker death fails fast, nothing survives
# ----------------------------------------------------------------------
def test_exit_before_ready_fails_fast(tmp_path, stub_identity):
    out = tmp_path / "out"
    started = time.monotonic()
    completed = run_fault(tmp_path, out, stub_identity, "exit_before_ready")
    wall = time.monotonic() - started
    assert completed.returncode == 1, completed.stderr
    assert wall < 60, f"readiness barrier hung ({wall:.1f}s)"
    session = read_session(out)
    assert session["status"] == "invalid"
    assert session["failure"]["reason"] == "worker_died"
    assert session["failure"]["detail"]["exitcode"] == 70
    assert session["shutdown"]["surviving_owned_processes"] == []


def test_die_while_dispatch_prompt_cancellation(tmp_path, stub_identity):
    out = tmp_path / "out"
    started = time.monotonic()
    completed = run_fault(tmp_path, out, stub_identity, "die_while_dispatch",
                          jobs=3)
    wall = time.monotonic() - started
    assert completed.returncode == 1, completed.stderr
    assert wall < 60, f"dispatch loop hung ({wall:.1f}s)"
    session = read_session(out)
    assert session["status"] == "invalid"
    assert session["failure"]["reason"] == "worker_died"
    assert session["failure"]["detail"]["exitcode"] == 72
    # Bounded cancellation ran: outstanding jobs cancelled, nothing survives.
    assert session["shutdown"]["surviving_owned_processes"] == []
    assert "cancelled_outstanding_jobs" in session["shutdown"]
    assert session["dispatch"] is None  # dispatch never completed


def test_hang_after_output_terminated_at_shutdown(tmp_path, stub_identity):
    out = tmp_path / "out"
    completed = run_fault(tmp_path, out, stub_identity, "hang_after_output",
                          jobs=1)
    assert completed.returncode == 1, completed.stderr
    session = read_session(out)
    assert session["status"] == "invalid"
    # The result itself arrived; the worker then hung past its sentinel.
    assert any(r["type"] == "job_result" for r in raw_records(out))
    assert session["failure"]["reason"] == "workers_force_terminated"
    assert session["shutdown"]["forced_terminations"] == [0]
    # SIGTERM went to OUR child only, and it did not survive.
    assert session["shutdown"]["surviving_owned_processes"] == []


def test_never_send_sentinel_marks_protocol_incomplete(tmp_path,
                                                       stub_identity):
    out = tmp_path / "out"
    completed = run_fault(tmp_path, out, stub_identity,
                          "never_send_sentinel", jobs=1)
    assert completed.returncode == 1, completed.stderr
    session = read_session(out)
    assert session["status"] == "invalid"
    assert session["failure"]["reason"] == "missing_worker_exit_record"
    # The worker exited cleanly but broke the shutdown protocol.
    assert session["shutdown"]["exitcodes"] == [0]
    assert session["shutdown"]["missing_worker_exit"] == [0]
    assert session["shutdown"]["surviving_owned_processes"] == []


# ----------------------------------------------------------------------
# Bounded queues under a slow drain (NOT invalidating, never qualifying)
# ----------------------------------------------------------------------
def test_slow_drain_bounded_queue_completes_without_deadlock(tmp_path,
                                                             stub_identity):
    """queue_depth=1 with a sleeping worker: the bounded input queue fills,
    dispatch interleaves with draining, and the run still completes."""
    out = tmp_path / "out"
    started = time.monotonic()
    completed = run_fault(tmp_path, out, stub_identity, "slow_drain",
                          jobs=4, workers=1, queue_depth=1)
    wall = time.monotonic() - started
    assert completed.returncode == 0, completed.stderr
    assert wall >= 0.8  # the sleeps really happened (4 x 0.2s serial)
    assert wall < 60
    session = read_session(out)
    assert session["status"] == "valid"
    assert session["counts"]["validated"] == 4
    # Non-invalidating, but a fault run is NEVER qualification data.
    assert session["measurement_class"] == "harness_selftest"
    assert session["qualification_valid"] is False
    assert any("fault" in lim for lim in session["limitations"])
