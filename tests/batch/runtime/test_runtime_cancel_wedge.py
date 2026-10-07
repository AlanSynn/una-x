"""REGRESSION (campaign una_10x task #20/#24): the silent cancel-wedge.

A worker-row failure with a backlogged inbox used to wedge the batch
INSIDE its own failure path: ``_Pool.cancel()`` terminated the row
workers BEFORE touching the queues, leaving the inbox feeder thread
blocked in ``send_bytes`` on a full pipe whose only remaining read end
was the coordinator's own undrained ``_reader`` (no EPIPE — the parent
holds it), and the ``join_thread()`` that followed hung forever.  The
failure path's re-raise (runtime.py, the ``except BaseException``
handler) therefore never executed and the original worker-row
exception was swallowed — a silent wedge instead of a loud failure
(observed 4/4 times in the N00 shakedowns; evidence:
campaign_runs/una_10x/tenx-20261005T2331Z/n00/wedge_triage_console.txt
and wedge_stacks_attempt4.txt).

Backlog geometry (decisive, measured): the suite's pre-existing
cancellation tests pass even on defective code — a 2-row inbox
(+1-row-backlog) flushes completely into the pipe buffer, the feeder
is idle at cancel time, and join_thread() returns.  The wedge needs
QUEUED BYTES BEYOND THE PIPE BUFFER (~64 KiB on macOS) so the feeder
is provably mid-flush when cancel() runs.  The scenarios below pin
that geometry two ways: engine rows whose injected job payloads far
exceed the pipe buffer, and a raw-bytes unit frame on the pool seam.

Clean-path control: one scenario runs the SAME 64-row + payload
geometry with NO failure injection and must complete normally —
the fix provably does not perturb the happy path (cancel() is
failure-path-only; close() is untouched).  Mirrors the live evidence
from the campaign triage: a failure-free sibling of the same repro
completed clean in 25.16 s, rc=0.

The contract under regression, per the runtime's own serial-parity law
("a worker row's failure is the serial loop's failure: re-raise the
original exception, same type/message"):
(a) the ORIGINAL worker-row exception must surface to the RunBatch
    caller, with the (empty) committed prefix attached;
(b) the cancellation must return in bounded time — no hang, no leaked
    live workers, no staging left behind.

Subprocess guard: a wedge cannot be bounded in-process — the stuck
main thread sits in a threading state-lock acquire, and signal-based
interrupts are delivery-unreliable on macOS with live extra threads
(measured: interrupt_main never landed; the demonstration process had
to be killed externally, sample at
/tmp/n00_wedge_repro/pytest_wedge_sample.txt).  So each scenario runs
in a CHILD pytest process that the runner test kills at a hard bound;
a child's in-thread faulthandler watchdog dumps every stack to its
captured stderr at 60 s for forensics before the kill.  A wedge
therefore lands as a loud runner-test failure carrying the child's
stack dump — never as a hung suite.
"""
from __future__ import annotations

import faulthandler
import os
import subprocess
import sys
import threading
import time

import pytest

from conftest import hidden_residue

pytestmark = pytest.mark.batch_runtime

HERE = os.path.dirname(os.path.abspath(__file__))
CHILD = os.path.join(HERE, "test_runtime_cancel_wedge.py")

# A healthy child: interpreter + conftest + engine imports + the batch,
# observed ~15-40 s cold.  A wedged child never finishes; the dump
# fires at 60 s and the kill at 150 s.
CHILD_DUMP_S = 60.0
CHILD_BOUND_S = 150.0

_GUARD_ENV = "UNA_CANCEL_WEDGE_CHILD"


def _run_child(node, tmp_path):
    """Run one scenario as a guarded child pytest process."""
    env = dict(os.environ)
    env[_GUARD_ENV] = "1"
    proc = subprocess.Popen(
        [sys.executable, "-m", "pytest", f"{CHILD}::{node}", "-q", "-x"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        cwd=HERE, env=env, start_new_session=True)
    try:
        out, err = proc.communicate(timeout=CHILD_BOUND_S)
    except subprocess.TimeoutExpired:
        os.killpg(os.getpgid(proc.pid), 9)
        out, err = proc.communicate()
        pytest.fail(
            f"WEDGED: {node} was killed at the {CHILD_BOUND_S:.0f}s "
            f"bound (the defective ordering hangs cancel(); child "
            f"stderr tail carries the faulthandler dump):\n{err[-4000:]}")
    assert proc.returncode == 0, (node, proc.returncode, out[-2000:],
                                  err[-2000:])
    assert "1 passed" in out, (node, out[-2000:])


def _assert_bounded_shutdown(out):
    """(b) of the regression contract: this run's staging removed by
    ownership (the single cleanup owner runs on cancellation alike)."""
    assert not os.path.exists(out) or not hidden_residue(out)


# ---------------------------------------------------------------------------
# Runner tests (safe in-process; execute the scenarios as children).
# ---------------------------------------------------------------------------

def test_worker_row_failure_propagates_without_wedging(tmp_path):
    """Row 0 fails inside the worker (pre_run injected MemoryError -> a
    failed RowResult, worker stays alive) while every other row sits in
    a backlog whose queued bytes exceed the pipe buffer.  The scheduler
    must raise the ORIGINAL exception at the cursor and the failure
    path's cancel must return."""
    _run_child("test_child_row_failure_surfaces_and_cancels", tmp_path)


def test_worker_death_cancels_with_backlog(tmp_path):
    """The died-in-flight path shares cancel(): a worker that exits
    while the inbox still holds undispatched jobs must not wedge the
    cancellation either (same ordering defect, different trigger)."""
    _run_child("test_child_worker_death_cancels", tmp_path)


def test_cancel_returns_with_freshly_killed_readers(tmp_path):
    """Unit frame on the defect seam itself: a real spawn pool whose
    workers are terminated while the inbox feeder holds ~0.5 MiB of
    queued bytes must return from cancel().  No engine row needed —
    the geometry alone decides (this exact frame wedged the pre-fix
    tree and required an external kill)."""
    _run_child("test_child_pool_cancel_backlog_returns", tmp_path)


def test_healthy_backlog_completes_normally(tmp_path):
    """Clean-path control: the same 64-row + 32 KiB-payload geometry
    with NO failure injection completes normally — every row commits,
    the report is published, and the child interpreter exits cleanly
    (no exit-time wedge from the backlog on the healthy path).  Proves
    the fix does not perturb the happy path.  Rows are named r00..r63
    (equal-length, prefix-distinct stems) so the planner admits all 64
    to workers and the inbox genuinely carries the whole ~2 MiB
    backlog; default row0..row63 names would serialize 60 rows on the
    engine's stem-prefix hazard and run them in the parent."""
    _run_child("test_child_healthy_backlog_completes", tmp_path)


# ---------------------------------------------------------------------------
# Child scenarios (skipped in-process; executed via _run_child).
# ---------------------------------------------------------------------------

def _dump_after_bound(dump_s: float = CHILD_DUMP_S):
    """Forensics-only watchdog: dump every thread stack to stderr once,
    then keep the process alive for the parent's hard kill.  (No
    interrupt_main: signal delivery is thread-luck on macOS and the
    dump is what matters.)"""
    def _fire():
        # The dump must be PRECEDED by the bound: this thread exists
        # only for a child that never finishes, so sleep first — an
        # undelayed dump would fire at test start and masquerade as a
        # wedge signal in every failing child (it did: preserved console
        # healthy_control_child_console_20261007T0458Z.txt shows a
        # spurious "no completion after 60s" inside a 13.12s run).
        time.sleep(dump_s)
        # sys.__stderr__: the child's pytest captures sys.stderr and
        # only re-emits it on a clean unwind, which is exactly what a
        # wedge never does - write the dump to the real stderr so the
        # parent's kill-report carries it.
        print("\n*** CANCEL-WEDGE WATCHDOG: no completion after "
              f"{dump_s:.0f}s - dumping all thread stacks ***",
              file=sys.__stderr__, flush=True)
        faulthandler.dump_traceback(file=sys.__stderr__)
        threading.Event().wait(CHILD_BOUND_S)
    t = threading.Thread(target=_fire, daemon=True)
    t.start()


# Backlog knob: each engine job carries an inert injected spec whose
# 32 KiB arg rides inside the pickled job (a spec whose phase never
# fires, so execution is untouched).  Only worker-admitted rows are
# dispatched to the inbox, and a few such jobs already put queued bytes
# far beyond the pipe buffer — the wedge scenarios dispatch 3 backlog
# jobs (~96 KiB) behind the failing row 0, which is what left the feeder
# provably mid-flush at cancel time (the suite's 2-row tests, well under
# one pipe buffer, cannot).
_INERT_32K = [{"phase": "n00_never", "action": "touch",
               "arg": "x" * (32 * 1024)}]


def test_child_row_failure_surfaces_and_cancels(make_batch, tmp_path):
    if os.environ.get(_GUARD_ENV) != "1":
        pytest.skip("guarded scenario: runs via _run_child subprocess")
    from dataclasses import replace

    out = str(tmp_path / "out")

    def extra(i, s):
        if i == 0:
            s.inject_specs = [
                {"phase": "pre_run", "action": "raise_memory"}]
        else:
            s.inject_specs = _INERT_32K

    p = make_batch(64, out_folder=out, extra=extra)
    _dump_after_bound()
    with pytest.raises(MemoryError, match="injected OOM") as ei:
        p.RunBatch("accessibility", parallel=True, workers=1,
                   execution=replace(p.execution, queue_depth=64))
    # (a) original exception surfaced, prefix law respected
    assert p.batch_report is None
    assert ei.value.batch_rows == ()          # nothing had committed
    _assert_bounded_shutdown(out)


def test_child_worker_death_cancels(make_batch, tmp_path):
    if os.environ.get(_GUARD_ENV) != "1":
        pytest.skip("guarded scenario: runs via _run_child subprocess")

    out = str(tmp_path / "out")

    def extra(i, s):
        if i == 0:
            s.inject_specs = [{"phase": "startup", "action": "exit",
                               "arg": 1}]
        else:
            s.inject_specs = _INERT_32K

    p = make_batch(64, out_folder=out, extra=extra)
    _dump_after_bound()
    with pytest.raises(Exception, match="died") as ei:
        p.RunBatch("accessibility", parallel=True, workers=1)
    assert p.batch_report is None
    assert ei.value.batch_rows == ()
    _assert_bounded_shutdown(out)


def test_child_pool_cancel_backlog_returns(tmp_path):
    if os.environ.get(_GUARD_ENV) != "1":
        pytest.skip("guarded scenario: runs via _run_child subprocess")
    from urban_network_analysis.batch import runtime as rt

    _dump_after_bound()
    pool = rt._Pool(1, 1)
    pool.start()
    # Raw bytes, not RowJobs — the worker's failure handling is
    # irrelevant; cancel() reaps it either way.  64 x 8 KiB = 512 KiB.
    payload = b"x" * 8192
    for _ in range(64):
        pool.inbox.put(payload)
    pool.cancel()             # terminates workers, then closes queues
    assert pool.cancelled is True
    assert all(not proc.is_alive() for proc in pool.procs)


def test_child_healthy_backlog_completes(make_batch, tmp_path):
    if os.environ.get(_GUARD_ENV) != "1":
        pytest.skip("guarded scenario: runs via _run_child subprocess")
    from dataclasses import replace

    out = str(tmp_path / "out")

    def extra(i, s):
        # Equal-length prefix-distinct stems (see the runner's
        # docstring): otherwise the planner's stem-prefix hazard
        # serializes 60 of these 64 rows and they run in the parent,
        # where Settings.Validation rejects the test-transport
        # inject_specs attribute (workers get a stripped state,
        # runtime.py _snapshot_row/_job_state — the parent path does
        # not).
        s.name = f"r{i:02d}"
        s.inject_specs = _INERT_32K

    p = make_batch(64, out_folder=out, extra=extra)
    # A healthy 64-row run is quick (observed ~5-8 s warm, one worker)
    # but must never dump, so arm the forensics watchdog later than the
    # wedge scenarios' 60 s regardless.
    _dump_after_bound(dump_s=90.0)
    # Same schedule and payload geometry as the wedge scenarios, minus
    # the failure: the only delta is the healthy path itself.  All 64
    # jobs flow through the inbox (~2 MiB) and are fully drained.
    p.RunBatch("accessibility", parallel=True, workers=1,
               execution=replace(p.execution, queue_depth=64))
    # Happy path unperturbed: report published, every row's artifacts
    # (<stem>.feather + <stem>.geojson) public under the output folder,
    # staging fully retired.
    assert p.batch_report is not None
    published = {n for n in os.listdir(out) if not n.startswith(".")}
    assert published == {f"r{i:02d}{ext}" for i in range(64)
                         for ext in (".feather", ".geojson")}, \
        sorted(published)[:8]
    assert not hidden_residue(out)
