"""REGRESSION (DEF-24, ported from perf/una-10x ab357c6): the silent
cancel-wedge.

A worker-row failure with a backlogged inbox used to wedge the batch
INSIDE its own failure path: ``_Pool.cancel()`` terminated the row
workers BEFORE touching the queues, leaving the inbox feeder thread
blocked in ``send_bytes`` on a full pipe whose only remaining read end
was the coordinator's own undrained inbox reader (no EPIPE — the parent
holds it), and the ``join_thread()`` that followed hung forever.  The
failure path's re-raise (runtime.py, the ``except BaseException``
handler) therefore never executed and the original worker-row exception
was swallowed — a silent wedge instead of a loud failure.

Backlog geometry (decisive, measured upstream): the suite's pre-existing
cancellation tests pass even on defective code — a 2-row inbox
(+1-row-backlog) flushes completely into the pipe buffer, the feeder is
idle at cancel time, and join_thread() returns.  The wedge needs QUEUED
BYTES BEYOND THE PIPE BUFFER (~64 KiB on Linux) so the feeder is
provably mid-flush when cancel() runs.  The scenarios below pin that
geometry two ways: engine rows whose injected job payloads far exceed
the pipe buffer, and a raw-bytes unit frame on the pool seam.

The contract under regression, per the runtime's own serial-parity law
("a worker row's failure is the serial loop's failure: re-raise the
original exception, same type/message"):
(a) the ORIGINAL worker-row exception must surface to the RunBatch
    caller, with the (empty) committed prefix attached;
(b) the cancellation must return in bounded time — no hang, no leaked
    live workers, no staging left behind.

Subprocess guard: a wedge cannot be bounded in-process — the stuck main
thread sits in a threading state-lock acquire, and signal-based
interrupts are delivery-unreliable with live extra threads.  So each
scenario runs in a CHILD pytest process that the runner test kills at a
hard bound; a child's in-thread faulthandler watchdog dumps every stack
to its captured stderr at 60 s for forensics before the kill.  A wedge
therefore lands as a loud runner-test failure carrying the child's
stack dump — never as a hung suite.
"""
from __future__ import annotations

import faulthandler
import os
import subprocess
import sys
import threading

import pytest

pytestmark = pytest.mark.batch_runtime

HERE = os.path.dirname(os.path.abspath(__file__))
CHILD = os.path.join(HERE, "test_runtime_cancel_wedge.py")

# A healthy child: interpreter + conftest + engine imports + the batch.
# A wedged child never finishes; the dump fires at 60 s and the kill at
# 150 s.
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


def _no_staging_residue(out):
    """(b) of the regression contract: this run's staging removed by
    ownership (the single cleanup owner runs on cancellation alike).
    The cohort/prepared store roots are excluded: on the merged tree
    they persist by design beside the row outputs, survive aborts
    alike, and are identity-verified at every reuse (the same
    convention as the runtime suite's ``runtime_residue`` helper)."""
    return not os.path.exists(out) or \
        not [n for n in os.listdir(out)
             if n.startswith(".una-batch")
             and n not in (".una-batch-cohort", ".una-batch-prepared")]


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
    the geometry alone decides."""
    _run_child("test_child_pool_cancel_backlog_returns", tmp_path)


# ---------------------------------------------------------------------------
# Child scenarios (skipped in-process; executed via _run_child).
# ---------------------------------------------------------------------------

def _dump_after_bound():
    """Forensics-only watchdog: dump every thread stack to stderr once,
    then keep the process alive for the parent's hard kill."""
    def _fire():
        # sys.__stderr__: the child's pytest captures sys.stderr and
        # only re-emits it on a clean unwind, which is exactly what a
        # wedge never does - write the dump to the real stderr so the
        # parent's kill-report carries it.
        print("\n*** CANCEL-WEDGE WATCHDOG: no completion after "
              f"{CHILD_DUMP_S:.0f}s - dumping all thread stacks ***",
              file=sys.__stderr__, flush=True)
        faulthandler.dump_traceback(file=sys.__stderr__)
        threading.Event().wait(CHILD_BOUND_S)
    t = threading.Thread(target=_fire, daemon=True)
    t.start()


# Backlog knob: each engine job carries an inert injected spec whose
# 32 KiB arg rides inside the pickled job (a spec whose phase never
# fires, so execution is untouched) — with 63 queued jobs the inbox
# holds ~2 MiB, far beyond the pipe buffer, so the feeder is provably
# mid-flush at cancel time.
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
    assert _no_staging_residue(out)


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
    assert _no_staging_residue(out)


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
    pool.cancel()             # abandons the backlog, reaps, closes
    assert pool.cancelled is True
    assert all(not proc.is_alive() for proc in pool.procs)
