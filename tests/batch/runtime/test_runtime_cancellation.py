"""Cancellation semantics: deadline, worker loss, KeyboardInterrupt
(dossier 04 step 7, dossier 05).

Every cancellation path must: stop dispatch, reap the pool, remove THIS
run's staging directories by ownership (never other paths), preserve the
committed prefix, leave ``batch_report`` None (serial parity), and raise
``BatchCancelledError`` with the prefix outcomes attached.
"""
from __future__ import annotations

import os

import pytest

from urban_network_analysis.batch.report import BatchCancelledError

pytestmark = pytest.mark.batch_runtime


def _proc_alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def _stems(folder):
    return {n.split(".")[0] for n in os.listdir(folder)}


def test_deadline_cancel_preserves_prefix_and_reaps_workers(
        make_batch, tmp_path):
    out = tmp_path / "out"
    worker_pids = tmp_path / "pids"

    def extra(i, s):
        if i >= 1:
            s.inject_specs = [
                {"phase": "pre_run", "action": "touch",
                 "arg": str(worker_pids / f"w{i}")},
                {"phase": "pre_run", "action": "sleep", "arg": 60},
            ]

    p = make_batch(4, out_folder=out, extra=extra)
    from dataclasses import replace
    # 20s: above a loaded node's worker spawn+import+row0 run (the
    # deadline budgets work, not spawn readiness), far below the injected
    # 60s sleeps so the cancel always fires mid-flight.  How much prefix
    # exists at cancel time depends on node speed, so the test pins the
    # INVARIANT — the reported rows are exactly the rows whose artifacts
    # reached the public tree (publication happens only at commit) —
    # not a fixed prefix (an 8s budget lost row 0 on a jammed node).
    with pytest.raises(BatchCancelledError, match="deadline") as ei:
        p.RunBatch("accessibility", parallel=True, workers=2,
                   execution=replace(p.execution, timeout_s=20))

    assert p.batch_report is None
    rows = ei.value.batch_rows
    published = {s for s in _stems(out) if s.startswith("row")}
    assert {f"row{r.index}" for r in rows} == published
    for r in rows:
        assert r.phase == "COMMITTED"
        for f in r.output_files:
            assert os.path.exists(f)
    # this run's staging removed by ownership
    assert not [n for n in os.listdir(out) if n.startswith(".una-batch")]

    # every worker that entered a cancelled row was reaped
    pid_files = sorted(worker_pids.iterdir()) if worker_pids.exists() \
        else []
    assert all(not _proc_alive(int(open(f).read())) for f in pid_files)


def test_worker_death_in_flight_cancels_cleanly(make_batch, tmp_path):
    out = tmp_path / "out"

    def extra(i, s):
        s.inject_specs = ([{"phase": "startup", "action": "exit",
                            "arg": 1}] if i == 0 else [])

    p = make_batch(2, out_folder=out, extra=extra)
    with pytest.raises(BatchCancelledError, match="died") as ei:
        p.RunBatch("accessibility", parallel=True, workers=1)

    assert p.batch_report is None
    assert ei.value.batch_rows == ()          # nothing had committed
    assert not os.path.exists(out) or \
        not [n for n in os.listdir(out)
             if n.startswith(("row", ".una-batch"))]


def test_keyboard_interrupt_cancels_and_cleans(make_batch, tmp_path,
                                               monkeypatch):
    from urban_network_analysis.batch import runtime as rt

    out = tmp_path / "out"
    orig_recv = rt._Pool.recv
    calls = {"n": 0}

    def recv(self, timeout):
        calls["n"] += 1
        raise KeyboardInterrupt()

    monkeypatch.setattr(rt._Pool, "recv", recv)

    p = make_batch(2, out_folder=out)
    with pytest.raises(BatchCancelledError,
                       match="KeyboardInterrupt") as ei:
        p.RunBatch("accessibility", parallel=True, workers=1)

    assert calls["n"] >= 1
    assert p.batch_report is None
    assert ei.value.batch_rows == ()
    assert not os.path.exists(out) or \
        not [n for n in os.listdir(out)
             if n.startswith(("row", ".una-batch"))]


def test_cancellation_is_batchcancellederror_not_bare_kill(
        make_batch, tmp_path):
    """The deadline path raises the typed error with the prefix attached —
    not a raw TimeoutError/RuntimeError with no recovery information."""
    from dataclasses import replace

    def extra(i, s):
        if i >= 1:
            s.inject_specs = [{"phase": "pre_run", "action": "sleep",
                               "arg": 60}]

    p = make_batch(3, out_folder=tmp_path / "out", extra=extra)
    with pytest.raises(BatchCancelledError) as ei:
        p.RunBatch("accessibility", parallel=True, workers=1,
                   execution=replace(p.execution, timeout_s=2))
    exc = ei.value
    assert "deadline" in str(exc)
    assert hasattr(exc, "batch_rows")
