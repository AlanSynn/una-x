"""Functional parallel overlap and bounded pending work (dossiers 04/05).

Overlap is proven with a filesystem handshake barrier, not a noisy timing
assertion: row 0 creates its marker and then BLOCKS until row 1's marker
exists; row 1 blocks until row 0's marker exists.  Both markers can only
exist if both rows were in flight simultaneously in separate processes.
A runtime that serializes worker rows deadlocks and each side fails its
injected wait within seconds, failing the test.
"""
from __future__ import annotations

import os
import queue
from dataclasses import replace

import pytest

pytestmark = pytest.mark.batch_runtime


def _barrier_extra(markers, wait_s=20):
    def extra(i, s):
        if i == 0:
            s.inject_specs = [
                {"phase": "pre_run", "action": "touch", "arg": str(markers[0])},
                {"phase": "pre_run", "action": "wait_for",
                 "arg_path": str(markers[1]), "arg": wait_s},
                {"phase": "pre_run", "action": "touch", "arg": str(markers[2])},
            ]
        else:
            s.inject_specs = [
                {"phase": "pre_run", "action": "wait_for",
                 "arg_path": str(markers[0]), "arg": wait_s},
                {"phase": "pre_run", "action": "touch", "arg": str(markers[1])},
            ]
    return extra


def test_two_independent_rows_run_concurrently(make_batch, tmp_path,
                                               monkeypatch):
    from urban_network_analysis.batch import runtime as rt

    markers = [tmp_path / f"m{i}.done" for i in range(3)]
    orig_job_for = rt._job_for

    def job_for(una, plan, i, staging_dir, analysis, script_output_folder,
                need_state):
        job = orig_job_for(una, plan, i, staging_dir, analysis,
                           script_output_folder, need_state)
        specs = getattr(una.projects[i], "inject_specs", None)
        if specs:
            object.__setattr__(job, "inject", specs)  # frozen dataclass
        return job

    monkeypatch.setattr(rt, "_job_for", job_for)

    p = make_batch(2, out_folder=tmp_path / "out",
                   extra=_barrier_extra(markers))
    p.RunBatch("accessibility", parallel=True, workers=2)

    # m2 is created by row 0 only AFTER it observed row 1's marker — its
    # existence is the handshake proof (no timing involved).
    assert markers[2].exists(), "handshake never completed: rows did not overlap"
    assert markers[1].exists()
    assert [r.status for r in p.batch_report.rows] == ["ran", "ran"]


def test_serialized_runtime_fails_the_barrier(make_batch, tmp_path,
                                              monkeypatch):
    """Non-vacuity guard for the commit machinery: a runtime that receives
    worker results but never advances its commit loop (every result is
    held back from the cursor) must never publish anything and must end
    in a bounded, observable cancellation — not a silent success or an
    infinite spin.  The injected waits (5s) are SHORTER than the deadline
    (15s) so the held results exist to prove the drain happened."""
    from urban_network_analysis.batch import runtime as rt

    markers = [tmp_path / f"s{i}.done" for i in range(3)]
    orig_recv = rt._Pool.recv
    held = []

    def recv(self, timeout):
        result = orig_recv(self, timeout)   # raises Empty when idle
        held.append(result)                 # hold: never reaches the cursor
        raise queue.Empty()

    monkeypatch.setattr(rt._Pool, "recv", recv)

    out = tmp_path / "out"
    p = make_batch(2, out_folder=out,
                   extra=_barrier_extra(markers, wait_s=5))
    with pytest.raises(Exception) as ei:
        p.RunBatch("accessibility", parallel=True, workers=2,
                   execution=replace(p.execution, timeout_s=15))
    assert held, "sabotage never saw a worker result"
    # nothing was ever published: no row folders, no staging residue
    assert not os.path.exists(out) or \
        not [n for n in os.listdir(out)
             if n.startswith(("row", ".una-batch"))]
    assert "deadline" in str(ei.value)
    assert hasattr(ei.value, "batch_rows")


def test_bounded_pending_dispatch(make_batch, tmp_path, monkeypatch):
    """The parent must never enqueue the whole workload: outstanding jobs
    stay bounded by workers + queue_depth (dossier 05).  With 2 workers
    and queue_depth 2 the bound is 4; greedy dispatch fills it but never
    exceeds it, and every row runs in a spawn worker (never the parent)."""
    import os

    from urban_network_analysis.batch import runtime as rt

    orig_dispatch = rt._Pool.dispatch
    calls = {"n": 0}
    seen_max = {"value": 0}

    def dispatch(self, job):
        calls["n"] += 1
        seen_max["value"] = max(seen_max["value"], self.in_flight + 1)
        return orig_dispatch(self, job)

    monkeypatch.setattr(rt._Pool, "dispatch", dispatch)

    p = make_batch(6, out_folder=tmp_path / "out")
    p.RunBatch("accessibility", parallel=True, workers=2,
               execution=replace(p.execution, queue_depth=2))
    # pending bound = 2 workers + queue_depth 2 = 4
    assert calls["n"] == 6            # every row dispatched exactly once
    assert seen_max["value"] <= 4
    assert seen_max["value"] >= 2     # overlap actually exercised
    pids = {r.worker_pid for r in p.batch_report.rows}
    assert pids and len(pids) <= 2
    assert pids.isdisjoint({os.getpid()})
