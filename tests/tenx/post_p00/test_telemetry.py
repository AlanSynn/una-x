"""P01M opt-in phase telemetry (post_p00 design 01, campaign una_10x).

Laws under test: disabled-by-default (no report change, no files); the
enabled event schema (nonnegative integer clocks, null for unavailable
CPU, nested parent_phase_id); top-level-only summaries; bounded buffers
with an explicit dropped counter; worker result_put cost never rides the
result it measures (it lives only in the streamed workerpool JSONL); and
telemetry-on runs still produce bitwise-identical artifacts.
"""
from __future__ import annotations

import json
import os

import pytest

from conftest import tree_bytes

from urban_network_analysis.batch.telemetry import (
    SCHEMA, PhaseRecorder, summarize_events)

pytestmark = pytest.mark.tenx_post_p00


# ---------------------------------------------------------------------------
# Unit: recorder mechanics.
# ---------------------------------------------------------------------------

def test_disabled_by_default(monkeypatch):
    monkeypatch.delenv("UNA_PHASE_TELEMETRY", raising=False)
    assert PhaseRecorder.maybe(source="probe") is None
    rec = PhaseRecorder("rid", "probe")
    rec.enabled = False
    with rec.phase("x"):
        pass
    assert rec.events_snapshot() == ()
    assert rec.dropped == 0
    assert rec.summarize() == {}


def test_nested_spans_carry_parent_and_summarize_top_level_only(monkeypatch):
    monkeypatch.setenv("UNA_PHASE_TELEMETRY", "1")
    rec = PhaseRecorder("rid", "unit")
    with rec.phase("outer"):
        with rec.phase("inner"):
            pass
    events = rec.events_snapshot()
    assert len(events) == 2
    outer, inner = events
    assert outer["parent_phase_id"] is None
    assert inner["parent_phase_id"] == outer["phase_id"]
    assert outer["schema"] == SCHEMA == inner["schema"]
    for ev in events:
        for key in ("start_mono_ns", "end_mono_ns", "process_cpu_start_ns",
                    "process_cpu_end_ns"):
            assert isinstance(ev[key], int) and ev[key] >= 0
        assert ev["end_mono_ns"] > ev["start_mono_ns"]
        assert ev["status"] == "ok"
    # top-level-only: the outer span's wall already contains the inner's
    summary = rec.summarize()
    assert set(summary) == {"outer"}
    wall = summary["outer"]["wall_s_sum"]
    assert wall >= summary_events_wall_of_inner(events)
    assert wall >= 0.0


def summary_events_wall_of_inner(events):
    inner = [e for e in events if e["phase"] == "inner"][0]
    return (inner["end_mono_ns"] - inner["start_mono_ns"]) / 1e9


def test_null_cpu_clock_recorded_as_null(monkeypatch):
    import urban_network_analysis.batch.telemetry as tel

    monkeypatch.setenv("UNA_PHASE_TELEMETRY", "1")
    rec = PhaseRecorder("rid", "unit")
    real = tel.time.thread_time_ns

    def boom():
        raise OSError("thread time unavailable")

    tel.time.thread_time_ns = boom
    try:
        with rec.phase("x"):
            pass
    finally:
        tel.time.thread_time_ns = real
    ev = rec.events_snapshot()[0]
    assert ev["thread_cpu_start_ns"] is None
    assert ev["thread_cpu_end_ns"] is None
    assert rec.summarize()["x"]["thread_cpu_unavailable"] == 1


def test_bounded_buffer_drops_explicitly(monkeypatch):
    monkeypatch.setenv("UNA_PHASE_TELEMETRY", "1")
    rec = PhaseRecorder("rid", "unit", max_events=3)
    for i in range(5):
        with rec.phase(f"p{i}"):
            pass
    assert rec.dropped == 2
    assert len(rec.events_snapshot()) == 3
    summary = rec.summarize()
    assert sum(a["count"] for a in summary.values()) == 3


def test_exception_recorded_and_reraised(monkeypatch):
    monkeypatch.setenv("UNA_PHASE_TELEMETRY", "1")
    rec = PhaseRecorder("rid", "unit")
    with pytest.raises(ValueError):
        with rec.phase("boom"):
            raise ValueError("x")
    ev = rec.events_snapshot()[0]
    assert ev["status"] == "ValueError"
    assert ev["end_mono_ns"] is not None


def test_flush_jsonl_appends_and_offsets(tmp_path, monkeypatch):
    monkeypatch.setenv("UNA_PHASE_TELEMETRY", "1")
    rec = PhaseRecorder("rid", "unit")
    with rec.phase("a"):
        pass
    path = tmp_path / "ev.jsonl"
    n1 = rec.flush_jsonl(str(path))
    n2 = rec.flush_jsonl(str(path), since=n1)   # nothing new
    with rec.phase("b"):
        pass
    n3 = rec.flush_jsonl(str(path), since=n1)
    assert (n1, n2, n3) == (1, 0, 1)
    lines = open(path).read().splitlines()
    assert len(lines) == 2
    assert [json.loads(l)["phase"] for l in lines] == ["a", "b"]


# ---------------------------------------------------------------------------
# Integration: the public RunBatch surface, env off vs on.
# ---------------------------------------------------------------------------

def test_parallel_run_default_has_no_telemetry(make_batch, tmp_path,
                                               monkeypatch, telemetry_extra):
    monkeypatch.delenv("UNA_PHASE_TELEMETRY", raising=False)
    monkeypatch.delenv("UNA_PHASE_DIR", raising=False)
    p = make_batch(2, out_folder=tmp_path / "out")
    p.RunBatch("accessibility", parallel=True, workers=1)
    assert telemetry_extra(p) is None
    extra = p.batch_report.extra
    # Expected expectation change (una_10x P02): the prepared route's
    # counters block joins "cohort" in extra — runtime-only reporting
    # per the P02 contract (prepared_contract.json @ f139dd3); telemetry
    # itself remains env-gated and absent here.
    assert set(extra) <= {"cohort", "preparation"}, set(extra)
    # no telemetry files anywhere under the output tree
    leftovers = [rel for rel in tree_bytes(tmp_path)
                 if "phase_events" in rel]
    assert leftovers == []


def test_parallel_run_enabled_reports_phases_and_files(
        make_batch, tmp_path, monkeypatch, telemetry_extra):
    phdir = tmp_path / "ph"
    monkeypatch.setenv("UNA_PHASE_TELEMETRY", "1")
    monkeypatch.setenv("UNA_PHASE_DIR", str(phdir))
    monkeypatch.setenv("UNA_PHASE_RUN_ID", "tel-fixed-run")

    p = make_batch(3, out_folder=tmp_path / "out")
    p.RunBatch("accessibility", parallel=True, workers=2)

    tel = telemetry_extra(p)
    assert tel is not None, "telemetry block missing from report extra"
    assert tel["schema"] == SCHEMA
    assert tel["run_id"] == "tel-fixed-run"
    assert tel["diagnostic_only"] is True
    assert tel["dropped_events"] == 0
    assert tel["coordinator_jsonl"] and os.path.isfile(
        tel["coordinator_jsonl"])

    # coordinator phases over the real route
    for phase in ("plan", "pool_start", "dispatch", "result_recv",
                  "commit", "pool_close", "rehydrate"):
        assert phase in tel["coordinator"], \
            f"missing coordinator phase {phase}: {sorted(tel['coordinator'])}"
    assert tel["coordinator"]["dispatch"]["count"] == 3
    assert tel["coordinator"]["commit"]["count"] == 3
    # result_recv is a per-ARRIVAL marker, never a per-tick span (a
    # per-tick span floods any bounded buffer with idle noise)
    assert tel["coordinator"]["result_recv"]["count"] == 3

    # worker in-row phases rode the results
    assert "run" in tel["workers"]
    assert tel["workers"]["run"]["count"] == 3
    assert len(tel["worker_rows"]) == 3
    assert all(v["events"] > 0 and v["dropped"] == 0
               for v in tel["worker_rows"].values())

    # workerpool streaming JSONL: result_put lives ONLY there, never in
    # the events that ride a RowResult.
    wp_files = [f for f in os.listdir(phdir)
                if f.startswith("phase_events.workerpool.")]
    assert wp_files, "no workerpool telemetry files"
    riding_phases = set()
    for f in os.listdir(phdir):
        if not f.endswith(".jsonl"):
            continue
        for line in open(phdir / f):
            ev = json.loads(line)
            assert ev["schema"] == SCHEMA
            assert ev["run_id"] == "tel-fixed-run"
            for key in ("start_mono_ns", "end_mono_ns",
                        "process_cpu_start_ns", "process_cpu_end_ns"):
                if ev[key] is not None:
                    assert isinstance(ev[key], int) and ev[key] >= 0
            riding_phases.add((ev["source"], ev["phase"]))
    assert ("workerpool", "result_put") in riding_phases
    assert ("workerpool", "inbox_get") in riding_phases
    # the result-riding worker events contain no pool-loop phases
    assert ("worker", "result_put") not in riding_phases

    # row attribution on workerpool result_put spans
    put_rows = set()
    for f in wp_files:
        for line in open(phdir / f):
            ev = json.loads(line)
            if ev["phase"] == "result_put":
                put_rows.add(ev["row_id"])
    assert put_rows == {0, 1, 2}


def test_enabled_run_artifacts_bitwise_match_serial(
        make_batch, tmp_path, monkeypatch):
    """Diagnostics must never perturb outputs (telemetry is never the
    comparator, and an enabled run is still a valid parity run)."""
    monkeypatch.setenv("UNA_PHASE_TELEMETRY", "1")
    monkeypatch.setenv("UNA_PHASE_DIR", str(tmp_path / "ph"))
    monkeypatch.setenv("UNA_PHASE_RUN_ID", "tel-parity")
    par = make_batch(3, out_folder=tmp_path / "par")
    par.RunBatch("accessibility", parallel=True, workers=2)

    monkeypatch.delenv("UNA_PHASE_TELEMETRY", raising=False)
    ser = make_batch(3, out_folder=tmp_path / "ser")
    ser.RunBatch("accessibility")

    assert tree_bytes(tmp_path / "ser", skip_hidden=True) == \
        tree_bytes(tmp_path / "par", skip_hidden=True)


def test_summarize_events_ignores_nested_and_incomplete():
    events = [
        {"phase": "a", "parent_phase_id": None,
         "start_mono_ns": 0, "end_mono_ns": 1_000_000_000,
         "process_cpu_start_ns": 0, "process_cpu_end_ns": 500_000_000,
         "thread_cpu_start_ns": None, "thread_cpu_end_ns": None},
        {"phase": "b", "parent_phase_id": "a-1",
         "start_mono_ns": 0, "end_mono_ns": 2_000_000_000,
         "process_cpu_start_ns": 0, "process_cpu_end_ns": 2_000_000_000,
         "thread_cpu_start_ns": 0, "thread_cpu_end_ns": 100},
    ]
    out = summarize_events(events)
    assert set(out) == {"a"}
    assert out["a"]["count"] == 1
    assert out["a"]["wall_s_sum"] == 1.0
    assert out["a"]["process_cpu_s_sum"] == 0.5
    assert out["a"]["thread_cpu_unavailable"] == 1
