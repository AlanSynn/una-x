"""P01M phase-telemetry tests (post_p00 designs/01_telemetry.md).

Proves the four properties the design requires of the diagnostics
plumbing, on the PUBLIC parallel RunBatch over the frozen smoke grid:

1. Default-absent schema: with ``UNA_BATCH_PHASES`` unset, the new
   fields keep their defaults and report.extra carries no telemetry —
   normal runs are unchanged (WS04's compatibility leg).
2. When enabled, worker phase spans ride the existing trusted RowResult
   payload into every committed row outcome (never a second channel),
   with monotone wall stamps and null-safe CPU deltas.
3. When enabled, coordinator spans and per-row dispatch/recv transport
   stamps (H2) land in report.extra, with dispatch-mono <= recv-mono
   causality per row.
4. Bounded buffers: the span bound drops excess into a counter instead
   of growing without bound; journal records stay phase-free so
   checkpoint identity/bytes are untouched.
"""
from __future__ import annotations

import json
import pickle
import time

import pytest

import urban_network_analysis.batch.telemetry as telemetry
from urban_network_analysis.batch.checkpoint import JournalRow

SPAN_KEYS = {"phase", "start_mono_ns", "end_mono_ns",
             "proc_cpu_ns", "thr_cpu_ns"}

WORKER_PHASES = {"bootstrap_import", "una_construct", "settings_bind",
                 "run", "stage_scan"}

COORD_PHASES = {"pool_start", "dispatch", "staged_verify",
                "publish", "state_delta", "journal", "rehydrate",
                "pool_close"}


# ---------------------------------------------------------------------------
# Unit: recorder semantics (no pool needed).
# ---------------------------------------------------------------------------

def test_disabled_recorder_is_noop():
    rec = telemetry.PhaseRecorder()
    assert rec.on is False
    with rec.span("anything"):
        pass
    assert rec.snapshot() == ()
    assert rec.dropped_events == 0
    assert telemetry.opt_span(None, "x").__class__ is telemetry._NullSpan


def test_span_bound_drops_excess_into_counter(monkeypatch):
    monkeypatch.setenv("UNA_BATCH_PHASES", "1")
    monkeypatch.setattr(telemetry, "MAX_SPANS", 2)
    rec = telemetry.PhaseRecorder()
    for name in ("a", "b", "c", "d"):
        with rec.span(name):
            pass
    kept = rec.snapshot()
    assert [s["phase"] for s in kept] == ["a", "b"]   # first-come, kept
    assert rec.dropped_events == 2                    # c and d dropped
    assert all(set(s) == SPAN_KEYS for s in kept)


def test_span_monotone_and_cpu_nonnegative(monkeypatch):
    monkeypatch.setenv("UNA_BATCH_PHASES", "1")
    rec = telemetry.PhaseRecorder()
    with rec.span("outer"):
        with rec.span("inner"):
            pass
    spans = rec.snapshot()
    # Spans record at completion, so exit order is inner-then-outer; the
    # nesting itself is proven by wall-clock containment.
    by_name = {s["phase"]: s for s in spans}
    assert set(by_name) == {"outer", "inner"}
    outer, inner = by_name["outer"], by_name["inner"]
    assert outer["start_mono_ns"] <= inner["start_mono_ns"]
    assert inner["end_mono_ns"] <= outer["end_mono_ns"]
    for s in spans:
        assert set(s) == SPAN_KEYS
        assert s["end_mono_ns"] >= s["start_mono_ns"] > 0
        assert isinstance(s["proc_cpu_ns"], int) and s["proc_cpu_ns"] >= 0
        assert s["thr_cpu_ns"] is None or (
            isinstance(s["thr_cpu_ns"], int) and s["thr_cpu_ns"] >= 0)


def test_thread_cpu_unavailable_is_null_not_zero(monkeypatch):
    monkeypatch.setenv("UNA_BATCH_PHASES", "1")
    monkeypatch.delattr(time, "thread_time_ns", raising=False)
    rec = telemetry.PhaseRecorder()
    with rec.span("no_thread_clock"):
        pass
    s = rec.snapshot()[0]
    assert s["thr_cpu_ns"] is None
    assert isinstance(s["proc_cpu_ns"], int)


def test_snapshot_returns_copies(monkeypatch):
    monkeypatch.setenv("UNA_BATCH_PHASES", "1")
    rec = telemetry.PhaseRecorder()
    with rec.span("x"):
        pass
    first = rec.snapshot()
    first[0]["phase"] = "mutated"
    assert rec.snapshot()[0]["phase"] == "x"


def test_accumulate_totals_off_and_on(monkeypatch):
    rec = telemetry.PhaseRecorder()
    rec.accumulate("w", 5)                     # disabled: no-op
    assert rec.totals == {}
    monkeypatch.setenv("UNA_BATCH_PHASES", "1")
    rec2 = telemetry.PhaseRecorder()
    rec2.accumulate("w", 5)
    rec2.accumulate("w", 7)
    rec2.accumulate("other", 1)
    assert rec2.totals == {"w": 12, "other": 1}


def test_journal_row_schema_has_no_phases():
    # Checkpoint identity/bytes stay untouched: the journaled row record
    # never gains telemetry fields (diagnostics ride the report only).
    row = JournalRow(phase="COMMITTED", files={}, capture=None,
                     settings={}, timings={})
    assert set(row) == {"phase", "files", "capture", "settings", "timings"}
    assert "phases" not in row
    assert "dropped_events" not in row


# ---------------------------------------------------------------------------
# Integration: public parallel RunBatch over the smoke grid.
# ---------------------------------------------------------------------------

def _report(una):
    assert una.batch_report is not None
    return una.batch_report


def test_default_off_schema_unchanged(make_batch):
    p = make_batch(2)
    p.RunBatch("accessibility", parallel=True, workers=2)
    rep = _report(p)
    for row in rep.rows:
        assert row.phases == ()
        assert row.dropped_events == 0
        assert row.status in ("ran", "skipped")
    # No telemetry keys in extra when disabled (cohort block may exist).
    assert "coordinator_phases" not in rep.extra
    assert "row_transport" not in rep.extra
    assert "telemetry_dropped_events" not in rep.extra


def test_enabled_end_to_end_spans_transport_and_causality(
        make_batch, monkeypatch, tmp_path):
    import os
    from dataclasses import replace

    monkeypatch.setenv("UNA_BATCH_PHASES", "1")
    p = make_batch(2)
    ckpt = str(tmp_path / "ckpt")
    p.RunBatch("accessibility", parallel=True, workers=2,
               execution=replace(p.execution, checkpoint=ckpt))
    rep = _report(p)
    ran = [r for r in rep.rows if r.status == "ran"]
    assert ran, "expected worker-ran rows"

    # 1. Worker phases rode the trusted payload into every outcome.
    for row in ran:
        assert row.phases, "worker spans must reach the outcome"
        assert row.dropped_events == 0
        names = {s["phase"] for s in row.phases}
        assert WORKER_PHASES <= names
        for s in row.phases:
            assert set(s) == SPAN_KEYS
            assert s["end_mono_ns"] >= s["start_mono_ns"] > 0

    # 2. Coordinator spans + transport stamps in extra (H2).
    assert rep.extra["telemetry_dropped_events"] == 0
    coord = {s["phase"] for s in rep.extra["coordinator_phases"]}
    assert COORD_PHASES <= coord
    transport = rep.extra["row_transport"]
    dispatched = {str(r.index) for r in ran}
    assert dispatched <= set(transport)
    for key, stamps in transport.items():
        d, v = stamps.get("dispatch_mono_ns"), stamps.get("recv_mono_ns")
        assert d is not None and v is not None
        assert d <= v, "a row cannot be received before it is dispatched"

    # 3. The recv loop's blocking is a phase TOTAL (high-frequency ticks
    #    must never flood the bounded span buffer), and the buffer held:
    #    nothing was dropped on this small run.
    totals = rep.extra.get("phase_totals", {})
    assert isinstance(totals.get("result_recv_wait_ns"), int)
    assert totals["result_recv_wait_ns"] >= 0
    assert rep.extra["telemetry_dropped_events"] == 0

    # 4. Spans are plain JSON/pickle-able payloads (transport safety).
    payload = json.dumps(
        {"rows": [list(r.phases) for r in ran],
         "coord": list(rep.extra["coordinator_phases"]),
         "transport": rep.extra["row_transport"]})
    assert "phase" in payload
    pickle.dumps(rep)

    # 5. End-to-end journal hygiene: checkpoint row records stay
    #    phase-free even with telemetry on (identity/bytes untouched).
    doc = json.load(open(os.path.join(ckpt, "journal.json"),
                         encoding="utf-8"))
    for rec in (doc.get("rows") or {}).values():
        assert "phases" not in rec
        assert "dropped_events" not in rec


def test_enabled_serial_route_untouched(make_batch, monkeypatch):
    # Serial RunBatch has no phase telemetry (parallel-runtime-only) and
    # must run unaffected with the variable set: its rows are the staged
    # RowOutcome type, which carries no phases field at all.
    monkeypatch.setenv("UNA_BATCH_PHASES", "1")
    p = make_batch(2)
    p.RunBatch("accessibility")
    rep = _report(p)
    for row in rep.rows:
        assert not hasattr(row, "phases")
