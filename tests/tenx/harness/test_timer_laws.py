"""H00 negative tests: the timer span laws (benchmarks/una_10x/timing).

Completion criterion: "Harness negative tests prove failures cannot
become valid windows" - every mutant here must be rejected, and the
recomputed tamper check must catch edited retained snapshots.
"""
from __future__ import annotations

import pytest

from benchmarks.una_10x import PublicCallTimer, TimerViolation
from benchmarks.una_10x.timing import (PHASE_IN, PHASE_POST,
                                       timer_violations)


def _run_primary():
    t = PublicCallTimer()
    t.start_primary()
    t.end_primary()
    t.start_post_hash()
    t.end_post_hash()
    return t


def test_happy_window_snapshot_is_clean_and_spans_nested():
    t = PublicCallTimer()
    t.start_primary()
    with t.span("work", detail=1):
        with t.span("inner"):
            pass
    t.end_primary()
    t.start_post_hash()
    t.end_post_hash()
    snap = t.assert_clean()
    assert snap["violations"] == []
    assert snap["primary_s"] > 0 and snap["post_hash_s"] > 0
    outer, inner = snap["spans"]   # spans list is enter-ordered
    assert outer["depth"] == 0 and inner["depth"] == 1
    assert outer["phase"] == inner["phase"] == PHASE_IN
    assert outer["start"] <= inner["start"] <= inner["end"] <= outer["end"]


def test_primary_without_start_is_invalid():
    with pytest.raises(TimerViolation):
        PublicCallTimer().end_primary()


def test_double_primary_start_is_invalid():
    t = PublicCallTimer()
    t.start_primary()
    with pytest.raises(TimerViolation):
        t.start_primary()


def test_open_span_blocks_boundary_completion():
    t = PublicCallTimer()
    t.start_primary()
    cm = t.span("leak")
    cm.__enter__()
    with pytest.raises(TimerViolation):
        t.end_primary()
    cm.__exit__(None, None, None)


def test_pre_primary_span_leaking_into_boundary_is_a_violation():
    t = PublicCallTimer()
    with t.span("early_setup"):
        t.start_primary()          # boundary started inside the span
    t.end_primary()
    t.start_post_hash()
    t.end_post_hash()
    snap = t.snapshot()
    assert any("leaked" in v for v in snap["violations"]), snap["violations"]


def test_post_hash_before_primary_end_is_invalid():
    t = PublicCallTimer()
    with pytest.raises(TimerViolation):
        t.start_post_hash()
    t.start_primary()
    with pytest.raises(TimerViolation):
        t.start_post_hash()
    t.end_primary()
    t.start_post_hash()
    with pytest.raises(TimerViolation):
        t.start_post_hash()
    t.end_post_hash()
    with pytest.raises(TimerViolation):
        t.end_post_hash()


def test_tampered_retained_snapshot_is_rejected():
    # overlapping sibling spans: the "summed twice" defect, injected into
    # a retained record the way a buggy aggregator would
    snap = _run_primary().snapshot()
    s0 = snap["primary_start"]
    snap["spans"] = [
        {"name": "a", "phase": PHASE_IN, "depth": 0, "tags": {},
         "start": s0 + 1, "end": s0 + 3},
        {"name": "b", "phase": PHASE_IN, "depth": 0, "tags": {},
         "start": s0 + 2, "end": s0 + 4},
    ]
    bad = timer_violations(snap)
    assert any("overlap" in v for v in bad), bad
    # a clean copy of the same snapshot must stay clean: the detector is
    # not vacuously failing
    assert timer_violations(_run_primary().snapshot()) == []


def test_escaping_and_backwards_spans_are_violations():
    snap = _run_primary().snapshot()
    ps, pe = snap["primary_start"], snap["primary_end"]
    snap["spans"] = [
        {"name": "escape", "phase": PHASE_IN, "depth": 0, "tags": {},
         "start": ps + 1, "end": pe + 1},
        {"name": "backwards", "phase": PHASE_IN, "depth": 0, "tags": {},
         "start": ps + 2, "end": ps + 1},
        {"name": "premature_hash", "phase": PHASE_POST, "depth": 0,
         "tags": {}, "start": pe - 1, "end": pe + 2},
    ]
    bad = timer_violations(snap)
    assert len(bad) == 3, bad
    assert any("escapes" in v for v in bad)
    assert any("backwards" in v for v in bad)
    assert any("post-primary" in v for v in bad)
