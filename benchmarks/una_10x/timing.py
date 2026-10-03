"""una_10x boundary timer (H00 action: separate diagnostic nested spans,
primary wall and post-hash wall).

BENCHMARKS.md "Primary timer and successful throughput": the primary wall
starts immediately before the required public call and ends after it
returns with the required files, state, validation and checkpoint
guarantees satisfied (public_call_complete).  Post-boundary comparison
hashes sit outside service timing but must still be timed and retained
(post-hash wall).  Diagnostic spans (touches, counters, stage probes) are
recorded separately and must be exclusive or explicitly nested - never
summed into the primary wall.

`timer_violations` is a pure function over snapshot dicts, so *retained*
window records can be re-checked for tampering after the fact; a stored
window whose spans leak into or overlap the primary wall is invalid, not
merely suspicious.
"""
from __future__ import annotations

from contextlib import contextmanager
import time

from benchmarks.platform.errors import HarnessViolation

from .errors import TimerViolation

PHASE_PRE = "pre_primary"
PHASE_IN = "inside_primary"
PHASE_POST = "post_primary"


def timer_violations(snap: dict) -> list[str]:
    """Structural span laws over a snapshot dict (pure; tamper-checkable).

    * a span must run forwards;
    * same-phase sibling/nested spans must be disjoint or contained -
      partial overlap is the "summed twice" defect;
    * inside-primary spans must lie fully within the primary interval;
    * pre-primary spans must close before the primary wall starts;
    * post-primary spans must open after it ends;
    * the post-hash wall must start at or after the primary boundary end.
    """
    bad: list[str] = []
    ps, pe = snap.get("primary_start"), snap.get("primary_end")
    hs, he = snap.get("post_hash_start"), snap.get("post_hash_end")
    for s in snap.get("spans", []):
        name = s.get("name") or "<anon>"
        st, en = s.get("start"), s.get("end")
        if st is None or en is None:
            bad.append(f"span {name!r} not closed")
            continue
        if en < st:
            bad.append(f"span {name!r} runs backwards")
            continue
        phase = s.get("phase")
        if phase == PHASE_IN:
            if ps is not None and st < ps:
                bad.append(f"span {name!r} starts before the primary wall")
            if pe is not None and en > pe:
                bad.append(f"span {name!r} escapes the primary boundary")
        elif phase == PHASE_PRE:
            if ps is not None and en > ps:
                bad.append(f"pre-primary span {name!r} leaked into the "
                           "timed boundary")
        elif phase == PHASE_POST:
            if pe is not None and st < pe:
                bad.append(f"post-primary span {name!r} started inside the "
                           "service boundary")
    spans = [s for s in snap.get("spans", []) if s.get("end") is not None]
    for i in range(len(spans)):
        for j in range(i + 1, len(spans)):
            a, b = spans[i], spans[j]
            if a.get("phase") != b.get("phase"):
                continue
            disjoint = a["end"] <= b["start"] or b["end"] <= a["start"]
            nested = (a["start"] <= b["start"] and b["end"] <= a["end"]) or \
                     (b["start"] <= a["start"] and a["end"] <= b["end"])
            if not disjoint and not nested:
                bad.append("diagnostic spans %r/%r partially overlap "
                           "(exclusive-or-nested law; never summed twice)"
                           % (a.get("name"), b.get("name")))
    if hs is not None and pe is not None and hs < pe:
        bad.append("post-hash wall started inside the primary service "
                   "boundary (hashes must be timed outside service timing)")
    return bad


class PublicCallTimer:
    """Primary wall / post-hash wall / nested diagnostic spans.

    Usage law (BENCHMARKS.md): start_primary() immediately before the
    public call; end_primary() only once required outputs/state are
    satisfied; comparison hashing between start_post_hash()/end_post_hash();
    everything else goes through span() context managers.
    """

    definitions = {
        "primary_wall": "perf_counter; start immediately before the public "
                        "call, end after return with required files, state, "
                        "validation and checkpoint/cache/publication "
                        "guarantees satisfied (public_call_complete)",
        "post_hash_wall": "timed, retained, outside service timing; "
                          "identical policy across arms",
        "diagnostic_spans": "separate, exclusive or explicitly nested; "
                            "never summed into the primary wall",
        "throughput": "N_valid / primary_wall only",
    }

    def __init__(self):
        self._t: dict[str, float] = {}
        self._spans: list[dict] = []
        self._open: list[dict] = []

    # -- primary wall ----------------------------------------------------
    def start_primary(self) -> None:
        if "primary_start" in self._t:
            raise TimerViolation("primary wall already started")
        self._t["primary_start"] = time.perf_counter()

    def end_primary(self) -> None:
        if "primary_start" not in self._t:
            raise TimerViolation("primary wall never started; there is no "
                                 "window to validate")
        if "primary_end" in self._t:
            raise TimerViolation("primary wall ended twice")
        open_spans = [s["name"] for s in self._spans if s["end"] is None]
        if open_spans:
            raise TimerViolation(
                "diagnostic spans still open at end_primary: %r (spans must "
                "close before the boundary completes)" % open_spans)
        self._t["primary_end"] = time.perf_counter()

    # -- post-hash wall ---------------------------------------------------
    def start_post_hash(self) -> None:
        if "primary_end" not in self._t:
            raise TimerViolation("post-hash wall must start only after the "
                                 "primary boundary ends")
        if "post_hash_start" in self._t:
            raise TimerViolation("post-hash wall already started")
        self._t["post_hash_start"] = time.perf_counter()

    def end_post_hash(self) -> None:
        if "post_hash_start" not in self._t:
            raise TimerViolation("post-hash wall never started")
        if "post_hash_end" in self._t:
            raise TimerViolation("post-hash wall ended twice")
        self._t["post_hash_end"] = time.perf_counter()

    # -- diagnostic spans --------------------------------------------------
    @contextmanager
    def span(self, name: str, **tags):
        if "primary_start" not in self._t:
            phase = PHASE_PRE
        elif "primary_end" not in self._t:
            phase = PHASE_IN
        else:
            phase = PHASE_POST
        rec = {"name": name, "phase": phase, "depth": len(self._open),
               "tags": tags, "start": time.perf_counter(), "end": None}
        self._open.append(rec)
        self._spans.append(rec)
        try:
            yield rec
        finally:
            rec["end"] = time.perf_counter()
            self._open.pop()

    # -- snapshot ----------------------------------------------------------
    def snapshot(self) -> dict:
        ps, pe = self._t.get("primary_start"), self._t.get("primary_end")
        hs, he = self._t.get("post_hash_start"), self._t.get("post_hash_end")
        snap = {
            "primary_start": ps,
            "primary_end": pe,
            "primary_s": (pe - ps) if ps is not None and pe is not None
            else None,
            "post_hash_start": hs,
            "post_hash_end": he,
            "post_hash_s": (he - hs) if hs is not None and he is not None
            else None,
            "spans": [dict(s) for s in self._spans],
            "definitions": dict(self.definitions),
        }
        snap["violations"] = timer_violations(snap)
        return snap

    def assert_clean(self) -> dict:
        snap = self.snapshot()
        if snap["violations"]:
            raise TimerViolation("timer law violations: %s"
                                 % snap["violations"])
        return snap
