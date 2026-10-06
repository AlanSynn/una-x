"""Bounded phase telemetry for batch diagnostics (post-P00 P01M).

Env-gated by ``UNA_BATCH_PHASES=1``: with the variable unset every
recorder is a no-op and the optional report fields keep their defaults,
so normal runs are unaffected (schema stays default-absent).  The
variable is inherited by spawned worker processes through the normal
environment, so one setting instruments a whole run.

This is diagnostics infrastructure, NOT a second result transport:
worker spans ride inside the existing trusted ``RowResult`` payload and
coordinator spans land in ``BatchExecutionReport.extra`` (post_p00
designs/01_telemetry.md: "Do not log through a second blocking result
channel").  Recorders are bounded — at most ``MAX_SPANS`` spans are
retained; further spans increment ``dropped_events`` instead of growing
without bound.  Diagnostics only: never cache/checkpoint/report
identity, never primary timing evidence.
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional, Tuple

__all__ = ["ENV_VAR", "MAX_SPANS", "PhaseRecorder", "opt_span"]

ENV_VAR = "UNA_BATCH_PHASES"
MAX_SPANS = 64

#: Wall-clock clock domain: the host's monotonic clock (ns), per
#: designs/01 — spans never compare clocks across hosts.
_MONO = time.monotonic_ns


def _cpu_pair() -> Tuple[int, Optional[int]]:
    """(process CPU ns, thread CPU ns or None when the platform lacks
    thread CPU clocks — unavailable is None, never 0)."""
    proc = time.process_time_ns()
    try:
        thread: Optional[int] = time.thread_time_ns()
    except (AttributeError, OSError):
        thread = None
    return proc, thread


class _NullSpan:
    """Shared no-op context manager for disabled telemetry."""

    __slots__ = ()

    def __enter__(self) -> "_NullSpan":
        return self

    def __exit__(self, *exc: Any) -> bool:
        return False


_NULL_SPAN = _NullSpan()


class _Span:
    """One bounded phase span; records itself into its recorder on exit.

    An exception inside the span does not suppress it and the span is
    still recorded (``__exit__`` returns False); diagnostics must not
    alter control flow."""

    __slots__ = ("_rec", "_name", "_t0", "_c0")

    def __init__(self, rec: "PhaseRecorder", name: str) -> None:
        self._rec = rec
        self._name = name
        self._t0 = _MONO()
        self._c0 = _cpu_pair()

    def __enter__(self) -> "_Span":
        return self

    def __exit__(self, *exc: Any) -> bool:
        rec = self._rec
        # Module-global lookup so MAX_SPANS stays patchable in tests.
        if len(rec.spans) >= MAX_SPANS:
            rec.dropped_events += 1
            return False
        end = _MONO()
        c1 = _cpu_pair()
        rec.spans.append({
            "phase": self._name,
            "start_mono_ns": self._t0,
            "end_mono_ns": end,
            "proc_cpu_ns": c1[0] - self._c0[0],
            "thr_cpu_ns": (c1[1] - self._c0[1])
            if c1[1] is not None and self._c0[1] is not None else None,
        })
        return False


class PhaseRecorder:
    """Bounded span collector; a no-op unless telemetry is enabled.

    Single-threaded by construction: the worker's ``_run_row`` and the
    batch coordinator loop each run on one thread, so plain list append
    is safe.  ``snapshot()`` returns copies — the recorder can keep
    mutating after a snapshot is taken for transport.

    High-frequency phases (the coordinator's recv loop ticks every
    ``_TICK_S``) must NOT become spans — they would flood the bounded
    buffer and evict the low-frequency commit phases.  Use
    :meth:`accumulate` for those: a name -> total-ns counter."""

    __slots__ = ("spans", "dropped_events", "totals", "on")

    def __init__(self) -> None:
        self.spans: List[Dict[str, Any]] = []
        self.dropped_events = 0
        self.totals: Dict[str, int] = {}
        self.on: bool = os.environ.get(ENV_VAR) == "1"

    def span(self, name: str):
        if not self.on:
            return _NULL_SPAN
        return _Span(self, name)

    def accumulate(self, name: str, ns: int) -> None:
        """Add ``ns`` to a named total (bounded: one dict entry per
        distinct name, never per event)."""
        if self.on:
            self.totals[name] = self.totals.get(name, 0) + ns

    def snapshot(self) -> Tuple[Dict[str, Any], ...]:
        return tuple(dict(s) for s in self.spans)


def opt_span(recorder: Optional[PhaseRecorder], name: str):
    """``span()`` that tolerates a None recorder (telemetry disabled at
    the caller, or a coordinator running without instrumentation)."""
    return recorder.span(name) if recorder is not None else _NULL_SPAN
