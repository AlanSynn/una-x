"""Opt-in causal phase telemetry for the parallel batch runtime (post-P00 P01M).

Design 01 (``campaigns/una_10x/post_p00/designs/01_telemetry.md``).  The
recorder is DISABLED unless the environment enables it
(``UNA_PHASE_TELEMETRY=1``): disabled recorders are inert null contexts, so
the default route keeps its exact behavior, report surface, and timings —
diagnostics must never be the comparator (BENCHMARKS.md).  Enabled runs are
diagnostic runs: phase events are runtime-only observations, never
cache/checkpoint identity and never required state (INTERFACES.md's timing
law).

Event schema ``una-10x-phase-event-1``: every counter is a nonnegative
integer on one host's monotonic clock; a CPU clock a platform cannot
observe is ``null``, never a zero observation (design 01).  Nested spans
carry ``parent_phase_id``; summaries sum TOP-LEVEL spans only (same-thread
exclusive spans — a parent's wall includes its children, and process CPU
sampled under overlapping threads must not be added twice).

Buffers are bounded with an explicit ``dropped`` counter: a run whose
counter is nonzero cannot claim interval completeness (design 01).  Worker
processes flush ``result_put`` and pool-loop events to per-process JSONL
files under ``UNA_PHASE_DIR`` as they happen (single short O_APPEND writes
— crash fragments survive); in-row phase events also ride home on the
:class:`~urban_network_analysis.batch.worker.RowResult` for the report
path.

Phase vocabulary at this stage is COARSE: the worker's ``run`` span covers
the whole ``RunAccessibility`` call (per-stage hooks arrive with P02's
opt-in ``Add*`` boundaries); ``source_decode``, ``topology``,
``engine_build`` etc. of design 01 are therefore not yet separable here.
"""

from __future__ import annotations

import itertools
import json
import os
import threading
import time
import uuid
from collections import deque
from contextlib import contextmanager
from typing import Any, Dict, Iterator, Optional

__all__ = ["PhaseRecorder", "telemetry_enabled", "summarize_events", "SCHEMA"]

SCHEMA = "una-10x-phase-event-1"

ENV_ENABLED = "UNA_PHASE_TELEMETRY"
ENV_DIR = "UNA_PHASE_DIR"
ENV_RUN_ID = "UNA_PHASE_RUN_ID"

_MAX_EVENTS_DEFAULT = 4096
_MAX_LINE_BYTES = 1 << 20   # a malformed event must not write an endless line

_ids = itertools.count(1)
_ids_lock = threading.Lock()


def telemetry_enabled() -> bool:
    """True when the environment opted into diagnostics."""
    return os.environ.get(ENV_ENABLED) == "1"


class PhaseRecorder:
    """Bounded per-process phase-event recorder (inert unless enabled).

    Construct through :meth:`maybe` — it returns a recorder only when the
    environment opted in, so disabled call sites share one ``if rec:``
    branch and pay no other cost.  ``phase()`` on a disabled recorder is a
    null context manager, but the intended pattern keeps even that behind
    the ``if rec:`` guard.
    """

    __slots__ = ("run_id", "source", "enabled", "_events", "_dropped",
                 "_stack", "_row_id", "_lock", "_max")

    def __init__(self, run_id: str, source: str,
                 max_events: int = _MAX_EVENTS_DEFAULT) -> None:
        self.run_id = run_id
        self.source = source
        self.enabled = telemetry_enabled()
        self._max = max(1, int(max_events))
        self._events: deque = deque()
        self._dropped = 0
        self._stack: list = []
        self._row_id: Optional[int] = None
        # Reentrant: phase() holds the lock while opening a span and
        # _new_event (and the finally-block's pop) re-enter it.
        self._lock = threading.RLock()

    @classmethod
    def maybe(cls, source: str, run_id: Optional[str] = None,
              max_events: int = _MAX_EVENTS_DEFAULT) -> Optional["PhaseRecorder"]:
        """Recorder when telemetry is enabled, else ``None``.  ``run_id``
        falls back to the coordinator-injected ``UNA_PHASE_RUN_ID`` so
        every process of one run shares the identity."""
        if not telemetry_enabled():
            return None
        return cls(run_id or os.environ.get(ENV_RUN_ID)
                   or uuid.uuid4().hex[:12], source, max_events)

    # -- per-row identity ---------------------------------------------------

    def begin_row(self, row_id: Optional[int]) -> None:
        """Tag subsequent events with the logical row being executed."""
        self._row_id = row_id

    def row_events(self) -> tuple:
        """The buffered events of the current row (bounded; the worker
        attaches this snapshot to the row's ``RowResult``)."""
        with self._lock:
            row_id = self._row_id
            return tuple(e for e in self._events
                         if e.get("row_id") == row_id)

    # -- spans ---------------------------------------------------------------

    # Recorder-owned event keys: contextual ``fields`` may annotate
    # anything else (e.g. row_id) but never rewrite these; a collision is
    # silently ignored — telemetry must not change behavior, even on a
    # caller mistake.
    _PROTECTED = frozenset((
        "schema", "run_id", "source", "pid", "phase_id",
        "parent_phase_id", "phase", "start_mono_ns", "end_mono_ns",
        "process_cpu_start_ns", "process_cpu_end_ns",
        "thread_cpu_start_ns", "thread_cpu_end_ns", "status"))

    @contextmanager
    def phase(self, name: str, **fields: Any) -> Iterator[None]:
        """Record one span around a block.  Exceptions do not suppress the
        event: it is recorded with ``status`` = the exception class name and
        the exception re-raises (telemetry never changes failure behavior).
        Contextual ``fields`` annotate the event (overriding the event's
        default ``row_id`` placeholder); recorder-owned keys above are
        protected and an attempt to set one is ignored.
        """
        if not self.enabled:
            yield
            return
        with self._lock:
            parent = self._stack[-1] if self._stack else None
            event = self._new_event(name, parent)
            self._stack.append(event["phase_id"])
        start_proc = time.process_time_ns()
        try:
            start_thread = time.thread_time_ns()
        except OSError:                          # pragma: no cover — platform
            start_thread = None
        try:
            yield
        except BaseException as exc:             # noqa: BLE001 — recorded, re-raised
            event["status"] = type(exc).__name__
            raise
        else:
            event["status"] = "ok"
        finally:
            try:
                end_thread = time.thread_time_ns()
            except OSError:                      # pragma: no cover — platform
                end_thread = None
            event["process_cpu_end_ns"] = time.process_time_ns()
            event["thread_cpu_end_ns"] = end_thread
            event["thread_cpu_start_ns"] = start_thread
            event["end_mono_ns"] = time.monotonic_ns()
            for key, value in fields.items():
                if key not in self._PROTECTED:
                    event[key] = value
            self._append(event)
            with self._lock:
                if self._stack and self._stack[-1] == event["phase_id"]:
                    self._stack.pop()

    def _new_event(self, name: str, parent: Optional[str]) -> Dict[str, Any]:
        with self._lock:
            phase_id = f"{self.source}-{next(_ids)}"
        return {
            "schema": SCHEMA,
            "run_id": self.run_id,
            "source": self.source,
            "row_id": self._row_id,
            "pid": os.getpid(),
            "phase_id": phase_id,
            "parent_phase_id": parent,
            "phase": name,
            "start_mono_ns": time.monotonic_ns(),
            "end_mono_ns": None,
            "process_cpu_start_ns": time.process_time_ns(),
            "process_cpu_end_ns": None,
            "thread_cpu_start_ns": None,
            "thread_cpu_end_ns": None,
            "status": None,
        }

    def _append(self, event: Dict[str, Any]) -> None:
        with self._lock:
            if len(self._events) >= self._max:
                self._dropped += 1
                return
            self._events.append(event)

    # -- output ---------------------------------------------------------------

    def events_snapshot(self) -> tuple:
        """Bounded event tuple in START order (the buffer fills in
        completion order; start order keeps artifacts deterministic)."""
        with self._lock:
            return tuple(sorted(self._events,
                                key=lambda e: (e["start_mono_ns"],
                                               e["phase_id"])))

    @property
    def dropped(self) -> int:
        with self._lock:
            return self._dropped

    def flush_jsonl(self, path: str, since: int = 0) -> int:
        """Append buffered events from index ``since`` as JSONL (one short
        O_APPEND write per line — crash fragments survive; concurrent
        writers do not tear lines under the local filesystem's
        atomic-append guarantee for small writes).  Returns the number of
        lines written; the caller keeps ``since + written`` as the next
        offset for streaming flushes.  Ordering caveat: snapshots are in
        START order, so a streaming ``since`` offset is stable only while
        span starts are monotonic with completions (true for the
        sequential pool-loop spans this recorder streams; arbitrarily
        overlapping spans could duplicate lines across flushes — the
        phase_id in each line keeps any such duplicates attributable)."""
        if not self.enabled:
            return 0
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        written = 0
        with open(path, "a", encoding="utf-8") as f:
            for event in self.events_snapshot()[since:]:
                line = json.dumps(event, separators=(",", ":"),
                                  default=str)[:_MAX_LINE_BYTES]
                f.write(line + "\n")
                written += 1
        return written

    def summarize(self) -> Dict[str, Dict[str, Any]]:
        """Per-phase aggregates of this recorder's TOP-LEVEL spans."""
        with self._lock:
            events = list(self._events)
        return summarize_events(events)


def summarize_events(events) -> Dict[str, Dict[str, Any]]:
    """Per-phase wall/CPU aggregates over TOP-LEVEL spans only (design 01:
    parents include children; process CPU under overlapping threads must
    never be summed twice — top-level same-thread spans are disjoint by
    construction in each process)."""
    out: Dict[str, Dict[str, Any]] = {}
    for event in events:
        if event["parent_phase_id"] is not None:
            continue
        agg = out.setdefault(event["phase"], {
            "count": 0, "wall_s_sum": 0.0, "process_cpu_s_sum": 0.0,
            "thread_cpu_s_sum": 0.0, "max_wall_s": 0.0,
            "thread_cpu_unavailable": 0})
        agg["count"] += 1
        wall = (event["end_mono_ns"] - event["start_mono_ns"])
        proc = (event["process_cpu_end_ns"]
                - event["process_cpu_start_ns"])
        agg["wall_s_sum"] += wall / 1e9
        agg["process_cpu_s_sum"] += proc / 1e9
        agg["max_wall_s"] = max(agg["max_wall_s"], wall / 1e9)
        if event["thread_cpu_start_ns"] is None \
                or event["thread_cpu_end_ns"] is None:
            agg["thread_cpu_unavailable"] += 1
        else:
            agg["thread_cpu_s_sum"] += (
                    event["thread_cpu_end_ns"]
                    - event["thread_cpu_start_ns"]) / 1e9
    for agg in out.values():
        for key in ("wall_s_sum", "process_cpu_s_sum",
                    "thread_cpu_s_sum", "max_wall_s"):
            agg[key] = round(agg[key], 6)
    return out
