"""Simultaneous recursive process-tree RSS sampler (HARNESS.md behavior 7).

A daemon thread walks the coordinator's whole process tree (self + all
descendants, recursively) at a fixed declared interval and records, per
tick, one snapshot taken at a single instant:

  * tree_rss_bytes_sum   — conservative summed RSS (shared pages between
                           processes can double-count; always labelled so)
  * tree_uss_bytes_sum   — where psutil exposes USS (unique set)
  * tree_pss_bytes_sum   — where psutil exposes PSS (proportional set)
  * n_processes          — total process count in the tree
  * per-pid rss/uss/pss/name (capped list)
  * walk overhead (per-tick duration; total reported at stop)

Peaks are SAMPLED peaks, never guaranteed caps.  Watchdog thresholds are
enforced through a threading.Event the pool loop polls:
  * memory budget: summed tree RSS above --memory-budget-mib
  * external pressure stop: system available RAM below
    max(1 GiB, 0.10 * physical)  (RESOURCES.md policy)

Both arms run identical sampling code at the identical declared interval.
"""
from __future__ import annotations

import threading
import time

import psutil

from harness.spec import (
    PRESSURE_STOP_FRACTION,
    PRESSURE_STOP_MIN_BYTES,
    SAMPLER_INTERVAL_S,
)

MAX_STORED_SAMPLES = 4000   # decimation bound for long runs (recorded)
MAX_PIDS_PER_SAMPLE = 64


class WatchdogTrip(Exception):
    def __init__(self, reason: str, detail: dict) -> None:
        super().__init__(reason)
        self.reason = reason
        self.detail = detail


class ProcessTreeSampler:
    """Sample coordinator + all descendants at a fixed declared interval."""

    def __init__(self, root_pid: int | None = None,
                 interval_s: float = SAMPLER_INTERVAL_S,
                 memory_budget_bytes: int | None = None,
                 label: str = "coordinator_tree") -> None:
        self.root_pid = root_pid if root_pid is not None else psutil.Process().pid
        self.interval_s = interval_s
        self.memory_budget_bytes = memory_budget_bytes
        self.label = label

        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._samples: list[dict] = []
        self._total_samples = 0
        self._overhead_seconds = 0.0
        self.peak = {
            "tree_rss_bytes_sum": 0,
            "tree_uss_bytes_sum": None,
            "tree_pss_bytes_sum": None,
            "n_processes": 0,
            "peak_rss_at_utc": None,
        }
        self.trip: WatchdogTrip | None = None
        self.trip_event = threading.Event()
        self._vm = psutil.virtual_memory

    # ------------------------------------------------------------------
    def start(self) -> None:
        self._thread = threading.Thread(
            target=self._loop, name=f"una-tree-sampler-{self.label}", daemon=True)
        self._thread.start()

    def stop(self) -> dict:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        return self.summary()

    # ------------------------------------------------------------------
    def _walk_tree(self) -> dict:
        """One simultaneous snapshot of the whole tree at this instant."""
        root = psutil.Process(self.root_pid)
        procs = [root]
        procs.extend(root.children(recursive=True))  # ALL descendants, recursive

        rss_sum = 0
        uss_sum = 0
        pss_sum = 0
        have_uss = have_pss = False
        per_pid = []
        for proc in procs:
            info = None
            try:
                info = proc.memory_full_info()
            except (psutil.NoSuchProcess, psutil.ZombieProcess):
                continue  # process vanished mid-walk; excluded, counted below
            except psutil.AccessDenied:
                # macOS denies USS/PSS to non-root callers: fall back to the
                # plain RSS metric rather than dropping the process from the
                # tree accounting entirely.
                try:
                    info = proc.memory_info()
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            rss = info.rss
            rss_sum += rss
            uss = getattr(info, "uss", None)
            pss = getattr(info, "pss", None)
            if uss is not None:
                have_uss = True
                uss_sum += uss
            if pss is not None:
                have_pss = True
                pss_sum += pss
            if len(per_pid) < MAX_PIDS_PER_SAMPLE:
                try:
                    per_pid.append({"pid": proc.pid, "name": proc.name()[:40], "rss": rss})
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    per_pid.append({"pid": proc.pid, "name": "?", "rss": rss})
        return {
            "rss_sum": rss_sum,
            "uss_sum": uss_sum if have_uss else None,
            "pss_sum": pss_sum if have_pss else None,
            "n": len(procs),
            "available_bytes": self._available_bytes(),
            "per_pid": per_pid,
        }

    @staticmethod
    def _available_bytes() -> int | None:
        try:
            return psutil.virtual_memory().available
        except psutil.Error:
            return None

    def _loop(self) -> None:
        while not self._stop.is_set():
            t0 = time.perf_counter()
            try:
                snapshot = self._walk_tree()
            except psutil.Error:
                self._stop.wait(self.interval_s)
                continue
            walk_seconds = time.perf_counter() - t0
            now_utc = time.time()

            # System memory context at the SAME instant as the tree walk.
            try:
                sm = psutil.swap_memory()
                swap_used = sm.used
                swap_percent = sm.percent
            except psutil.Error:
                swap_used = None
                swap_percent = None

            with self._lock:
                self._total_samples += 1
                self._overhead_seconds += walk_seconds
                sample = {
                    "t": round(now_utc, 3),
                    "tree_rss_bytes_sum": snapshot["rss_sum"],
                    "tree_uss_bytes_sum": snapshot["uss_sum"],
                    "tree_pss_bytes_sum": snapshot["pss_sum"],
                    "n_processes": snapshot["n"],
                    "system_available_bytes": snapshot["available_bytes"],
                    "swap_used_bytes": swap_used,
                    "swap_percent": swap_percent,
                    "walk_seconds": round(walk_seconds, 6),
                    "per_pid": snapshot["per_pid"],
                }
                if len(self._samples) < MAX_STORED_SAMPLES:
                    self._samples.append(sample)
                else:
                    # Decimate deterministically: keep every 2nd stored sample.
                    del self._samples[::2]
                    self._samples.append(sample)

                if snapshot["rss_sum"] > self.peak["tree_rss_bytes_sum"]:
                    self.peak.update({
                        "tree_rss_bytes_sum": snapshot["rss_sum"],
                        "tree_uss_bytes_sum": snapshot["uss_sum"],
                        "tree_pss_bytes_sum": snapshot["pss_sum"],
                        "n_processes": snapshot["n"],
                        "peak_rss_at_utc": now_utc,
                    })

            if self.memory_budget_bytes is not None and \
                    snapshot["rss_sum"] > self.memory_budget_bytes and \
                    not self.trip_event.is_set():
                self._trip(WatchdogTrip(
                    "memory_budget_exceeded",
                    {"tree_rss_bytes_sum": snapshot["rss_sum"],
                     "budget_bytes": self.memory_budget_bytes,
                     "note": "summed-RSS sampled peak is conservative; shared "
                             "pages can double-count"}))
                return

            vm = self._vm()
            pressure_stop = max(PRESSURE_STOP_MIN_BYTES,
                                PRESSURE_STOP_FRACTION * vm.total)
            if vm.available < pressure_stop and not self.trip_event.is_set():
                self._trip(WatchdogTrip(
                    "external_pressure_stop",
                    {"available_bytes": vm.available,
                     "pressure_stop_available_bytes": pressure_stop}))
                return

            # Hold the declared interval measured from tick start (walk time
            # included), so the declared rate is honest under load.
            elapsed = time.perf_counter() - t0
            self._stop.wait(max(0.001, self.interval_s - elapsed))

    def _trip(self, trip: WatchdogTrip) -> None:
        with self._lock:
            if self.trip is None:
                self.trip = trip
        self.trip_event.set()
        self._stop.set()

    # ------------------------------------------------------------------
    def summary(self) -> dict:
        with self._lock:
            return {
                "label": self.label,
                "root_pid": self.root_pid,
                "interval_s": self.interval_s,
                "metric_semantics": (
                    "conservative summed resident set of coordinator + all "
                    "descendants at one instant; shared pages can double-count"),
                "samples_recorded": len(self._samples),
                "samples_total": self._total_samples,
                "decimated": self._total_samples > len(self._samples),
                "sampler_overhead_seconds": round(self._overhead_seconds, 6),
                "avg_walk_seconds": round(
                    self._overhead_seconds / max(1, self._total_samples), 6),
                "peak": dict(self.peak),
                "trip": (
                    {"reason": self.trip.reason, "detail": self.trip.detail}
                    if self.trip else None),
                "samples": list(self._samples),
                "guarantee": "sampled peaks only; sampling is not a hard cap",
            }
