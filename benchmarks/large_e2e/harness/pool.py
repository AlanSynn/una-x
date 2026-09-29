"""Coordinator-side worker pool (HARNESS.md behavior 5).

  * Workers are mp spawn processes (never a fork of an initialized runtime).
  * The input queue is bounded (maxsize = --queue-depth) AND the result
    channel is bounded; dispatch is interleaved with draining — the
    coordinator never enqueues all jobs before reading results.
  * Every put/get uses a short polling timeout; an overall monotonic
    deadline bounds every phase.
  * Worker liveness is checked on every poll; abrupt worker exit fails the
    run promptly, cancels outstanding owned jobs, and retains results
    already received.
  * Shutdown terminates/joins ONLY the campaign-owned children recorded in
    this module; there is no blanket kill.
"""
from __future__ import annotations

import multiprocessing as mp
import queue as queue_module
import time

from harness import worker as worker_module
from harness.spec import (
    POLL_TIMEOUT_S,
    PUT_TIMEOUT_S,
    READY_TIMEOUT_S,
    SHUTDOWN_JOIN_S,
)


class PoolFailure(Exception):
    def __init__(self, reason: str, detail: dict | None = None) -> None:
        super().__init__(reason)
        self.reason = reason
        self.detail = detail or {}


class WorkerPool:
    """Owns W spawned workers; safe, bounded lifecycle for the whole session."""

    def __init__(self, spec, worker_env: dict, worker_config: dict) -> None:
        self.spec = spec
        self.worker_env = worker_env
        self.worker_config = worker_config
        self.ctx = mp.get_context("spawn")   # spawn only: never fork a JIT runtime
        self.job_queue = self.ctx.Queue(maxsize=spec.queue_depth)
        self.result_queue = self.ctx.Queue(maxsize=spec.result_queue_maxsize())
        self.processes: list = []
        self.readies: dict[int, dict] = {}
        self.exits: dict[int, dict] = {}
        self.failure: PoolFailure | None = None
        self.retained_results: list[dict] = []
        self.peak_in_flight = 0
        self.forced_terminations: list[int] = []
        self.started_monotonic: float | None = None

    # ------------------------------------------------------------------
    def spawn_workers(self) -> None:
        self.started_monotonic = time.monotonic()
        for worker_id in range(self.spec.workers):
            proc = self.ctx.Process(
                target=worker_module.spawn_entry,
                args=(worker_id, self.job_queue, self.result_queue,
                      dict(self.worker_config, worker_id=worker_id)),
                daemon=False,   # we terminate/join explicitly, only our own
                name=f"una-e2e-worker-{worker_id}",
            )
            proc.start()
            self.processes.append(proc)

    # ------------------------------------------------------------------
    def _check_liveness(self) -> None:
        """Prompt failure on abrupt worker exit (no infinite readiness wait)."""
        for idx, proc in enumerate(self.processes):
            if proc.exitcode is None:
                continue
            raise PoolFailure(
                "worker_died",
                {"worker_id": idx, "pid": proc.pid,
                 "exitcode": proc.exitcode,
                 "phase": "observed_during_pool_operation"})

    def _check_deadline(self, deadline: float, phase: str) -> None:
        if time.monotonic() > deadline:
            raise PoolFailure("session_timeout",
                              {"phase": phase, "timeout_s": self.spec.timeout_s})

    def _check_trip(self, sampler_summary_fn) -> None:
        if sampler_summary_fn is not None:
            summary = sampler_summary_fn()
            if summary and summary.get("trip"):
                trip = summary["trip"]
                raise PoolFailure(trip["reason"], trip["detail"])

    # ------------------------------------------------------------------
    def wait_ready(self, sampler_summary_fn=None) -> dict:
        """Readiness barrier with a monotonic deadline and liveness polling."""
        deadline = self.started_monotonic + min(READY_TIMEOUT_S,
                                                self.spec.timeout_s)
        while len(self.readies) < self.spec.workers:
            self._check_deadline(deadline, "readiness_barrier")
            self._check_liveness()
            self._check_trip(sampler_summary_fn)
            try:
                message = self.result_queue.get(timeout=POLL_TIMEOUT_S)
            except queue_module.Empty:
                continue
            self._route(message)
        return dict(self.readies)

    # ------------------------------------------------------------------
    def _route(self, message: dict) -> None:
        mtype = message.get("type")
        if mtype in ("worker_ready", "worker_failed_guard", "worker_failed_warmup"):
            self.readies[message["worker_id"]] = message
            if mtype != "worker_ready":
                raise PoolFailure(
                    "worker_startup_failed",
                    {"worker_id": message["worker_id"],
                     "error": message.get("error")})
        elif mtype in ("job_result", "job_failed"):
            self.retained_results.append(message)
        elif mtype == "worker_exit":
            self.exits[message["worker_id"]] = message
        else:
            raise PoolFailure("unexpected_message", {"message_type": mtype})

    # ------------------------------------------------------------------
    def dispatch_and_drain(self, job_specs: list[dict],
                           sampler_summary_fn=None) -> dict:
        """Interleaved bounded dispatch + drain until every job has a result.

        In-flight work is capped at --queue-depth because producers block on
        the bounded queue; the honest batch window therefore includes any
        producer blocking.
        """
        deadline = time.monotonic() + self.spec.timeout_s
        t_dispatch0 = time.monotonic()
        next_job = 0
        outstanding: set[int] = set()
        last_application_event_monotonic: float | None = None
        duplicate_ids: dict[int, int] = {}
        # S01 producer-wait instrumentation (recorded spans, never wall
        # deltas): per-job coordinator-side span around the blocking
        # bounded-queue put, from dispatch attempt to put success.
        producer_waits: dict[int, int] = {}

        while next_job < len(job_specs) or outstanding:
            self._check_deadline(deadline, "dispatch_drain")
            self._check_liveness()
            self._check_trip(sampler_summary_fn)

            dispatched_now = False
            if next_job < len(job_specs) and len(outstanding) < self.spec.queue_depth:
                job_spec = job_specs[next_job]
                job_spec["dispatched_utc"] = time.time()
                t_put_attempt = time.monotonic()
                while True:
                    self._check_liveness()
                    try:
                        self.job_queue.put(job_spec, timeout=PUT_TIMEOUT_S)
                        break
                    except queue_module.Full:
                        self._check_deadline(deadline, "dispatch_put")
                producer_waits[job_spec["job_id"]] = int(
                    (time.monotonic() - t_put_attempt) * 1e9)
                outstanding.add(job_spec["job_id"])
                self.peak_in_flight = max(self.peak_in_flight, len(outstanding))
                next_job += 1
                dispatched_now = True

            # Drain at least one result per loop pass (blocking-ish short
            # poll); drain extra results opportunistically without blocking
            # so bursts never wedge the bounded result channel.
            drained_any = False
            while True:
                try:
                    timeout = POLL_TIMEOUT_S if not (dispatched_now and drained_any) \
                        else 0.0
                    message = self.result_queue.get(timeout=timeout)
                except queue_module.Empty:
                    break
                drained_any = True
                job_id = message.get("job_id")
                if m_is_job(message) and job_id is not None:
                    duplicate_ids[job_id] = duplicate_ids.get(job_id, 0) + 1
                    if message["type"] == "job_result" or message["type"] == "job_failed":
                        outstanding.discard(job_id)
                        if message["type"] == "job_result":
                            last_application_event_monotonic = time.monotonic()
                self._route(message)

            if not dispatched_now and not drained_any:
                time.sleep(POLL_TIMEOUT_S / 4)

        t_all_received = time.monotonic()
        return {
            "dispatch_start_monotonic": t_dispatch0,
            "last_application_event_monotonic": last_application_event_monotonic,
            "all_received_monotonic": t_all_received,
            "duplicate_job_id_counts": {k: v for k, v in duplicate_ids.items() if v > 1},
            "peak_in_flight": self.peak_in_flight,
            "producer_wait_ns_by_job": producer_waits,
            "producer_wait_total_ns": sum(producer_waits.values()),
            "producer_wait_nonzero_jobs": sum(
                1 for v in producer_waits.values() if v > 0),
            "producer_wait_semantics": (
                "coordinator-side recorded span per job around the blocking "
                "bounded-queue put (dispatch attempt to put success), "
                "including put-retry polling; S01 queue-2W criterion consumes "
                "ONLY these spans, never wall deltas"),
        }

    # ------------------------------------------------------------------
    def shutdown(self, sampler_summary_fn=None) -> dict:
        """Graceful sentinel shutdown, bounded; terminate ONLY owned workers.

        During shutdown, worker deaths are recorded (missing exits, nonzero
        exit codes) rather than raised: every owned process is still
        terminate/joined so nothing survives, and the summary carries the
        protocol violation.
        """
        shutdown_wait_s = min(SHUTDOWN_JOIN_S * 2,
                              max(self.spec.timeout_s * 0.5, 5.0))
        deadline = time.monotonic() + shutdown_wait_s
        for _ in self.processes:
            try:
                self.job_queue.put(None, timeout=PUT_TIMEOUT_S)
            except queue_module.Full:
                break
        while len(self.exits) < len(self.processes) and time.monotonic() < deadline:
            try:
                message = self.result_queue.get(timeout=POLL_TIMEOUT_S)
            except queue_module.Empty:
                continue
            if message.get("type") == "worker_exit":
                self._route(message)
        missing_exit = [i for i in range(len(self.processes)) if i not in self.exits]

        join_deadline = time.monotonic() + SHUTDOWN_JOIN_S
        for proc in self.processes:
            proc.join(timeout=max(0.0, join_deadline - time.monotonic()))
        for idx, proc in enumerate(self.processes):
            if proc.is_alive():
                proc.terminate()  # SIGTERM to our own child only
                self.forced_terminations.append(idx)
        for proc in self.processes:
            proc.join(timeout=SHUTDOWN_JOIN_S)

        survivors = [proc.pid for proc in self.processes if proc.is_alive()]
        return {
            "worker_exits": self.exits,
            "missing_worker_exit": missing_exit,
            "forced_terminations": self.forced_terminations,
            "surviving_owned_processes": survivors,
            "exitcodes": [proc.exitcode for proc in self.processes],
        }

    # ------------------------------------------------------------------
    def cancel_and_terminate(self) -> dict:
        """Failure path: cancel outstanding owned jobs, retain received
        results, terminate/join only owned children, never hang."""
        cancelled = 0
        while True:
            try:
                self.job_queue.get(timeout=0.05)
                cancelled += 1
            except queue_module.Empty:
                break
        for proc in self.processes:
            if proc.is_alive():
                proc.terminate()
                self.forced_terminations.append(self.processes.index(proc))
        for proc in self.processes:
            proc.join(timeout=SHUTDOWN_JOIN_S)
        survivors = [proc.pid for proc in self.processes if proc.is_alive()]
        return {
            "cancelled_outstanding_jobs": cancelled,
            "forced_terminations": self.forced_terminations,
            "surviving_owned_processes": survivors,
            "retained_result_count": len(self.retained_results),
        }


def m_is_job(message: dict) -> bool:
    return message.get("type") in ("job_result", "job_failed")
