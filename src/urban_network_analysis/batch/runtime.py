"""Parallel RunBatch coordinator (dossier 04 steps 5-7, dossier 05).

Scheduling model
----------------
``plan_batch`` decides, per row, whether it is *proven independent*
(worker-admissible) or must run serially with a reason.  The coordinator
executes every row in **caller order at a single commit cursor**:

* **worker rows** are dispatched greedily (bounded by
  ``workers + ExecutionOptions.queue_depth`` pending jobs — the parent
  concurrently feeds and drains, never enqueueing the whole workload)
  into persistent spawn-safe worker processes.  Workers write to a
  private staging directory on the same filesystem as the row's public
  output folder; speculation is invisible until commit.
* **serial rows and plan-held error rows** run in the parent process, on
  the caller's live instance, when the cursor reaches them — the exact
  serial statements, so exceptions, settings mutations and artifacts are
  serial-exact by construction.  An error row therefore raises the *real*
  exception the serial loop would raise, after every earlier row has
  committed (finish-the-prefix / earliest-ordered-failure).
* **skipped rows** have their bind transition (output-folder defaulting)
  applied at the cursor, exactly as the serial loop mutates them before
  the skip gate (BATCH_PLAN NOTE-2).  Dispatch never starts rows at or
  beyond a plan-known error position.

At each commit the coordinator: verifies the staged bytes against the
worker's digests, publishes them by ``os.replace`` (stronger than serial,
which writes output files in place one by one) with every published byte
fsynced before its public name flips, applies the row's
settings delta onto the caller's project object (identity preserved),
applies the row's observable flags, folds composite capture in
``plan.fold_order`` (caller order — the fold is float order-sensitive),
appends the checkpoint journal record, re-checks the planner's decided
naming state against the applied state (``_check_prefix_decisions`` — the
executable slice of dossier 04's observable-prefix invariant, checked
after EVERY commit), and only then advances the cursor.
After the last row it rehydrates the final engine/topology state from the
last ran row's worker bundle — never by rerunning the scientific job —
and finalizes the composite.

Public state parity contract: after the call, settings/projects, topology,
engine objects, result flags, ``resolved_gravity_cap``,
``composite_result(s)`` and every published artifact match the serial
route bitwise (tests compare arrays by bytes across W=1,2,4 vs serial).
Two documented diagnostic-only deviations: ``topology.logger.log_list``
holds the last row's worker-process log rather than the serial call
sequence (log timestamps are not bitwise-comparable anyway), and a
*resumed* batch rehydrates its final state through one private
state-only rerun of the last committed row (a crash destroyed the live
bundle; the rerun publishes nothing and double-counts nothing — the
journal's capture payloads, not the rerun, feed the composite fold).
A resume that re-runs rows AFTER the prior run had already finalized
the composite re-folds the FULL capture set from the journal bytes in
``plan.fold_order`` before re-finalizing — never just the re-run rows'
captures, which would overwrite the published composite with a subset
rebuild (BATCH_REVIEW MAJOR-1).

Cancellation (deadline, KeyboardInterrupt, worker loss) stops dispatch,
reaps the pool, removes this run's staging directories by ownership
(never any other path), preserves the committed prefix and raises
:class:`~urban_network_analysis.batch.report.BatchCancelledError` with
the prefix's outcomes attached.  ``UNA.batch_report`` stays None on any
raise, matching the serial route.

Transportability is an admission requirement in both directions: the
coordinator probe-pickles each row job before dispatch (an untransportable
job fails its own row loudly through the normal ordering path instead of
silently killing the inbox feeder thread), and the worker degrades an
untransportable result (drop the live exception object, then reduce to a
text-only failure descriptor) so the failure always reaches the
coordinator with the original type/message.  Results travel over a
SYNCHRONOUS channel (``SimpleQueue``): a worker's result put reaches the
pipe before that worker starts the next job, so an abrupt worker death
(fault injection, OOM kill) can never truncate an already-completed
sibling result that the drain-then-raise protocol is owed.
"""

from __future__ import annotations

import base64
import builtins
import copy
import io
import json
import multiprocessing as mp
import os
import pickle
import queue
import shutil
import tempfile
import time
import uuid
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from ..Execution import (EffectiveExecution, ExecutionOptions, RowOutcome)
from .checkpoint import (CheckpointJournal, JournalRow, _canonical,
                         sha256_bytes, sha256_file)
from .plan import BatchPlan, InvariantViolation, plan_batch
from .report import (BatchCancelledError, BatchCheckpointError,
                     BatchExecutionReport, RowExecutionOutcome)
from .telemetry import PhaseRecorder, opt_span
from .worker import (RowJob, RowResult, _unpack_state_bundle)

__all__ = ["run_batch_parallel"]

_TICK_S = 0.02
_READY_TIMEOUT_S = 120.0
_IN_WORKER_ENV = "_UNA_BATCH_IN_WORKER"


# ---------------------------------------------------------------------------
# Batch identity (dossier 05: code/model/profile/input — never timings).
# ---------------------------------------------------------------------------

def _package_dir() -> str:
    from .. import __file__ as pkg_init
    return os.path.dirname(os.path.abspath(pkg_init))


def _code_identity() -> Dict[str, str]:
    """sha256 of every .py in the installed package, keyed by relpath."""
    root = _package_dir()
    out: Dict[str, str] = {}
    for dirpath, dirnames, names in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d != "__pycache__")
        for name in sorted(names):
            if name.endswith(".py"):
                full = os.path.join(dirpath, name)
                out[os.path.relpath(full, root)] = sha256_file(full)
    return out


def _input_identity(plan: BatchPlan) -> Dict[str, Dict[str, str]]:
    out: Dict[str, Dict[str, str]] = {}
    for rp in plan.rows:
        if rp.row.input_paths:
            out[str(rp.row.index)] = {p: sha256_file(p)
                                      for p in rp.row.input_paths}
    return out


def _settings_closure(plan: BatchPlan, una) -> Dict[str, dict]:
    """Canonical projection of every row's Settings state — the identity's
    ``model`` quarter (dossier 05/06): each row's result depends on its
    whole Settings closure (radius/weights/OD model, naming fields,
    composite flags), so a changed value must reject resume rather than
    silently reuse the stale committed row.  Computed only for
    checkpointed batches (the only identity consumers), whose journal
    writes already require every value to have a canonical form."""
    out: Dict[str, dict] = {}
    for rp in plan.rows:
        s = una.projects[rp.row.index]
        # ``inject_specs`` is a test-transport field (see _job_for).
        state = {k: v for k, v in vars(s).items() if k != "inject_specs"}
        out[str(rp.row.index)] = _canonical(state)
    return out


def _plan_summary(plan: BatchPlan) -> dict:
    return {
        "rows": [{"index": rp.row.index,
                  "validation": rp.validation.status,
                  "admission": rp.admission} for rp in plan.rows],
        "commit_order": list(plan.commit_order),
    }


def _transport_identity() -> Dict[str, str]:
    """Fingerprints of pieces the package walk cannot see: the stdlib-only
    spawn shim (outside the package dir) and the numerical environment
    (a different python/numpy can change float behavior)."""
    import platform

    import una_batch_worker_main
    return {
        "spawn_shim": sha256_file(una_batch_worker_main.__file__),
        "python": platform.python_version(),
        "numpy": np.__version__,
    }


def _batch_identity(plan: BatchPlan, analysis: str,
                    effective: EffectiveExecution, una) -> Tuple[str, dict]:
    detail = {
        "analysis": analysis,
        "semantic_profile": effective.options.semantic_profile,
        "backend": effective.backend,
        "code": _code_identity(),
        "inputs": _input_identity(plan),
        "settings": _settings_closure(plan, una),
        "plan": _plan_summary(plan),
        "transport": _transport_identity(),
    }
    blob = json.dumps(detail, sort_keys=True).encode("utf-8")
    return sha256_bytes(blob), detail


# ---------------------------------------------------------------------------
# Staging (private, same-filesystem, ownership-bounded cleanup).
# ---------------------------------------------------------------------------

class _Staging:
    """Per-run private staging directories, cleaned up by ownership.

    Staging lives under the row's public output folder so publishing is a
    same-filesystem ``os.replace``; the hidden root name carries this
    run's id, and cleanup touches only paths we created.
    """

    def __init__(self, run_id: str):
        self.run_id = run_id
        self._roots: Dict[str, str] = {}
        self._dirs: Dict[int, str] = {}

    def dir_for(self, row_index: int, public_folder: str) -> str:
        base = os.path.abspath(public_folder)
        root = self._roots.get(base)
        if root is None:
            root = os.path.join(base, f".una-batch-staging-{self.run_id}")
            self._roots[base] = root
        d = os.path.join(root, f"row{row_index}")
        self._dirs[row_index] = d
        return d

    def discard(self, row_index: int) -> None:
        d = self._dirs.pop(row_index, None)
        if d and os.path.isdir(d):
            shutil.rmtree(d, ignore_errors=True)

    def cleanup(self) -> None:
        for idx in list(self._dirs):
            self.discard(idx)
        for root in self._roots.values():
            if not os.path.isdir(root):
                continue
            for name in os.listdir(root):
                if name.startswith("row"):
                    shutil.rmtree(os.path.join(root, name),
                                  ignore_errors=True)
            try:
                os.rmdir(root)
            except OSError:
                pass


def _fsync_file(path: str) -> None:
    """Make a staged file's bytes durable BEFORE its public name flips
    (an os.replace is atomic for the name, not for unflushed data)."""
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _fsync_dir(path: str) -> None:
    """Best-effort directory fsync (same convention as the checkpoint
    journal's): some filesystems refuse directory fsync; the rename's
    durability then rides on the file fsync plus the journal record,
    which carries every published file's digest for resume
    verification either way."""
    try:
        dfd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(dfd)
    except OSError:
        pass
    finally:
        os.close(dfd)


def _publish_staged(staging_dir: str, public_folder: str,
                    files: Tuple[Tuple[str, int, str], ...]) -> Tuple[str, ...]:
    """Publish staged artifacts by ``os.replace`` — atomic per file
    (multi-file rows appear file-by-file; the journal record naming the
    full set is written after, so resume verification never accepts a
    partially published row).  Every published byte is fsynced before
    its public name flips, and the destination directory afterwards."""
    published = []
    dest_dirs = []
    for relpath, _size, _digest in files:
        src = os.path.join(staging_dir, relpath)
        dest = os.path.join(public_folder, relpath)
        d = os.path.dirname(dest) or public_folder
        os.makedirs(d, exist_ok=True)
        _fsync_file(src)
        os.replace(src, dest)
        published.append(dest)
        dest_dirs.append(d)
    for d in dict.fromkeys(dest_dirs):   # distinct dirs, first-seen order
        _fsync_dir(d)
    return tuple(sorted(published))


def _verify_staged(staging_dir: str,
                   files: Tuple[Tuple[str, int, str], ...]) -> None:
    """Coordinator-side digest verification of the staged bytes.

    A failure is a COMMIT INTEGRITY error (the worker produced or left
    corrupt bytes), never a user cancellation — raising
    :class:`BatchCheckpointError` keeps ``except BatchCancelledError``
    handlers from swallowing corruption as a benign cancel
    (BATCH_REVIEW MINOR-2).  The batch still stops: the coordinator's
    generic failure path cancels dispatch and preserves the committed
    prefix."""
    for relpath, size, digest in files:
        path = os.path.join(staging_dir, relpath)
        if not os.path.exists(path):
            raise BatchCheckpointError(
                f"staged artifact {relpath!r} vanished before commit")
        if os.path.getsize(path) != size or sha256_file(path) != digest:
            raise BatchCheckpointError(
                f"staged artifact {relpath!r} failed digest verification "
                f"before commit; refusing to publish")


# ---------------------------------------------------------------------------
# Worker pool (persistent spawn processes, readiness-gated, reaped).
# ---------------------------------------------------------------------------

class _ResultChannel:
    """Synchronous worker->coordinator result transport.

    ``multiprocessing.Queue.put`` only hands the object to a feeder
    thread; the bytes reach the pipe when that thread happens to be
    scheduled.  A worker that dies abruptly (the tests' fault injection,
    an OOM kill) before the flush truncates every result still buffered
    in its process — the closure regression demonstrated a COMPLETED
    row's result being lost exactly this way, so the crash-resume prefix
    came back empty.  ``SimpleQueue.put`` pickles and writes the result
    to the pipe synchronously under a lock (multi-writer safe on POSIX),
    so once it returns, death cannot lose the result.  ``get`` restores
    ``Queue``'s timeout semantics via the reader's poll.
    """

    def __init__(self, ctx):
        self._q = ctx.SimpleQueue()

    def put(self, obj) -> None:
        self._q.put(obj)

    def get(self, timeout: Optional[float] = None):
        if timeout is not None and not self._q._poll(timeout):
            raise queue.Empty
        return self._q.get()

    # Queue-compatible teardown surface (``_Pool.cancel`` closes every
    # channel).  A synchronous transport holds nothing in a buffer, so
    # there is no feeder thread to join.
    def close(self) -> None:
        self._q.close()

    def join_thread(self) -> None:
        pass


class _Pool:
    def __init__(self, size: int, threads_per_worker: int):
        self.size = size
        self.threads_per_worker = threads_per_worker
        self.ctx = mp.get_context("spawn")
        self.inbox = self.ctx.Queue()
        self.outbox = _ResultChannel(self.ctx)
        self.ready = self.ctx.Queue()
        self.procs: List[mp.process.BaseProcess] = []
        self.pids: set = set()
        self.in_flight = 0
        self.cancelled = False

    def start(self) -> None:
        # Spawn target = the stdlib-only shim: it applies the thread
        # budgets BEFORE its first heavy import (a target inside the
        # package would import numpy — and OpenBLAS's environment-sized
        # pool — before pool_main could set anything).
        from una_batch_worker_main import pool_main as _spawn_main
        for _ in range(self.size):
            p = self.ctx.Process(
                target=_spawn_main,
                args=(self.inbox, self.outbox, self.ready,
                      self.threads_per_worker, 0),
                daemon=False)   # engines may spawn their own pools
            p.start()
            self.procs.append(p)
        deadline = time.monotonic() + _READY_TIMEOUT_S
        readied = 0
        while readied < self.size:
            if time.monotonic() > deadline:
                self.cancel()
                raise BatchCancelledError(
                    "row-worker pool failed to become ready")
            try:
                msg, pid = self.ready.get(timeout=_TICK_S)
            except queue.Empty:
                if any(not p.is_alive() for p in self.procs):
                    self.cancel()
                    raise BatchCancelledError(
                        "a row worker died before signalling readiness")
                continue
            if msg == "ready":
                self.pids.add(pid)
                readied += 1

    def dispatch(self, job: RowJob) -> None:
        self.inbox.put(job)
        self.in_flight += 1

    def recv(self, timeout: float) -> RowResult:
        result = self.outbox.get(timeout=timeout)
        self.in_flight -= 1
        return result

    def any_dead(self) -> bool:
        return any(not p.is_alive() for p in self.procs)

    def close(self) -> None:
        if self.cancelled:
            return          # cancel() already reaped workers and closed queues
        for _ in self.procs:
            self.inbox.put(None)
        deadline = time.monotonic() + 10.0
        for p in self.procs:
            p.join(timeout=max(0.0, deadline - time.monotonic()))
        for p in self.procs:
            if p.is_alive():
                p.terminate()
        for p in self.procs:
            p.join(timeout=5.0)

    def cancel(self) -> None:
        self.cancelled = True
        for p in self.procs:
            if p.is_alive():
                p.terminate()
        for p in self.procs:
            p.join(timeout=5.0)
        for q in (self.inbox, self.outbox, self.ready):
            try:
                q.close()
                q.join_thread()
            except (ValueError, OSError):
                pass


# ---------------------------------------------------------------------------
# Settings delta application (identity-preserving).
# ---------------------------------------------------------------------------

def _decode_journal_value(value: Any) -> Any:
    """Invert ``checkpoint._canonical`` for values the journal stored in
    canonical form: ndarray markers become fresh, writable arrays with
    exactly the recorded dtype/shape/bytes (C-order)."""
    if isinstance(value, dict) and set(value) == {"__ndarray__"}:
        spec = value["__ndarray__"]
        raw = base64.b64decode(spec["b64"])
        arr = np.frombuffer(raw, dtype=spec["dtype"]).reshape(
            spec["shape"])
        return arr.copy()          # frombuffer is read-only
    return value


def _apply_settings_delta(obj, post: Dict[str, Any]) -> None:
    """Apply a worker's post-run Settings snapshot onto the caller's
    project object, preserving object identity (the serial loop mutates
    the project's own object; it never replaces it)."""
    for key, raw in post.items():
        value = _decode_journal_value(raw)
        try:
            current = getattr(obj, key)
        except AttributeError:
            setattr(obj, key, value)
            continue
        if current is value:
            continue
        if hasattr(current, "__dataclass_fields__") \
                and isinstance(value, dict):
            # Frozen dataclass (e.g. ``execution``) restored from a
            # journal's canonical dict: engines never mutate it, so the
            # caller's instance is already correct — never swap the class
            # for its dict projection.
            continue
        try:
            if current == value:
                continue
        except Exception:
            pass
        setattr(obj, key, value)


# ---------------------------------------------------------------------------
# Journal helpers.
# ---------------------------------------------------------------------------

def _journal_row(journal: Optional[CheckpointJournal], row_index: int,
                 phase: str, files: Optional[dict] = None,
                 capture: Optional[dict] = None,
                 settings: Optional[dict] = None,
                 timings: Optional[dict] = None) -> None:
    if journal is None:
        return
    journal.record_row(row_index, JournalRow(
        phase=phase,
        files=files or {},
        capture=capture,
        settings=settings,
        timings=timings or {}))


def _check_prefix_decisions(plan: BatchPlan, i: int, una,
                            published: Tuple[str, ...]) -> None:
    """Executable slice of the observable-prefix invariant, checked after
    EVERY commit (dossier 04: "check after every committed row"): the
    applied public naming state must equal the planner's decided values
    for this row.  A mismatch means the coordinator drifted from the plan
    — a coding error, raised loudly as InvariantViolation."""
    d = plan.rows[i].row
    s = una.projects[i]
    if s.output_folder != d.output_folder:
        raise InvariantViolation(
            row=i,
            detail=f"output folder {s.output_folder!r} != planned "
                   f"{d.output_folder!r}")
    if s.output_file_name != d.output_stem:
        raise InvariantViolation(
            row=i,
            detail=f"output stem {s.output_file_name!r} != planned "
                   f"{d.output_stem!r}")
    base = os.path.abspath(d.output_folder) + os.sep if d.output_folder \
        else None
    for f in published:
        if base is None or not os.path.abspath(f).startswith(base):
            raise InvariantViolation(
                row=i,
                detail=f"published artifact {f!r} outside the planned "
                       f"output folder {d.output_folder!r}")


def _dir_file_map(folder: str) -> Dict[str, dict]:
    out = {}
    if os.path.isdir(folder):
        for root, _dirs, names in os.walk(folder):
            for name in names:
                path = os.path.join(root, name)
                out[os.path.relpath(path, folder)] = {
                    "bytes": os.path.getsize(path),
                    "sha256": sha256_file(path),
                }
    return out


def _npy_bytes(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    np.save(buf, np.asarray(arr), allow_pickle=False)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Job construction.
# ---------------------------------------------------------------------------

def _job_for(una, plan: BatchPlan, i: int, staging_dir: str, analysis: str,
             script_output_folder: Optional[str],
             need_state: bool,
             cohort_root: Optional[str] = None,
             cohort_residency_cap_bytes: int = 0,
             prepared_root: Optional[str] = None) -> RowJob:
    s = una.projects[i]
    # ``inject_specs`` is a test-transport field, not a Settings field: it
    # must never reach the worker's Settings (Validation() rejects unknown
    # attributes) — the snapshot carries the row's real state only.
    state = {k: v for k, v in vars(s).items() if k != "inject_specs"}
    return RowJob(
        job_id=uuid.uuid4().hex,
        row_index=i,
        analysis=analysis,
        settings_state=copy.deepcopy(state),
        script_output_folder=script_output_folder,
        staging_dir=staging_dir,
        capture=bool(una._composite_active and s.batch_composite_output),
        need_state=need_state,
        verbosity=0,
        cohort_root=cohort_root,
        cohort_residency_cap_bytes=cohort_residency_cap_bytes,
        prepared_root=prepared_root,
    )


# ---------------------------------------------------------------------------
# Commits.
# ---------------------------------------------------------------------------

def _commit_worker_row(una, plan: BatchPlan, journal, i: int,
                       result: RowResult, public_folder: str,
                       effective: EffectiveExecution,
                       outcomes: Dict[int, RowOutcome],
                       manifests: Dict[int, Tuple[str, ...]],
                       bundles: Dict[int, dict],
                       tele: Optional[PhaseRecorder] = None) -> None:
    with opt_span(tele, "staged_verify"):
        _verify_staged(result.staging_dir, result.files)
    os.makedirs(public_folder, exist_ok=True)
    with opt_span(tele, "publish"):
        published = _publish_staged(result.staging_dir, public_folder,
                                    result.files)

    with opt_span(tele, "state_delta"):
        _apply_settings_delta(una.projects[i], result.settings_post)
        una.settings = una.projects[i]
        una.has_flow_results = result.has_flow_results
        una.has_centrality_results = result.has_centrality_results
        if result.cap_resolved:
            una.resolved_gravity_cap = result.resolved_gravity_cap
        if result.capture is not None:
            una._composite_captured.append(result.capture)
        if result.state_bundle is not None:
            bundles[i] = _unpack_state_bundle(result.state_bundle)

    capture_meta = None
    if journal is not None and result.capture is not None:
        with opt_span(tele, "composite"):
            capture_meta = {
                **journal.write_capture(i, result.capture[0],
                                        _npy_bytes(result.capture[1])),
                "column": result.capture[0],
                "metric": result.capture[2],
                "join_path": result.capture[3],
            }

    outcomes[i] = RowExecutionOutcome(
        index=i, name=plan.rows[i].row.name, status="ran",
        semantic_profile=plan.rows[i].row.semantic_profile,
        backend=effective.backend,
        admission=plan.rows[i].admission,
        admission_reasons=plan.rows[i].admission_reasons,
        phase="COMMITTED",
        worker_pid=result.worker_pid,
        peak_rss_bytes=result.peak_rss_bytes,
        output_files=published,
        timings=dict(result.timings),
        phases=result.phases,
        dropped_events=result.dropped_events)
    manifests[i] = published

    if journal is not None:
        with opt_span(tele, "journal"):
            _journal_row(journal, i, phase="COMMITTED",
                         files={rel: {"bytes": size, "sha256": digest}
                                for rel, size, digest in result.files},
                         capture=capture_meta,
                         settings={k: v for k, v
                                   in result.settings_post.items()},
                         timings=dict(result.timings))
    _check_prefix_decisions(plan, i, una, published)


def _commit_parent_row(una, plan: BatchPlan, journal, i: int,
                       analysis: str, script_output_folder: Optional[str],
                       effective: EffectiveExecution, n: int,
                       outcomes: Dict[int, RowOutcome],
                       manifests: Dict[int, Tuple[str, ...]]) -> None:
    """The exact serial row statements, executed on the caller's instance
    once everything before the cursor has committed.  Exceptions here are
    the serial exceptions and propagate verbatim."""
    s = una.projects[i]
    una.settings = s

    if not (s.output_folder or '').strip():
        s.output_folder = script_output_folder or \
            os.path.join(s.data_folder, "Results")

    missing = [f for f in ('network_file', 'origins_file', 'destinations_file')
               if not (getattr(s, f, None) or '').strip()]
    if missing:
        una.topology.logger.log(
            'RunBatch',
            f"Row {i + 1}/{n}: skipped — missing required fields: {missing}",
            v=1)
        outcomes[i] = RowExecutionOutcome(
            index=i, name=s.name, status="skipped",
            semantic_profile=s.execution.semantic_profile,
            backend=effective.backend,
            detail=f"missing required fields: {missing}",
            admission=plan.rows[i].admission,
            admission_reasons=plan.rows[i].admission_reasons,
            phase="SKIPPED", worker_pid=0)
        _journal_row(journal, i, phase="SKIPPED")
        return

    if s.output_file_name == "Results":
        s.output_file_name = s.name

    una.topology.logger.log(
        'RunBatch', f"Row {i + 1}/{n}: '{s.name}'", v=1)

    before = _dir_file_map(s.output_folder) if journal is not None else None

    if analysis == 'flow':
        una.RunFlow()
        engine = una.flow
    else:
        una.RunAccessibility()
        engine = una.accessibility

    una._capture_batch_row(s, engine)

    files_meta = None
    capture_meta = None
    if journal is not None:
        after = _dir_file_map(s.output_folder)
        files_meta = {rel: meta for rel, meta in after.items()
                      if before.get(rel) != meta}
        capture_meta = None
        if una._composite_active and s.batch_composite_output \
                and una._composite_captured:
            latest = una._composite_captured[-1]
            capture_meta = {
                **journal.write_capture(i, latest[0],
                                        _npy_bytes(latest[1])),
                "column": latest[0],
                "metric": latest[2],
                "join_path": latest[3],
            }

    outcomes[i] = RowExecutionOutcome(
        index=i, name=s.name, status="ran",
        semantic_profile=s.execution.semantic_profile,
        backend=effective.backend,
        admission=plan.rows[i].admission,
        admission_reasons=plan.rows[i].admission_reasons,
        phase="COMMITTED", worker_pid=0,
        output_files=tuple(sorted(
            os.path.join(s.output_folder, rel)
            for rel in (files_meta or {}))) if journal is not None else ())
    if journal is not None:
        manifests[i] = tuple(sorted(
            os.path.join(s.output_folder, rel) for rel in files_meta))
        _journal_row(journal, i, phase="COMMITTED", files=files_meta,
                     capture=capture_meta,
                     settings=copy.deepcopy(vars(s)))
    else:
        manifests[i] = ()


# ---------------------------------------------------------------------------
# Rehydration (dossier 04 step 7 — never rerun the scientific job).
# ---------------------------------------------------------------------------

def _state_only_rerun(una, plan: BatchPlan, i: int, analysis: str,
                      script_output_folder: Optional[str],
                      threads_per_worker: int,
                      run_timeout_s: float) -> Optional[dict]:
    """Rebuild final engine/topology state for a resumed batch by running
    the last committed row once more with publication disabled (throwaway
    staging).  Publishes nothing; the journal's capture payloads, not this
    run, feed the composite fold.  Returns the state bundle or None."""
    if not plan.rows[i].row.exact_settings_type:
        return None            # a Settings subclass may behave differently
    tmp = tempfile.mkdtemp(prefix="una-batch-state-")
    try:
        ctx = mp.get_context("spawn")
        inbox, outbox, ready = ctx.Queue(), ctx.Queue(), ctx.Queue()
        from una_batch_worker_main import pool_main as _spawn_main
        p = ctx.Process(target=_spawn_main,
                        args=(inbox, outbox, ready, threads_per_worker, 0))
        p.start()
        try:
            ready.get(timeout=_READY_TIMEOUT_S)
            job = _job_for(una, plan, i, os.path.join(tmp, "row"),
                           analysis, script_output_folder, need_state=True)
            inbox.put(job)
            result = outbox.get(timeout=run_timeout_s)
        finally:
            inbox.put(None)
            p.join(timeout=5.0)
            if p.is_alive():
                p.terminate()
                p.join(timeout=5.0)
        if result is not None and result.status == "ran" \
                and result.state_bundle is not None:
            return _unpack_state_bundle(result.state_bundle)
        return None
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _rehydrate(una, plan: BatchPlan, bundles: Dict[int, dict],
               outcomes: Dict[int, RowOutcome], journal, effective,
               analysis: str, script_output_folder: Optional[str],
               notes_runtime: List[str], parent_ran: set) -> None:
    """Apply the last ran committed row's observable state to the caller's
    instance.  Rows that ran in the parent this call already hold live
    state; worker rows come from their bundle; any other row (resumed
    batch) falls back to one state-only rerun."""
    ran = [i for i in plan.commit_order
           if getattr(outcomes.get(i), "status", None) == "ran"]
    if not ran:
        return
    last = ran[-1]

    bundle = bundles.get(last)
    if bundle is None and last not in parent_ran:
        # Neither a bundle nor live state exists (resume path): one
        # private state-only rerun, publishing nothing (documented
        # deviation).  The budget scales with the row's recorded runtime
        # so a legitimately long final row is not killed by the spawn
        # readiness constant.
        record = ((journal.load() if journal is not None else None)
                  or {}).get("rows", {}).get(str(last)) or {}
        recorded = sum(float(v) for v in
                       (record.get("timings") or {}).values()
                       if isinstance(v, (int, float)))
        budget = max(_READY_TIMEOUT_S, 2.0 * recorded + 60.0)
        bundle = _state_only_rerun(una, plan, last, analysis,
                                   script_output_folder,
                                   effective.options.threads_per_worker,
                                   run_timeout_s=budget)
        if bundle is None:
            # Serial leaves the last row's live engine/topology state on
            # the caller's instance; failing to rehydrate would deviate
            # from that public state silently — refuse loudly instead
            # (dossier 04 step 7).  The journal is untouched: a retry
            # reuses every committed row and re-attempts only this.
            raise BatchCancelledError(
                f"final engine/topology state could not be rehydrated "
                f"(state-only rerun of row {last + 1} unavailable or "
                f"unsuccessful)")
        notes_runtime.append(
            f"row {last + 1}: final state rehydrated via a private "
            f"state-only rerun (resume path; no publication, no double "
            f"count)")
    if bundle is None:
        return            # parent-run row this call: live state on `una`

    una.topology = bundle["topology"]
    if bundle["analysis"] == 'flow':
        una.flow = bundle["engine"]
    else:
        una.accessibility = bundle["engine"]
    una.has_flow_results = bundle["has_flow_results"]
    una.has_centrality_results = bundle["has_centrality_results"]
    if bundle["cap_resolved"]:
        una.resolved_gravity_cap = bundle["resolved_gravity_cap"]


# ---------------------------------------------------------------------------
# Entry point.
# ---------------------------------------------------------------------------

def run_batch_parallel(una, analysis: str, *, workers: Optional[int],
                       requested: ExecutionOptions,
                       effective: EffectiveExecution,
                       script_output_folder: Optional[str],
                       notes: Tuple[str, ...],
                       _on_commit: Optional[Callable[[int], None]] = None
                       ) -> None:
    """Execute ``una.projects`` in parallel; raise like the serial route on
    failure.  Sets ``una.batch_report`` only on success (serial parity).

    Guarded entry point (dossier 05): nested parallel batches are refused
    — a worker process re-running user module-level code that calls
    ``RunBatch(parallel=True)`` again fails here instead of forking a
    recursion of pools.  Caller-side entry scripts still follow the
    standard multiprocessing-spawn rule of an ``if __name__ == "__main__"``
    guard around module-level code."""
    if os.environ.get(_IN_WORKER_ENV) == "1":
        raise RuntimeError(
            "RunBatch(parallel=True) was re-entered inside a batch worker "
            "process; nested parallel batches are not supported (guard "
            "module-level entry code with `if __name__ == '__main__'`)")
    projects = una.projects
    plan = plan_batch(projects, script_output_folder=script_output_folder,
                      analysis=analysis)
    n = len(plan.rows)
    commit_order = tuple(plan.commit_order)
    error_at = next((rp.row.index for rp in plan.rows
                     if rp.validation.status == "error"), None)

    journal: Optional[CheckpointJournal] = None
    committed: set = set()
    notes_runtime: List[str] = list(notes)
    if requested.checkpoint:
        journal = CheckpointJournal(requested.checkpoint)
        identity, identity_detail = _batch_identity(plan, analysis, effective,
                                                    una)
        doc = journal.load()
        if doc is None:
            journal.create(identity, identity_detail)
        else:
            journal.verify_identity(identity)
            committed = journal.committed_row_indexes()
            for idx in sorted(committed):
                # Artifacts AND capture payloads verify against their
                # recorded digests; a row failing either is dropped from
                # the generation and re-run (never mixed in unverified).
                if not journal.verify_row_artifacts(
                        idx, _public_folder(plan, idx)) \
                        or not journal.verify_row_capture(idx):
                    doc = journal.load()
                    rows = dict(doc["rows"])
                    rows.pop(str(idx), None)
                    journal._write({**doc, "rows": rows})
                    committed.discard(idx)
            if committed:
                notes_runtime += [
                    f"resuming from checkpoint generation "
                    f"{(journal.load() or {}).get('generation')}: "
                    f"{len(committed)} committed row(s) reused"]
    composite_already_finalized = bool(journal and journal.composite_finalized())

    dispatchable = []
    for i in plan.worker_admissible:
        rp = plan.rows[i]
        if rp.validation.status != "ok":
            continue
        if error_at is not None and i >= error_at:
            continue
        if i in committed:
            continue
        dispatchable.append(i)
    dispatch_set = set(dispatchable)

    ran_todo = [i for i in commit_order
                if i not in committed
                and plan.rows[i].validation.status == "ok"]
    expected_last = ran_todo[-1] if ran_todo else \
        (commit_order[-1] if commit_order else None)

    workers_effective = _effective_workers(workers, effective,
                                           len(dispatchable))

    staging = _Staging(uuid.uuid4().hex[:12])
    una._init_batch_compositor()
    folded_this_call = {"value": False}

    # Cohort route (C30, dossier 06): eligible rows are the
    # proven-independent (worker-admissible) rows of an accessibility
    # batch.  The shared store root is runtime-only state beside the
    # first dispatched row's public folder (staging-root convention:
    # hidden, content-addressed, persists for warm resumes, safe to
    # delete); the per-worker residency cap comes from RESOURCES.md
    # sizing via the cohort module.  Grouping itself is runtime-
    # resolved: workers derive trace identities after the graph build.
    cohort_root = None
    cohort_cap = 0
    cohort_rows: Dict[int, dict] = {}
    prepared_rows: Dict[int, dict] = {}
    if analysis == 'accessibility' and dispatchable:
        from .cohort import per_worker_residency_cap
        # The root is STABLE across runs and resumes (no run id): warm
        # restarts and exact repeats must hit, and a crashed run's trace
        # serves the resumed one — identity-verified at every load, so
        # reuse is always content-proven.  Content addressing + the
        # single flight make concurrent sharers safe.
        cohort_root = os.path.join(_public_folder(plan, dispatchable[0]),
                                   ".una-batch-cohort")
        cohort_cap = per_worker_residency_cap(workers_effective)
        notes_runtime.append(
            f"cohort route: {len(dispatchable)} eligible row(s); "
            f"residency cap {cohort_cap} bytes/worker")

    # Prepared run-wide tier (una_10x P02 increment C): one shared run
    # store beside the first dispatched row's public folder (the C30
    # cohort-root convention — stable across resumes, safe to delete,
    # identity-verified at every load).  Workers attach it per row; the
    # kill switch (UNA_PREPARED=0) keeps workers from ever creating it,
    # so the paired control leaves the tree exactly as before.
    prepared_root = None
    if dispatchable:
        prepared_root = os.path.join(_public_folder(plan, dispatchable[0]),
                                     ".una-batch-prepared")

    outcomes: Dict[int, RowOutcome] = {}
    manifests: Dict[int, Tuple[str, ...]] = {}
    bundles: Dict[int, dict] = {}
    buffered: Dict[int, RowResult] = {}
    parent_ran: set = set()
    # P01M diagnostics: coordinator phase spans + per-row dispatch/recv
    # transport stamps (H2).  No-op unless UNA_BATCH_PHASES=1; surfaces
    # in the report's extra — never journal/checkpoint state.
    tele = PhaseRecorder()
    dispatch_stamps: Dict[int, int] = {}
    recv_stamps: Dict[int, int] = {}
    pool = _Pool(workers_effective, effective.options.threads_per_worker)
    pool_started = False
    deadline = None

    try:
        if dispatchable:
            with opt_span(tele, "pool_start"):
                pool.start()
            pool_started = True
            notes_runtime.append(
                f"row-worker pool: {workers_effective} spawn worker(s) x "
                f"{effective.options.threads_per_worker} thread(s); pending "
                f"bound = workers + queue_depth="
                f"{effective.options.queue_depth}")
        # The deadline budgets WORK, not worker-process spawn readiness
        # (on a loaded node spawning can take tens of seconds; charging it
        # to timeout_s would cancel healthy batches before row 1).
        deadline = (time.monotonic() + effective.options.timeout_s
                    if effective.options.timeout_s else None)

        cursor = 0
        dispatch_ptr = 0
        dead_seen = False       # a worker death has been observed

        def _pump() -> None:
            """Greedy bounded dispatch (dossier 05): keep every admitted
            row whose predecessors allow it in flight, up to the pending
            bound (workers + queue_depth), independent of the commit
            cursor.  Results are buffered and committed here in caller
            order — speculation is staged and invisible (dossier 04)."""
            nonlocal dispatch_ptr
            if dead_seen:
                return          # never feed a pool with a dead worker
            while dispatch_ptr < len(dispatchable) \
                    and _pending_bound(pool, buffered, effective) > 0:
                j = dispatchable[dispatch_ptr]
                dispatch_ptr += 1
                job = _job_for(
                    una, plan, j, staging.dir_for(
                        j, _public_folder(plan, j)),
                    analysis, script_output_folder,
                    need_state=(j == expected_last),
                    cohort_root=cohort_root,
                    cohort_residency_cap_bytes=cohort_cap,
                    prepared_root=prepared_root)
                # Transportability is an admission requirement (see
                # _job_transport_failure): fail THIS row loudly through the
                # normal ordering path (earliest failure, prefix
                # preserved) instead of a silent feeder-thread death.
                probe = _job_transport_failure(job)
                if probe is not None:
                    buffered[j] = RowResult(
                        job_id=job.job_id, row_index=j, status="failed",
                        error_type="RuntimeError", error_msg=probe)
                    continue
                with opt_span(tele, "dispatch"):
                    dispatch_stamps[j] = time.monotonic_ns()
                    pool.dispatch(job)

        while cursor < n:
            if deadline is not None and time.monotonic() > deadline:
                raise BatchCancelledError(
                    f"deadline of {effective.options.timeout_s}s exceeded "
                    f"at row {cursor + 1}/{n}")

            _pump()
            if not dead_seen and pool.in_flight > 0 and pool.any_dead():
                # A dead worker's already-sent results survive in the
                # outbox pipe: drain them so the committed prefix extends
                # across every receivable result before the cancellation
                # fires (the failure is exposed at the earliest row the
                # finished work cannot cover — dossier 04's ordering).
                while True:
                    try:
                        drained = pool.recv(timeout=0)
                    except queue.Empty:
                        break
                    if drained.row_index in dispatch_set \
                            and drained.row_index not in committed:
                        buffered[drained.row_index] = drained
                        recv_stamps[drained.row_index] = time.monotonic_ns()
                dead_seen = True
            if dead_seen and not (cursor in dispatch_set
                                  and cursor in buffered):
                raise BatchCancelledError(
                    "a row worker process died while jobs were in flight")

            rp = plan.rows[cursor]
            i = cursor

            if rp.validation.status == "skipped":
                # bind_settings runs for skipped rows too (BATCH_PLAN NOTE-2)
                s = projects[i]
                una.settings = s
                if not (s.output_folder or '').strip():
                    s.output_folder = script_output_folder or \
                        os.path.join(s.data_folder, "Results")
                una.topology.logger.log(
                    'RunBatch',
                    f"Row {i + 1}/{n}: skipped — missing required fields: "
                    f"{list(rp.validation.missing)}", v=1)
                outcomes[i] = RowExecutionOutcome(
                    index=i, name=rp.row.name, status="skipped",
                    semantic_profile=rp.row.semantic_profile,
                    backend=effective.backend,
                    detail=f"missing required fields: "
                           f"{list(rp.validation.missing)}",
                    admission=rp.admission,
                    admission_reasons=rp.admission_reasons,
                    phase="SKIPPED")
                _check_prefix_decisions(plan, i, una, ())
                _journal_row(journal, i, phase="SKIPPED")
                cursor += 1
                continue

            if i in dispatch_set:
                if i in buffered:
                    result = buffered.pop(i)
                    if result.status == "failed":
                        # A worker row's failure is the serial loop's
                        # failure: re-raise the original exception (same
                        # type/message) once the prefix has committed.
                        raise _worker_failure(result)
                    _commit_worker_row(
                        una, plan, journal, i, result,
                        _public_folder(plan, i), effective,
                        outcomes, manifests, bundles, tele=tele)
                    if result.cohort_stats is not None:
                        cohort_rows[i] = result.cohort_stats
                    if result.prepared_stats is not None:
                        prepared_rows[i] = result.prepared_stats
                    folded_this_call["value"] = folded_this_call["value"] \
                        or result.capture is not None
                    staging.discard(i)
                    if _on_commit is not None:
                        _on_commit(i)
                    cursor += 1
                    continue
                if not pool_started:
                    raise RuntimeError(
                        "dispatchable rows exist but the worker pool "
                        "never started")
                try:
                    # result_recv is a high-frequency wait (one recv tick
                    # per _TICK_S): accumulated as a total, never a span —
                    # per-tick spans would flood the bounded buffer and
                    # evict the commit phases (P01M telemetry laws).
                    t_wait = time.monotonic_ns()
                    result = pool.recv(timeout=_TICK_S)
                    tele.accumulate(
                        "result_recv_wait_ns",
                        time.monotonic_ns() - t_wait)
                    recv_stamps[result.row_index] = time.monotonic_ns()
                    if result.row_index in dispatch_set \
                            and result.row_index not in committed:
                        buffered[result.row_index] = result
                except queue.Empty:
                    pass
                continue

            if i in committed:
                # Resumed: reuse the committed unit.  Its journal record
                # already holds the settings post-state and capture payload;
                # the fold happens here in caller order unless the previous
                # run already finalized the composite.  (A record can only
                # be SKIPPED here if the plan's validation statuses had
                # changed — impossible under the identity, which binds the
                # plan summary; skipped rows are re-transitioned at the
                # cursor-skip branch above and never reach this branch.)
                record = (journal.load() or {}).get("rows", {}).get(
                    str(i)) or {}
                reused_files = tuple(sorted(
                    os.path.join(_public_folder(plan, i), rel)
                    for rel in (record.get("files") or {})))
                outcomes[i] = RowExecutionOutcome(
                    index=i, name=rp.row.name, status="ran",
                    semantic_profile=rp.row.semantic_profile,
                    backend=effective.backend,
                    admission=rp.admission,
                    admission_reasons=rp.admission_reasons,
                    phase="COMMITTED", worker_pid=0,
                    output_files=reused_files)
                manifests[i] = reused_files
                if journal is not None:
                    if record.get("capture") \
                            and not composite_already_finalized:
                        data = journal.read_capture(record["capture"])
                        arr = np.load(io.BytesIO(data), allow_pickle=False)
                        una._composite_captured.append((
                            record["capture"].get("column", ""),
                            arr,
                            record["capture"].get("metric", ""),
                            record["capture"].get("join_path")))
                        folded_this_call["value"] = True
                    _apply_settings_delta(
                        projects[i], record.get("settings") or {})
                    una.settings = projects[i]
                # The reuse path is a commit class like any other: the
                # re-applied state must still satisfy the planner's
                # decided naming state (this is exactly where stale/mixed
                # reuse would first surface).
                _check_prefix_decisions(plan, i, una, reused_files)
                cursor += 1
                continue

            # Serial-class or plan-error row: exact serial statements, run
            # once everything before the cursor has committed.
            _commit_parent_row(una, plan, journal, i, analysis,
                               script_output_folder, effective, n,
                               outcomes, manifests)
            _check_prefix_decisions(plan, i, una, manifests.get(i, ()))
            parent_ran.add(i)
            folded_this_call["value"] = folded_this_call["value"] or bool(
                una._composite_active
                and projects[i].batch_composite_output)
            if _on_commit is not None:
                _on_commit(i)
            cursor += 1
    except BatchCancelledError as exc:
        _cancel(pool, staging)
        _attach_rows(exc, outcomes)
        raise
    except KeyboardInterrupt:
        exc = BatchCancelledError(
            "KeyboardInterrupt while the batch coordinator owned workers")
        _cancel(pool, staging)
        _attach_rows(exc, outcomes)
        raise exc from None
    except BaseException as exc:
        _cancel(pool, staging)
        _attach_rows(exc, outcomes)
        raise
    finally:
        if pool_started:
            with opt_span(tele, "pool_close"):
                pool.close()
        # Single cleanup owner: remove this run's staging by ownership on
        # success and cancellation alike (row dirs were discarded at their
        # commits; the per-folder hidden ROOTS remain).
        staging.cleanup()

    # ---- rehydration, composite finalization, report ----------------------
    try:
        with opt_span(tele, "rehydrate"):
            _rehydrate(una, plan, bundles, outcomes, journal, effective,
                       analysis, script_output_folder, notes_runtime,
                       parent_ran)
    except BatchCancelledError as exc:
        _attach_rows(exc, outcomes)
        raise
    except queue.Empty as exc:
        cexc = BatchCancelledError(
            "final-state rehydration stalled: the state-only rerun "
            "exceeded its budget")
        _attach_rows(cexc, outcomes)
        raise cexc from exc
    except KeyboardInterrupt as exc:
        cexc = BatchCancelledError(
            f"final-state rehydration interrupted: {exc!r}")
        _attach_rows(cexc, outcomes)
        raise cexc from None

    if not (journal is not None and composite_already_finalized
            and not folded_this_call["value"]):
        if journal is not None and composite_already_finalized:
            # BATCH_REVIEW MAJOR-1: this call re-ran rows after the prior
            # run had already finalized the composite, so the in-memory
            # accumulator holds only THIS call's captures — finalizing it
            # would silently overwrite the published composite with a
            # subset rebuild.  The full capture set lives in the journal
            # (every commit, worker or parent, journaled its payload, and
            # resume verified every payload digest): rebuild from those
            # bytes in plan.fold_order (== commit order, the float
            # order-sensitive fold order) and finalize that instead.
            rows = (journal.load() or {}).get("rows", {})
            captured = []
            for i in plan.fold_order:
                meta = (rows.get(str(i)) or {}).get("capture")
                if not meta:
                    continue
                data = journal.read_capture(meta)
                arr = np.load(io.BytesIO(data), allow_pickle=False)
                captured.append((meta.get("column", ""), arr,
                                 meta.get("metric", ""),
                                 meta.get("join_path")))
            una._composite_captured = captured
        una._finalize_batch_composite()
    if journal is not None and not composite_already_finalized:
        journal.record_composite_finalized()

    # Cohort aggregation (dossier 06 counters: unique searches vs rows,
    # store decodes, resident hits, folds, refusals — runtime-resolved
    # identities grouped here only for reporting).
    cohort_summary = None
    if cohort_rows:
        totals = {
            "rows": 0, "builds": 0, "faultins": 0, "resident_hits": 0,
            "singleflight_lost": 0, "replays": 0, "folds_done": 0,
            "build_s": 0.0, "faultin_s": 0.0, "replay_s": 0.0,
        }
        refusals: Dict[str, int] = {}
        identities = set()
        for stats in cohort_rows.values():
            for k in totals:
                totals[k] += stats.get(k, 0)
            for k, v in (stats.get("refusals") or {}).items():
                refusals[k] = refusals.get(k, 0) + v
            if stats.get("trace_id"):
                identities.add(stats["trace_id"])
        totals["refusals"] = refusals
        totals["distinct_trace_identities"] = len(identities)
        totals["per_row"] = {str(i): cohort_rows[i]
                             for i in sorted(cohort_rows)}
        totals["store_root"] = cohort_root
        totals["residency_cap_bytes_per_worker"] = cohort_cap
        cohort_summary = totals
        notes_runtime.append(
            f"cohort route: {len(cohort_rows)} row(s) attempted, "
            f"{totals['builds']} trace build(s) over "
            f"{totals['distinct_trace_identities']} distinct identit"
            f"{'y' if totals['distinct_trace_identities'] == 1 else 'ies'}, "
            f"{totals['faultins']} store decode(s), "
            f"{totals['resident_hits']} resident hit(s), "
            f"{totals['replays']} replay(s)"
            + (f"; refusals {refusals}" if refusals else ""))

    # Prepared run-wide tier totals (una_10x P02 increment C): folded
    # like the cohort counters — runtime-only reporting, never observable
    # serial state.  Absent from `extra` entirely when no row attached a
    # run store (the paired control's report is byte-shaped as before).
    prepared_summary = None
    if prepared_rows:
        prep_totals: Dict[str, Any] = {}
        for stats in prepared_rows.values():
            for group, counters in stats.items():
                if not isinstance(counters, dict):
                    continue
                g = prep_totals.setdefault(group, {})
                for k, v in counters.items():
                    g[k] = g.get(k, 0) + int(v)
        prep_totals["per_row"] = {str(i): prepared_rows[i]
                                  for i in sorted(prepared_rows)}
        prep_totals["store_root"] = prepared_root
        prepared_summary = prep_totals
        run = prep_totals.get("run", {})
        notes_runtime.append(
            f"prepared reuse: {run.get('cold_builds', 0)} cold build(s), "
            f"{run.get('tier2_hits', 0)} run-store hit(s), "
            f"{run.get('contention_waits', 0)} contention wait(s)"
            + (f", {run.get('corruption_rebuilds', 0)} corruption rebuild(s)"
               if run.get("corruption_rebuilds") else "")
            + (f", {run.get('tier2_refusals', 0)} run-tier refusal(s)"
               if run.get("tier2_refusals") else ""))

    una.batch_report = BatchExecutionReport(
        requested=requested,
        effective=effective,
        parallel_requested=True,
        workers_requested=workers,
        rows=tuple(outcomes[i] for i in sorted(outcomes)),
        notes=tuple(notes),
        workers_effective=workers_effective,
        pools=("row_workers",) if workers_effective else (),
        commit_order=commit_order,
        fold_order=tuple(plan.fold_order),
        worker_admissible=tuple(plan.worker_admissible),
        serialized=tuple(plan.serialized),
        cancelled=False,
        checkpoint_generation=(journal.load() or {}).get("generation")
        if journal else None,
        output_manifest={str(i): manifests[i] for i in sorted(manifests)},
        notes_runtime=tuple(notes_runtime),
        extra=_report_extra(cohort_summary, prepared_summary, tele,
                            dispatch_stamps, recv_stamps),
    )


# ---------------------------------------------------------------------------
# Small helpers shared above.
# ---------------------------------------------------------------------------

def _report_extra(cohort_summary, prepared_summary, tele: PhaseRecorder,
                  dispatch_stamps: Dict[int, int],
                  recv_stamps: Dict[int, int]) -> Dict[str, Any]:
    extra: Dict[str, Any] = {}
    if cohort_summary is not None:
        extra["cohort"] = cohort_summary
    if prepared_summary is not None:
        extra["prepared"] = prepared_summary
    if tele.on:
        # P01M diagnostics (post_p00 designs/01): coordinator phase spans
        # and the per-row dispatch/recv transport stamps behind H2.  A
        # transport-refused row (probe failure) has a dispatch stamp but
        # never a recv stamp — its entry carries the keys it has.
        # Diagnostics only — never cache/checkpoint identity, never
        # primary timing evidence.
        extra["coordinator_phases"] = list(tele.snapshot())
        extra["telemetry_dropped_events"] = tele.dropped_events
        if tele.totals:
            extra["phase_totals"] = dict(tele.totals)
        extra["row_transport"] = {
            str(i): {k: v for k, v in
                     (("dispatch_mono_ns", dispatch_stamps.get(i)),
                      ("recv_mono_ns", recv_stamps.get(i)))
                     if v is not None}
            for i in sorted(set(dispatch_stamps) | set(recv_stamps))}
    return extra


def _public_folder(plan: BatchPlan, i: int) -> str:
    return plan.rows[i].row.output_folder


def _job_transport_failure(job: RowJob) -> Optional[str]:
    """None when the job is transportable to a worker process; else the
    failure message for its row.  The pickle would otherwise happen in the
    parent's feeder thread, where a failure kills the thread silently,
    strands every later job and surfaces only as a misleading deadline."""
    try:
        pickle.dumps(job)
    except Exception as pexc:               # noqa: BLE001 — probe, not handler
        return (f"row job could not be transported to a worker process "
                f"(unpicklable state): {pexc!r}")
    return None


def _effective_workers(workers: Optional[int], effective: EffectiveExecution,
                       n_dispatchable: int) -> int:
    if n_dispatchable <= 0:
        return 0
    requested = workers
    if requested is None:
        requested = effective.options.cpu_budget or os.cpu_count() or 1
    return max(1, min(int(requested), n_dispatchable))


def _pending_bound(pool: _Pool, buffered: dict,
                   effective: EffectiveExecution) -> int:
    bound = pool.size + effective.options.queue_depth
    return bound - pool.in_flight - len(buffered)


def _worker_failure(result: RowResult) -> BaseException:
    if result.error_exc is not None:
        return result.error_exc
    # The live exception object could not cross the process boundary:
    # reconstruct the serial failure from the recorded failure descriptor.
    # Picklable args rebuild the EXACT serial construction (type and
    # message, including __str__ transformations such as KeyError's
    # repr-quoting); builtin exception classes without transportable args
    # are rebuilt from the recorded text; anything else raises a
    # RuntimeError carrying the original type name and message verbatim.
    if result.error_args is not None:
        exc_type = getattr(builtins, result.error_type, None)
        if isinstance(exc_type, type) \
                and issubclass(exc_type, BaseException):
            try:
                return exc_type(*result.error_args)
            except Exception:
                pass
    return RuntimeError(
        f"row worker failed with {result.error_type}: {result.error_msg}")


def _attach_rows(exc: BaseException,
                 outcomes: Dict[int, RowOutcome]) -> None:
    """Carry the committed prefix's outcomes on the raised exception
    (type and message stay verbatim for serial parity)."""
    try:
        exc.batch_rows = tuple(outcomes[i] for i in sorted(outcomes))
    except Exception:
        pass


def _cancel(pool: _Pool, staging: _Staging) -> None:
    """Stop dispatch and reap the pool.  Journal records of the committed
    prefix stand; staging removal belongs to the run-level ``finally``
    (the single cleanup owner — it runs on cancellation and success
    alike, by ownership)."""
    if pool.size:
        pool.cancel()
