"""Spawn-safe batch row worker (dossier 04 step 5, dossier 05 pools).

Each worker process owns fresh mutable UNA/Settings/Topology — it never
receives live parent objects, only an immutable job descriptor carrying a
full ``vars()`` snapshot of the row's pre-bind Settings state.  The worker
reproduces the serial loop's row transitions exactly (bind output folder →
missing-field skip gate → ``'Results'``→name substitution → engine run →
composite capture) with two runtime-only deviations, both invisible to
public state:

1. **Staging redirect.**  After the bind/skip/substitute transitions the
   row's ``output_folder`` is pointed at a private staging directory on
   the same filesystem as the real output folder, so a speculative row's
   artifacts never appear at public paths before its commit (the
   observable-prefix invariant).  The returned settings delta reports the
   *public* folder value the serial loop would have left, and the
   coordinator publishes staging contents by atomic rename at commit.
2. **Deferred public state.**  Settings mutations the engines make
   (cluster sizing, gravity-cap writeback, ``Validation()`` coercion) are
   returned as a post-run snapshot for the coordinator to apply onto the
   caller's project object at commit time, preserving object identity.

Module import stays light (no geopandas/numba/UNA at import time): the
spawned interpreter applies its thread-budget environment variables
BEFORE the heavy analysis stack is imported, so Numba/BLAS pools are
budgeted per worker (``threads_per_worker``) instead of deriving from
transient available cores (dossier 05).

Fault-injection points (``job.inject``) exist solely for the dossier-05
failure-injection tests: a phase name plus an action
(``exit``/``sleep``/``raise_memory``/``raise_ki``).  They are consumed
here and nowhere in public code paths.
"""

from __future__ import annotations

import _thread
import dataclasses
import hashlib
import os
import pickle
import queue
import resource
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

__all__ = ["RowJob", "RowResult", "pool_main"]

_CHUNK = 1 << 20


@dataclass(frozen=True)
class RowJob:
    """Immutable descriptor for one row's work (everything picklable)."""

    job_id: str
    row_index: int
    analysis: str                      # "accessibility" | "flow"
    settings_state: Dict[str, Any]     # vars() of the pre-bind Settings
    script_output_folder: Optional[str]
    staging_dir: str                   # private output root (same FS)
    capture: bool                      # row requests composite output
    need_state: bool                   # carry the rehydration bundle
    verbosity: int = 0
    inject: Optional[Dict[str, str]] = None


@dataclass(frozen=True)
class RowResult:
    """Worker outcome descriptor: staged file digests + deferred state.

    ``settings_post`` is the full post-run ``vars()`` snapshot; the
    coordinator diffs it against its own pre-dispatch snapshot and applies
    the delta onto the caller's project object (identity preserved).
    ``files`` lists every staged artifact as (relpath, bytes, sha256),
    verified again by the coordinator before publication.  ``error_exc``
    carries the original raised exception object (same-run trusted
    transport) so the parent can re-raise the serial type/message; the
    text fallbacks cover exceptions that resist pickling.
    """

    job_id: str
    row_index: int
    status: str                                  # "ran" | "failed"
    settings_post: Dict[str, Any] = dataclasses.field(default_factory=dict)
    capture: Optional[Tuple[str, Any, str, Optional[str]]] = None
    files: Tuple[Tuple[str, int, str], ...] = ()
    staging_dir: str = ""
    state_bundle: Optional[dict] = None
    # Observable flags after this row (the serial loop leaves the last
    # row's values on the instance; the coordinator applies them at commit
    # time so the prefix invariant holds after every commit, not just at
    # the end).  ``cap_resolved`` guards ``resolved_gravity_cap``: a fresh
    # worker instance has no pre-batch stale value, so the coordinator
    # keeps its own when the row did not resolve one.
    has_flow_results: bool = False
    has_centrality_results: bool = False
    cap_resolved: bool = False
    resolved_gravity_cap: Optional[dict] = None
    peak_rss_bytes: int = 0
    worker_pid: int = 0
    timings: Dict[str, float] = dataclasses.field(default_factory=dict)
    error_exc: Optional[BaseException] = None
    error_type: str = ""
    error_msg: str = ""
    # The exception's args when THEY are transportable even when the
    # exception object is not: the parent rebuilds the exact serial
    # construction (type and message, __str__ transformations included).
    error_args: Optional[Tuple[Any, ...]] = None


# ---------------------------------------------------------------------------
# Thread budgets — applied before the heavy stack is imported.
# ---------------------------------------------------------------------------

def _thread_env(threads_per_worker: int) -> Dict[str, str]:
    n = str(max(1, int(threads_per_worker)))
    return {
        "OMP_NUM_THREADS": n,
        "OPENBLAS_NUM_THREADS": n,
        "MKL_NUM_THREADS": n,
        "NUMBA_NUM_THREADS": n,
        "VECLIB_MAXIMUM_THREADS": n,
    }


_HEAVY = {}


def _load_analysis():
    """Import the analysis stack once per worker process."""
    if "UNA" not in _HEAVY:
        from .. import UNA as _UNA
        _HEAVY["UNA"] = _UNA
    return _HEAVY["UNA"]


# ---------------------------------------------------------------------------
# Fault injection (tests only).
# ---------------------------------------------------------------------------

def _inject(phase: str, spec) -> None:
    """Apply the fault-injection specs for ``phase``.  ``spec`` is None, a
    single dict, or a list of dicts; each carries ``phase``, ``action``
    and ``arg``.  Actions: exit / sleep / raise_memory / raise_ki /
    touch (create a marker file) / wait_for (block until a file exists).
    touch/wait_for exist so tests can prove real cross-process overlap
    through filesystem evidence instead of noisy timing assertions."""
    if not spec:
        return
    specs = spec if isinstance(spec, (list, tuple)) else [spec]
    for item in specs:
        if not item or item.get("phase") != phase:
            continue
        action = item.get("action")
        if action == "exit":
            os._exit(int(item.get("arg", 1)))
        elif action == "sleep":
            time.sleep(float(item.get("arg", 60)))
        elif action == "raise_memory":
            raise MemoryError("injected OOM (worker fault-injection test)")
        elif action == "raise_ki":
            raise KeyboardInterrupt("injected (worker fault-injection test)")
        elif action == "touch":
            path = str(item.get("arg", ""))
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                f.write(str(os.getpid()))
        elif action == "wait_for":
            path = str(item.get("arg_path", ""))
            deadline = time.monotonic() + float(item.get("arg", 60))
            while not os.path.exists(path):
                if time.monotonic() > deadline:
                    raise TimeoutError(
                        f"injected wait_for timed out: {path!r}")
                time.sleep(0.01)
        else:
            raise RuntimeError(f"unknown inject action {action!r}")


# ---------------------------------------------------------------------------
# Lock-aware transport for live engine objects.
#
# Engines carry runtime-only ownership references (e.g. the flow engines'
# ``_assigned_stats_lock = threading.Lock()``) that are deliberately never
# serialized — and threading locks are unpicklable, so a state bundle
# containing a live engine would die silently in the result queue's
# feeder thread (the parent would then wait forever).  The transport
# wrapper strips _thread-module lock objects on pickle and rebuilds FRESH
# locks on unpickle: the engine arrives as its real class with real
# arrays and a new, uncontended lock — the lock is a runtime reference,
# not observable state.
# ---------------------------------------------------------------------------

def _lock_attrs(obj) -> list:
    return [k for k, v in vars(obj).items()
            if type(v).__module__ == "_thread"]


class _LockSafe:
    """Pickle-proxy for live objects holding thread locks."""

    def __init__(self, obj):
        self._obj = obj

    def __getstate__(self):
        cls = type(self._obj)
        state = {
            "class": cls,
            "attrs": {k: v for k, v in vars(self._obj).items()
                      if type(getattr(self._obj, k, None)).__module__
                      != "_thread"},
            "locks": [k for k in _lock_attrs(self._obj)],
        }
        return state

    def __setstate__(self, state):
        cls = state["class"]
        obj = cls.__new__(cls)
        for k, v in state["attrs"].items():
            setattr(obj, k, v)
        for k in state["locks"]:
            setattr(obj, k, threading.Lock() if k.endswith("lock")
                    or "lock" in k.lower() else threading.RLock())
        self._obj = obj

    def unwrap(self):
        return self._obj


def _pack_state_bundle(bundle: Optional[dict]) -> Optional[dict]:
    """Wrap lock-bearing members of a state bundle for transport."""
    if bundle is None:
        return None
    packed = dict(bundle)
    for key in ("topology", "engine"):
        obj = packed.get(key)
        if obj is not None and _lock_attrs(obj):
            packed[key] = _LockSafe(obj)
    return packed


def _unpack_state_bundle(bundle: Optional[dict]) -> Optional[dict]:
    if bundle is None:
        return None
    unpacked = dict(bundle)
    for key in ("topology", "engine"):
        obj = unpacked.get(key)
        if isinstance(obj, _LockSafe):
            unpacked[key] = obj.unwrap()
    return unpacked


# ---------------------------------------------------------------------------
# The serial row sequence, executed on a fresh instance.
# ---------------------------------------------------------------------------

def _sha256_file(path: str) -> Tuple[int, str]:
    h = hashlib.sha256()
    size = 0
    with open(path, "rb") as f:
        while True:
            chunk = f.read(_CHUNK)
            if not chunk:
                break
            size += len(chunk)
            h.update(chunk)
    return size, h.hexdigest()


def _scan_staging(staging_dir: str) -> Tuple[Tuple[str, int, str], ...]:
    entries = []
    for root, _dirs, names in os.walk(staging_dir):
        for name in names:
            path = os.path.join(root, name)
            size, digest = _sha256_file(path)
            entries.append((os.path.relpath(path, staging_dir), size, digest))
    return tuple(sorted(entries))


def _run_row(job: RowJob) -> RowResult:
    """Execute one row exactly as the serial loop would, on a fresh UNA."""
    t0 = time.perf_counter()
    UNA = _load_analysis()
    timings = {"import_s": time.perf_counter() - t0}

    una = UNA(verbosity=job.verbosity)

    s = una.settings
    s.__dict__.update(dict(job.settings_state))

    # -- bind_settings (serial: applies to every row, skipped ones too) ----
    if not (s.output_folder or '').strip():
        s.output_folder = job.script_output_folder or \
            os.path.join(s.data_folder, "Results")

    # Workers only ever receive planner-admitted, validation-ok rows; the
    # skip gate is re-evaluated anyway so a scheduling bug can never turn a
    # skipped row into a run (fail loudly instead of writing artifacts).
    missing = [f for f in ('network_file', 'origins_file', 'destinations_file')
               if not (getattr(s, f, None) or '').strip()]
    if missing:
        raise RuntimeError(
            f"worker received row {job.row_index} whose required fields are "
            f"missing: {missing} — the planner routes skipped rows to the "
            f"coordinator, never here")

    # -- substitute_output_name (after the skip gate, serial order) --------
    if s.output_file_name == "Results":
        s.output_file_name = s.name

    # -- staging redirect (runtime-only; public value restored below) ------
    public_output_folder = s.output_folder
    os.makedirs(job.staging_dir, exist_ok=True)
    s.output_folder = job.staging_dir

    _inject("pre_run", job.inject)
    t1 = time.perf_counter()
    if job.analysis == 'flow':
        una.RunFlow()
        engine = una.flow
    else:
        una.RunAccessibility()
        engine = una.accessibility
    timings["run_s"] = time.perf_counter() - t1
    _inject("post_run", job.inject)

    # -- composite capture (the serial loop's capture, on this instance) ---
    capture = None
    if job.capture:
        una._composite_active = True
        una._composite_captured = []
        una._capture_batch_row(s, engine)
        if una._composite_captured:
            col_name, arr, metric, join_path = una._composite_captured[0]
            capture = (col_name, arr, metric, join_path)

    # -- rehydration bundle (only for the plan's expected last row) --------
    state_bundle = None
    if job.need_state:
        _inject("pre_state", job.inject)
        state_bundle = _pack_state_bundle({
            "analysis": job.analysis,
            "topology": una.topology,
            "engine": engine,
            "has_flow_results": una.has_flow_results,
            "has_centrality_results": una.has_centrality_results,
            "resolved_gravity_cap": getattr(una, "resolved_gravity_cap", None),
            "cap_resolved": getattr(una, "resolved_gravity_cap", None) is not None,
            "settings_post": vars(s),
        })

    # -- deferred settings delta: the public folder value is what the
    #    serial loop would have left on the object; the staging path must
    #    never leak into observable state.
    post = dict(vars(s))
    post["output_folder"] = public_output_folder

    timings["stage_scan_s"] = 0.0
    t2 = time.perf_counter()
    files = _scan_staging(job.staging_dir)
    timings["stage_scan_s"] = time.perf_counter() - t2

    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    return RowResult(
        job_id=job.job_id,
        row_index=job.row_index,
        status="ran",
        settings_post=post,
        capture=capture,
        files=files,
        staging_dir=job.staging_dir,
        state_bundle=state_bundle,
        has_flow_results=una.has_flow_results,
        has_centrality_results=una.has_centrality_results,
        cap_resolved=getattr(una, "resolved_gravity_cap", None) is not None,
        resolved_gravity_cap=getattr(una, "resolved_gravity_cap", None),
        peak_rss_bytes=int(rss),
        worker_pid=os.getpid(),
        timings=timings,
    )


def _run_job_safely(job: RowJob) -> RowResult:
    """Run one job, converting any exception into a failed RowResult so the
    pool loop survives to drain further messages (the coordinator decides
    what the failure means for ordering/cancellation).  Transportability
    is enforced by the pool loop via :func:`_transport_safe` — a silent
    feeder-thread death would leave the coordinator waiting forever."""
    try:
        return _run_row(job)
    except BaseException as exc:            # noqa: BLE001 — reported, not hidden
        return _failed_result(job, exc)


def _failed_result(job: RowJob, exc: BaseException) -> RowResult:
    pid = os.getpid()
    try:
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    except OSError:
        rss = 0
    try:
        transportable_args = tuple(exc.args)
        pickle.dumps(transportable_args)
    except Exception:                       # noqa: BLE001 — probe, not handler
        transportable_args = None
    return RowResult(
        job_id=job.job_id,
        row_index=job.row_index,
        status="failed",
        error_exc=exc,
        error_type=type(exc).__name__,
        error_msg=str(exc),
        error_args=tuple(transportable_args) if transportable_args is not None
        else None,
        worker_pid=pid,
        peak_rss_bytes=int(rss),
        timings={},
    )


def _transport_safe(result: RowResult) -> RowResult:
    """Guarantee the result crosses the process boundary, degrading in
    place when it cannot: first drop the live exception object (the
    parent rebuilds the original type/message from the recorded text and
    transportable args), then reduce to a text-only failure descriptor.
    An untransportable result would kill the outbox feeder thread
    silently — nothing afterwards would ever be delivered and the
    coordinator would spin to an unrelated deadline."""
    try:
        pickle.dumps(result)
        return result
    except Exception:                       # noqa: BLE001 — probe, not handler
        pass
    stripped = dataclasses.replace(result, error_exc=None)
    try:
        pickle.dumps(stripped)
        return stripped
    except Exception:                       # noqa: BLE001 — converted, loud
        return RowResult(
            job_id=result.job_id,
            row_index=result.row_index,
            status="failed",
            error_type="RuntimeError",
            error_msg=(f"row result could not be transported from the "
                       f"worker (unpicklable state; runtime-only "
                       f"references must never be serialized)"),
            worker_pid=result.worker_pid,
            peak_rss_bytes=result.peak_rss_bytes,
        )


# ---------------------------------------------------------------------------
# Persistent pool entry point (spawn target — must stay module-level).
# ---------------------------------------------------------------------------

def pool_main(inbox, outbox, ready, threads_per_worker: int = 1,
              verbosity: int = 0) -> None:
    """Worker process loop: signal ready, run jobs until a None sentinel.

    Reusable across jobs without retaining mutable state: every job builds
    its own fresh UNA/Settings/Topology inside :func:`_run_row`; nothing
    from a previous job survives it (the dossier-05 worker-reuse rule).
    """
    for key, value in _thread_env(threads_per_worker).items():
        os.environ[key] = value
    os.environ["_UNA_BATCH_IN_WORKER"] = "1"
    try:
        ready.put(("ready", os.getpid()))
    except Exception:                        # pragma: no cover — queue broken
        return
    while True:
        try:
            job = inbox.get()
        except (KeyboardInterrupt, EOFError, OSError):
            return
        if job is None:
            return
        _inject("startup", getattr(job, "inject", None))
        outbox.put(_transport_safe(_run_job_safely(job)))
