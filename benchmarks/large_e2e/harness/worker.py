"""Campaign worker process (spawn target) — one worker, one process.

Responsibilities (HARNESS.md behaviors 2, 3, 4, 5, 6, 9, 10):

  * Applies the coordinator-controlled environment BEFORE importing any
    numerical library (NUMBA_NUM_THREADS, BLAS/OpenMP pins, NUMBA_CACHE_DIR;
    PYTHONPATH absent in installed mode).
  * Validates the arm identity in-process: resolution-based site-packages
    containment, repository-shadow rejection, .pth/editable rejection,
    dist-info presence, whole-tree module hashes, version and interpreter
    match.
  * Installs the writer semaphore adapter around the engine's EXISTING
    export boundary (Engines.Base export entry methods) — production export
    policy is untouched.
  * Runs one warm-up job (batch mode) matching the manifest analysis, into
    its own unique directory; hashes/metadata recorded before the tree is
    discarded.
  * Consumes job specs from a bounded queue: per job a FRESH UNA/Settings/
    Topology and a unique output directory; times the REAL public method
    through the ONE shared dispatcher; verifies outputs; reports a rich
    record on success AND failure.

This module must stay light at import time (stdlib + psutil): the mp spawn
child imports it before the controlled environment is applied.
"""
from __future__ import annotations

import json
import os
import queue as queue_module
import shutil
import sys
import threading
import time
import traceback
from pathlib import Path

import psutil

from harness.dispatch import (
    execute_analysis,
    extract_signature,
    ANALYSIS_TO_ENGINE_ATTR,
    ODM_DEFAULT_FORMAT,
    ODM_DEFAULT_SPEED,
)
from harness.manifest import hash_file
from harness.spec import ValidationError

PACKAGE_IMPORT_NAME = "urban_network_analysis"
SELF_SAMPLE_INTERVAL_S = 0.1
RESULT_PUT_DEADLINE_S = 30.0
SPIN_SLEEP_S = 0.05


# ----------------------------------------------------------------------
# Environment
# ----------------------------------------------------------------------
def apply_worker_env(worker_env: dict) -> dict:
    """Replace os.environ with the coordinator-controlled environment.

    Called before ANY numerical import so NUMBA_NUM_THREADS, the BLAS/OpenMP
    pins and NUMBA_CACHE_DIR are effective from first import (behavior 3).
    """
    os.environ.clear()
    os.environ.update(worker_env)
    return dict(os.environ)


def inspect_thread_pools() -> dict:
    """Requested counts are not proof of effective counts: record actuals.

    Everything is best-effort with None for unavailable: the harness runs
    identically on arms whose environment lacks numba (harness self-test
    stub) and on real arms.
    """
    observed: dict = {
        "numba_present": False,
        "numba_num_threads_requested_env": os.environ.get("NUMBA_NUM_THREADS"),
        "numba_num_threads_effective": None,
        "numba_threading_layer": None,
        "numba_threading_layer_available": False,
        "numba_cache_dir": os.environ.get("NUMBA_CACHE_DIR"),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "openblas_num_threads": os.environ.get("OPENBLAS_NUM_THREADS"),
        "mkl_num_threads": os.environ.get("MKL_NUM_THREADS"),
        "veclib_maximum": os.environ.get("VECLIB_MAXIMUM"),
        "numexpr_num_threads": os.environ.get("NUMEXPR_NUM_THREADS"),
    }
    try:
        import numba
        from numba.np.ufunc import parallel as nb_parallel
    except Exception:
        return observed
    observed["numba_present"] = True
    try:
        observed["numba_num_threads_effective"] = int(nb_parallel.get_num_threads())
    except Exception:
        pass
    try:
        observed["numba_threading_layer"] = numba.threading_layer()
        observed["numba_threading_layer_available"] = True
    except Exception:
        # threading_layer() raises before any target was compiled; that is an
        # honest "not yet initialized", not a fabricated value.
        observed["numba_threading_layer"] = None
    return observed


# ----------------------------------------------------------------------
# Writer semaphore adapter (behavior 6)
# ----------------------------------------------------------------------
class WriterAdapter:
    """Shared-semaphore bound around the engine's EXISTING export boundary.

    The three engine export entry methods on urban_network_analysis.Engines
    .Base are wrapped in-process; the production export code, its write
    order and its file formats are untouched.  The adapter only limits how
    many export calls may be active simultaneously across the pool and
    records the actual peak.
    """

    WRAPPED_METHODS = (
        "ExportAccessibilityResults",
        "ExportFlowResult",
        "ExportODM",
    )

    def __init__(self, limit: int) -> None:
        import threading
        self.limit = int(limit)
        self.semaphore = threading.Semaphore(self.limit)
        self.lock = threading.Lock()
        self.active = 0
        self.peak_active = 0
        self.total_calls = 0
        self.wrapped: list[str] = []

    def install(self) -> list[str]:
        from urban_network_analysis.Engines import Base as engine_base

        for method_name in self.WRAPPED_METHODS:
            original = getattr(engine_base.Base, method_name, None)
            if original is None:
                continue
            adapter = self

            def wrapped(self_engine, *args, __original=original,
                        __adapter=adapter, **kwargs):
                with __adapter.semaphore:
                    with __adapter.lock:
                        __adapter.active += 1
                        __adapter.total_calls += 1
                        __adapter.peak_active = max(
                            __adapter.peak_active, __adapter.active)
                    try:
                        return __original(self_engine, *args, **kwargs)
                    finally:
                        with __adapter.lock:
                            __adapter.active -= 1

            setattr(engine_base.Base, method_name, wrapped)
            self.wrapped.append(method_name)
        if not self.wrapped:
            raise ValidationError(
                "writer adapter found no engine export boundary to wrap; "
                "refusing to claim writer bounding")
        return list(self.wrapped)

    def snapshot(self) -> dict:
        with self.lock:
            return {
                "writer_limit": self.limit,
                "writer_peak_active": self.peak_active,
                "writer_total_calls": self.total_calls,
                "writer_active_now": self.active,
                "writer_wrapped_methods": list(self.wrapped),
                "writer_adapter_scope": (
                    "shared semaphore around existing Engines.Base export "
                    "entry methods; production export policy unchanged"),
            }


# ----------------------------------------------------------------------
# Identity guards (behavior 2) — in-process, before engine use
# ----------------------------------------------------------------------
def validate_identity_and_import(identity_path: Path, repo_root: Path,
                                 diagnostic: bool,
                                 diagnostic_source_root: Path | None) -> tuple[dict, dict]:
    """Run every identity guard, then import the engine package.

    Returns (identity_summary, import_record).  Raises ValidationError on
    any guard failure — never falls back to another arm or source tree.
    """
    from harness.identity import (
        load_identity, guard_no_pythonpath, check_under_site_packages,
        guard_repository_shadow, scan_pth_files, scan_editable_install,
        require_dist_info, hash_package_tree, tree_digest,
        verify_imported_modules, PACKAGE_IMPORT_NAME,
    )

    identity = load_identity(identity_path)
    stripped_sys_path: list[str] = []   # entries removed for being in-repo

    if diagnostic:
        if identity.kind != "diagnostic_source":
            raise ValidationError(
                "diagnostic mode requires an identity_kind=diagnostic_source "
                f"identity (got {identity.kind})")
    else:
        if identity.kind != "installed_wheel":
            raise ValidationError(
                "installed mode requires an identity_kind=installed_wheel "
                f"identity (got {identity.kind})")
        guard_no_pythonpath(os.environ)

        # The coordinator bootstraps ITS OWN harness directory onto sys.path
        # (spawn children inherit sys.path).  Every entry inside the harness
        # repository is stripped here and recorded, BEFORE the engine import;
        # guard_repository_shadow below then verifies none remain, so the
        # engine can never resolve through the repository.
        repo = Path(repo_root).resolve()
        kept: list[str] = []
        for entry in sys.path:
            try:
                inside = Path(entry).resolve().relative_to(repo) is not None
            except (OSError, ValueError):
                inside = False
            (stripped_sys_path if inside else kept).append(entry)
        sys.path[:] = kept

    # Emulate venv site setup: put the declared installed locations on
    # sys.path (a real arm venv already has them; this is a no-op there and
    # is how the harness self-test stub becomes importable).  Recorded, and
    # the guards below still decide whether the import is acceptable.
    injected: list[str] = []
    if identity.kind == "installed_wheel":
        for sp in identity.site_packages:
            sp_str = str(sp.resolve())
            if sp_str not in sys.path:
                sys.path.append(sp_str)
                injected.append(sp_str)
    elif diagnostic_source_root is not None:
        root_str = str(diagnostic_source_root.resolve())
        if root_str not in sys.path:
            sys.path.insert(0, root_str)
            injected.append(root_str)

    t_import0 = time.perf_counter_ns()
    package = __import__(PACKAGE_IMPORT_NAME)
    t_import1 = time.perf_counter_ns()

    package_root = Path(package.__file__).resolve()
    if package_root.name == "__init__.py":
        package_root = package_root.parent

    # Version string alone is never identity, but a mismatch is still fatal.
    package_version = getattr(package, "__version__", None)
    if package_version is None or str(package_version) != identity.package_version:
        raise ValidationError(
            f"package version mismatch: identity={identity.package_version!r} "
            f"imported={package_version!r}")
    interpreter = os.path.realpath(sys.executable)
    if interpreter != os.path.realpath(identity.interpreter):
        raise ValidationError(
            f"interpreter mismatch: identity={identity.interpreter!r} "
            f"actual={interpreter!r}")

    if identity.kind == "installed_wheel":
        check_under_site_packages(package_root, identity.site_packages)
        guard_repository_shadow(package_root, repo_root, list(sys.path))
        pth_findings = scan_pth_files(identity.site_packages)
        if pth_findings:
            raise ValidationError(
                f"installed worker refuses .pth hooks in declared "
                f"site-packages (editable/source shadowing): {pth_findings}")
        editable_findings = scan_editable_install(identity.site_packages)
        if editable_findings:
            raise ValidationError(
                f"installed worker refuses editable-install markers: "
                f"{editable_findings}")
        dist_info = require_dist_info(identity.site_packages)
    else:
        if diagnostic_source_root is None:
            raise ValidationError("diagnostic mode requires a source root")
        resolved_root = diagnostic_source_root.resolve()
        try:
            package_root.relative_to(resolved_root)
        except ValueError:
            raise ValidationError(
                f"diagnostic import resolved outside the declared source "
                f"root: {package_root} not under {resolved_root}")
        dist_info = None

    # Whole-tree hash comparison: changed/missing/extra files invalidate the
    # warm identity ('changed cache/module source hash' negative test).
    actual_tree = hash_package_tree(package_root)
    if actual_tree != identity.module_hashes:
        changed = sorted(k for k in actual_tree
                         if k in identity.module_hashes
                         and actual_tree[k] != identity.module_hashes[k])
        missing = sorted(k for k in identity.module_hashes if k not in actual_tree)
        extra = sorted(k for k in actual_tree if k not in identity.module_hashes)
        raise ValidationError(
            f"package tree does not match identity module hashes "
            f"(changed={changed[:8]}, missing={missing[:8]}, extra={extra[:8]})")

    verify_imported_modules(dict(sys.modules), identity)

    summary = {
        "identity_kind": identity.kind,
        "identity_arm": identity.arm,
        "identity_sha256": identity.identity_sha256,
        "identity_fingerprint": identity.fingerprint,
        "package_root": str(package_root),
        "package_version": str(package_version),
        "package_tree_sha256": identity.package_tree_sha256,
        "tree_files_verified": len(actual_tree),
        "interpreter": interpreter,
        "dist_info": dist_info,
        "site_packages_declared": [str(p) for p in identity.site_packages],
    }
    import_record = {
        "import_ns": t_import1 - t_import0,
        "sys_path_injected": injected,
        "sys_path_stripped_repo_entries": stripped_sys_path,
        "pythonpath_env": os.environ.get("PYTHONPATH"),
    }
    return summary, import_record


# ----------------------------------------------------------------------
# Settings assembly and result extraction
# ----------------------------------------------------------------------
def jsonable(value):
    try:
        import numpy as np
    except Exception:
        np = None
    if np is not None:
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, (np.integer, np.floating, np.bool_)):
            return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    return repr(value)


def build_settings(settings_template: dict, output_root: Path,
                   settings_cls) -> object:
    """Fresh Settings per job with the manifest overrides applied."""
    try:
        import numpy as np
    except Exception:
        np = None
    settings = settings_cls()
    for key, value in settings_template.items():
        if key == "knn_weights" and np is not None:
            value = np.asarray(value, dtype=np.float64)
        setattr(settings, key, value)
    settings.output_folder = str(output_root)
    return settings


def settings_to_jsonable(settings) -> dict:
    if hasattr(settings, "ToDict"):
        try:
            return {k: jsonable(v) for k, v in settings.ToDict(compact=False).items()}
        except Exception:
            pass
    return {k: jsonable(v) for k, v in vars(settings).items()}


def load_settings_cls():
    module = __import__(f"{PACKAGE_IMPORT_NAME}.Settings",
                        fromlist=["Settings"])
    return getattr(module, "Settings")


def load_una_cls():
    module = __import__(f"{PACKAGE_IMPORT_NAME}.UNA", fromlist=["UNA"])
    return getattr(module, "UNA")


# ----------------------------------------------------------------------
# Output verification (worker side; coordinator re-verifies independently)
# ----------------------------------------------------------------------
def collect_outputs(output_root: Path,
                    required_patterns: list[str]) -> dict:
    """Hash all outputs, reject zero-size placeholders, check companion sets."""
    import fnmatch
    files = []
    for path in sorted(output_root.rglob("*")):
        if path.is_file():
            size, sha = hash_file(path)
            files.append({
                "path": str(path),
                "relpath": str(path.relative_to(output_root)),
                "bytes": size,
                "sha256": sha,
            })
    zero_size = [f["relpath"] for f in files if f["bytes"] == 0]
    if zero_size:
        raise ValidationError(
            f"zero-size placeholder output files present: {zero_size}")
    if not files:
        raise ValidationError(f"no output files produced under {output_root}")
    pattern_report = {}
    for pattern in required_patterns:
        matches = [f for f in files if fnmatch.fnmatch(f["relpath"], pattern)]
        pattern_report[pattern] = len(matches)
        if not matches:
            raise ValidationError(
                f"missing required companion output matching {pattern!r} "
                f"under {output_root}")
    return {
        "files": files,
        "n_output_files": len(files),
        "output_bytes_total": sum(f["bytes"] for f in files),
        "required_pattern_matches": pattern_report,
    }


# ----------------------------------------------------------------------
# Self RSS sampling (worker-local; coordinator tree sampler is authoritative)
# ----------------------------------------------------------------------
class SelfSampler:
    def __init__(self) -> None:
        import threading
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.peak_rss = 0

    def start(self) -> None:
        self._thread = threading.Thread(target=self._loop, daemon=True,
                                        name="una-worker-self-sampler")
        self._thread.start()

    def _loop(self) -> None:
        proc = psutil.Process()
        while not self._stop.is_set():
            try:
                self.peak_rss = max(self.peak_rss, proc.memory_info().rss)
            except psutil.Error:
                pass
            self._stop.wait(SELF_SAMPLE_INTERVAL_S)

    def stop(self) -> int:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        return self.peak_rss


# ----------------------------------------------------------------------
# Job execution (behavior 9: the REAL public method, timed once)
# ----------------------------------------------------------------------
def run_job(job_spec: dict, una_cls, settings_cls, writer_snapshot_fn) -> dict:
    """Execute one job with explicit timer boundaries.

    application_ns: immediately before UNA construction through return from
    the requested public method (required exports are synchronous inside the
    public call).  Settings assembly is timed separately and included in the
    whole-job wall.  The constructor is counted exactly once; no gravity-cap
    pre-resolution happens outside RunFlow.
    """
    analysis = job_spec["analysis"]
    output_root = Path(job_spec["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)

    record: dict = {
        "job_id": job_spec["job_id"],
        "analysis": analysis,
        "output_root": str(output_root),
        "status": "success",
    }
    t_wall0 = time.perf_counter_ns()

    t_set0 = time.perf_counter_ns()
    settings = build_settings(job_spec["settings"], output_root, settings_cls)
    t_set1 = time.perf_counter_ns()
    record["settings_assembly_ns"] = t_set1 - t_set0
    record["settings_before_run"] = settings_to_jsonable(settings)

    t_app0 = time.perf_counter_ns()
    record["application_start_ns"] = t_app0
    una = una_cls(verbosity=0)          # fresh UNA/Topology per job, counted once
    una.settings = settings             # fresh Settings per job

    # Flow stripes: when a non-default logical stripe count was requested,
    # set it through the public topology attribute the AggregateFlow engine
    # reads; this mirrors user-facing configuration, not a source change.
    stripes_requested = job_spec.get("flow_stripes")
    if analysis == "flow" and stripes_requested is not None:
        una.topology.num_threads = int(stripes_requested)

    if job_spec.get("fault") == "worker_raise":
        raise RuntimeError("injected fault: worker_raise")

    if job_spec.get("fault") == "slow_drain":
        time.sleep(0.2)

    outcome = execute_analysis(
        una, analysis, job_spec.get("odm_options"))
    t_app1 = time.perf_counter_ns()
    record["application_end_ns"] = t_app1
    record["application_ns"] = t_app1 - t_app0
    record["method_called"] = outcome.method_name
    record["engine_signature"] = outcome.signature

    signature = outcome.signature
    if job_spec.get("fault") == "wrong_engine_signature":
        signature = dict(signature)
        signature["engine_attr"] = "flow" if analysis != "flow" else "accessibility"
    record["engine_signature"] = signature

    record["engine_config_observed"] = {
        "flow_stripes_requested": stripes_requested,
        "topology_num_threads": getattr(getattr(una, "topology", None),
                                        "num_threads", None),
        "engine_num_threads": getattr(
            getattr(una, ANALYSIS_TO_ENGINE_ATTR[analysis], None),
            "num_threads", None),
        "topology_num_clusters": getattr(getattr(una, "topology", None),
                                         "num_clusters", None),
    }
    if hasattr(una, "resolved_gravity_cap"):
        record["resolved_gravity_cap"] = jsonable(una.resolved_gravity_cap)
    record["settings_after_run"] = settings_to_jsonable(settings)

    if job_spec.get("fault") == "zero_size_output":
        # Zero-byte placeholder emitted before worker verification: the
        # worker must refuse it (and the coordinator would too).
        (output_root / "placeholder.bin").write_bytes(b"")

    if job_spec.get("fault") == "missing_companion_file":
        # Delete one output matching the first required pattern before the
        # worker verifies its own companion set.
        import fnmatch
        pattern = job_spec["required_output_patterns"][0]
        for candidate in sorted(output_root.rglob("*")):
            if candidate.is_file() and fnmatch.fnmatch(
                    str(candidate.relative_to(output_root)), pattern):
                candidate.unlink()
                break

    th0 = time.perf_counter_ns()
    outputs = collect_outputs(output_root, job_spec["required_output_patterns"])
    th1 = time.perf_counter_ns()
    record["output_hash_start_ns"] = th0
    record["output_hash_end_ns"] = th1
    record["output_hashes_ns"] = th1 - th0
    record.update({
        "n_output_files": outputs["n_output_files"],
        "output_bytes_total": outputs["output_bytes_total"],
        "output_files": outputs["files"],
        "required_pattern_matches": outputs["required_pattern_matches"],
    })

    if job_spec.get("fault") == "altered_bytes":
        # Mutate AFTER hashing/verification, BEFORE the coordinator re-hash:
        # proves the coordinator's independent re-hash catches post-report
        # byte changes.
        victim = Path(outputs["files"][0]["path"])
        with open(victim, "ab") as handle:
            handle.write(b"\x00FAULT")

    record["job_wall_ns"] = time.perf_counter_ns() - t_wall0
    record["rss_at_end_bytes"] = psutil.Process().memory_info().rss
    record["writer"] = writer_snapshot_fn() if writer_snapshot_fn else None
    return record


# ----------------------------------------------------------------------
# Worker main loop (behavior 5: bounded queues, short polls, prompt failure)
# ----------------------------------------------------------------------
def worker_main(worker_id: int, job_queue, result_queue, config: dict) -> None:
    """Spawn-target body.  config keys are documented in pool.spawn_workers."""
    poll_s = config["poll_timeout_s"]
    put_s = config["put_timeout_s"]
    fault = config.get("test_fault")

    worker_env = config["worker_env"]
    apply_worker_env(worker_env)
    # Heavy/optional imports happen only AFTER the controlled env is applied.
    pools = inspect_thread_pools()

    self_sampler = SelfSampler()
    self_sampler.start()

    record: dict = {
        "type": "worker_ready", "worker_id": worker_id, "pid": os.getpid(),
        "worker_env": worker_env, "thread_pools_observed": pools,
    }

    try:
        identity_summary, import_record = validate_identity_and_import(
            Path(config["identity_path"]), Path(config["repo_root"]),
            config["diagnostic"],
            Path(config["diagnostic_source_root"])
            if config.get("diagnostic_source_root") else None)
    except ValidationError as exc:
        record["type"] = "worker_failed_guard"
        record["error"] = {"type": "ValidationError", "message": str(exc)}
        _put_message(result_queue, record, put_s)
        self_sampler.stop()
        return
    except Exception as exc:  # noqa: BLE001 - import failure is a guard failure
        record["type"] = "worker_failed_guard"
        record["error"] = {
            "type": type(exc).__name__, "message": str(exc),
            "traceback_tail": traceback.format_exc()[-2000:],
            "note": "the engine import itself failed under the controlled "
                    "environment; nothing may run in this worker",
        }
        _put_message(result_queue, record, put_s)
        self_sampler.stop()
        return

    record.update({
        "identity": identity_summary,
        "import_record": import_record,
    })

    settings_cls = load_settings_cls()
    una_cls = load_una_cls()

    # Writer semaphore adapter, identical in every arm.
    writer = WriterAdapter(config["writer_limit"])
    writer.install()
    record["writer"] = writer.snapshot()

    # Warm-up (batch mode): matches the manifest analysis/profile
    # specialization, unique directory, recorded then discarded.
    warmup_ns = None
    if config.get("warmup"):
        t_w0 = time.perf_counter_ns()
        try:
            warm_root = Path(config["warmup_root"]) / f"worker_{worker_id}"
            warm_spec = {
                "job_id": f"warmup_w{worker_id}",
                "analysis": config["manifest"]["analysis"],
                "settings": dict(config["manifest"]["settings"]),
                "output_root": str(warm_root),
                "required_output_patterns": config["manifest"]["required_output_patterns"],
                "odm_options": config["manifest"].get("odm_options"),
                "flow_stripes": config.get("flow_stripes"),
                "fault": None,
            }
            warm_record = run_job(warm_spec, una_cls, settings_cls,
                                  writer.snapshot)
            warmup_ns = time.perf_counter_ns() - t_w0
            record["warmup"] = {
                "warmup_ns": warmup_ns,
                "warmup_output_root": warm_record["output_root"],
                "warmup_output_files": [
                    {"relpath": f["relpath"], "bytes": f["bytes"],
                     "sha256": f["sha256"]}
                    for f in warm_record["output_files"]],
                "warmup_application_ns": warm_record["application_ns"],
                "warmup_peak_rss_bytes": self_sampler.peak_rss,
                "warmup_discarded_after_hash": True,
            }
            shutil.rmtree(warm_root, ignore_errors=True)
        except Exception as exc:  # noqa: BLE001 - warm-up failure kills readiness
            record["type"] = "worker_failed_warmup"
            record["error"] = {
                "type": type(exc).__name__, "message": str(exc),
                "traceback_tail": traceback.format_exc()[-2000:],
            }
            _put_message(result_queue, record, put_s)
            self_sampler.stop()
            return

    record["warmup_ns"] = warmup_ns
    record["startup_complete_ns"] = time.perf_counter_ns()

    if fault == "exit_before_ready":
        # Abrupt death BEFORE the readiness record: the coordinator barrier
        # must fail fast instead of waiting forever.
        os._exit(70)
    _put_message(result_queue, record, put_s)

    jobs_processed = 0
    while True:
        try:
            job_spec = job_queue.get(timeout=poll_s)
        except queue_module.Empty:
            continue
        if job_spec is None:
            break

        if fault == "die_while_dispatch" and jobs_processed == 0:
            os._exit(72)

        try:
            job_spec["fault"] = fault
            job_record = run_job(job_spec, una_cls, settings_cls,
                                 writer.snapshot)
            job_record.update({
                "type": "job_result",
                "worker_id": worker_id,
                "pid": os.getpid(),
                "identity_fingerprint": identity_summary["identity_fingerprint"],
                "identity_sha256": identity_summary["identity_sha256"],
                "identity_kind": identity_summary["identity_kind"],
                "package_tree_sha256": identity_summary["package_tree_sha256"],
                "measurement_class": config["measurement_class"],
                "started_utc": job_spec.get("dispatched_utc"),
                "thread_pools_observed": pools,
            })
            _put_message(result_queue, job_record, put_s)

            if fault == "duplicate_job_id" and job_spec["job_id"] == 0:
                dupe = dict(job_record)
                dupe["duplicate_of_job_id"] = 0
                _put_message(result_queue, dupe, put_s)

            if fault == "hang_after_output" and jobs_processed == 0:
                while True:
                    time.sleep(1.0)  # simulate a hung worker post-output
        except ValidationError as exc:
            _put_message(result_queue, {
                "type": "job_failed", "worker_id": worker_id,
                "pid": os.getpid(), "job_id": job_spec["job_id"],
                "analysis": job_spec["analysis"],
                "error": {"type": "ValidationError", "message": str(exc)},
                "measurement_class": config["measurement_class"],
            }, put_s)
        except Exception as exc:  # noqa: BLE001 - report and keep pool alive
            _put_message(result_queue, {
                "type": "job_failed", "worker_id": worker_id,
                "pid": os.getpid(), "job_id": job_spec["job_id"],
                "analysis": job_spec["analysis"],
                "error": {
                    "type": type(exc).__name__, "message": str(exc),
                    "traceback_tail": traceback.format_exc()[-2000:],
                },
                "measurement_class": config["measurement_class"],
            }, put_s)
        jobs_processed += 1

    peak_rss = self_sampler.stop()
    if fault == "never_send_sentinel":
        # Exit WITHOUT the worker_exit record: the coordinator must still
        # shut down boundedly and mark the protocol incomplete.
        return
    _put_message(result_queue, {
        "type": "worker_exit", "worker_id": worker_id, "pid": os.getpid(),
        "jobs_processed": jobs_processed,
        "phase_peak_rss_bytes": peak_rss,
        "writer": writer.snapshot(),
        "exit_reason": "sentinel",
    }, put_s)


def _put_message(result_queue, message: dict, put_timeout_s: float) -> None:
    """Bounded result channel put with a hard monotonic deadline."""
    deadline = time.monotonic() + RESULT_PUT_DEADLINE_S
    while True:
        try:
            result_queue.put(message, timeout=put_timeout_s)
            return
        except queue_module.Full:
            if time.monotonic() > deadline:
                # Result channel stuck: die loudly instead of hanging forever.
                os._exit(73)


def spawn_entry(worker_id: int, job_queue, result_queue, config: dict) -> None:
    """mp spawn target: fresh interpreter, controlled env, worker_main."""
    worker_main(worker_id, job_queue, result_queue, config)


def main() -> None:  # pragma: no cover - standalone debugging helper
    print("This module is a multiprocessing spawn target; use run.py.")
    raise SystemExit(2)


if __name__ == "__main__":  # pragma: no cover
    main()
