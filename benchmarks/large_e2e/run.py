#!/usr/bin/env python3
"""UNA large-e2e benchmark harness entry point (campaign task H02).

Run by ABSOLUTE path using each arm's own venv Python, from a neutral
working directory:

    <arm-python> <repo>/benchmarks/large_e2e/run.py \\
      --manifest <workload.json> --arm <label> --identity <arm_identity.json> \\
      --mode single|batch --jobs <K> --workers <W> --numba-threads <H> \\
      --flow-stripes <Kflow-or-default> --queue-depth <Q> --writer-limit <L> \\
      --cpu-budget <C> --memory-budget-mib <M> --timeout-s <T> \\
      --cache-root <unique-cache-root> --out <new-run-directory>

Every argument is validated before job execution.  There are no implicit
absolute laptop paths: every path arrives via CLI.  The harness runs
identically for the B0 arm and every candidate arm.

Exit codes:
  0  valid run, all jobs validated
  2  rejected before any work (validation, rehash, admission, collision)
  1  run executed but INVALID (job failure, guard failure, watchdog,
     incomplete shutdown) — never qualification data

Arm identity files are generated with:
    <arm-python> <repo>/benchmarks/large_e2e/harness/identity.py \\
        --make --arm <label> --kind installed_wheel|diagnostic_source ...
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import psutil  # noqa: E402

from harness import faults as fault_registry  # noqa: E402
from harness import pool as pool_module  # noqa: E402
from harness import sampler as sampler_module  # noqa: E402
from harness import verify as verify_module  # noqa: E402
from harness import session as session_module  # noqa: E402
from harness.manifest import Manifest, load_manifest  # noqa: E402
from harness.identity import Identity, load_identity  # noqa: E402
from harness.spec import (  # noqa: E402
    DIAG_ONLY_ENV,
    POLL_TIMEOUT_S,
    RUN_EXIT,
    SINGLE_ENV_THREAD_VARS,
    VALIDATION_EXIT,
)
from harness.spec import RunSpec, ValidationError, parse_and_validate  # noqa: E402

REPO_ROOT = HERE.parents[2]   # .../benchmarks/large_e2e/run.py -> repo root
RAW_DIR_NAME = "raw"


class _SystemAdapter:
    """Narrow psutil view used by admission (faked in unit tests)."""

    @staticmethod
    def cpu_count() -> int:
        return psutil.cpu_count(logical=True) or 1

    @staticmethod
    def virtual_memory():
        return psutil.virtual_memory()

    @staticmethod
    def disk_usage(path):
        return psutil.disk_usage(str(path))


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)


def _reject_before_work(spec: RunSpec, reason: str, detail: dict) -> int:
    """Write a rejection record into --out (when possible) and exit 2."""
    try:
        spec.out_dir.mkdir(parents=True, exist_ok=True)
        _write_json(spec.out_dir / "session.json", {
            "schema_version": 1,
            "run_id": None,
            "status": "rejected_before_work",
            "arm": spec.arm,
            "measurement_class": spec.measurement_class,
            "qualification_valid": False,
            "installed_qualified": False,
            "requested_config": spec.describe(),
            "failure": {"reason": reason, "detail": detail},
            "recorded_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        })
    except OSError:
        pass
    print(f"REJECTED BEFORE WORK [{reason}]: {detail}", file=sys.stderr)
    return VALIDATION_EXIT


def _trip_snapshot(sampler: sampler_module.ProcessTreeSampler):
    trip = sampler.trip
    return {"reason": trip.reason, "detail": trip.detail} if trip else None


def main(argv: list[str] | None = None) -> int:
    try:
        spec = parse_and_validate(argv)
    except ValidationError as exc:
        print(f"REJECTED BEFORE WORK [arguments]: {exc}", file=sys.stderr)
        return VALIDATION_EXIT

    # ---- fault flag validation (test-only; inert when absent) -----------
    try:
        fault_registry.validate_fault(spec.test_fault)
    except ValueError as exc:
        return _reject_before_work(spec, "unknown_test_fault", {"error": str(exc)})

    # ---- arm identity (structural, before anything else) ----------------
    try:
        identity = load_identity(spec.identity_path)
        if spec.diagnostic and identity.kind != "diagnostic_source":
            raise ValidationError(
                "--diagnostic-source-root requires identity_kind="
                f"diagnostic_source (got {identity.kind})")
        if not spec.diagnostic and identity.kind != "installed_wheel":
            raise ValidationError(
                "installed mode requires identity_kind=installed_wheel "
                f"(got {identity.kind}); source diagnostics must use the "
                "distinctly named --diagnostic-source-root mode")
    except ValidationError as exc:
        return _reject_before_work(spec, "identity_invalid", {"error": str(exc)})

    # ---- environment policy --------------------------------------------
    try:
        env_policy = spec.validate_environment(dict(os.environ))
    except ValidationError as exc:
        return _reject_before_work(spec, "environment_policy", {"error": str(exc)})

    # ---- manifest (structural; settings keys vs arm identity) -----------
    try:
        manifest = load_manifest(spec.manifest_path, identity.settings_fields)
    except ValidationError as exc:
        return _reject_before_work(spec, "manifest_invalid", {"error": str(exc)})

    # ---- path/collision preflight ---------------------------------------
    try:
        spec.validate_paths(manifest.input_dirs())
    except ValidationError as exc:
        return _reject_before_work(spec, "path_collision", {"error": str(exc)})

    # ---- resource admission ---------------------------------------------
    try:
        spec.validate_admission(_SystemAdapter(), analysis=manifest.analysis)
    except ValidationError as exc:
        return _reject_before_work(spec, "admission_refused", {"error": str(exc)})

    # ======================================================================
    # Whole-session accounting starts HERE (rehash inside the window).
    # ======================================================================
    session_start_monotonic = time.monotonic()
    spec.out_dir.mkdir(parents=True, exist_ok=True)
    spec.cache_root.mkdir(parents=True, exist_ok=True)
    raw_dir = spec.out_dir / RAW_DIR_NAME
    raw_dir.mkdir(exist_ok=True)
    run_id = session_module.new_run_id()

    # Worker environment: built once, applied by every worker BEFORE any
    # numerical import (behavior 3).  Installed mode has no PYTHONPATH; the
    # diagnostic mode pins the declared source root explicitly.
    worker_env = {key: os.environ[key]
                  for key in ("PATH", "HOME", "TMPDIR", "LANG")
                  if key in os.environ}
    worker_env["NUMBA_NUM_THREADS"] = str(spec.numba_threads)
    for var in SINGLE_ENV_THREAD_VARS:
        worker_env[var] = "1"
    worker_env["NUMBA_CACHE_DIR"] = str(spec.cache_root)
    if spec.diagnostic:
        worker_env["PYTHONPATH"] = str(spec.diagnostic_source_root)
        # Diagnostic-only engine-behavior variables (e.g. NUMBA_DISABLE_JIT
        # for harness smoke runs) are propagated ONLY here; installed mode
        # rejects them outright in validate_environment.
        for var in DIAG_ONLY_ENV:
            if var in os.environ:
                worker_env[var] = os.environ[var]
                env_policy["diagnostic_env_propagated"] = {
                    var: os.environ[var]}
    env_policy["worker_env"] = dict(worker_env)

    worker_config = {
        "identity_path": str(spec.identity_path),
        "repo_root": str(REPO_ROOT),
        "diagnostic": spec.diagnostic,
        "diagnostic_source_root": (
            str(spec.diagnostic_source_root) if spec.diagnostic else None),
        "measurement_class": spec.measurement_class,
        "worker_env": worker_env,
        "writer_limit": spec.writer_limit,
        "poll_timeout_s": POLL_TIMEOUT_S,
        "put_timeout_s": 0.5,
        "test_fault": spec.test_fault,
        "flow_stripes": spec.flow_stripes,
        "warmup": spec.mode == "batch",
        "warmup_root": str(spec.out_dir / "warmup"),
        "manifest": {
            "analysis": manifest.analysis,
            "settings": manifest.settings,
            "required_output_patterns": list(manifest.required_output_patterns),
            "odm_options": manifest.odm_options,
        },
    }

    # ---- input rehash (behavior 8): reject mutated inputs before work ----
    try:
        rehash_report = manifest.rehash_inputs()
    except ValidationError as exc:
        return _reject_before_work(spec, "input_rehash_mismatch",
                                   {"error": str(exc)})
    _write_json(raw_dir / "input_rehash.json", rehash_report)

    session_record = session_module.build_session_record(
        spec, manifest, identity, run_id, session_start_monotonic,
        rehash_report, env_policy)
    _write_json(raw_dir / "session_config.json", {
        "run_id": run_id,
        "requested_config": spec.describe(),
        "admission": spec.admission,
        "workload": manifest.describe(),
        "identity": identity.describe(),
        "env_policy": env_policy,
    })

    job_specs = _build_job_specs(spec, manifest)
    sampler = sampler_module.ProcessTreeSampler(
        interval_s=0.25,
        memory_budget_bytes=spec.memory_budget_mib * 1024 * 1024,
        label=f"run_{run_id}")

    failure: dict | None = None
    dispatch_summary: dict | None = None
    shutdown_summary: dict = {}
    results: list[dict] = []
    readies: dict[int, dict] = {}
    active: list[pool_module.WorkerPool] = []

    sampler.start()
    try:
        results, readies, dispatch_summary, shutdown_summary = _execute_mode(
            spec, worker_config, job_specs, sampler, active)
    except pool_module.PoolFailure as exc:
        failure = {"reason": exc.reason, "detail": exc.detail}
    except Exception as exc:  # noqa: BLE001 - fail loudly, never hang
        import traceback
        failure = {"reason": "unexpected_coordinator_error",
                   "detail": {"type": type(exc).__name__, "message": str(exc),
                              "traceback_tail": traceback.format_exc()[-2000:]}}
    finally:
        if failure is not None and active:
            # Prompt bounded cancellation: cancel outstanding owned jobs,
            # retain results already received, terminate/join ONLY owned
            # children.  Never masks the original failure.
            try:
                shutdown_summary = active[0].cancel_and_terminate()
                results.extend(active[0].retained_results)
                for worker_id, ready in active[0].readies.items():
                    readies.setdefault(worker_id, ready)
            except Exception:  # noqa: BLE001
                shutdown_summary = {"cancel_error": True,
                                    "surviving_owned_processes": []}
        sampler.stop()

    # ---- raw records on success AND failure (behavior 10) -----------------
    for message in results:
        mtype = message.get("type")
        if mtype == "job_result":
            _write_json(raw_dir / f"job_{message['job_id']:05d}.json", message)
        elif mtype == "job_failed":
            suffix = message.get("job_id", "unknown")
            _write_json(raw_dir / f"job_{suffix}_FAILED.json", message)
    for worker_id, ready in sorted(readies.items()):
        _write_json(raw_dir / f"worker_{worker_id}_ready.json", ready)

    failed_count = sum(1 for m in results if m.get("type") == "job_failed")
    succeeded_count = sum(1 for m in results if m.get("type") == "job_result")

    # ---- end-of-run validation (HARNESS.md 'End' definition) --------------
    verification: dict | None = None
    if failure is None:
        try:
            verification = verify_module.verify_results(
                job_specs, results, manifest.describe(), shutdown_summary)
        except verify_module.RunInvalid as exc:
            failure = {"reason": exc.reason, "detail": exc.detail}
        except Exception as exc:  # noqa: BLE001
            failure = {"reason": "verification_error",
                       "detail": {"type": type(exc).__name__,
                                  "message": str(exc)}}

    # ---- frozen time definitions ------------------------------------------
    whole_session_s = time.monotonic() - session_start_monotonic
    if spec.mode == "batch":
        times = _batch_times(dispatch_summary, verification,
                             whole_session_s)
    else:
        times = _single_times(dispatch_summary, whole_session_s)

    counts = {
        "submitted": len(job_specs),
        "validated": (verification["validated_job_count"]
                      if verification is not None else 0),
        "succeeded_before_validation": succeeded_count,
        "failed": failed_count,
        "canceled": max(0, len(job_specs) - succeeded_count - failed_count)
        if failure is not None else 0,
        "note": "failed jobs never enter successful throughput; validated "
                "count is 0 unless the WHOLE run validates end-to-end",
    }

    session_record = session_module.finalize_session(
        session_record, spec, times, counts, verification,
        sampler.summary(), shutdown_summary, dispatch_summary, failure)

    _write_json(spec.out_dir / "session.json", session_record)
    sampler_summary = sampler.summary()
    _write_json(raw_dir / "samples.jsonl", {
        "note": "decimated process-tree samples; peaks/overhead in "
                "session.json process_tree_memory",
        "summary": {k: v for k, v in sampler_summary.items() if k != "samples"},
    })
    for index, sample in enumerate(sampler_summary["samples"]):
        _write_json(raw_dir / "samples" / f"sample_{index:05d}.json", sample)

    status = session_record["status"]
    print(f"run {run_id}: status={status} "
          f"validated={counts['validated']} failed={counts['failed']} "
          f"canceled={counts['canceled']} "
          f"measurement_class={spec.measurement_class}")
    if status != "valid":
        print(f"failure: {failure}", file=sys.stderr)
        return RUN_EXIT
    return 0


# ----------------------------------------------------------------------
# Mode execution: single = fresh pool per job; batch = one persistent pool.
# Both modes drive the SAME dispatcher (harness.dispatch) and the SAME
# pool machinery — there is no separately maintained logic.
# ----------------------------------------------------------------------
def _execute_mode(spec: RunSpec, worker_config: dict, job_specs: list[dict],
                  sampler: sampler_module.ProcessTreeSampler,
                  active: list):
    """Returns (results, readies, dispatch_summary, shutdown_summary)."""
    if spec.mode == "batch":
        worker_pool = pool_module.WorkerPool(spec, worker_config["worker_env"],
                                             worker_config)
        active.append(worker_pool)
        worker_pool.spawn_workers()
        worker_pool.wait_ready(sampler_summary_fn=lambda: _trip_snapshot(sampler))
        dispatch_summary = worker_pool.dispatch_and_drain(
            job_specs, sampler_summary_fn=lambda: _trip_snapshot(sampler))
        shutdown_summary = worker_pool.shutdown()
        active.clear()
        return (worker_pool.retained_results, worker_pool.readies,
                dispatch_summary, shutdown_summary)

    # Single mode: ONE fresh spawned process per job (cold process each).
    all_results: list[dict] = []
    all_readies: dict[int, dict] = {}
    shutdown_total: dict = {"worker_exits": {}, "missing_worker_exit": [],
                            "forced_terminations": [],
                            "surviving_owned_processes": [], "exitcodes": []}
    per_job_cold: list[dict] = []

    for job_spec in job_specs:
        single_spec = _single_spec(spec)
        worker_pool = pool_module.WorkerPool(
            single_spec, worker_config["worker_env"],
            dict(worker_config, warmup=False))
        active.append(worker_pool)
        t0 = time.monotonic()
        worker_pool.spawn_workers()
        worker_pool.wait_ready(sampler_summary_fn=lambda: _trip_snapshot(sampler))
        ready = worker_pool.readies.get(0, {})
        all_readies[len(all_readies)] = ready
        local = worker_pool.dispatch_and_drain(
            [job_spec], sampler_summary_fn=lambda: _trip_snapshot(sampler))
        cold_ns = time.monotonic() - t0
        app_wall = None
        if local["last_application_event_monotonic"] is not None:
            app_wall = (local["last_application_event_monotonic"]
                        - local["dispatch_start_monotonic"])
        per_job_cold.append({
            "job_id": job_spec["job_id"],
            "cold_process_ns": int(cold_ns * 1e9),
            "spawn_to_application_done_s": app_wall,
            "peak_in_flight": local["peak_in_flight"],
        })
        summary = worker_pool.shutdown()
        active.clear()
        for key in ("missing_worker_exit", "forced_terminations",
                    "surviving_owned_processes", "exitcodes"):
            shutdown_total[key].extend(summary[key] or [])
        shutdown_total["worker_exits"].update(summary["worker_exits"])
        all_results.extend(worker_pool.retained_results)

    dispatch_summary = {"mode": "single", "per_job_cold": per_job_cold}
    return all_results, all_readies, dispatch_summary, shutdown_total


def _single_spec(spec: RunSpec) -> RunSpec:
    """Per-job shallow copy with workers=1 and queue depth 1."""
    import copy
    single = copy.copy(spec)
    single.workers = 1
    single.queue_depth = 1
    return single


def _build_job_specs(spec: RunSpec, manifest: Manifest) -> list[dict]:
    """One spec per job: unique per-job output root, no shared live objects."""
    specs = []
    for job_id in range(spec.jobs):
        job_out = spec.out_dir / "outputs" / f"job_{job_id:05d}"
        specs.append({
            "job_id": job_id,
            "analysis": manifest.analysis,
            "settings": dict(manifest.settings),
            "output_root": str(job_out),
            "required_output_patterns": list(manifest.required_output_patterns),
            "odm_options": manifest.odm_options,
            "flow_stripes": spec.flow_stripes,
        })
    return specs


def _batch_times(dispatch_summary: dict | None, verification: dict | None,
                 whole_session_s: float) -> dict:
    """Frozen batch time definitions (all four windows, explicitly labelled)."""
    times: dict = {"mode": "batch"}
    if dispatch_summary is None:
        times["whole_session_ns"] = int(whole_session_s * 1e9)
        return times
    dispatch0 = dispatch_summary["dispatch_start_monotonic"]
    last_app = dispatch_summary.get("last_application_event_monotonic")
    all_received = dispatch_summary.get("all_received_monotonic")
    verified_at = (verification or {}).get("_verified_at_monotonic")

    if last_app is not None:
        wall = last_app - dispatch0
        times["warm_batch_application_wall_ns"] = int(wall * 1e9)
        times["warm_batch_application_wall_s"] = wall
    if verified_at is not None:
        wall = verified_at - dispatch0
        times["validated_batch_wall_ns"] = int(wall * 1e9)
        times["validated_batch_wall_s"] = wall
    else:
        times["validated_batch_wall_ns"] = None
        times["validated_batch_wall_s"] = None
        times["validated_batch_wall_note"] = (
            "no validated batch wall: the run did not validate end-to-end")
    times["whole_session_ns"] = int(whole_session_s * 1e9)
    return times


def _single_times(dispatch_summary: dict | None,
                  whole_session_s: float) -> dict:
    times: dict = {
        "mode": "single",
        "whole_session_ns": int(whole_session_s * 1e9),
        "note": "per-job cold_process_ns values are in dispatch.per_job_cold; "
                "each job runs in a fresh spawned process and includes its "
                "own import/JIT/output/exit window (fresh cache whenever a "
                "new --cache-root is designated for the compile-cold test)",
    }
    if dispatch_summary is not None:
        total = sum(entry["cold_process_ns"]
                    for entry in dispatch_summary["per_job_cold"])
        times["cold_process_ns_sum_over_jobs"] = total
    return times


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
