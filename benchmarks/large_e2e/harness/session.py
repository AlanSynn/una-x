"""Session record assembly (HARNESS.md behavior 10 + BENCHMARKS.md fields).

Builds the parent session record with all required per-run fields: run_id,
arm, identities, workload, complete Settings, requested/effective engine,
logical stripes and physical pools, queue/writer limits and observed counts,
CPU/RAM admission, cache/JIT profile, every frozen time definition,
successful/failed counts, outputs and signatures, process-tree memory,
return statuses, environment, measurement class and limitations.

Structural honesty rule: a diagnostic-source or self-test run can never
carry qualification_valid=true, regardless of content.
"""
from __future__ import annotations

import hashlib
import json
import time
import uuid
from pathlib import Path

HARNESS_FILES = ("run.py", "harness/__init__.py", "harness/spec.py",
                 "harness/manifest.py", "harness/identity.py",
                 "harness/dispatch.py", "harness/faults.py",
                 "harness/sampler.py", "harness/worker.py", "harness/pool.py",
                 "harness/verify.py", "harness/session.py")


def harness_tree_sha256(base_dir: Path) -> str:
    """SHA-256 over the harness implementation files (sorted, self-proof)."""
    digest = hashlib.sha256()
    for relpath in sorted(HARNESS_FILES):
        path = base_dir / relpath
        digest.update(relpath.encode("utf-8"))
        digest.update(b"\x00")
        if path.exists():
            with open(path, "rb") as handle:
                digest.update(handle.read())
        digest.update(b"\x00")
    return digest.hexdigest()


def new_run_id() -> str:
    return f"una-e2e-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"


def build_session_record(spec, manifest, identity, run_id: str,
                         session_start_monotonic: float,
                         rehash_report: dict | None,
                         env_policy: dict) -> dict:
    """Session record shell filled in before workers spawn."""
    if rehash_report is not None:
        rehash_report = dict(rehash_report)
        rehash_report.pop("files", None)  # per-file detail lives in raw/
    record = {
        "schema_version": 1,
        "run_id": run_id,
        "arm": spec.arm,
        "measurement_class": spec.measurement_class,
        "qualification_valid": False,   # only finalize() may set this true
        "installed_qualified": spec.measurement_class == "installed",
        "recorded_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "harness_sha256": harness_tree_sha256(Path(__file__).resolve().parents[1]),
        "workload": manifest.describe() if manifest else None,
        "input_rehash": rehash_report,
        "identity": identity.describe() if identity else None,
        "requested_config": spec.describe(),
        "env_policy": env_policy,
        "limitations": [],
        "time_definitions": {
            "application_ns": "immediately before UNA construction through "
                              "return from the requested public method "
                              "(synchronous required exports included)",
            "cold_process_ns": "before spawning the interpreter through "
                               "child exit (single mode; import/JIT/outputs "
                               "included; fresh cache when --cache-root is new)",
            "warm_batch_application_wall_ns": "timed dispatch through the "
                                              "final application-completion "
                                              "event received",
            "validated_batch_wall_ns": "timed dispatch through receipt and "
                                       "successful verification of all job "
                                       "results",
            "whole_session_ns": "process/pool setup, warm-up, work, "
                                "validation and shutdown",
        },
        "hashing_note": "input rehash precedes the session inside "
                        "whole-session accounting; output hashing by workers "
                        "blocks worker reuse, so it is reported separately "
                        "and NOT subtracted from any window",
    }
    record["whole_session_start_monotonic"] = session_start_monotonic
    return record


def finalize_session(record: dict, spec, times: dict, counts: dict,
                     verification: dict | None, sampler_summary: dict,
                     shutdown_summary: dict, dispatch_summary: dict | None,
                     failure: dict | None) -> dict:
    """Close out the session record after validation/shutdown."""
    record["times_ns"] = times
    record["counts"] = counts
    if verification is not None:
        verification = dict(verification)
        verification.pop("_verified_at_monotonic", None)
    record["verification"] = verification
    record["process_tree_memory"] = {
        key: sampler_summary[key]
        for key in ("label", "root_pid", "interval_s", "metric_semantics",
                    "samples_recorded", "samples_total", "decimated",
                    "sampler_overhead_seconds", "avg_walk_seconds", "peak",
                    "trip", "guarantee")
    }
    record["shutdown"] = shutdown_summary
    record["dispatch"] = dispatch_summary
    record["failure"] = failure

    validated = counts.get("validated", 0)
    failed = counts.get("failed", 0)
    canceled = counts.get("canceled", 0)

    # Primary successful throughput: validated jobs over the full validated
    # batch wall; never the submitted-job count.  Any failure disqualifies.
    validated_wall_s = times.get("validated_batch_wall_s")
    throughput = None
    if verification is not None and validated_wall_s and validated_wall_s > 0:
        throughput = validated / validated_wall_s
    record["throughput"] = {
        "primary_jobs_per_s": throughput,
        "definition": "validated_jobs / validated_batch_wall_seconds",
        "descriptive_total_jobs_over_application_window_per_s": (
            (validated / times["warm_batch_application_wall_s"])
            if verification is not None
            and times.get("warm_batch_application_wall_s") else None),
        "note": "the descriptive value uses the application-only window and "
                "is NOT the primary metric; hashing delays worker reuse and "
                "is never subtracted",
    }

    if failure is not None:
        status = "invalid"
        record["limitations"].append(f"run invalid: {failure.get('reason')}")
    elif verification is None:
        status = "invalid"
        record["limitations"].append("run invalid: end-of-run validation "
                                     "did not complete")
    else:
        status = "valid"

    # Structural qualification rule: only a fully valid INSTALLED run with
    # zero injected faults and zero failures can be qualification data.
    qualification_valid = (
        status == "valid"
        and spec.measurement_class == "installed"
        and spec.test_fault is None
        and failed == 0 and canceled == 0
        and verification is not None
    )
    record["qualification_valid"] = qualification_valid
    record["installed_qualified"] = (
        qualification_valid and spec.measurement_class == "installed")
    if spec.measurement_class == "diagnostic_source":
        record["limitations"].append(
            "diagnostic source mode: records are structurally NOT "
            "installed-qualified and cannot enter qualification")
    if spec.test_fault is not None:
        record["limitations"].append(
            "harness self-test fault injection active; never qualification data")

    record["status"] = status
    record["whole_session_ns"] = times.get("whole_session_ns")
    record["counts"]["successful"] = validated
    record["return_status"] = {
        "session_status": status,
        "validated_jobs": validated,
        "failed_jobs": failed,
        "canceled_jobs": canceled,
    }
    return record
