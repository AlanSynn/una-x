"""Coordinator-side end-of-run validation (HARNESS.md 'End' definition).

End requires: all expected jobs present, no duplicate ids, correct analysis
and engine/result signatures, artifact signatures that re-hash to the
worker-reported values with no zero-size placeholders and no missing
companion files, no failures/timeouts, all writers returned and all workers
shut down.  Any failed job disqualifies the comparison block: failed jobs
never enter successful throughput and no qualified status is emitted.
"""
from __future__ import annotations

import fnmatch
import time
from pathlib import Path

from harness.manifest import hash_file


class RunInvalid(Exception):
    def __init__(self, reason: str, detail: dict | None = None) -> None:
        super().__init__(reason)
        self.reason = reason
        self.detail = detail or {}


EXPECTED_ENGINE_ATTR = {"accessibility": "accessibility", "flow": "flow",
                        "odm": "od"}


def verify_results(job_specs: list[dict], results: list[dict],
                   manifest_desc: dict, shutdown_summary: dict) -> dict:
    """Validate a full result set; raises RunInvalid on any violation.

    Returns a verification report embedded into the session record.
    """
    expected_ids = {spec["job_id"] for spec in job_specs}
    successful: list[dict] = []
    failed: list[dict] = []
    seen_ids: dict[int, int] = {}

    for message in results:
        job_id = message.get("job_id")
        seen_ids[job_id] = seen_ids.get(job_id, 0) + 1
        if message["type"] == "job_result":
            successful.append(message)
        else:
            failed.append(message)

    duplicates = {job_id: count for job_id, count in seen_ids.items() if count > 1}
    if duplicates:
        raise RunInvalid("duplicate_job_id", {"duplicates": duplicates})

    missing_ids = sorted(expected_ids - seen_ids.keys())
    if missing_ids:
        raise RunInvalid("missing_job_results", {"missing_job_ids": missing_ids})

    unexpected_ids = sorted(seen_ids.keys() - expected_ids)
    if unexpected_ids:
        raise RunInvalid("unexpected_job_results",
                         {"unexpected_job_ids": unexpected_ids})

    if failed:
        raise RunInvalid("job_failures_present", {
            "failed_job_ids": sorted(m.get("job_id") for m in failed),
            "errors": [
                {"job_id": m.get("job_id"), "error": m.get("error")}
                for m in failed[:16]
            ],
            "note": "any failed job disqualifies the comparison block; "
                    "successful counts below reflect only validated jobs",
        })

    # Independent coordinator re-hash: worker-reported artifact signatures
    # must match the bytes on disk right now (catches altered bytes, deleted
    # files and placeholders reported as good).
    rehash_seconds_total = 0.0
    validated: list[dict] = []
    for message in successful:
        job_id = message["job_id"]
        analysis = message.get("analysis")
        expected_attr = EXPECTED_ENGINE_ATTR[analysis]
        signature = message.get("engine_signature") or {}
        if signature.get("engine_attr") != expected_attr:
            raise RunInvalid("wrong_engine_signature", {
                "job_id": job_id,
                "analysis": analysis,
                "expected_engine_attr": expected_attr,
                "reported_engine_attr": signature.get("engine_attr"),
            })

        t0 = time.perf_counter()
        for entry in message["output_files"]:
            path = Path(entry["path"])
            size, sha = hash_file(path)
            if size != entry["bytes"] or sha != entry["sha256"]:
                raise RunInvalid("artifact_signature_mismatch", {
                    "job_id": job_id,
                    "file": entry["relpath"],
                    "reported": {"bytes": entry["bytes"], "sha256": entry["sha256"]},
                    "actual": {"bytes": size, "sha256": sha},
                    "note": "output changed between worker hashing and "
                            "coordinator verification (altered bytes or "
                            "deleted companion file)",
                })
            if size == 0:
                raise RunInvalid("zero_size_output",
                                 {"job_id": job_id, "file": entry["relpath"]})
        rehash_seconds_total += time.perf_counter() - t0

        relpaths = [entry["relpath"] for entry in message["output_files"]]
        for pattern in manifest_desc["required_output_patterns"]:
            if not any(fnmatch.fnmatch(rel, pattern) for rel in relpaths):
                raise RunInvalid("missing_required_companion_file", {
                    "job_id": job_id, "pattern": pattern,
                    "files": relpaths,
                })
        validated.append(message)

    # Writers and workers must all be accounted for.
    forced = shutdown_summary.get("forced_terminations") or []
    if forced:
        raise RunInvalid("workers_force_terminated", {
            "worker_ids": forced,
            "note": "a worker had to be terminated at shutdown (e.g. hung "
                    "after output); the protocol did not complete cleanly"})
    missing_exit = shutdown_summary.get("missing_worker_exit") or []
    if missing_exit:
        raise RunInvalid("missing_worker_exit_record", {
            "worker_ids": missing_exit,
            "note": "a worker exited without its sentinel record; the "
                    "shutdown protocol was incomplete"})
    survivors = shutdown_summary.get("surviving_owned_processes") or []
    if survivors:
        raise RunInvalid("surviving_owned_processes", {"pids": survivors})

    writer_peaks = []
    for ready_or_exit in shutdown_summary.get("worker_exits", {}).values():
        writer = ready_or_exit.get("writer") or {}
        writer_peaks.append({
            "worker_id": ready_or_exit.get("worker_id"),
            "peak_active": writer.get("writer_peak_active"),
            "limit": writer.get("writer_limit"),
            "active_now": writer.get("writer_active_now"),
        })
    for worker in writer_peaks:
        if worker["active_now"] not in (0, None):
            raise RunInvalid("writers_not_returned", {"worker": worker})
        if worker["peak_active"] is not None and worker["limit"] is not None \
                and worker["peak_active"] > worker["limit"]:
            raise RunInvalid("writer_limit_exceeded", {"worker": worker})

    return {
        "validated_job_ids": sorted(m["job_id"] for m in validated),
        "validated_job_count": len(validated),
        "duplicate_check": "passed",
        "signature_check": "passed",
        "rehash_seconds_total": round(rehash_seconds_total, 6),
        "writer_peak_per_worker": writer_peaks,
        "all_writers_returned": True,
        "all_workers_shutdown": True,
        "_verified_at_monotonic": time.monotonic(),
    }
