"""Measurement records (HARNESS.md "Measurement records"; control 8).

Every session record must carry the full field set before it is admissible:
hashes, model+profile, fingerprints, workload class, obligations, requested
AND effective backend, pools/stripes, device, cache state, all timing
boundaries, simultaneous memory sampling, counts and fault status.
`validate_record` fails thin records; `write_session` is append-only per
unique run ID.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from .errors import RecordAuditError

REQUIRED_TOP = (
    "run_id", "harness_version", "campaign", "profile", "model",
    "workload_class", "requested_backend", "effective_backend",
    "backend_counters", "dispatch_called", "timings", "cache_state",
    "export_synced", "outputs", "child_identity", "input_hashes",
    "settings_fingerprint", "resource_policy", "counts",
    "fault_status",
)

REQUIRED_TIMINGS = ("child_wall_s", "child_import_s", "child_compute_s",
                    "engagement_wall_s", "launch_to_exit_s")

REQUIRED_HASHES = ("python", "module_file", "installed_tree_sha256")


def validate_record(record: dict) -> None:
    """Thin/incomplete records fail the audit (control 8)."""
    missing = [k for k in REQUIRED_TOP if k not in record]
    if missing:
        raise RecordAuditError(f"record missing required fields: {missing}")
    timings = record.get("timings") or {}
    miss_t = [k for k in REQUIRED_TIMINGS if timings.get(k) is None]
    if miss_t:
        raise RecordAuditError(f"record timings missing boundaries: {miss_t}")
    ident = record.get("child_identity") or {}
    miss_h = [k for k in REQUIRED_HASHES if not ident.get(k)]
    if miss_h:
        raise RecordAuditError(
            f"identity hashes incomplete: {miss_h} (wheel/kernel identity "
            "must be recorded, not assumed)")
    if record.get("memory") in (None, {}):
        raise RecordAuditError(
            "record has no process-tree memory sampling (parent-memory-only "
            "or absent sampler fails the audit; control 8)")
    mem = record["memory"]
    if not mem.get("peak_is_simultaneous_tree_sum"):
        raise RecordAuditError(
            "memory peak is not a simultaneous whole-tree sum (summed "
            "noncoexisting per-worker maxima are inadmissible)")
    if not mem.get("per_sample"):
        raise RecordAuditError("memory sampling retained no samples")
    if int(mem.get("samples_taken", 0)) <= 0:
        raise RecordAuditError("sampler took no samples")
    if not record.get("export_synced"):
        raise RecordAuditError(
            "export synchronization marker absent (close is not fsync "
            "durability; unsynchronized results fail the audit; control 8)")
    if record.get("requested_backend") in ("native", "gpu") and \
            not record.get("backend_counters", {}).get(
                f"{record['requested_backend']}_module_loaded"):
        raise RecordAuditError(
            f"backend {record['requested_backend']} claimed without "
            "engagement counters")
    counts = record.get("counts") or {}
    for k in ("completed", "failed", "cancelled"):
        if k not in counts:
            raise RecordAuditError(f"counts missing {k} "
                                   "(missing jobs cannot pass as success)")
    if record.get("fault_status") is None:
        raise RecordAuditError("fault/timeout/OOM status not recorded")


def write_session(out_dir: str | Path, run_id: str, records: list[dict],
                  commands: list[str] | None = None) -> Path:
    """Append-only session store: refuses to touch an existing run ID's
    files.  Raw session JSONL and the exact command file live together."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl = out_dir / f"session_{run_id}.jsonl"
    cmd = out_dir / f"command_{run_id}.txt"
    for p in (jsonl, cmd):
        if p.exists():
            raise RecordAuditError(
                f"append-only custody violation: {p} already exists "
                f"(run IDs must be unique; never overwrite raw sessions)")
    with open(jsonl, "a", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, sort_keys=True, default=str) + "\n")
        f.flush()
        os.fsync(f.fileno())
    with open(cmd, "a", encoding="utf-8") as f:
        for line in commands or []:
            f.write(line + "\n")
        f.flush()
        os.fsync(f.fileno())
    return jsonl
