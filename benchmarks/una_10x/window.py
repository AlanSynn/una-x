"""una_10x window admissibility (H00 actions: true dispatch/import
identity/counts, byte-bounded queues; completion: failures cannot become
valid windows).

A una_10x measurement window record is admissible only when every law
below holds.  `make_window_record` validates before returning, so an
invalid window cannot even be produced, and `validate_window` is pure
enough to re-check retained records for tampering.
"""
from __future__ import annotations

from benchmarks.platform.identity import is_source_shadow

from .errors import WindowInvalid
from .timing import timer_violations

# The only public entry points a una_10x window may legitimately trace.
LEGAL_PUBLIC_ENTRIES = ("UNA.RunAccessibility", "UNA.RunODM",
                        "UNA.RunFlow", "UNA.RunBatch")

# BENCHMARKS.md cache regimes (a window must name one).
WINDOW_CLASSES = ("FIRST_COHORT", "RESIDENT_DISTINCT",
                  "REPEAT_IDENTICAL", "COLD")

REQUIRED_TOP = ("run_id", "harness_version", "campaign", "window_class",
                "cohort_digest", "timer_snapshot", "dispatch_called",
                "dispatch_expected", "counts", "queue", "resource_policy",
                "identity", "memory", "outputs", "fault_status",
                "export_synced")

REQUIRED_IDENTITY = ("module_file", "installed_tree_sha256",
                     "distribution_version")


class BoundedPendingBytes:
    """Byte-capped pending-result descriptors (bounded submission).

    Pending descriptors accumulate until a result completes; the cap is a
    hard failure, not a silent queue growth.  Peak is recorded so the
    ledger can be audited after the window.
    """

    def __init__(self, cap_bytes: int):
        if cap_bytes <= 0:
            raise WindowInvalid("pending-bytes cap must be positive")
        self.cap_bytes = int(cap_bytes)
        self.pending_bytes = 0
        self.peak_bytes = 0

    def submit(self, nbytes: int) -> None:
        if nbytes < 0:
            raise WindowInvalid("negative descriptor size")
        if self.pending_bytes + nbytes > self.cap_bytes:
            raise WindowInvalid(
                f"pending descriptor bytes {self.pending_bytes + nbytes} "
                f"would exceed cap {self.cap_bytes}; window cannot drain "
                "unbounded (byte-bounded queue law)")
        self.pending_bytes += nbytes
        self.peak_bytes = max(self.peak_bytes, self.pending_bytes)

    def complete(self, nbytes: int) -> None:
        if nbytes < 0 or nbytes > self.pending_bytes:
            raise WindowInvalid(
                f"completing {nbytes} bytes with {self.pending_bytes} "
                "pending is inconsistent")

        self.pending_bytes -= nbytes


def _short(name: str) -> str:
    return name.rsplit(".", 1)[-1]


def validate_window(record: dict, *, source_root: str | None = None) -> None:
    """Raise WindowInvalid unless *record* is an admissible una_10x window.

    Every branch names the failing law; a window that fails any law is
    invalid forever - there is no repair-in-place path through this
    function.
    """
    missing = [k for k in REQUIRED_TOP if k not in record]
    if missing:
        raise WindowInvalid(f"window record missing required fields: "
                            f"{missing} (thin records cannot pass)")

    snap = record["timer_snapshot"]
    # tamper law: stored violations are re-derivable; a scrubbed or edited
    # snapshot still fails because the laws are recomputed from the spans
    violations = list(snap.get("violations") or []) \
        + timer_violations(snap)
    if snap.get("primary_s") is None:
        violations.append("primary wall never completed")
    elif snap["primary_s"] <= 0:
        violations.append("primary wall is not positive")
    if snap.get("post_hash_s") is None:
        violations.append("post-hash wall missing (hashes must be timed "
                          "and retained, outside service timing)")
    elif snap["post_hash_s"] <= 0:
        violations.append("post-hash wall is not positive")
    if violations:
        raise WindowInvalid(f"timer law violations: {violations}")

    expected = record["dispatch_expected"]
    if expected not in LEGAL_PUBLIC_ENTRIES:
        raise WindowInvalid(f"dispatch expectation {expected!r} is not a "
                            "legal public entry point")
    called = record["dispatch_called"]
    called_legal = sorted(set(called) & set(LEGAL_PUBLIC_ENTRIES))
    if expected not in called:
        raise WindowInvalid(
            f"expected dispatch {expected} was not traced; captured "
            f"{called} (wrong-analysis success is forbidden)")
    if called_legal != [expected]:
        raise WindowInvalid(
            f"traced public entries {called_legal} diverge from the single "
            f"expected entry {expected} (true-API dispatch law)")

    counts = record["counts"]
    for k in ("submitted", "completed", "failed", "cancelled"):
        if k not in counts:
            raise WindowInvalid(f"counts missing {k}")
    total = (counts["completed"] + counts["failed"] + counts["cancelled"])
    if total != counts["submitted"]:
        raise WindowInvalid(
            f"counts do not close: {counts['completed']}+{counts['failed']}"
            f"+{counts['cancelled']} != {counts['submitted']} submitted "
            "(no inflated successful N)")
    for k in ("failed", "cancelled"):
        if counts[k]:
            raise WindowInvalid(
                f"window holds {counts[k]} {k} jobs; a failed or cancelled "
                "window is never a valid measurement (failures cannot "
                "become valid windows)")
    if counts["completed"] <= 0:
        raise WindowInvalid("window validated zero jobs")

    q = record["queue"]
    for k in ("max_workers", "queue_depth", "peak_inflight", "bounded"):
        if k not in q:
            raise WindowInvalid(f"queue ledger missing {k}")
    bound = min(int(q["max_workers"]), int(q["queue_depth"]))
    if not q["bounded"]:
        raise WindowInvalid("queue ledger claims unbounded submission")
    if int(q["peak_inflight"]) > bound:
        raise WindowInvalid(
            f"observed {q['peak_inflight']} in-flight > declared bound "
            f"{bound} (max_workers={q['max_workers']}, "
            f"queue_depth={q['queue_depth']})")

    policy = record["resource_policy"]
    if not isinstance(policy, dict) or "memory_limit_bytes" not in policy:
        raise WindowInvalid("resource_policy must declare memory_limit_bytes")
    mem = record["memory"]
    for k in ("peak_is_simultaneous_tree_sum", "per_sample", "samples_taken"):
        if k not in mem:
            raise WindowInvalid(f"memory ledger missing {k}")
    if not mem["peak_is_simultaneous_tree_sum"]:
        raise WindowInvalid("memory peak is not a simultaneous whole-tree "
                            "sum (per-worker maxima are inadmissible)")
    if not mem["per_sample"] or int(mem["samples_taken"]) <= 0:
        raise WindowInvalid("memory sampler retained no samples")
    limit = policy["memory_limit_bytes"]
    if limit is not None and mem.get("peak_simultaneous_rss_bytes",
                                     0) > int(limit):
        raise WindowInvalid(
            f"tree peak {mem['peak_simultaneous_rss_bytes']} exceeds "
            f"declared memory_limit_bytes {limit}")

    ident = record["identity"]
    miss_h = [k for k in REQUIRED_IDENTITY if not ident.get(k)]
    if miss_h:
        raise WindowInvalid(
            f"identity hashes incomplete: {miss_h} (wheel/kernel identity "
            "must be recorded, not assumed)")
    if source_root is not None and is_source_shadow(ident, source_root):
        raise WindowInvalid(
            f"imported module file {ident['module_file']} sits under the "
            f"source tree {source_root} (shadowing the installed wheel)")
    if "site-packages" not in str(ident["module_file"]):
        raise WindowInvalid(
            f"module imports from {ident['module_file']!r}, not an "
            "installed site-packages tree")

    if record["window_class"] not in WINDOW_CLASSES:
        raise WindowInvalid(
            f"window_class {record['window_class']!r} is not a BENCHMARKS.md "
            "cache regime")
    digest = record["cohort_digest"]
    if not isinstance(digest, str) or len(digest) < 32:
        raise WindowInvalid("cohort_digest missing or too short to bind "
                            "the exact job set")

    outputs = record["outputs"]
    if not outputs:
        raise WindowInvalid("window lists no output artifacts; required "
                            "CSV/GeoJSON/Feather outputs belong inside the "
                            "primary wall")
    for out in outputs:
        for k in ("path", "bytes", "sha256"):
            if k not in out:
                raise WindowInvalid(f"output entry missing {k}: {out}")
        if int(out["bytes"]) <= 0:
            raise WindowInvalid(f"output {out.get('path')!r} is empty")
        sha = out["sha256"]
        if (not isinstance(sha, str) or len(sha) != 64
                or set(sha) - set("0123456789abcdef")):
            raise WindowInvalid(f"output {out.get('path')!r} sha256 is not "
                                "a 64-hex digest")

    if not record["export_synced"]:
        raise WindowInvalid("export synchronization marker absent (close is "
                            "not fsync durability)")
    if record["fault_status"] != "none":
        raise WindowInvalid(
            f"fault_status={record['fault_status']!r}; a window with a "
            "recorded fault is never valid")


def make_window_record(*, run_id: str, window_class: str, cohort_digest: str,
                       timer_snapshot: dict, dispatch_expected: str,
                       dispatch_called, counts: dict, queue: dict,
                       resource_policy: dict, identity: dict, memory: dict,
                       outputs: list, fault_status: str = "none",
                       export_synced: bool = True,
                       campaign: str = "una_10x",
                       harness_version: str = "una-10x-h00-h1",
                       extra: dict | None = None) -> dict:
    """Assemble a window record and refuse to return an inadmissible one."""
    record = {
        "run_id": run_id,
        "harness_version": harness_version,
        "campaign": campaign,
        "window_class": window_class,
        "cohort_digest": cohort_digest,
        "timer_snapshot": timer_snapshot,
        "dispatch_expected": dispatch_expected,
        "dispatch_called": list(dispatch_called),
        "counts": dict(counts),
        "queue": dict(queue),
        "resource_policy": dict(resource_policy),
        "identity": dict(identity),
        "memory": dict(memory),
        "outputs": list(outputs),
        "fault_status": fault_status,
        "export_synced": export_synced,
    }
    if extra:
        record.update(extra)
    validate_window(record)
    return record
