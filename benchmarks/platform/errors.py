"""Typed harness violations (campaign una-platform-2026-09, HARNESS.md).

Every negative control in tests/platform_harness asserts that one of these is
raised by an intentionally broken candidate.  The harness never converts a
violation into a warning: a failed check invalidates the session.
"""
from __future__ import annotations


class HarnessViolation(Exception):
    """Base class for every harness check failure."""

    check_id = "HARNESS_VIOLATION"


class IdentityError(HarnessViolation):
    """Installed-path identity does not match the frozen expectation
    (source tree shadowing the installed wheel, wrong wheel hash, ...)."""

    check_id = "IDENTITY"


class ManifestError(HarnessViolation):
    """Frozen manifest missing/invalid, or workload content drifted from the
    fingerprint recorded at freeze time.  Raised before any timing."""

    check_id = "MANIFEST"


class ModelDispatchError(HarnessViolation):
    """Requested analysis/model does not match the executed public entry
    point captured by the dispatch trace."""

    check_id = "MODEL_DISPATCH"


class EngagementError(HarnessViolation):
    """A backend claim (native/GPU) lacks substantive execution evidence
    (loaded compiled module, nonzero work counters)."""

    check_id = "ENGAGEMENT"


class OutputValidationError(HarnessViolation):
    """A declared output obligation is missing, empty or malformed."""

    check_id = "OUTPUT_VALIDATION"


class EmptyComparisonError(HarnessViolation):
    """A comparison was requested with an empty file/row set — vacuous
    evidence is a failure, not a pass."""

    check_id = "EMPTY_COMPARISON"


class BitwiseMismatch(HarnessViolation):
    """Two arms that must be byte-identical differ (dtype, shape or bytes)."""

    check_id = "BITWISE_MISMATCH"


class SafetyViolation(HarnessViolation):
    """A run attempted to write a git-tracked (committed) file."""

    check_id = "SAFETY_WRITE_GUARD"


class OutputDirReuseError(HarnessViolation):
    """A run targeted an output directory already holding a completed
    manifest from a different run ID."""

    check_id = "OUTPUT_DIR_REUSE"


class RecordAuditError(HarnessViolation):
    """A measurement record is missing required fields/boundaries
    (parent-only memory, absent sync markers, ...)."""

    check_id = "RECORD_AUDIT"


class ProvenanceError(HarnessViolation):
    """Oracle provenance is absent, incomplete, or the oracle was produced
    by the candidate itself."""

    check_id = "PROVENANCE"


class ResourceLedgerError(HarnessViolation):
    """Enforced resources (workers/queue/memory) diverge from the declared
    policy."""

    check_id = "RESOURCE_LEDGER"


class BatchRunFailed(HarnessViolation):
    """A batch window ended with failed/cancelled jobs or missing outputs.
    Reported as a failure of the window, never counted as throughput."""

    check_id = "BATCH_RUN"
