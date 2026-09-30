"""UNA-X platform campaign harness (una-platform-2026-09, HARNESS task).

Relocatable, installed-path-truthful benchmark/validation harness:
- identity:  installed import roots and wheel digests, source-shadow rejection
- manifest:  frozen workload manifests with pre-timing fingerprint checks
- guards:    output-dir custody and committed-file write prevention
- sampler:   simultaneous whole-process-tree memory sampling
- runner:    one real public-API job + engagement leg + bounded batch windows
- validation: obligation/schema checks and bitwise (dtype+shape+bytes) compare
- provenance: oracle admissibility
- records:   complete measurement records, append-only custody
"""
from .errors import (BatchRunFailed, BitwiseMismatch, EmptyComparisonError,
                     EngagementError, HarnessViolation, IdentityError,
                     ManifestError, ModelDispatchError, OutputDirReuseError,
                     OutputValidationError, ProvenanceError,
                     RecordAuditError, ResourceLedgerError, SafetyViolation)

__all__ = [n for n in dir() if not n.startswith("_")]
