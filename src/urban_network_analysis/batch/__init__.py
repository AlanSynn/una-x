"""Batch planning/runtime package (dossier 04).

`plan` is the pure planner: it derives the row DAG (alias, input/output and
output-conflict hazards), holds per-row validation outcomes in caller
order, admits only proven-independent rows to worker execution, and states
the ordered transition specification and observable-prefix invariant that a
coordinator must reproduce.  It runs no analysis and mutates nothing.

The runtime that executes a plan with real worker overlap (bounded reorder
staging, deterministic commit, cancellation/checkpoint/resume) is a later
task in this campaign's DAG; until it lands, RunBatch's serial route is the
only admitted execution.
"""
from .plan import (
    BATCH_TRANSITIONS,
    ROW_TRANSITIONS,
    BatchPlan,
    Hazard,
    InvariantViolation,
    RowDescriptor,
    RowPlan,
    RowValidation,
    plan_batch,
    verify_prefix,
)

__all__ = [
    "BATCH_TRANSITIONS",
    "ROW_TRANSITIONS",
    "BatchPlan",
    "Hazard",
    "InvariantViolation",
    "RowDescriptor",
    "RowPlan",
    "RowValidation",
    "plan_batch",
    "verify_prefix",
]
