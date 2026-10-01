"""Batch planning/runtime package (dossier 04).

`plan` is the pure planner: it derives the row DAG (alias, input/output and
output-conflict hazards), holds per-row validation outcomes in caller
order, admits only proven-independent rows to worker execution, and states
the ordered transition specification and observable-prefix invariant that a
coordinator must reproduce.  It runs no analysis and mutates nothing, and
importing this package must stay light (no numpy/analysis stack — the
planner is importable from any environment).

`runtime` (BATCH_EXEC) executes a plan with real worker overlap: bounded
reorder staging, deterministic caller-order commit, cancellation and
checkpoint/resume.  It is imported lazily by ``UNA.RunBatch`` — never from
this ``__init__`` — so the planner package stays dependency-free.
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
