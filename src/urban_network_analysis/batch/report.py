"""Batch-execution reporting and typed cancellation/checkpoint errors.

BATCH_EXEC's extension of the staged reporting ABI (contract §6): the
serial route keeps reporting plain :class:`~urban_network_analysis.Execution.
RowOutcome`/:class:`~urban_network_analysis.Execution.BatchReport` values
unchanged; the parallel route reports these subclasses, which are
isinstance-compatible with the staged types (existing field names were
frozen by the EXECUTION task and are not renamed — see the staged-delivery
note in ``Execution.py``).

Defined here, not in ``Execution.py``, because the batch runtime owns this
surface: ``Execution.py`` is the profile/admission contract (owned by the
EXECUTION task); everything parallel-execution-specific lives under
``urban_network_analysis.batch``.

Import weight: light (dataclasses + the Execution contract only).  The
heavy analysis stack is imported by :mod:`.runtime`/:mod:`.worker`, never
here, so ``urban_network_analysis.batch.__init__`` stays plan-pure.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

from ..Execution import (
    BatchReport,
    CapabilityError,
    RowOutcome,
)

__all__ = [
    "BatchCancelledError",
    "BatchCheckpointError",
    "RowExecutionOutcome",
    "BatchExecutionReport",
]


class BatchCancelledError(CapabilityError):
    """A parallel batch was cancelled before completion.

    Raised on a caller deadline (``ExecutionOptions.timeout_s``), a
    KeyboardInterrupt while the coordinator owns workers, or the loss of a
    worker process.  The committed serial prefix (artifacts, settings
    mutations, report rows) is preserved and the exception carries the
    prefix's outcomes in ``batch_rows`` — cancellation never counts failed
    or unfinished rows as throughput (dossier 05).
    """

    def __init__(self, message: str, batch_rows: Tuple[RowOutcome, ...] = ()):
        super().__init__(message)
        self.batch_rows = tuple(batch_rows)


class BatchCheckpointError(CapabilityError):
    """A checkpointed batch cannot be resumed as requested.

    Raised when the on-disk journal's recorded identity (code, model,
    profile, inputs) does not match this call — the dossier-05 rule that a
    restart rejects changed identity rather than mixing generations — or
    when the journal itself is unusable and no safe interpretation exists.
    Recoverable damage (a corrupt row record) is quarantined instead: the
    affected row is re-run, not fatal.
    """

    def __init__(self, message: str, batch_rows: Tuple[RowOutcome, ...] = ()):
        super().__init__(message)
        self.batch_rows = tuple(batch_rows)


@dataclass(frozen=True, kw_only=True)
class RowExecutionOutcome(RowOutcome):
    """One row's outcome under the parallel runtime (extends :RowOutcome:).

    ``admission``/``admission_reasons`` come from the planner (dossier 04
    step 4 — serial rows always carry the reason they could not be proven
    independent).  ``phase`` is the dossier-05 transaction state at report
    time; a completed call only ever reports ``COMMITTED`` rows (plus
    ``SKIPPED``), because uncommitted work is cancelled, not reported as
    success.  ``timings`` are diagnostics, never cache/checkpoint identity
    (INTERFACES.md).
    """

    admission: str = "serial"
    admission_reasons: Tuple[str, ...] = ()
    phase: str = "COMMITTED"
    worker_pid: Optional[int] = None
    peak_rss_bytes: Optional[int] = None
    output_files: Tuple[str, ...] = ()
    timings: Dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True, kw_only=True)
class BatchExecutionReport(BatchReport):
    """The parallel route's report (extends :BatchReport:).

    Adds what INTERFACES.md requires beyond the staged fields: effective
    worker count and every independent pool, the planner's commit/fold
    orders with the worker-admissible and serialized row sets, cancellation
    state, the checkpoint generation reused/advanced by this call, and the
    output manifest (committed artifact paths per row).  ``rows`` holds
    :RowExecutionOutcome: values in caller row order for the processed
    prefix; on a raised failure the same tuples attach to the raised
    exception instead (serial parity: ``batch_report`` stays None when a
    batch call aborts).
    """

    workers_effective: int = 0
    pools: Tuple[str, ...] = ()
    commit_order: Tuple[int, ...] = ()
    fold_order: Tuple[int, ...] = ()
    worker_admissible: Tuple[int, ...] = ()
    serialized: Tuple[int, ...] = ()
    cancelled: bool = False
    checkpoint_generation: Optional[int] = None
    output_manifest: Dict[str, Tuple[str, ...]] = field(default_factory=dict)
    notes_runtime: Tuple[str, ...] = ()
    extra: Dict[str, Any] = field(default_factory=dict)
