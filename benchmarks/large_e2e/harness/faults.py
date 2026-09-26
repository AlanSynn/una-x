"""Test-only fault injection registry (HARNESS.md negative tests).

Faults exist to prove that the coordinator fails fast, cancels bounded work
and leaves no surviving owned processes.  They are injected ONLY through
run.py's --test-fault flag, force measurement_class="harness_selftest" and
qualification_valid=false, and are rejected outright in any qualification
context.  This is not a hidden arm: with --test-fault absent the fault
machinery is completely inert.
"""
from __future__ import annotations

# name -> description of the injected behavior and what the coordinator must do
FAULTS: dict[str, str] = {
    # Dispatch/verification invalidity
    "wrong_engine_signature": (
        "worker reports a flow engine signature for an accessibility job; "
        "coordinator must mark the run INVALID"),
    "zero_size_output": (
        "worker emits a zero-byte placeholder output file; run must be INVALID"),
    "duplicate_job_id": (
        "worker sends a second result reusing a job id; run must be INVALID"),
    "missing_companion_file": (
        "worker deletes one required companion output before reporting; "
        "run must be INVALID"),
    "altered_bytes": (
        "worker mutates an output file after hashing it, before the "
        "coordinator re-hash; run must be INVALID"),
    # Liveness / shutdown
    "worker_raise": (
        "worker raises mid-job; job recorded failed and run INVALID, pool "
        "keeps draining"),
    "exit_before_ready": (
        "worker exits before the readiness barrier; coordinator fails fast "
        "with no hang"),
    "die_while_dispatch": (
        "worker dies on first job receipt while the coordinator is still "
        "dispatching; prompt cancellation, retained received results"),
    "hang_after_output": (
        "worker sends its result then hangs forever; coordinator must "
        "terminate it at shutdown with no surviving owned processes"),
    "never_send_sentinel": (
        "worker exits without the worker_exit record; coordinator must "
        "complete bounded shutdown and mark the run INVALID"),
    # Bounded-queue exercise (NOT invalidating)
    "slow_drain": (
        "worker sleeps briefly per job so the input queue fills; used to "
        "prove bounded queues complete without deadlock"),
}

INVALIDATING_FAULTS = frozenset(FAULTS) - {"slow_drain"}


def validate_fault(name: str | None) -> str | None:
    if name is None:
        return None
    if name not in FAULTS:
        raise ValueError(
            f"unknown --test-fault {name!r}; known faults: {sorted(FAULTS)}")
    return name
