"""GPU backend capability surface (dossier-10 ABI shapes, src-contracts).

This module SPEAKS the shape of
src/urban_network_analysis/backends/contracts.py
(KernelCapabilities / CapabilityRegistryEntry) without importing it: gpu/
stays import-light and src-free so that detection never loads the engine.
tests/tenx/gpu/test_registry_abi_compat.py pins the structural
compatibility against the real contracts.py bytes (standalone load).

LAWS BOUND HERE:
  - qualified is ALWAYS False in this build until a qualification run
    admits a cell: a registry entry without qualification evidence chooses
    the trusted reference under auto and never qualifies a forced route
    (contracts.py).
  - hardware capability detection is not engagement: reporting what a host
    probes as absent is the whole honest capability of a detection surface.
  - THE GATE STORE IS DURABLE AND EVIDENCE-BACKED (G10R F2): the sole
    authority for ``cell_is_gated_open`` is a JSON store file whose entries
    name an evidence artifact; the artifact's bytes are re-verified by
    sha256 at every read.  There is NO in-RAM gate state: an in-process
    caller can no more open a cell than it can forge the evidence bytes.
    Only gpu/kernels/validate_device_bits.py running on a real device
    writes passed entries (record_gate).

Store contract (``UNA_GPU_GATE_STORE`` env or explicit path argument):
  {"cells": {"<cell_id>": {"bit_validated": bool, "association": str,
                           "evidence_ref": str, "evidence_sha256": str,
                           "recorded_utc": str}}}
A cell is OPEN iff an entry exists with bit_validated true, an association
from the certified tree set, and an evidence file whose CURRENT sha256
equals the recorded one.  Every other state — missing store, missing
entry, missing evidence, changed evidence, unknown association — is CLOSED
with a typed reason.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Optional

from ..runtime.availability import detect_nvcc, probe_driver
from ..runtime.errors import HardwareUnavailable, ParityGateNotRun  # noqa: F401  (re-export)

__all__ = [
    "BACKEND", "STAGE", "SEMANTIC_MATH_VERSION", "SUPPORTED_PROFILES",
    "ISA_OR_DEVICE_REQUIREMENTS", "capability_entry", "GATE_STORE_ENV",
    "resolve_gate_store", "record_gate", "gate_state", "cell_is_gated_open",
    "cell_association", "availability_report",
    "HardwareUnavailable", "ParityGateNotRun",
]

BACKEND = "gpu"
STAGE = "accessibility.fold"

# The math identity this region would execute: the legacy fastmath fold plan
# with its numba pin (typed identities include the compiler that produced
# the pinned bits).
SEMANTIC_MATH_VERSION = ("plan.accessibility.fold.reach_gravity_knn.v1."
                         "legacy_fastmath/numba-0.67.0")

SUPPORTED_PROFILES = ("una_legacy",)

ISA_OR_DEVICE_REQUIREMENTS = (
    "cuda_class_device",       # Apple Metal/MPS/MLX are not qualification targets
    "hardware_fp64",           # FP32 substitution is forbidden (dossier 08)
    "cell_bit_validation_gate",  # per-cell gate against pinned legacy bits
)

GATE_STORE_ENV = "UNA_GPU_GATE_STORE"

DEFAULT_CELL = "una_legacy.fold.default"


def capability_entry(binary_fingerprint: str, cell_id: str,
                     memory_estimate_bytes: Optional[int] = None,
                     crossover: Optional[str] = None,
                     workload_features: Optional[Mapping] = None,
                     ) -> dict:
    """A CapabilityRegistryEntry-shaped dict, unqualified until a
    qualification run records admissible evidence.

    ``qualified=False`` is structural here: only the G11 qualification
    deliverable (measured crossover + passed gate evidence on real
    hardware) can justify True, and that admission happens in the
    campaign register, not by editing this constant.
    """
    if not binary_fingerprint:
        raise ValueError("binary_fingerprint required (nvcc/target string "
                         "or an explicit unavailable marker)")
    return {
        "capabilities": {
            "backend": BACKEND,
            "binary_fingerprint": binary_fingerprint,
            "supported_profiles": SUPPORTED_PROFILES,
            "stage": STAGE,
            "isa_or_device_requirements": ISA_OR_DEVICE_REQUIREMENTS,
            "known_refusals": ("HardwareUnavailable", "ParityGateNotRun",
                               "RegionCanceled", "BudgetExceeded"),
            "qualified": False,
            "evidence_ref": None,
        },
        "semantic_math_version": SEMANTIC_MATH_VERSION,
        "workload_features": dict(workload_features or {}),
        "memory_estimate_bytes": memory_estimate_bytes,
        "crossover": crossover,
    }


def resolve_gate_store(explicit: Optional[str] = None) -> Optional[str]:
    """Explicit argument wins, else ``UNA_GPU_GATE_STORE``, else None.

    None means "no gate store configured": every cell reads CLOSED, which
    is the correct fail-closed default for any process that has not been
    pointed at recorded gate evidence.
    """
    if explicit:
        return str(explicit)
    env = os.environ.get(GATE_STORE_ENV)
    return str(env) if env else None


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_store(store_path: str) -> dict:
    with open(store_path, "r", encoding="utf-8") as fh:
        doc = json.load(fh)
    if not isinstance(doc, dict) or not isinstance(doc.get("cells"), dict):
        raise ValueError(f"gate store {store_path!r}: malformed document "
                         "(expected {'cells': {...}})")
    return doc


def record_gate(store_path: str, cell_id: str, *, passed: bool,
                association: str = "", evidence_ref: str = "") -> dict:
    """Durable gate write — the ONLY writer, harness-only by contract.

    A passed entry requires an association from the certified tree set AND
    an evidence_ref naming an EXISTING file; the file's sha256 is recorded
    and re-verified on every future read.  CPU-only callers cannot
    legitimately pass ``passed=True`` because they cannot produce the
    device evidence file the entry binds to — the store enforces the
    evidence, not anyone's word.

    Writes atomically (tmp file + os.replace).  Returns the stored entry.
    """
    from ..kernels.host_twins import REDUCTION_TREES  # lazy: import-light
    if passed:
        if association not in REDUCTION_TREES:
            raise ValueError(f"certified association must be one of "
                             f"{REDUCTION_TREES}, got {association!r}")
        if not evidence_ref:
            raise ValueError("a passed gate requires an evidence_ref to "
                             "real device evidence")
        if not os.path.isfile(evidence_ref):
            raise ValueError(f"evidence_ref {evidence_ref!r} does not name "
                             "an existing file; no evidence, no gate")
        evidence_sha256 = _sha256_file(evidence_ref)
    else:
        if association:
            raise ValueError("a refused/closed entry carries no association")
        evidence_sha256 = (_sha256_file(evidence_ref)
                           if evidence_ref and os.path.isfile(evidence_ref)
                           else "")
    path = Path(store_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = _read_store(store_path) if path.is_file() else {"cells": {}}
    entry = {
        "bit_validated": bool(passed),
        "association": association if passed else "",
        "evidence_ref": evidence_ref,
        "evidence_sha256": evidence_sha256,
        "recorded_utc": datetime.now(timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"),
    }
    doc["cells"][cell_id] = entry
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n",
                   encoding="utf-8")
    os.replace(tmp, path)
    return dict(entry)


_CLOSED = {"open": False, "reason": "", "association": "",
           "evidence_ref": ""}


def gate_state(cell_id: str,
               store_path: Optional[str] = None) -> dict:
    """AUTHORITATIVE read of the durable evidence-backed store.

    Re-reads the store file AND re-verifies the evidence bytes (sha256) on
    every call — there is no cache and no in-RAM authority.  Any of: store
    unconfigured/missing/malformed, entry absent, entry not passed,
    association outside the certified set, evidence file missing or CHANGED
    → closed, with the reason stated.  A changed evidence file closes the
    cell: the recorded verdict binds to bytes, not to a path.
    """
    from ..kernels.host_twins import REDUCTION_TREES  # lazy: import-light
    resolved = resolve_gate_store(store_path)
    if not resolved:
        return {**_CLOSED,
                "reason": f"no gate store configured ({GATE_STORE_ENV} "
                          "unset and no explicit path)"}
    try:
        doc = _read_store(resolved)
    except FileNotFoundError:
        return {**_CLOSED, "reason": f"gate store {resolved!r} not found"}
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return {**_CLOSED, "reason": f"gate store unreadable: {exc}"}
    entry = doc["cells"].get(cell_id)
    if not isinstance(entry, dict):
        return {**_CLOSED,
                "reason": f"cell {cell_id!r} has no entry in {resolved!r}"}
    if not entry.get("bit_validated"):
        return {**_CLOSED,
                "reason": f"cell {cell_id!r} recorded bit_validated=False",
                "evidence_ref": entry.get("evidence_ref", "")}
    association = entry.get("association", "")
    if association not in REDUCTION_TREES:
        return {**_CLOSED,
                "reason": f"cell {cell_id!r} association {association!r} "
                          f"outside the certified set {REDUCTION_TREES}",
                "evidence_ref": entry.get("evidence_ref", "")}
    evidence_ref = entry.get("evidence_ref", "")
    if not evidence_ref or not os.path.isfile(evidence_ref):
        return {**_CLOSED,
                "reason": f"cell {cell_id!r} evidence file missing "
                          f"({evidence_ref!r})",
                "evidence_ref": evidence_ref}
    recorded = entry.get("evidence_sha256", "")
    try:
        current = _sha256_file(evidence_ref)
    except OSError as exc:
        return {**_CLOSED,
                "reason": f"cell {cell_id!r} evidence unreadable: {exc}",
                "evidence_ref": evidence_ref}
    if current != recorded:
        return {**_CLOSED,
                "reason": f"cell {cell_id!r} evidence bytes changed "
                          f"(recorded {recorded[:12]}…, current "
                          f"{current[:12]}…); verdict binds to bytes",
                "evidence_ref": evidence_ref}
    return {"open": True, "reason": "evidence-verified",
            "association": association, "evidence_ref": evidence_ref}


def cell_is_gated_open(cell_id: str,
                       store_path: Optional[str] = None) -> bool:
    """True iff gate_state opens the cell (durable, evidence-verified)."""
    return gate_state(cell_id, store_path)["open"]


def cell_association(cell_id: str,
                     store_path: Optional[str] = None) -> str:
    """The certified association, or "" when the cell is closed."""
    return gate_state(cell_id, store_path)["association"]


def availability_report() -> dict:
    """Real probes, reported verbatim — the honest capability of this host."""
    driver = probe_driver()
    nvcc = detect_nvcc()
    return {
        "driver": {
            "available": driver.available,
            "reason": driver.reason,
            "detail": driver.detail,
            "device_count": driver.device_count,
            "fp64_capable": driver.fp64_capable,
            "device_names": list(driver.device_names),
        },
        "nvcc": {"on_path": nvcc.on_path, "version": nvcc.version,
                 "detail": nvcc.detail},
        "not_a_qualification_target_note": (
            "Apple GPU (Metal/MPS) presence does not satisfy any GPU "
            "obligation; qualification requires CUDA-class FP64 hardware"),
    }
