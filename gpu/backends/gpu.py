"""GPU backend capability surface (dossier-10 ABI shapes, src-contracts).

This module SPEAKS the shape of
src/urban_network_analysis/backends/contracts.py
(KernelCapabilities / CapabilityRegistryEntry) without importing it: gpu/
stays import-light and src-free so that detection never loads the engine.
tests/tenx/gpu/test_registry_abi_compat.py pins the structural
compatibility against the real contracts.py bytes (standalone load).

LAWS BOUND HERE:
  - qualified is ALWAYS False in this build: qualification is G11's
    real-device deliverable; a registry entry without qualification
    evidence chooses the trusted reference under auto and never qualifies
    a forced route (contracts.py).
  - hardware capability detection is not engagement: reporting what a host
    probes as absent is the whole honest capability of this build.
"""

from __future__ import annotations

from typing import Mapping, Optional

from ..runtime.availability import detect_nvcc, probe_driver
from ..runtime.errors import HardwareUnavailable, ParityGateNotRun  # noqa: F401  (re-export)

__all__ = [
    "BACKEND", "STAGE", "SEMANTIC_MATH_VERSION", "SUPPORTED_PROFILES",
    "ISA_OR_DEVICE_REQUIREMENTS", "capability_entry", "cell_gate",
    "cell_is_gated_open", "availability_report",
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

# In-memory cell-gate state; the durable gate evidence lands at G11 in the
# qualification run directory.  A passed gate is set ONLY by
# gpu/kernels/validate_device_bits.py on a real device — never by a
# CPU-only test, which is why tests exercise the gate through refusals.
_CELL_GATES: dict = {}


def capability_entry(binary_fingerprint: str, cell_id: str,
                     memory_estimate_bytes: Optional[int] = None,
                     crossover: Optional[str] = None,
                     workload_features: Optional[Mapping[str, str]] = None,
                     ) -> dict:
    """A CapabilityRegistryEntry-shaped dict, permanently unqualified here.

    ``qualified=False`` is structural in this build: G10 cannot qualify a
    GPU route without hardware, and must not pretend otherwise.
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


def cell_gate(cell_id: str, *, passed: bool, association: str,
              evidence_ref: str) -> dict:
    """Record a cell's bit-validation outcome (gate harness only, in-RAM).

    Passing requires an association from the certified tree set AND a
    non-empty evidence reference pointing at real device evidence.  CPU-only
    callers must never pass ``passed=True``; the parameter exists so the
    gate harness (running on the qualification device) can open a cell.
    """
    from ..kernels.host_twins import REDUCTION_TREES
    if passed:
        if association not in REDUCTION_TREES:
            raise ValueError(f"certified association must be one of "
                             f"{REDUCTION_TREES}, got {association!r}")
        if not evidence_ref:
            raise ValueError("a passed gate requires an evidence_ref to "
                             "real device evidence")
    _CELL_GATES[cell_id] = {
        "bit_validated": bool(passed),
        "association": association if passed else "",
        "evidence_ref": evidence_ref,
    }
    return dict(_CELL_GATES[cell_id])


def cell_is_gated_open(cell_id: str) -> bool:
    state = _CELL_GATES.get(cell_id)
    return bool(state and state["bit_validated"])


def cell_association(cell_id: str) -> str:
    state = _CELL_GATES.get(cell_id)
    return state["association"] if state else ""


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
