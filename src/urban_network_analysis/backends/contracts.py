"""Backend-neutral kernel ABI: typed inputs/outputs/capabilities/context.

Delivered by campaign task EXECUTION per dossier 10 ("Define
KernelInputs/Outputs/Capabilities/ExecutionContext with dtype/layout,
logical reduction plan, stage identity, owner, cancellation and memory
budget").  Public code and engine code program against THESE types; they
never import CUDA/Numba/pybind types.  Only numpy is imported here (a core
package dependency), so importing this module is cheap and safe everywhere.

Staged-delivery note (recorded in the EXECUTION receipt): the field NAMES
below are the frozen ABI that NATIVE_CORE / GPU_DEVICE / CACHE_GRAPH /
CPU_LAYOUT code against.  Registration, engagement counters, and the
capability registry consult do not exist yet — an empty capability registry
is exactly what makes ``backend='native'|'gpu'`` admission raise (see
``urban_network_analysis.Execution.admit_execution``) and what keeps ``auto``
on the trusted reference route.  No stub execution route is provided and none
may be presented as native/GPU capability.
"""

from dataclasses import dataclass, field
from typing import Literal, Mapping, Optional, Tuple

import numpy as np

__all__ = [
    "ArrayDescriptor", "ArrayResult",
    "KernelInputs", "KernelOutputs",
    "KernelCapabilities", "CapabilityRegistryEntry",
    "ExecutionContext", "kernel_inputs_fingerprint",
]


def _require_non_empty(field_name: str, value: str) -> None:
    """Required-string check shared by every ABI type (Execution.py uses the
    same whitespace-sensitive rule for the public option types)."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string, "
                         f"got {value!r}")


def _require_sha256_hex(field_name: str, value: str) -> None:
    """content_hash is the identity a cache key or parity check consumes —
    it must be a real sha256 hex digest, never invented (review F3).  The
    producer is ``hashlib.sha256(...).hexdigest()`` — lowercase, 64 chars."""
    _require_non_empty(field_name, value)
    if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(
            f"{field_name} must be a 64-char lowercase sha256 hex string "
            f"(hashlib hexdigest form), got {value!r}")


def _require_non_empty_members(field_name: str, values) -> None:
    for i, v in enumerate(values):
        _require_non_empty(f"{field_name}[{i}]", v)


@dataclass(frozen=True, kw_only=True)
class ArrayDescriptor:
    """Immutable description of one kernel-visible array.

    ``content_hash`` is the sha256 hex of the array's raw bytes (C order) —
    the identity a cache key or a parity check consumes; it is computed by
    the producer, never invented.
    """

    name: str
    dtype: str                      # numpy dtype str, e.g. "float64"
    shape: Tuple[int, ...]
    layout: Literal["dense", "csr", "coo", "geometry"] = "dense"
    content_hash: str               # sha256 hex of raw bytes (C order)
    owner: str                      # owning scope (e.g. "topology", "engine.flow")

    def __post_init__(self):
        _require_non_empty("ArrayDescriptor.name", self.name)
        _require_non_empty("ArrayDescriptor.dtype", self.dtype)
        _require_non_empty("ArrayDescriptor.owner", self.owner)
        _require_sha256_hex("ArrayDescriptor.content_hash", self.content_hash)
        if self.layout not in ("dense", "csr", "coo", "geometry"):
            raise ValueError(
                f"layout must be one of 'dense'|'csr'|'coo'|'geometry', "
                f"got {self.layout!r}")
        # shapes come from numpy .shape — validate cheaply without forcing int
        if any(dim < 0 for dim in self.shape):
            raise ValueError(
                f"ArrayDescriptor.shape must be non-negative, "
                f"got {self.shape!r}")


@dataclass(frozen=True, kw_only=True)
class ArrayResult:
    """One kernel output array.  The container is frozen; the ndarray is not
    — ownership discipline (no in-place mutation after publication) is the
    producer's obligation and is checked by parity tests downstream."""

    name: str
    array: np.ndarray


@dataclass(frozen=True, kw_only=True)
class KernelInputs:
    """Ordered, immutable kernel request (dossier 10).

    ``semantic_profile`` and ``logical_reduction_plan`` travel with EVERY
    kernel invocation — profile/model/partition and ownership are part of
    the input identity, never ambient state.
    """

    stage: str                      # e.g. "accessibility.reach", "flow.assign"
    semantic_profile: str
    logical_reduction_plan: str
    arrays: Tuple[ArrayDescriptor, ...]
    owner: str                      # admission owner (e.g. "UNA.RunAccessibility")

    def __post_init__(self):
        _require_non_empty("KernelInputs.stage", self.stage)
        _require_non_empty("KernelInputs.semantic_profile",
                           self.semantic_profile)
        _require_non_empty("KernelInputs.logical_reduction_plan",
                           self.logical_reduction_plan)
        _require_non_empty("KernelInputs.owner", self.owner)
        names = [a.name for a in self.arrays]
        if len(names) != len(set(names)):
            raise ValueError(
                f"KernelInputs.arrays has duplicate names: {names}")


def kernel_inputs_fingerprint(inputs: KernelInputs) -> str:
    """Stable sha256 hex over a KernelInputs' numerical identity.

    Covers stage, semantic profile, reduction plan, and every array's
    name/dtype/shape/layout/content hash.  (Owner is deliberately excluded:
    two owners sending numerically identical work share a fingerprint.)
    """
    import hashlib
    import json

    payload = {
        "stage": inputs.stage,
        "semantic_profile": inputs.semantic_profile,
        "logical_reduction_plan": inputs.logical_reduction_plan,
        "arrays": [
            {"name": a.name, "dtype": a.dtype, "shape": list(a.shape),
             "layout": a.layout, "content_hash": a.content_hash}
            for a in inputs.arrays
        ],
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


@dataclass(frozen=True, kw_only=True)
class KernelOutputs:
    """Typed kernel results plus honest completion status (dossier 10).

    ``status='incomplete'`` with ``next_logical_state`` is the restart
    contract for cancellation/checkpoint (FAULTS/BATCH_EXEC consume it);
    a kernel never reports 'complete' for partial work.
    """

    arrays: Tuple[ArrayResult, ...]
    status: Literal["complete", "incomplete"]
    next_logical_state: Optional[str] = None

    def __post_init__(self):
        if self.status not in ("complete", "incomplete"):
            raise ValueError(
                f"status must be 'complete'|'incomplete', got {self.status!r}")
        if self.status == "incomplete" and not self.next_logical_state:
            raise ValueError(
                "incomplete KernelOutputs must name next_logical_state "
                "(the restart point)")
        names = [r.name for r in self.arrays]
        if len(names) != len(set(names)):
            raise ValueError(
                f"KernelOutputs.arrays has duplicate names: {names}")


@dataclass(frozen=True, kw_only=True)
class KernelCapabilities:
    """What one backend binary claims it can do (dossier 10)."""

    backend: Literal["native", "gpu"]
    binary_fingerprint: str         # backend binary/compiler fingerprint
    supported_profiles: Tuple[str, ...]
    stage: str                      # stage/domain this capability covers
    isa_or_device_requirements: Tuple[str, ...]
    known_refusals: Tuple[str, ...] = ()
    qualified: bool = False         # calibrated end-to-end qualification state
    evidence_ref: Optional[str] = None  # retained evidence path for qualification

    def __post_init__(self):
        if self.backend not in ("native", "gpu"):
            raise ValueError(
                f"backend must be 'native'|'gpu', got {self.backend!r}")
        _require_non_empty("KernelCapabilities.binary_fingerprint",
                           self.binary_fingerprint)
        _require_non_empty("KernelCapabilities.stage", self.stage)
        _require_non_empty_members("KernelCapabilities.supported_profiles",
                                   self.supported_profiles)
        _require_non_empty_members(
            "KernelCapabilities.isa_or_device_requirements",
            self.isa_or_device_requirements)


@dataclass(frozen=True, kw_only=True)
class CapabilityRegistryEntry:
    """One registry row binding qualification evidence to a workload class.

    Dossier 10: entries bind semantic+math version, backend binary/compiler
    fingerprint, device/ISA, workload features, memory estimate, validation
    evidence and the calibrated end-to-end crossover.  Absent/stale/malformed
    entries choose trusted reference under auto; they never qualify a forced
    route.
    """

    capabilities: KernelCapabilities
    semantic_math_version: str
    workload_features: Mapping[str, str] = field(default_factory=dict)
    memory_estimate_bytes: Optional[int] = None
    crossover: Optional[str] = None  # calibrated threshold description/ref

    def __post_init__(self):
        _require_non_empty("CapabilityRegistryEntry.semantic_math_version",
                           self.semantic_math_version)


@dataclass(frozen=True, kw_only=True)
class ExecutionContext:
    """Runtime execution context handed to kernels (dossier 10).

    Carries the admitted budget, the logical schedule identity, physical
    resources, cancellation and ownership/provenance.  ``cancellation`` is a
    runtime-only ownership reference (never serialized into Settings) and
    ``None`` until the cancellation system lands (BATCH_EXEC/FAULTS).
    """

    semantic_profile: str
    logical_reduction_plan: str
    backend: str                    # the EFFECTIVE route (post-admission)
    cpu_budget: Optional[int]
    threads_per_worker: int
    memory_limit_bytes: Optional[int]
    workspace_limit_bytes: Optional[int]
    queue_depth: int
    writer_concurrency: int
    timeout_s: Optional[float]
    owner: str
    provenance: Mapping[str, str] = field(default_factory=dict)
    cancellation: Optional[object] = None  # runtime-only handle; never copied

    def __post_init__(self):
        _require_non_empty("ExecutionContext.semantic_profile",
                           self.semantic_profile)
        _require_non_empty("ExecutionContext.logical_reduction_plan",
                           self.logical_reduction_plan)
        _require_non_empty("ExecutionContext.owner", self.owner)
        # The context carries the route POST-admission: 'auto' is resolved
        # away by admit_execution, so only the three concrete routes are
        # valid here (review F3).
        if self.backend not in ("reference", "native", "gpu"):
            raise ValueError(
                f"ExecutionContext.backend must be one of "
                f"'reference'|'native'|'gpu' (post-admission effective "
                f"route), got {self.backend!r}")
