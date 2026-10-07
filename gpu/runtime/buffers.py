"""Buffer ownership for the GPU fold region.

Ownership law (gpu_region.json residency_model, dossier 08):
  - inputs are IMMUTABLE leases: a lease never exposes a writable view of
    caller memory; validation happens at take() and refuses (never coerces),
    following the N00 binding law (dtype/ndim/contiguity named on refusal);
  - outputs are OWNED by the region until handed off; handing off releases
    the region's claim;
  - scratch/staging is bounded by a declared byte budget; every allocation
    and release moves a ledger; the ledger reports outstanding and high-water
    bytes (measured, per the native scratch_peak_bytes precedent);
  - device memory allocation is fail-closed: without an attached device
    context it raises HardwareUnavailable.  No simulation.

Immutable internal leases do not permit public mutable aliasing: a lease
handed to the region can only be read through the read-only view the region
issues.
"""

from __future__ import annotations

import numpy as np

from .errors import BudgetExceeded, HardwareUnavailable

__all__ = [
    "InputLease", "OwnedBuffer", "ScratchLedger", "StagingPool",
    "require_c_contiguous",
]


def require_c_contiguous(name: str, array: np.ndarray) -> np.ndarray:
    """Typed layout validation: refuse, never coerce (N00 binding law)."""
    if not isinstance(array, np.ndarray):
        raise TypeError(f"{name} must be a numpy array, got "
                        f"{type(array).__name__}")
    if array.ndim != 1:
        raise ValueError(f"{name} must be 1-D, got ndim={array.ndim}")
    if len(array) and not array.flags["C_CONTIGUOUS"]:
        raise ValueError(f"{name} must be C-contiguous "
                         f"(strides={array.strides}); refusal, no copy")
    return array


class InputLease:
    """An immutable lease over caller input memory.

    The lease validates on construction and exposes ``readonly_view()`` —
    a view with writeable=False.  The caller's original array object is
    never exposed by the region while the lease is open.
    """

    __slots__ = ("_array", "_name", "_open")

    def __init__(self, name: str, array: np.ndarray, *, dtype: np.dtype):
        require_c_contiguous(name, array)
        if array.dtype != np.dtype(dtype):
            raise ValueError(f"{name} must have dtype {np.dtype(dtype)}, "
                             f"got {array.dtype}; refusal, no coercion "
                             "(typed identities include actual dtype bits)")
        object.__setattr__(self, "_array", array)
        object.__setattr__(self, "_name", name)
        object.__setattr__(self, "_open", True)

    @property
    def name(self) -> str:
        return self._name

    @property
    def nbytes(self) -> int:
        return int(self._array.nbytes)

    def readonly_view(self) -> np.ndarray:
        if not self._open:
            raise RuntimeError(f"lease {self._name!r} already released")
        view = self._array.view()
        view.flags["WRITEABLE"] = False
        return view

    def release(self) -> None:
        object.__setattr__(self, "_open", False)


class OwnedBuffer:
    """A region-owned buffer; the region releases it after handoff/teardown."""

    __slots__ = ("array", "name")

    def __init__(self, name: str, array: np.ndarray):
        self.name = name
        self.array = array

    @property
    def nbytes(self) -> int:
        return int(self.array.nbytes)


class ScratchLedger:
    """Byte ledger for outstanding scratch and its measured high-water."""

    __slots__ = ("_outstanding", "_high_water")

    def __init__(self) -> None:
        self._outstanding = 0
        self._high_water = 0

    @property
    def outstanding_bytes(self) -> int:
        return self._outstanding

    @property
    def high_water_bytes(self) -> int:
        return self._high_water

    def allocated(self, nbytes: int) -> None:
        self._outstanding += int(nbytes)
        if self._outstanding > self._high_water:
            self._high_water = self._outstanding

    def released(self, nbytes: int) -> None:
        self._outstanding -= int(nbytes)
        if self._outstanding < 0:
            raise RuntimeError("scratch ledger underflow: release without "
                               "matching allocation")


class StagingPool:
    """A bounded staging pool for host-side H2D/D2H buffers.

    The pool holds the declared byte budget for the region's per-tile
    staging; allocations beyond the budget refuse (BudgetExceeded) instead
    of silently growing.  Device-side allocation is a separate, fail-closed
    operation (see :func:`device_allocate`).
    """

    def __init__(self, budget_bytes: int):
        if int(budget_bytes) <= 0:
            raise ValueError("budget_bytes must be positive")
        self._budget = int(budget_bytes)
        self.ledger = ScratchLedger()

    @property
    def budget_bytes(self) -> int:
        return self._budget

    def allocate(self, name: str, shape, dtype=np.float64) -> OwnedBuffer:
        nbytes = int(np.prod(shape, dtype=np.int64)) * np.dtype(dtype).itemsize
        if self.ledger.outstanding_bytes + nbytes > self._budget:
            raise BudgetExceeded(
                f"{name}: {nbytes} B exceeds remaining staging budget "
                f"({self._budget - self.ledger.outstanding_bytes} B of "
                f"{self._budget} B)")
        self.ledger.allocated(nbytes)
        return OwnedBuffer(name, np.empty(shape, dtype=dtype))

    def release(self, buffer: OwnedBuffer) -> None:
        self.ledger.released(buffer.nbytes)


def device_allocate(nbytes: int, context=None) -> None:
    """Allocate device memory; FAILS CLOSED without an attached context.

    There is intentionally no return value on the refusal path: a host
    without a device context can never obtain device memory, and nothing
    here may hand back a host buffer masquerading as one.
    """
    if context is None:
        raise HardwareUnavailable(
            "device allocation requested without an attached CUDA context; "
            "this host has no usable FP64 CUDA device (fail-closed)")
    raise NotImplementedError(
        "device-context allocation lands with the G11 qualification wiring; "
        "the ownership ledger contract is defined and tested host-side")
