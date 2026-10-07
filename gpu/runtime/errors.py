"""Typed errors for the GPU fold region.

Refusal taxonomy (fail-closed by construction; nothing here is a fallback):

- ``HardwareUnavailable``  no CUDA-class driver/device on this host, or the
  device lacks hardware FP64.  G11 qualification is blocked by this state,
  never simulated around it.
- ``ParityGateNotRun``    a device exists but the requested cell has not
  passed the per-cell bit-validation gate (device exp / reduction-tree /
  tie-law reproduction against the pinned una_legacy fold bits).  Distinct
  from HardwareUnavailable on purpose: G10 ships the gate harness; until it
  runs on real hardware and passes, device dispatch is refused, silently
  degraded, or approximated by nothing.
- ``RegionCanceled``      cancellation observed at a tile barrier.  Never a
  truncated success (N00 law).
- ``BudgetExceeded``      staging/scratch beyond the declared byte budget.

Production callers must not catch these into a CPU fallback; the trusted CPU
route lives in src/urban_network_analysis and stays the default until a
route is qualified (G11) and wired by the integrator.
"""

__all__ = [
    "HardwareUnavailable",
    "ParityGateNotRun",
    "RegionCanceled",
    "BudgetExceeded",
    "RegionRefused",
]


class RegionRefused(Exception):
    """Base class for typed GPU-region refusals (fail-closed family)."""


class HardwareUnavailable(RegionRefused):
    """No usable CUDA-class FP64 device on this host."""


class ParityGateNotRun(RegionRefused):
    """The cell's bit-validation gate has not been run and passed."""


class RegionCanceled(RegionRefused):
    """Cancellation was observed at a tile barrier."""


class BudgetExceeded(RegionRefused):
    """A staging/scratch request exceeded the declared byte budget."""
