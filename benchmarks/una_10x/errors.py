"""una_10x harness violations (platform HarnessViolation subtypes)."""
from __future__ import annotations

from benchmarks.platform.errors import HarnessViolation


class TimerViolation(HarnessViolation):
    """Primary/post-hash/diagnostic span law broken (BENCHMARKS.md)."""


class WindowInvalid(HarnessViolation):
    """A measurement window fails admissibility; it can never be counted."""
