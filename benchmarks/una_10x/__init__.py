"""una_10x benchmark instrumentation (task H00).

Borrows the reviewed platform mechanics (benchmarks/platform, verdict
"REUSE - primary measurement vehicle" in h00_harness_audit.json) and adds
the campaign-specific instrumentation laws:

* timing.PublicCallTimer  - primary wall = public_call_complete, separate
  post-hash wall, diagnostic spans explicitly nested and never summed.
* window.BoundedPendingBytes / window.validate_window - a failed, thin,
  mis-dispatched, shadow-imported or queue-oversubscribed window can never
  become a valid una_10x measurement record.
* dispatch_probe.PublicDispatchProbe - true-public-API dispatch trace for
  engagement legs only (the timed run stays uninstrumented).

Harness negative tests live in tests/tenx/harness; completion criterion:
"Harness negative tests prove failures cannot become valid windows".
"""
from .errors import TimerViolation, WindowInvalid
from .timing import PublicCallTimer, timer_violations
from .window import (BoundedPendingBytes, LEGAL_PUBLIC_ENTRIES,
                     validate_window)

HARNESS_VERSION = "una-10x-h00-h1"

__all__ = ["HARNESS_VERSION", "TimerViolation", "WindowInvalid",
           "PublicCallTimer", "timer_violations", "BoundedPendingBytes",
           "LEGAL_PUBLIC_ENTRIES", "validate_window"]
