"""Backend-neutral ABI surface.

Only contract types are exposed here.  Backend implementations
(``native`` — NATIVE_CORE, ``gpu`` — GPU_DEVICE) are separate submodules that
land with their own tasks; importing this package must never import them and
must never require compiled or device libraries.
"""

from .contracts import (  # noqa: F401
    ArrayDescriptor,
    ArrayResult,
    CapabilityRegistryEntry,
    ExecutionContext,
    KernelCapabilities,
    KernelInputs,
    KernelOutputs,
    kernel_inputs_fingerprint,
)

__all__ = [
    "ArrayDescriptor",
    "ArrayResult",
    "CapabilityRegistryEntry",
    "ExecutionContext",
    "KernelCapabilities",
    "KernelInputs",
    "KernelOutputs",
    "kernel_inputs_fingerprint",
]
