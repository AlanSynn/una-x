"""NATIVE_CORE backend (dossier 08): the compiled CPU native backend.

Public surface:

- :func:`route_scope_access` — the integrated accessibility route with
  the native search+adjust core engaged when admitted (both semantic
  profiles), falling back to the pinned portable routes with a recorded
  reason otherwise.
- :func:`new_cancel_flag` — the cancellation handle the route accepts.
- :func:`native_capabilities` — the typed capability claim backed by the
  actually loaded binary (never a static assertion).
- :func:`is_available` / :func:`load_core` — engagement probes.
- :class:`NativeUnavailableError` — the typed refusal to engage.

Importing this package NEVER imports the compiled module and never
fails when it is absent (the loader refuses lazily, with a reason).
Importing it does pull the portable stack (``kernels``/``reference``,
via ``_route``) — the routes compose the core with the canonical fold
references and fall back to them; it never pulls the compiled binary.
"""
from ..contracts import KernelCapabilities
from ._loader import (  # noqa: F401
    EXPECTED_ABI_VERSION,
    NativeUnavailableError,
    load_core,
)
from ._route import (  # noqa: F401
    NativeBackendMismatch,
    NativeRunResult,
    route_scope_access,
)

__all__ = [
    "EXPECTED_ABI_VERSION",
    "KernelCapabilities",
    "NativeBackendMismatch",
    "NativeRunResult",
    "NativeUnavailableError",
    "is_available",
    "load_core",
    "native_capabilities",
    "new_cancel_flag",
    "route_scope_access",
]


def new_cancel_flag():
    """A fresh cancellation handle (this binding's CancelFlag)."""
    return load_core().CancelFlag()


def is_available():
    """True when the compiled core passes every engagement check."""
    try:
        load_core()
    except NativeUnavailableError:
        return False
    return True


def native_capabilities():
    """The capability claim of the ACTUALLY LOADED binary.

    ``qualified`` stays False until a qualification task calibrates it
    with retained evidence (contracts.py: absent/stale entries choose
    the trusted reference under auto; they never qualify a forced
    route).  Raises NativeUnavailableError when the core cannot engage.
    """
    core = load_core()
    from ...kernels.reductions import STAGE_FOLD  # local: defer numba import
    isa = core.isa_report()
    return KernelCapabilities(
        backend="native",
        binary_fingerprint=str(core.build_fingerprint()),
        supported_profiles=("una_legacy", "corrected_v1"),
        stage=STAGE_FOLD,
        # The COMPILED baseline (what the binary requires), not the
        # host's ceiling — a v4 host must not inflate the claim.
        isa_or_device_requirements=(str(isa["compiled_baseline"]),),
        known_refusals=(
            "una_legacy: compact route (admission-mirror refusal) stays "
            "on the pinned oracle",
            "una_legacy: non-C-contiguous admitted arrays delegate",
            "una_legacy: A1 destination terminals outside [0, V+d) delegate "
            "(Python negative-index semantics)",
            "corrected_v1: none — the reference validators are the typed "
            "admission contract and the core re-enforces them",
        ),
        qualified=False,
        evidence_ref=None,
    )
