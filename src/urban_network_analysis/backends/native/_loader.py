"""Loader for the NATIVE_CORE compiled module (dossier 08).

The compiled ``_native_core`` extension is an OPTIONAL accelerator: this
module is the only code that imports it, and a missing/stale/broken
binary is reported as :class:`NativeUnavailableError` (an ImportError)
— never silently swallowed into a different computation.  The route
(``_route``) turns that into the documented portable fallback; everything
else (Engagement provenance, capability claims) requires an actually
engaged core.

Refusal conditions (each checked, each reported with its reason):

- ``UNA_NATIVE_CORE_DISABLE=1`` in the environment (cold-import and
  native-absent fallback tests drive this);
- the extension is not importable (never built, wrong interpreter);
- a required symbol is missing (a stale or foreign stub module);
- ``abi_version()`` differs from :data:`EXPECTED_ABI_VERSION` (build
  skew between the C++ core and this Python surface).
"""
import importlib
import os

__all__ = [
    "EXPECTED_ABI_VERSION",
    "MODULE_NAME",
    "REQUIRED_SYMBOLS",
    "NativeUnavailableError",
    "load_core",
    "reset_loader_cache",
]

# Must match scope_core.hpp ABI_VERSION; the loader cross-checks the
# loaded module's self-report against this pin.
EXPECTED_ABI_VERSION = 1

MODULE_NAME = "urban_network_analysis.backends.native._native_core"

# Every symbol the route/binding surface consumes.  A module missing any
# of these is not this binding (stub or stale build) and is refused.
REQUIRED_SYMBOLS = (
    "run_legacy",
    "run_corrected",
    "search_labels_legacy",
    "search_labels_corrected",
    "check_domain_size",
    "status_codes",
    "isa_report",
    "build_fingerprint",
    "abi_version",
    "CancelFlag",
)


class NativeUnavailableError(ImportError):
    """The compiled native core cannot be engaged (with the reason)."""


_CORE = None  # loaded-module cache (idempotent engagement per process)


def load_core():
    """Return the compiled ``_native_core`` module or raise
    ``NativeUnavailableError`` with the specific refusal reason."""
    global _CORE
    if _CORE is not None:
        return _CORE

    if os.environ.get("UNA_NATIVE_CORE_DISABLE") == "1":
        raise NativeUnavailableError(
            "native core disabled by environment (UNA_NATIVE_CORE_DISABLE=1)")

    try:
        # import_module consults sys.modules FIRST: a poisoned/stale
        # sys.modules entry (stub, halted import) is refused here even
        # after a real load bound the parent-package attribute — the
        # `from . import _native_core` form would silently return that
        # attribute and never re-check (the fail-sentinel law).
        core = importlib.import_module(MODULE_NAME)
    except ImportError as exc:
        raise NativeUnavailableError(
            f"compiled module {MODULE_NAME} is not importable "
            f"(not built for this interpreter?): {exc}") from exc

    missing = [name for name in REQUIRED_SYMBOLS if not hasattr(core, name)]
    if missing:
        raise NativeUnavailableError(
            f"loaded module is missing required symbols {missing} "
            "(stale or foreign build)")

    try:
        abi = int(core.abi_version())
    except Exception as exc:  # a symbol that exists but does not work
        raise NativeUnavailableError(
            f"abi_version() failed on the loaded module: {exc}") from exc
    if abi != EXPECTED_ABI_VERSION:
        raise NativeUnavailableError(
            f"abi skew: loaded module reports abi_version={abi}, this "
            f"Python surface expects {EXPECTED_ABI_VERSION} (rebuild)")

    _CORE = core
    return _CORE


def reset_loader_cache():
    """Forget the cached module (test hook: the loader refuses again on
    the next load_core() so disable/stub conditions are re-provable)."""
    global _CORE
    _CORE = None
