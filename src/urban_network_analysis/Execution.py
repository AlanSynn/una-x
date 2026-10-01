"""Public execution contract: immutable options, typed admission, reports.

Delivered by campaign task EXECUTION per the frozen public schema in
``campaigns/una_platform/evidence/contract/contract.md`` §6 (binding):
``ExecutionOptions`` / ``CacheOptions`` are frozen, keyword-only dataclasses;
``backend='native'/'gpu'`` without a qualified capability raises a typed
admission error BEFORE scientific execution; ``backend='auto'`` may fall back
to the trusted reference route and records the reason.  No silent fallback.

Nothing in this module imports heavy analysis stack members (numba, sklearn)
or any backend implementation — public code requests semantic profiles,
resources and optional backends; it never imports CUDA/Numba/pybind types.
Backend-neutral kernel ABI types live in ``urban_network_analysis.backends.contracts``.

Staged delivery note (recorded in the EXECUTION receipt): the parallel batch
runtime, capability registry entries, and engine-side consumption of
``ExecutionContext`` are owned by later DAG tasks (BATCH_EXEC, NATIVE_CORE,
GPU_DEVICE, CACHE_GRAPH, CPU_LAYOUT).  Until those land, forced ``native``/
``gpu`` requests raise (nothing is qualified), ``auto`` resolves to the
trusted CPU reference with a recorded reason, and ``RunBatch(parallel=True)``
raises the typed explicit-mode error rather than silently running serial.
"""

from dataclasses import dataclass
from typing import Literal, Optional, Tuple

__all__ = [
    "Backend", "CacheMode", "CacheVerification",
    "CacheOptions", "ExecutionOptions",
    "CapabilityError", "BackendNotAvailableError", "ExecutionNotAdmittedError",
    "EffectiveExecution", "RowOutcome", "BatchReport",
    "admit_execution", "settings_serialization_fields",
]


Backend = Literal["auto", "reference", "native", "gpu"]
CacheMode = Literal["off", "memory", "disk"]
CacheVerification = Literal["never", "on_write", "on_read", "always"]

# The semantic profiles frozen by CONTRACT §1 are una_legacy, madina_legacy
# and corrected_v1 (corrected_vN versions forward-compatible).  Option
# construction validates structure only (non-empty string); the admission /
# kernel layer resolves a profile against profiles.json and fails closed on
# an unknown one — option construction never becomes a registry lookup.


class CapabilityError(RuntimeError):
    """Base class for capability/admission failures.

    Raised before scientific execution when a request cannot be honoured.
    Distinct from ValueError on purpose: callers can catch admission policy
    separately from input-validation errors.
    """


class BackendNotAvailableError(CapabilityError):
    """A forced backend request ('native'/'gpu') has no qualified capability.

    Per contract: explicit expert routes never silently fall back — the
    request must be relaxed (e.g. to 'auto' or 'reference') by the caller.
    """


class ExecutionNotAdmittedError(CapabilityError):
    """A requested execution mode is not admitted by this build.

    Staged-delivery example: ``RunBatch(parallel=True)`` before the parallel
    batch runtime exists — raising beats silently running serial (which would
    fake the flag), and beats dropping the request.
    """


def _validate_positive_int(name: str, value: int, minimum: int = 1) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise ValueError(
            f"{name} must be an integer >= {minimum}, got {value!r}")


def _validate_optional_positive_number(name: str, value, minimum: float = 0.0,
                                       integer: bool = False) -> None:
    if value is None:
        return
    if integer:
        if not isinstance(value, int) or isinstance(value, bool) or value <= minimum:
            raise ValueError(
                f"{name} must be a positive integer or None, got {value!r}")
        return
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= minimum:
        raise ValueError(
            f"{name} must be a number > {minimum} or None, got {value!r}")


def _validate_optional_name(name: str, value) -> None:
    if value is None:
        return
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"{name} must be a non-empty string or None, got {value!r}")


@dataclass(frozen=True, kw_only=True)
class CacheOptions:
    """Immutable cache policy (contract §6).

    ``verification`` must not be silently downgradable by expert options —
    numerical identity validation is always available at least 'on_read'.
    """

    mode: CacheMode = "off"
    directory: str = ".una-cache"
    max_memory_bytes: Optional[int] = None
    max_disk_bytes: Optional[int] = None
    verification: CacheVerification = "on_read"
    schema_version: int = 1

    def __post_init__(self):
        if self.mode not in ("off", "memory", "disk"):
            raise ValueError(
                f"cache.mode must be one of 'off'|'memory'|'disk', "
                f"got {self.mode!r}")
        if not isinstance(self.directory, str) or not self.directory.strip():
            raise ValueError(
                f"cache.directory must be a non-empty string, "
                f"got {self.directory!r}")
        _validate_optional_positive_number(
            "cache.max_memory_bytes", self.max_memory_bytes, integer=True)
        _validate_optional_positive_number(
            "cache.max_disk_bytes", self.max_disk_bytes, integer=True)
        if self.verification not in ("never", "on_write", "on_read", "always"):
            raise ValueError(
                f"cache.verification must be one of "
                f"'never'|'on_write'|'on_read'|'always', "
                f"got {self.verification!r}")
        _validate_positive_int("cache.schema_version", self.schema_version)


@dataclass(frozen=True, kw_only=True)
class ExecutionOptions:
    """Immutable, validated execution options (contract §6 — implement exactly).

    ``semantic_profile`` selects the numerical contract (una_legacy /
    madina_legacy / corrected_v1).  ``backend`` selects the route: 'reference'
    forces the trusted CPU implementation; 'native'/'gpu' are explicit expert
    routes that raise ``BackendNotAvailableError`` when unqualified; 'auto' may
    fall back to reference with a recorded reason.  Scheduling identity is
    separate from semantic identity.

    Runtime-only ownership references (device handles, cancellation tokens)
    never live here and are never serialized into Settings — see
    ``settings_serialization_fields`` for the one authoritative serialized
    representation of this object inside project files.
    """

    semantic_profile: str = "una_legacy"
    backend: Backend = "auto"
    device: Optional[str] = None
    cpu_budget: Optional[int] = None
    threads_per_worker: int = 1
    logical_reduction_plan: str = "canonical"  # CONTRACT §3 frozen partition
    memory_limit_bytes: Optional[int] = None
    workspace_limit_bytes: Optional[int] = None
    queue_depth: int = 64
    writer_concurrency: int = 1
    timeout_s: Optional[float] = None
    cache: CacheOptions = CacheOptions()
    checkpoint: Optional[str] = None

    def __post_init__(self):
        _validate_optional_name("semantic_profile", self.semantic_profile)
        if self.backend not in ("auto", "reference", "native", "gpu"):
            raise ValueError(
                f"backend must be one of 'auto'|'reference'|'native'|'gpu', "
                f"got {self.backend!r}")
        _validate_optional_name("device", self.device)
        _validate_optional_positive_number(
            "cpu_budget", self.cpu_budget, integer=True)
        _validate_positive_int("threads_per_worker", self.threads_per_worker)
        _validate_optional_name("logical_reduction_plan",
                                self.logical_reduction_plan)
        _validate_optional_positive_number(
            "memory_limit_bytes", self.memory_limit_bytes, integer=True)
        _validate_optional_positive_number(
            "workspace_limit_bytes", self.workspace_limit_bytes, integer=True)
        _validate_positive_int("queue_depth", self.queue_depth)
        _validate_positive_int("writer_concurrency", self.writer_concurrency)
        _validate_optional_positive_number("timeout_s", self.timeout_s)
        if not isinstance(self.cache, CacheOptions):
            raise ValueError(
                f"cache must be a CacheOptions instance, got "
                f"{type(self.cache).__name__}")
        _validate_optional_name("checkpoint", self.checkpoint)


# The ONE authoritative serialized representation of ExecutionOptions inside
# Settings/project files (contract §6: "execution.semantic_profile",
# "execution.backend", "execution.cache.mode", "execution.cache.directory" —
# runtime-only fields are not serialized).  Settings.ToDict emits exactly
# these dotted keys (always, defaults included: the serialized defaults are
# explicit), and Settings.ApplyRow/Load accept exactly these dotted keys.
settings_serialization_fields: Tuple[str, ...] = (
    "execution.semantic_profile",
    "execution.backend",
    "execution.cache.mode",
    "execution.cache.directory",
)


@dataclass(frozen=True, kw_only=True)
class EffectiveExecution:
    """What was actually admitted for a run: resolved route + reason.

    ``fallback_reason`` is None when the requested route was honoured
    verbatim; 'auto' downgrades record why (dossier 10: reason and effective
    route are always recorded).
    """

    backend: Backend
    fallback_reason: Optional[str]
    options: ExecutionOptions


def admit_execution(options: ExecutionOptions) -> EffectiveExecution:
    """Resolve and admit an :ExecutionOptions: BEFORE scientific execution.

    Contract / dossier-10 admission policy at this stage of the campaign:

    - ``backend='reference'``  -> honoured (trusted CPU route).
    - ``backend='native'|'gpu'`` -> ``BackendNotAvailableError`` — no
      capability registry entries exist yet; forced expert routes never fall
      back.
    - ``backend='auto'``       -> trusted reference with a recorded reason
      (absent/stale/malformed capability entries choose reference under auto).

    When NATIVE_CORE/GPU_DEVICE register qualified capabilities, the registry
    consult happens here and stays behind this exact signature.
    """
    if options.backend == "reference":
        return EffectiveExecution(backend="reference", fallback_reason=None,
                                  options=options)
    if options.backend in ("native", "gpu"):
        raise BackendNotAvailableError(
            f"backend={options.backend!r} was explicitly requested but no "
            f"qualified capability is registered for this build "
            f"(semantic_profile={options.semantic_profile!r}); expert routes "
            f"never fall back silently — use backend='auto' or 'reference'.")
    # auto
    return EffectiveExecution(
        backend="reference",
        fallback_reason=(
            "no qualified native/gpu capability entries are registered for "
            "this build; auto selected the trusted reference route"),
        options=options,
    )


@dataclass(frozen=True, kw_only=True)
class RowOutcome:
    """One batch row's committed outcome (serial compatibility route).

    BATCH_EXEC extends batch reporting (pools, per-row cache engagement,
    peak resources, manifest, checkpoint generation); these are the fields
    the serial route can state truthfully today.
    """

    index: int
    name: str
    status: Literal["ran", "skipped", "failed"]
    semantic_profile: str
    backend: str
    detail: Optional[str] = None


@dataclass(frozen=True, kw_only=True)
class BatchReport:
    """Diagnostics for one RunBatch call (read ``UNA.batch_report``).

    ``UNA.RunBatch`` keeps returning None; everything a caller needs to
    inspect lives here.  The serial route reports the requested and effective
    execution plus one :RowOutcome: per project row.  BATCH_EXEC extends this
    type when real parallelism lands (pool structure, fallback reasons per
    row, cache engagement, resource peaks) — the existing field names are
    part of the staged ABI and will not be renamed.
    """

    requested: ExecutionOptions
    effective: EffectiveExecution
    parallel_requested: bool
    workers_requested: Optional[int]
    rows: Tuple[RowOutcome, ...] = ()
    notes: Tuple[str, ...] = ()
