"""Routing policy for the NATIVE_CORE scope-access backend (dossier 08).

``route_scope_access`` is the integrated accessibility route with the
native core as the search+adjust executor for BOTH semantic profiles:

- ``una_legacy``: admission mirrors (``kernels.scope_access``
  ``scope_route_admits`` / ``tail_admits`` — the engine guard's pinned
  mirrors) decide the route ladder; the core executes the characterized
  two-phase schedule on the A3/A1 routes and the LEGACY fastmath fold
  stays on ``reductions.fold_reach_gravity_knn_legacy_plan`` (the
  recorded admission: the fold's bits include LLVM fastmath
  reassociation no declared C++ sequence reproduces).
- ``corrected_v1``: the reference validators (``reference.graph``) ARE
  the typed admission contract; the core executes the corrected
  schedule and the corrected numpy fold stays on
  ``reductions.fold_reach_gravity_knn_plan`` (platform-libm
  transcendentals retained per NUMERICS.md).

Layer law (dossier 08): THIS module owns POLICY — admission decisions,
layout/dtype canonicality, fallback to the pinned portable routes on
refusal.  The C++ core owns ENFORCEMENT — it re-checks everything it
reads and reports any disagreement as a loud status/error, never a
silent fallback mid-kernel.  A nonzero core status on an input this
module admitted is therefore a bug surfaced as
:class:`NativeBackendMismatch`, never papered over.

Fallback honesty: every delegation records a machine-readable
``fallback_reason`` in :class:`NativeRunResult`.  Engagement counters
(SPEC.md section 6) are reported only when the core actually ran.
"""
from dataclasses import dataclass
from typing import Mapping, Optional

import numpy as np

from ..contracts import ArrayResult, KernelOutputs
from ...kernels.reductions import (
    adjust_destination_distances_plan,
    fold_reach_gravity_knn_legacy_plan,
    fold_reach_gravity_knn_plan,
)
from ...kernels.scope_access import (
    ROUTE_A1,
    ROUTE_A3,
    ROUTE_COMPACT,
    ScopeAccessResult,
    integrated_scope_access_oracle,
    scope_route_admits,
    tail_admits,
)
from ...reference.graph import validate_cutoff, validate_csr, validate_terminals
from ...reference.search import corrected_scope_labels
from . import _loader

__all__ = [
    "NativeBackendMismatch",
    "NativeRunResult",
    "route_scope_access",
]

_LEGACY = "una_legacy"
_CORRECTED = "corrected_v1"


class NativeBackendMismatch(RuntimeError):
    """The core reported a condition the routing policy had already
    excluded — an admission/core disagreement (SPEC.md section 7:
    internal mismatch fails loudly; never silently re-routed)."""


@dataclass(frozen=True)
class NativeRunResult:
    """Route outcome: typed outputs, attribution, engagement, honesty.

    ``outputs.status == 'incomplete'`` (cancellation) carries
    ``next_logical_state`` and NO arrays — nothing partial is published.
    ``result`` is populated only for complete runs.
    """

    outputs: KernelOutputs
    result: Optional[ScopeAccessResult]
    route: str
    native_engaged: bool
    fallback_reason: Optional[str]
    engagement: Mapping[str, object]


def _incomplete(first_origin: int) -> KernelOutputs:
    return KernelOutputs(arrays=(), status="incomplete",
                         next_logical_state=f"origin-{int(first_origin)}")


def _engagement(core, rep, kept_count):
    """Full engagement dict: the core counters plus the wrapper-side
    kept count (the fold's ``np.where(d_distance <= cutoff)`` membership,
    summed over origins; NaN compares False and is never kept).  Empty
    on the portable fallback (the oracle reports no core counters — the
    fallback_reason records why)."""
    if core is None:
        return {}
    eng = dict(rep)
    eng["kept_count"] = int(kept_count)
    eng["abi_version"] = int(core.abi_version())
    eng["binary_fingerprint"] = str(core.build_fingerprint())
    return eng


def route_scope_access(
    profile,
    o_terminal_idxs,
    o_terminal_weights,
    adjacency_pointer,
    adjacency_vector,
    adjacency_vector_weights,
    flag,
    d_terminal_idxs,
    d_terminal_weights,
    d_weights,
    gravity_beta,
    gravity_plateau,
    gravity_logistic_midpoint,
    gravity_growth_rate,
    knn_decay,
    knn_weights,
    cutoff,
    *,
    threads=1,
    cancel=None,
):
    """Integrated accessibility route (search + adjust + fold) under the
    frozen ladder, with the native core engaged when admitted.

    Signature mirrors ``kernels.scope_access.integrated_scope_access_oracle``
    (the pinned portable route this module falls back to), plus the
    backend knobs ``threads`` (core worker count; outputs are
    origin-partitioned, so the count cannot move a bit) and ``cancel``
    (a ``new_cancel_flag()`` handle; polled between origins).

    Refusal is routing, not error: an unengaged or unadmitted input
    runs the exact pinned portable route (engine-mirror oracle for
    ``una_legacy``, reference composition for ``corrected_v1``) and the
    reason is recorded.  Cancellation on the fallback path is honored
    as a pre-check only (the portable routes have no cancellation hook).
    """
    if profile == _LEGACY:
        return _route_legacy(
            o_terminal_idxs, o_terminal_weights, adjacency_pointer,
            adjacency_vector, adjacency_vector_weights, flag,
            d_terminal_idxs, d_terminal_weights, d_weights,
            gravity_beta, gravity_plateau, gravity_logistic_midpoint,
            gravity_growth_rate, knn_decay, knn_weights, cutoff,
            threads=threads, cancel=cancel)
    if profile == _CORRECTED:
        return _route_corrected(
            o_terminal_idxs, o_terminal_weights, adjacency_pointer,
            adjacency_vector, adjacency_vector_weights, flag,
            d_terminal_idxs, d_terminal_weights, d_weights,
            gravity_beta, gravity_plateau, gravity_logistic_midpoint,
            gravity_growth_rate, knn_decay, knn_weights, cutoff,
            threads=threads, cancel=cancel)
    raise ValueError(
        f"profile must be {_LEGACY!r} or {_CORRECTED!r}, got {profile!r}")


# ---------------------------------------------------------------------------
# una_legacy
# ---------------------------------------------------------------------------

def _layout_problems(*named_arrays):
    """Binding layout-law violations (contiguity, natural alignment) for
    the admitted legacy arrays — the guard checks dtype/shape only, the
    core reads raw pointers, so this is the policy mirror of the
    binding's argument enforcement."""
    problems = []
    for name, a in named_arrays:
        if not a.flags["C_CONTIGUOUS"]:
            problems.append(f"{name}: not C-contiguous")
        elif not a.flags["ALIGNED"]:
            problems.append(f"{name}: not naturally aligned")
    return problems


def _canonical(a):
    """Bit-preserving canonical buffer for the validated corrected
    arrays: same dtype, same values; a fresh (aligned) allocation only
    when the validated view is strided or unaligned."""
    if a.flags["C_CONTIGUOUS"] and a.flags["ALIGNED"]:
        return a
    return np.array(a, copy=True)


def _route_legacy(o_terminal_idxs, o_terminal_weights, adjacency_pointer,
                  adjacency_vector, adjacency_vector_weights, flag,
                  d_terminal_idxs, d_terminal_weights, d_weights,
                  gravity_beta, gravity_plateau, gravity_logistic_midpoint,
                  gravity_growth_rate, knn_decay, knn_weights, cutoff,
                  *, threads, cancel):
    # Route ladder via the pinned admission mirrors (the engine guard's
    # decision, reproduced in pure Python and machine-checked by
    # tests/cpu_kernels).  A guard refusal is a COMPACT route: the core
    # only implements the characterized A3/A1 two-phase domain, so the
    # compact route stays on the pinned portable path wholesale.
    a1_admitted, _max_degree = scope_route_admits(
        adjacency_pointer, adjacency_vector, adjacency_vector_weights,
        flag, o_terminal_idxs, o_terminal_weights, cutoff)

    node_count = adjacency_pointer.shape[0] - 1
    d_count = d_terminal_weights.shape[0]
    if a1_admitted and tail_admits(d_terminal_idxs, d_count, node_count):
        route, read_bound = ROUTE_A3, node_count
    elif a1_admitted:
        # A1 route: the guard admitted but some destination terminal is
        # outside [0, V); the engine/oracle read the (V + d_count) label
        # domain with Python/index semantics there.
        route, read_bound = ROUTE_A1, node_count + d_count
    else:
        route, read_bound = ROUTE_COMPACT, 0

    core = None
    fallback_reason = None
    try:
        core = _loader.load_core()
    except _loader.NativeUnavailableError as exc:
        fallback_reason = f"native-unavailable: {exc}"

    _require_cancel_type(cancel, core)

    if route == ROUTE_COMPACT:
        return _legacy_fallback(
            "route-compact: admission mirror refused the two-phase domain",
            o_terminal_idxs, o_terminal_weights, adjacency_pointer,
            adjacency_vector, adjacency_vector_weights, flag,
            d_terminal_idxs, d_terminal_weights, d_weights,
            gravity_beta, gravity_plateau, gravity_logistic_midpoint,
            gravity_growth_rate, knn_decay, knn_weights, cutoff,
            cancel=cancel, route_tag=route)

    if core is None:
        return _legacy_fallback(
            fallback_reason, o_terminal_idxs, o_terminal_weights,
            adjacency_pointer, adjacency_vector, adjacency_vector_weights,
            flag, d_terminal_idxs, d_terminal_weights, d_weights,
            gravity_beta, gravity_plateau, gravity_logistic_midpoint,
            gravity_growth_rate, knn_decay, knn_weights, cutoff,
            cancel=cancel, route_tag=route)

    # Binding layout law (policy mirror): the guard checks dtype/shape
    # but not contiguity/alignment; the core reads raw pointers, so an
    # admitted-but-strided (or unaligned) array delegates instead of
    # copying (bit-preserving copies would also be honest; delegation is
    # the simpler single-source choice).
    problems = _layout_problems(
        ("adjacency_pointer", adjacency_pointer),
        ("adjacency_vector", adjacency_vector),
        ("adjacency_vector_weights", adjacency_vector_weights),
        ("adjacynct_vector_network_node", flag),
        ("o_terminal_idxs", o_terminal_idxs),
        ("o_terminal_weights", o_terminal_weights),
        ("d_terminal_idxs", d_terminal_idxs),
        ("d_terminal_weights", d_terminal_weights))
    if problems:
        return _legacy_fallback(
            f"layout: {problems}",
            o_terminal_idxs, o_terminal_weights, adjacency_pointer,
            adjacency_vector, adjacency_vector_weights, flag,
            d_terminal_idxs, d_terminal_weights, d_weights,
            gravity_beta, gravity_plateau, gravity_logistic_midpoint,
            gravity_growth_rate, knn_decay, knn_weights, cutoff,
            cancel=cancel, route_tag=route)

    # Destination read domain (policy): A3 reads [0, V) (tail_admits
    # guaranteed it); A1 reads [0, V + d_count) — the engine's label
    # domain, where negative terminals rely on Python negative-index
    # wrap semantics the pointer-reading core cannot reproduce.  Beyond
    # that, delegate to the oracle (whose IndexError/behavior IS the
    # existing pinned portable behavior on those inputs).
    if d_terminal_idxs.ndim != 2 or d_terminal_idxs.shape != (d_count, 2) \
            or d_terminal_weights.shape != (d_count, 2) \
            or d_terminal_weights.dtype != np.float64:
        return _legacy_fallback(
            "layout: destination terminal arrays off the admitted shape/dtype",
            o_terminal_idxs, o_terminal_weights, adjacency_pointer,
            adjacency_vector, adjacency_vector_weights, flag,
            d_terminal_idxs, d_terminal_weights, d_weights,
            gravity_beta, gravity_plateau, gravity_logistic_midpoint,
            gravity_growth_rate, knn_decay, knn_weights, cutoff,
            cancel=cancel, route_tag=route)
    if d_count and (int(d_terminal_idxs.min()) < 0
                    or int(d_terminal_idxs.max()) >= read_bound):
        return _legacy_fallback(
            f"a1-read-domain: destination terminal outside [0, {read_bound})",
            o_terminal_idxs, o_terminal_weights, adjacency_pointer,
            adjacency_vector, adjacency_vector_weights, flag,
            d_terminal_idxs, d_terminal_weights, d_weights,
            gravity_beta, gravity_plateau, gravity_logistic_midpoint,
            gravity_growth_rate, knn_decay, knn_weights, cutoff,
            cancel=cancel, route_tag=route)
    if flag.shape[0] != adjacency_vector.shape[0]:
        # The guard omits this condition; the portable route's behavior
        # on it (IndexError from the adjust scan) is the pinned one —
        # never engage the pointer-reading core on a mismatched flag.
        return _legacy_fallback(
            "layout: adjacynct_vector_network_node length != E",
            o_terminal_idxs, o_terminal_weights, adjacency_pointer,
            adjacency_vector, adjacency_vector_weights, flag,
            d_terminal_idxs, d_terminal_weights, d_weights,
            gravity_beta, gravity_plateau, gravity_logistic_midpoint,
            gravity_growth_rate, knn_decay, knn_weights, cutoff,
            cancel=cancel, route_tag=route)

    o_count = o_terminal_idxs.shape[0]
    out = np.empty((o_count, d_count), dtype=np.float64)
    # `cutoff` is the ONE charged conversion (int -> float64), the same
    # widening the compiled kernel's float64 parameter performs.
    rep = core.run_legacy(
        "a3" if route == ROUTE_A3 else "a1",
        adjacency_pointer, adjacency_vector, adjacency_vector_weights,
        flag, o_terminal_idxs, o_terminal_weights,
        d_terminal_idxs, d_terminal_weights,
        cutoff, int(threads), cancel, out)

    codes = core.status_codes()
    status = int(rep["status"])
    if status == int(codes["ERR_CANCELLED"]):
        return NativeRunResult(
            outputs=_incomplete(rep["first_incomplete_origin"]),
            result=None, route=route, native_engaged=True,
            fallback_reason=None, engagement=_engagement(core, rep, 0))
    if status != int(codes["OK"]):
        raise NativeBackendMismatch(
            f"core refused an input the admission mirrors accepted "
            f"(status={status} reason={rep['reason']!r}) — admission/core "
            "disagreement (SPEC.md section 7)")

    return _legacy_compose(
        out, route, o_terminal_idxs, adjacency_vector_weights, d_weights,
        cutoff, gravity_beta, gravity_plateau, gravity_logistic_midpoint,
        gravity_growth_rate, knn_decay, knn_weights, core, rep)


def _legacy_compose(adjusted, route, o_terminal_idxs, adjacency_vector_weights,
                    d_weights, cutoff, gravity_beta, gravity_plateau,
                    gravity_logistic_midpoint, gravity_growth_rate, knn_decay,
                    knn_weights, core, rep):
    """Fold + publication: the LEGACY fold plan per origin on the native
    adjust rows (the recorded admission keeps the fold on the numba
    transcription — identical plan calls, identical inputs, identical
    bits as the oracle's composition)."""
    o_count = adjusted.shape[0]
    reach = np.empty(o_count, dtype=o_terminal_idxs.dtype)
    gravity_exponential = np.empty(o_count,
                                   dtype=adjacency_vector_weights.dtype)
    gravity_logistic = np.empty(o_count,
                                dtype=adjacency_vector_weights.dtype)
    knn_access = np.empty(o_count, dtype=adjacency_vector_weights.dtype)
    kept = 0
    for o in range(o_count):
        d_distance = adjusted[o]
        kept += int(np.count_nonzero(d_distance <= cutoff))
        r, ge, gl, ka = fold_reach_gravity_knn_legacy_plan(
            d_distance, d_weights, cutoff, gravity_beta, gravity_plateau,
            gravity_logistic_midpoint, gravity_growth_rate, knn_decay,
            knn_weights)
        reach[o] = r
        gravity_exponential[o] = ge
        gravity_logistic[o] = gl
        knn_access[o] = ka

    result = ScopeAccessResult(reach=reach,
                               gravity_exponential=gravity_exponential,
                               gravity_logistic=gravity_logistic,
                               knn_access=knn_access,
                               route=route)
    outputs = KernelOutputs(
        arrays=(
            ArrayResult(name="reach", array=reach),
            ArrayResult(name="gravity_exponential", array=gravity_exponential),
            ArrayResult(name="gravity_logistic", array=gravity_logistic),
            ArrayResult(name="knn_access", array=knn_access),
        ),
        status="complete",
    )
    return NativeRunResult(outputs=outputs, result=result, route=route,
                           native_engaged=True, fallback_reason=None,
                           engagement=_engagement(core, rep, kept))


def _legacy_fallback(reason, o_terminal_idxs, o_terminal_weights,
                     adjacency_pointer, adjacency_vector,
                     adjacency_vector_weights, flag, d_terminal_idxs,
                     d_terminal_weights, d_weights, gravity_beta,
                     gravity_plateau, gravity_logistic_midpoint,
                     gravity_growth_rate, knn_decay, knn_weights, cutoff,
                     *, cancel, route_tag=ROUTE_COMPACT):
    """Portable route: the pinned oracle (raises its own typed refusal
    for out-of-domain dtypes exactly as today).  ``route_tag`` labels a
    pre-cancelled fallback (the oracle never ran, so only the ladder's
    decision is known); a completed fallback carries the ORACLE's own
    route attribution."""
    if cancel is not None and cancel.is_cancelled():
        return NativeRunResult(outputs=_incomplete(0), result=None,
                               route=route_tag, native_engaged=False,
                               fallback_reason=reason, engagement={})
    result = integrated_scope_access_oracle(
        o_terminal_idxs, o_terminal_weights, adjacency_pointer,
        adjacency_vector, adjacency_vector_weights, flag,
        d_terminal_idxs, d_terminal_weights, d_weights,
        gravity_beta, gravity_plateau, gravity_logistic_midpoint,
        gravity_growth_rate, knn_decay, knn_weights, cutoff)
    outputs = KernelOutputs(
        arrays=(
            ArrayResult(name="reach", array=result.reach),
            ArrayResult(name="gravity_exponential",
                        array=result.gravity_exponential),
            ArrayResult(name="gravity_logistic",
                        array=result.gravity_logistic),
            ArrayResult(name="knn_access", array=result.knn_access),
        ),
        status="complete",
    )
    return NativeRunResult(outputs=outputs, result=result, route=result.route,
                           native_engaged=False, fallback_reason=reason,
                           engagement={})


def _require_cancel_type(cancel, core):
    """The cancel handle must be this binding's CancelFlag (or None) —
    no duck-typed substitute crosses into the released-GIL region.  When
    the core is unavailable the handle never crosses one, so only the
    fallback's duck-typed ``is_cancelled`` pre-check consumes it."""
    if cancel is None or core is None:
        return
    if not isinstance(cancel, core.CancelFlag):
        raise TypeError(
            "cancel must be a native new_cancel_flag() handle or None, "
            f"got {type(cancel)!r}")


# ---------------------------------------------------------------------------
# corrected_v1
# ---------------------------------------------------------------------------

def _route_corrected(o_terminal_idxs, o_terminal_weights, adjacency_pointer,
                     adjacency_vector, adjacency_vector_weights, flag,
                     d_terminal_idxs, d_terminal_weights, d_weights,
                     gravity_beta, gravity_plateau, gravity_logistic_midpoint,
                     gravity_growth_rate, knn_decay, knn_weights, cutoff,
                     *, threads, cancel):
    # ``flag`` is accepted for call-site parity with the legacy route; the
    # corrected profile has no push-gate column (reference/search.py) and
    # never reads it.
    core = None
    fallback_reason = None
    try:
        core = _loader.load_core()
    except _loader.NativeUnavailableError as exc:
        fallback_reason = f"native-unavailable: {exc}"

    _require_cancel_type(cancel, core)
    if cancel is not None and cancel.is_cancelled() and core is None:
        return NativeRunResult(outputs=_incomplete(0), result=None,
                               route="corrected_reference", native_engaged=False,
                               fallback_reason=fallback_reason, engagement={})

    # Typed admission contract: the reference validators, in the
    # reference order, raise the reference errors — the route's refusal
    # surface IS the corrected contract (the core re-enforces the same
    # scan; disagreement is a loud mismatch, never a fallback).
    node_count, pointer, vector, weights = validate_csr(
        adjacency_pointer, adjacency_vector, adjacency_vector_weights)
    cutoff_f = validate_cutoff(cutoff)
    o_idxs, o_wts = validate_terminals(
        o_terminal_idxs, o_terminal_weights, node_count, what="origin")
    d_idxs, d_wts = validate_terminals(
        d_terminal_idxs, d_terminal_weights, node_count,
        what="destination")

    if core is None:
        return _corrected_fallback(
            fallback_reason, o_idxs, o_wts, pointer, vector, weights,
            cutoff_f, d_idxs, d_wts, d_weights, gravity_beta, gravity_plateau,
            gravity_logistic_midpoint, gravity_growth_rate, knn_decay,
            knn_weights)

    # Layout canonicality for the pointer-reading core: dtype and domain
    # are now validated; a strided or unaligned validated view is
    # canonicalized with a bit-preserving same-dtype copy (values
    # untouched, dtype equality already enforced by the validators).
    pointer = _canonical(pointer)
    vector = _canonical(vector)
    weights = _canonical(weights)
    o_idxs = _canonical(o_idxs)
    o_wts = _canonical(o_wts)
    d_idxs = _canonical(d_idxs)
    d_wts = _canonical(d_wts)

    o_count = o_idxs.shape[0]
    d_count = d_idxs.shape[0]
    out = np.empty((o_count, d_count), dtype=np.float64)
    rep = core.run_corrected(
        pointer, vector, weights, o_idxs, o_wts, d_idxs, d_wts,
        cutoff_f, int(threads), cancel, out)

    codes = core.status_codes()
    status = int(rep["status"])
    if status == int(codes["ERR_CANCELLED"]):
        return NativeRunResult(
            outputs=_incomplete(rep["first_incomplete_origin"]),
            result=None, route="corrected_native", native_engaged=True,
            fallback_reason=None, engagement=_engagement(core, rep, 0))
    if status in (int(codes["ERR_MEMORY"]),):
        raise MemoryError(
            f"native core allocation failure: {rep['reason']!r}")
    if status in (int(codes["ERR_OVERFLOW"]),):
        raise OverflowError(
            f"native core size arithmetic overflow: {rep['reason']!r}")
    if status != int(codes["OK"]):
        raise NativeBackendMismatch(
            f"core refused an input the reference validators admitted "
            f"(status={status} reason={rep['reason']!r}) — admission/core "
            "disagreement (SPEC.md section 7)")

    return _corrected_compose(
        out, o_idxs, d_weights, cutoff_f, gravity_beta, gravity_plateau,
        gravity_logistic_midpoint, gravity_growth_rate, knn_decay,
        knn_weights, core, rep, engaged=True)


def _corrected_compose(adjusted, o_idxs, d_weights, cutoff, gravity_beta,
                       gravity_plateau, gravity_logistic_midpoint,
                       gravity_growth_rate, knn_decay, knn_weights,
                       core, rep, engaged, fallback_reason=None):
    """Corrected fold per origin on the (native or reference) adjust
    rows (``reductions.fold_reach_gravity_knn_plan`` — the numpy plan
    whose transcendentals stay the platform libm via numpy, per
    NUMERICS).  Attribution honesty: ``engaged`` decides the route tag
    AND the engagement claim — a reference-composed result is never
    labeled native."""
    route = "corrected_native" if engaged else "corrected_reference"
    o_count = adjusted.shape[0]
    reach = np.empty(o_count, dtype=o_idxs.dtype)  # validated int64
    gravity_exponential = np.empty(o_count, dtype=np.float64)
    gravity_logistic = np.empty(o_count, dtype=np.float64)
    knn_access = np.empty(o_count, dtype=np.float64)
    kept = 0
    for o in range(o_count):
        d_distance = adjusted[o]
        kept += int(np.count_nonzero(d_distance <= cutoff))
        r, ge, gl, ka = fold_reach_gravity_knn_plan(
            d_distance, d_weights, cutoff, gravity_beta, gravity_plateau,
            gravity_logistic_midpoint, gravity_growth_rate, knn_decay,
            knn_weights)
        reach[o] = r
        gravity_exponential[o] = ge
        gravity_logistic[o] = gl
        knn_access[o] = ka

    result = ScopeAccessResult(reach=reach,
                               gravity_exponential=gravity_exponential,
                               gravity_logistic=gravity_logistic,
                               knn_access=knn_access,
                               route=route)
    outputs = KernelOutputs(
        arrays=(
            ArrayResult(name="reach", array=reach),
            ArrayResult(name="gravity_exponential", array=gravity_exponential),
            ArrayResult(name="gravity_logistic", array=gravity_logistic),
            ArrayResult(name="knn_access", array=knn_access),
        ),
        status="complete",
    )
    return NativeRunResult(outputs=outputs, result=result,
                           route=route, native_engaged=engaged,
                           fallback_reason=fallback_reason,
                           engagement=_engagement(core, rep, kept))


def _corrected_fallback(reason, o_idxs, o_wts, pointer, vector, weights,
                        cutoff, d_idxs, d_wts, d_weights, gravity_beta,
                        gravity_plateau, gravity_logistic_midpoint,
                        gravity_growth_rate, knn_decay, knn_weights):
    """Portable route: the reference composition (corrected search ->
    builtin-min adjust -> corrected fold plan), the exact pieces the
    native path is pinned bitwise against."""
    o_count = o_idxs.shape[0]
    out = np.empty((o_count, d_idxs.shape[0]), dtype=np.float64)
    for i in range(o_count):
        label = corrected_scope_labels(
            o_idxs[i:i + 1], o_wts[i:i + 1], pointer, vector, weights,
            cutoff)
        out[i, :] = adjust_destination_distances_plan(label, d_idxs, d_wts)
    return _corrected_compose(out, o_idxs, d_weights, cutoff, gravity_beta,
                              gravity_plateau, gravity_logistic_midpoint,
                              gravity_growth_rate, knn_decay, knn_weights,
                              core=None, rep={"status": 0}, engaged=False,
                              fallback_reason=reason)
