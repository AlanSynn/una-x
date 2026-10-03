"""Improved CPU arm: compacted-producer scope access (CPU_ALGORITHMS).

Contract-conformant port of CPU_LAYOUT candidate C4 (preserved in
evidence/tasks/CPU_LAYOUT/cpul-20261002T2305Z/rejected_implementation/,
measured wall-neutral, kept because it is the natural AOT/GPU port
shape and strictly reduces allocation) composed over the frozen
artifacts: per origin the characterized search schedule runs unchanged
(`scope_access.scope_search_schedule`), then ONE ordered collect
recomputes the legacy ``min(start, end) <= cutoff`` predicate with the
same builtin-min bits (`reductions.collect_kept`) and feeds the
prologue-free LEGACY kept fold
(`reductions.fold_reach_gravity_knn_kept_legacy_plan`, njit fastmath —
see reductions.py for why the numpy fold cannot carry una_legacy
parity).

NOT a default route: BENCHMARKS.md requires a positive paired
improvement for a default change, and the CPU_LAYOUT paired-alternating
windows gave speedup medians 0.9895 (W3 4T) / 1.0029 (W3 1T) / 0.8512
(W1 4T) / ~1.1 (W1 1T) — wall-neutral.  This module's role is the
improved-CPU comparison arm of the freeze: bitwise-equal outputs,
strictly lower allocation, attribution recorded below.
"""
import numpy as np

from .reductions import (
    collect_kept,
    fold_reach_gravity_knn_kept_legacy_plan,
)
from .scope_access import (
    PLAN_SCOPE_SEARCH,
    ROUTE_A1,
    ROUTE_A3,
    ROUTE_COMPACT,
    ScopeAccessResult,
    integrated_scope_access_oracle,
    scope_route_admits,
    scope_search_schedule,
    tail_admits,
)

__all__ = [
    "IMPROVED_ARM_ID",
    "ATTRIBUTION",
    "ATTRIBUTION_EVIDENCE",
    "compacted_admits",
    "scope_access_compacted",
]

IMPROVED_ARM_ID = "scope_access_compacted_v1"

# CPU_LAYOUT paired-alternating A/B attribution (run cpul-20261002T2305Z,
# bench_json/paired_*.json): speedup medians, compacted vs legacy, same
# process alternating windows; outputs bitwise-equal in every window.
ATTRIBUTION = {
    "W3_medium_proxy_4t": {"speedup_median": 0.9895, "verdict": "wall-neutral"},
    "W3_medium_proxy_1t": {"speedup_median": 1.0029, "verdict": "wall-neutral"},
    "W1_small_genuine_4t": {"speedup_median": 0.8512,
                            "verdict": "noise-dominated, not a win"},
    "W1_small_genuine_1t": {"speedup_median": 1.1,
                            "verdict": "sub-gate, single-window"},
    "outputs": "bitwise-equal vs legacy on the 22-test battery",
    "allocation": "strictly less (no mask, no boolean gathers, no "
                  "d_distances store pass)",
    "gate": "BENCHMARKS.md paired-improvement gate: no positive paired "
            "improvement -> not a default route",
}
ATTRIBUTION_EVIDENCE = (
    "campaigns/una_platform/evidence/tasks/CPU_LAYOUT/cpul-20261002T2305Z/ "
    "(receipt.json candidates.C4; raw evidence in the cpu-layout worktree)")

# Scratch-cap pattern of the C route: per-thread 16*d_count + 16 bytes
# (two float64 destination buffers) under the 256 MiB cap.
_COMPACTED_SCRATCH_CAP_BYTES = 268435456


def compacted_admits(adjacency_pointer, adjacency_vector,
                     adjacency_vector_weights, flag,
                     o_terminal_idxs, o_terminal_weights,
                     d_terminal_idxs, d_terminal_weights,
                     d_weights, cutoff):
    """Pure-Python mirror of CPU_LAYOUT ``_c_route_admits``: the A1
    guard plus the A3 tail check plus the collect's shape requirements
    plus the compacted scratch cap.  Same decision in both of the C
    module's dispatch modes by construction (its scan body was shared
    verbatim; this mirror consumes the same guards)."""
    a1_admitted, _ = scope_route_admits(
        adjacency_pointer, adjacency_vector, adjacency_vector_weights,
        flag, o_terminal_idxs, o_terminal_weights, cutoff)
    if not a1_admitted:
        return False
    n_count = adjacency_pointer.shape[0] - 1
    d_count = d_terminal_weights.shape[0]
    if not tail_admits(d_terminal_idxs, d_count, n_count):
        return False
    # The collect indexes the terminal arrays by destination index.
    if d_terminal_idxs.shape[1] != 2:
        return False
    if d_terminal_weights.shape[1] != 2:
        return False
    if d_weights.shape[0] != d_count:
        return False
    # Scratch cap: threads * (16*d + 16) bytes.
    if d_count > _COMPACTED_SCRATCH_CAP_BYTES:
        return False
    from .scope_access import _default_threads_bound
    threads_bound = _default_threads_bound()
    if threads_bound < 1:
        return False
    per_thread = 16 * d_count + 16
    if per_thread * threads_bound > _COMPACTED_SCRATCH_CAP_BYTES:
        return False
    return True


def scope_access_compacted(o_terminal_idxs, o_terminal_weights,
                           adjacency_pointer, adjacency_vector,
                           adjacency_vector_weights, flag,
                           d_terminal_idxs, d_terminal_weights,
                           d_weights, gravity_beta, gravity_plateau,
                           gravity_logistic_midpoint, gravity_growth_rate,
                           knn_decay, knn_weights, cutoff):
    """The improved arm driver: ``IMPROVED_ARM_ID`` over the frozen
    schedule + collect + kept fold.  Refused domains delegate to the
    oracle's ladder (route tag reports which arm ran: 'compacted' or
    the delegated legacy route tag)."""
    if not compacted_admits(
            adjacency_pointer, adjacency_vector, adjacency_vector_weights,
            flag, o_terminal_idxs, o_terminal_weights,
            d_terminal_idxs, d_terminal_weights, d_weights, cutoff):
        result = integrated_scope_access_oracle(
            o_terminal_idxs, o_terminal_weights, adjacency_pointer,
            adjacency_vector, adjacency_vector_weights, flag,
            d_terminal_idxs, d_terminal_weights, d_weights,
            gravity_beta, gravity_plateau, gravity_logistic_midpoint,
            gravity_growth_rate, knn_decay, knn_weights, cutoff)
        return ScopeAccessResult(
            reach=result.reach,
            gravity_exponential=result.gravity_exponential,
            gravity_logistic=result.gravity_logistic,
            knn_access=result.knn_access,
            route="delegated:" + result.route)

    o_count = o_terminal_idxs.shape[0]
    d_count = d_terminal_weights.shape[0]

    reach = np.empty(o_count, dtype=o_terminal_idxs.dtype)
    gravity_exponential = np.empty(o_count,
                                   dtype=adjacency_vector_weights.dtype)
    gravity_logistic = np.empty(o_count,
                                dtype=adjacency_vector_weights.dtype)
    knn_access = np.empty(o_count, dtype=adjacency_vector_weights.dtype)

    for o in range(o_count):
        labels = scope_search_schedule(
            o_terminal_idxs[o], o_terminal_weights[o],
            adjacency_pointer, adjacency_vector,
            adjacency_vector_weights, flag, cutoff,
            d_count=0, domain="a3")
        distances, kept_weights, kept = collect_kept(
            labels, d_terminal_idxs, d_terminal_weights, d_weights, cutoff)
        # una_legacy parity needs the LEGACY compiled kept fold: the
        # plain-numpy fold rounds the logistic/exponential aggregates
        # differently than the engine's fastmath compilation (1-ulp
        # class, characterized in reductions.py) — the arm's contract
        # is bitwise-equality with the engine driver.
        r, ge, gl, ka = fold_reach_gravity_knn_kept_legacy_plan(
            distances[:kept], kept_weights[:kept], cutoff,
            gravity_beta, gravity_plateau, gravity_logistic_midpoint,
            gravity_growth_rate, knn_decay, knn_weights)
        reach[o] = r
        gravity_exponential[o] = ge
        gravity_logistic[o] = gl
        knn_access[o] = ka

    return ScopeAccessResult(reach=reach,
                             gravity_exponential=gravity_exponential,
                             gravity_logistic=gravity_logistic,
                             knn_access=knn_access,
                             route="compacted")
