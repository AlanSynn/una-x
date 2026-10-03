"""Canonical reduction plans for the scope-access kernel family (CPU_ALGORITHMS).

These are the executable freeze of the final-reduction arithmetic that
SPEC.md section 5 pins for the accessibility kernels.  Every expression
is a byte-level transcription of the audited engine kernel
(`Engines/AccessibilityWElevation.py::reach_gravity_knn_access` and
`::adjust_destination_distances` as of baseline 16bf404d).  The
transcription exists in TWO compiled-in-profile variants, because the
profiles genuinely diverge here:

- ``*_plan`` (plain numpy + Python scalars) — the corrected_v1
  binary64 semantics: no reassociation, numpy's own reduction order,
  IEEE ``<=`` membership.
- ``*_legacy_plan`` (njit, fastmath=True, same array-expression
  structure as the engine) — the una_legacy semantics.  Platform
  characterization (this task's evidence, logistic-ulp probe): the
  compiled engine's ``np.sum`` over a computed expression is NOT
  bit-identical to the numpy call in general — LLVM fastmath
  reassociates the vectorized elementwise product/sum (observed 1-ulp
  delta, logistic branch, n=4) while scalar-loop compilations DO match
  numpy.  The conformance tests pin ``*_legacy_plan`` against the
  compiled engines bitwise, so any numba/LLVM upgrade that shifts the
  legacy rounding fails loudly; the numpy plans are pinned only to
  themselves (corrected_v1) and are NOT claimed to reproduce legacy
  bits.

Plan identity: the plan-ID strings exported here are the values that
flow into ``backends.contracts.KernelInputs.logical_reduction_plan``.
A port (native/GPU) must implement the plan its invocation names —
never a substitute reduction (no np.dot, no pairwise reorder, no
fastmath contraction beyond the profile's own compilation contract).
"""
import numpy as np

__all__ = [
    "STAGE_SCOPE_SEARCH", "STAGE_ADJUST", "STAGE_FOLD",
    "PLAN_SCOPE_SEARCH", "PLAN_ADJUST", "PLAN_FOLD", "PLAN_FOLD_KEPT",
    "PLAN_FOLD_LEGACY", "PLAN_FOLD_KEPT_LEGACY",
    "PLANS",
    "adjust_destination_distances_plan",
    "collect_kept",
    "fold_reach_gravity_knn_plan",
    "fold_reach_gravity_knn_kept_plan",
    "fold_reach_gravity_knn_legacy_plan",
    "fold_reach_gravity_knn_kept_legacy_plan",
]

# Stage identities (KernelInputs.stage values for this kernel family).
STAGE_SCOPE_SEARCH = "accessibility.scope_search"
STAGE_ADJUST = "accessibility.adjust"
STAGE_FOLD = "accessibility.reach_gravity_knn"

# Reduction-plan identities (KernelInputs.logical_reduction_plan values).
PLAN_SCOPE_SEARCH = "plan.accessibility.scope_search.heap_two_phase.v1"
PLAN_ADJUST = "plan.accessibility.adjust.builtin_min.v1"
PLAN_FOLD = "plan.accessibility.fold.reach_gravity_knn.v1"
PLAN_FOLD_KEPT = "plan.accessibility.fold.reach_gravity_knn_kept.v1"

# Legacy-profile plan identities, composed from the canonical IDs so the
# frozen literal appears exactly once (typo-proof under render noise).
PLAN_LEGACY_SUFFIX = ".legacy_fastmath.v1"
PLAN_FOLD_LEGACY = PLAN_FOLD + PLAN_LEGACY_SUFFIX
PLAN_FOLD_KEPT_LEGACY = PLAN_FOLD_KEPT + PLAN_LEGACY_SUFFIX

# Characterized dead-parameter fact (SPEC.md section 5): the engine
# fold's signature carries knn_gravity_plateau / knn_gravity_beta /
# knn_gravity_logistic_midpoint / knn_gravity_growth_rate, but the
# audited body never reads them — the knn branches bind the gravity_*
# scalars.  The canonical plan carries only the live parameters; the
# engine-boundary binding is pinned by test_reductions.py.
DEAD_ENGINE_FOLD_PARAMETERS = (
    "knn_gravity_plateau",
    "knn_gravity_beta",
    "knn_gravity_logistic_midpoint",
    "knn_gravity_growth_rate",
)


def adjust_destination_distances_plan(scope_weights, d_terminal_idxs,
                                      d_terminal_weights):
    """plan.accessibility.adjust.builtin_min.v1 (SPEC.md section 4).

    ``dist[i] = min(label[t0] + w0, label[t1] + w1)`` — builtin min,
    operand order (start, end).  Builtin min returns the FIRST operand
    on equality, so ``min(+0.0, -0.0) == +0.0`` and ``min(-0.0, +0.0)
    == -0.0``; NaN follows Python min semantics (min(NaN, x) = NaN,
    min(x, NaN) = x).  These bit rules are part of the plan.
    """
    d_count = d_terminal_idxs.shape[0]
    d_distances = np.empty(d_count, dtype=scope_weights.dtype)
    for i in range(d_count):
        dist_to_start = scope_weights[d_terminal_idxs[i, 0]] + d_terminal_weights[i, 0]
        dist_to_end = scope_weights[d_terminal_idxs[i, 1]] + d_terminal_weights[i, 1]
        d_distances[i] = min(dist_to_start, dist_to_end)
    return d_distances


def collect_kept(scope_weights, d_terminal_idxs, d_terminal_weights,
                 d_weights, cutoff):
    """Ordered compacting producer (CPU_LAYOUT candidate C4, proof L4).

    One pass over destinations in ascending index order recomputing the
    legacy predicate ``min(start, end) <= cutoff`` with the same
    builtin-min bits, compacting ``(dist, d_weights[i])`` of kept
    destinations.  Returns ``(distances, weights, kept)`` where the
    first two arrays hold the kept prefix in ascending destination
    order — exactly the values, in the order, that
    ``fold_reach_gravity_knn_plan`` would gather.  Buffers are exact
    length ``kept`` (the AOT/GPU port compacts into fixed scratch; the
    contract is the values and their order, not the allocation).
    """
    d_count = d_terminal_idxs.shape[0]
    dists = []
    weights = []
    for i in range(d_count):
        dist_to_start = scope_weights[d_terminal_idxs[i, 0]] + d_terminal_weights[i, 0]
        dist_to_end = scope_weights[d_terminal_idxs[i, 1]] + d_terminal_weights[i, 1]
        dist = min(dist_to_start, dist_to_end)
        if dist <= cutoff:
            dists.append(dist)
            weights.append(d_weights[i])
    distances = np.array(dists, dtype=np.float64)
    kept_weights = np.array(weights, dtype=np.float64)
    return distances, kept_weights, distances.shape[0]


def fold_reach_gravity_knn_plan(d_distance, d_weights, cutoff,
                                gravity_beta, gravity_plateau,
                                gravity_logistic_midpoint,
                                gravity_growth_rate, knn_decay,
                                knn_weights):
    """plan.accessibility.fold.reach_gravity_knn.v1 (SPEC.md section 5).

    Byte-level transcription of the audited engine fold.  Membership
    ``np.where(d_distance <= cutoff)[0]`` (closed radius; NaN compares
    False and is excluded), boolean gathers in ascending destination
    order, ``.sum()`` reductions, default ``np.argsort`` (its tie
    permutation is observable and contractual), the
    exponential/logistic/other knn decay branches reading the gravity_*
    scalars, and ``knn = min(kept, len(knn_weights))`` with the 0.0
    empty result.
    """
    n_reach_filter = np.where(d_distance <= cutoff)[0]
    n_distances = d_distance[n_reach_filter]
    n_weights = d_weights[n_reach_filter]

    reach = n_weights.sum()
    gravity_exponential = (n_weights / np.exp(gravity_beta * np.maximum(0, n_distances - gravity_plateau))).sum()
    gravity_logistic = (n_weights * (1 - 1 / (1 + np.exp(-gravity_growth_rate * (n_distances - gravity_plateau - gravity_logistic_midpoint))))).sum()

    knn = min(n_distances.shape[0], knn_weights.shape[0])
    if knn == 0:
        knn_access = 0.0
    else:
        idx = np.argsort(n_distances)
        n_distances_sorted = n_distances[idx]
        n_weights_sorted = n_weights[idx]
        n_distances_cropped = n_distances_sorted[:knn]
        n_weights_cropped = n_weights_sorted[:knn]
        knn_coefficients_cropped = knn_weights[:knn]

        if knn_decay == "exponential":
            knn_access = ((knn_coefficients_cropped * n_weights_cropped) * np.exp(-gravity_beta * np.maximum(0, n_distances_cropped - gravity_plateau))).sum()
        elif knn_decay == "logistic":
            knn_access = ((knn_coefficients_cropped * n_weights_cropped) * (1 - 1 / (1 + np.exp(-gravity_growth_rate * (n_distances_cropped - gravity_plateau - gravity_logistic_midpoint))))).sum()
        else:
            knn_access = (knn_coefficients_cropped * n_weights_cropped).sum()

    return reach, gravity_exponential, gravity_logistic, knn_access


def fold_reach_gravity_knn_kept_plan(n_distances, n_weights, cutoff,
                                     gravity_beta, gravity_plateau,
                                     gravity_logistic_midpoint,
                                     gravity_growth_rate, knn_decay,
                                     knn_weights):
    """plan.accessibility.fold.reach_gravity_knn_kept.v1 (improved arm).

    The same arithmetic on an already-compacted prefix — identical to
    ``fold_reach_gravity_knn_plan`` minus the np.where/gather prologue.
    Bitwise-equal to the base plan whenever ``(n_distances, n_weights)``
    equal the base plan's post-gather arrays (pinned by
    test_reductions.py::test_fold_kept_matches_base_plan).
    """
    reach = n_weights.sum()
    gravity_exponential = (n_weights / np.exp(gravity_beta * np.maximum(0, n_distances - gravity_plateau))).sum()
    gravity_logistic = (n_weights * (1 - 1 / (1 + np.exp(-gravity_growth_rate * (n_distances - gravity_plateau - gravity_logistic_midpoint))))).sum()

    knn = min(n_distances.shape[0], knn_weights.shape[0])
    if knn == 0:
        knn_access = 0.0
    else:
        idx = np.argsort(n_distances)
        n_distances_sorted = n_distances[idx]
        n_weights_sorted = n_weights[idx]
        n_distances_cropped = n_distances_sorted[:knn]
        n_weights_cropped = n_weights_sorted[:knn]
        knn_coefficients_cropped = knn_weights[:knn]

        if knn_decay == "exponential":
            knn_access = ((knn_coefficients_cropped * n_weights_cropped) * np.exp(-gravity_beta * np.maximum(0, n_distances_cropped - gravity_plateau))).sum()
        elif knn_decay == "logistic":
            knn_access = ((knn_coefficients_cropped * n_weights_cropped) * (1 - 1 / (1 + np.exp(-gravity_growth_rate * (n_distances_cropped - gravity_plateau - gravity_logistic_midpoint))))).sum()
        else:
            knn_access = (knn_coefficients_cropped * n_weights_cropped).sum()

    return reach, gravity_exponential, gravity_logistic, knn_access


# Plan registry: plan-ID -> (executor, stage, spec section, profiles).
# The engine's own compilation contract (AccessibilityWElevation.py
# lines 57-61): parallel=False, cache=True, nogil=True, fastmath=True.
# fastmath is what produces the una_legacy rounding; parallel must stay
# False to keep the reduction single-stranded.
from numba import njit as _njit


@_njit(cache=True, nogil=True, fastmath=True)
def _fold_reach_gravity_knn_legacy_body(d_distance, d_weights, cutoff,
                                        gravity_beta, gravity_plateau,
                                        gravity_logistic_midpoint,
                                        gravity_growth_rate, knn_decay,
                                        knn_weights):
    """plan...reach_gravity_knn.v1.legacy_fastmath — the una_legacy
    transcription: byte-level copy of the compiled engine's fold
    (accessibility engine lines 75-110), compiled with the engine's own
    flags.  Reproduces the engine's bits INCLUDING the LLVM fastmath
    reassociation of the vectorized elementwise/sum pipelines that plain
    numpy does not perform (characterized 1-ulp class).  This function —
    not the numpy plan — is the una_legacy parity artifact and the
    portable porting source for NATIVE_CORE."""
    n_reach_filter = np.where(d_distance <= cutoff)[0]
    n_distances = d_distance[n_reach_filter]
    n_weights = d_weights[n_reach_filter]

    reach = n_weights.sum()
    gravity_exponential = (n_weights / np.exp(gravity_beta * np.maximum(0, n_distances - gravity_plateau))).sum()
    gravity_logistic = (n_weights * (1 - 1 / (1 + np.exp(-gravity_growth_rate * (n_distances - gravity_plateau - gravity_logistic_midpoint))))).sum()

    knn = min(n_distances.shape[0], knn_weights.shape[0])

    if knn == 0:
        knn_access = 0.0
    else:
        idx = np.argsort(n_distances)
        n_distances_sorted = n_distances[idx]
        n_weights_sorted = n_weights[idx]

        n_distances_cropped = n_distances_sorted[:knn]
        n_weights_cropped = n_weights_sorted[:knn]
        knn_coefficients_cropped = knn_weights[:knn]

        if knn_decay == "exponential":
            knn_access = ((knn_coefficients_cropped * n_weights_cropped) * np.exp(-gravity_beta * np.maximum(0, n_distances_cropped - gravity_plateau))).sum()
        elif knn_decay == "logistic":
            knn_access = ((knn_coefficients_cropped * n_weights_cropped) * (1 - 1 / (1 + np.exp(-gravity_growth_rate * (n_distances_cropped - gravity_plateau - gravity_logistic_midpoint))))).sum()
        else:
            knn_access = (knn_coefficients_cropped * n_weights_cropped).sum()

    return reach, gravity_exponential, gravity_logistic, knn_access


@_njit(cache=True, nogil=True, fastmath=True)
def _fold_reach_gravity_knn_kept_legacy_body(n_distances, n_weights, cutoff,
                                             gravity_beta, gravity_plateau,
                                             gravity_logistic_midpoint,
                                             gravity_growth_rate, knn_decay,
                                             knn_weights):
    """plan...reach_gravity_knn_kept.v1.legacy_fastmath — the kept-prefix
    legacy fold: the legacy body minus the membership prologue (the
    improved arm's collect feeds the already-gathered prefix).  Identity
    with the base legacy plan on equal post-gather inputs is pinned by
    the conformance suite."""
    reach = n_weights.sum()
    gravity_exponential = (n_weights / np.exp(gravity_beta * np.maximum(0, n_distances - gravity_plateau))).sum()
    gravity_logistic = (n_weights * (1 - 1 / (1 + np.exp(-gravity_growth_rate * (n_distances - gravity_plateau - gravity_logistic_midpoint))))).sum()

    knn = min(n_distances.shape[0], knn_weights.shape[0])

    if knn == 0:
        knn_access = 0.0
    else:
        idx = np.argsort(n_distances)
        n_distances_sorted = n_distances[idx]
        n_weights_sorted = n_weights[idx]

        n_distances_cropped = n_distances_sorted[:knn]
        n_weights_cropped = n_weights_sorted[:knn]
        knn_coefficients_cropped = knn_weights[:knn]

        if knn_decay == "exponential":
            knn_access = ((knn_coefficients_cropped * n_weights_cropped) * np.exp(-gravity_beta * np.maximum(0, n_distances_cropped - gravity_plateau))).sum()
        elif knn_decay == "logistic":
            knn_access = ((knn_coefficients_cropped * n_weights_cropped) * (1 - 1 / (1 + np.exp(-gravity_growth_rate * (n_distances_cropped - gravity_plateau - gravity_logistic_midpoint))))).sum()
        else:
            knn_access = (knn_coefficients_cropped * n_weights_cropped).sum()

    return reach, gravity_exponential, gravity_logistic, knn_access


def fold_reach_gravity_knn_legacy_plan(d_distance, d_weights, cutoff,
                                       gravity_beta, gravity_plateau,
                                       gravity_logistic_midpoint,
                                       gravity_growth_rate, knn_decay,
                                       knn_weights):
    """Pure-Python entry for the una_legacy fold plan (jit-compiles its
    body on first call with the engine's fastmath contract)."""
    return _fold_reach_gravity_knn_legacy_body(
        np.ascontiguousarray(d_distance, dtype=np.float64),
        np.ascontiguousarray(d_weights, dtype=np.float64),
        cutoff, gravity_beta, gravity_plateau, gravity_logistic_midpoint,
        gravity_growth_rate, knn_decay,
        np.ascontiguousarray(knn_weights, dtype=np.float64))


def fold_reach_gravity_knn_kept_legacy_plan(n_distances, n_weights, cutoff,
                                            gravity_beta, gravity_plateau,
                                            gravity_logistic_midpoint,
                                            gravity_growth_rate, knn_decay,
                                            knn_weights):
    """Pure-Python entry for the kept-prefix una_legacy fold plan."""
    return _fold_reach_gravity_knn_kept_legacy_body(
        np.ascontiguousarray(n_distances, dtype=np.float64),
        np.ascontiguousarray(n_weights, dtype=np.float64),
        cutoff, gravity_beta, gravity_plateau, gravity_logistic_midpoint,
        gravity_growth_rate, knn_decay,
        np.ascontiguousarray(knn_weights, dtype=np.float64))


PLANS = {
    PLAN_SCOPE_SEARCH: ("scope_search_schedule", STAGE_SCOPE_SEARCH,
                        "SPEC.md section 3", ("una_legacy",)),
    PLAN_ADJUST: ("adjust_destination_distances_plan", STAGE_ADJUST,
                  "SPEC.md section 4", ("una_legacy", "corrected_v1")),
    PLAN_FOLD: ("fold_reach_gravity_knn_plan", STAGE_FOLD,
                "SPEC.md section 5", ("corrected_v1",)),
    PLAN_FOLD_KEPT: ("fold_reach_gravity_knn_kept_plan", STAGE_FOLD,
                     "SPEC.md section 5 (compacted arm)", ("corrected_v1",)),
    PLAN_FOLD_LEGACY: ("fold_reach_gravity_knn_legacy_plan", STAGE_FOLD,
                       "SPEC.md section 5 (legacy fastmath transcription)",
                       ("una_legacy",)),
    PLAN_FOLD_KEPT_LEGACY: ("fold_reach_gravity_knn_kept_legacy_plan",
                            STAGE_FOLD,
                            "SPEC.md section 5 (kept legacy transcription)",
                            ("una_legacy",)),
}
