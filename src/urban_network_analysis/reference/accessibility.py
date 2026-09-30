"""Corrected_v1 reach / gravity / KNN accessibility reference (SCIENCE).

Structure mirrors the legacy kernel's intended model (same formulas, so
the delta against legacy is exactly the registered defects, not a
different model), with corrected_v1 arithmetic:

- reach accumulates in binary64 and returns float64
  (BUG-FRAC-WEIGHT-TRUNC corrected rule: the weighted reach is a real
  quantity; int64 truncation is the legacy behavior, retained there);
- every accumulation is an ordered left-to-right binary64 sum in
  destination order -- not ``np.sum``'s pairwise reduction (declared
  corrected order; legacy pairwise remains legacy);
- radius membership is ``distance <= cutoff`` (closed, as legacy);
- plateau clamp is declared as ``t if t > 0.0 else 0.0`` so a -0.0
  difference clamps to +0.0;
- KNN sorts with a stable sort (ties broken by ascending destination
  index, declared), then applies coefficients/weights/decay in the
  same association order as the legacy formulas;
- ONE declared decay-parameter set (beta/plateau/midpoint/growth)
  drives both the gravity and the KNN decays, matching the integrated
  driver's wiring (``knn_gravity_beta = gravity_beta`` etc.).  The
  legacy kernel declares knn_gravity_* parameters its body never reads
  (BUG-TEA-KNN-DEAD-PARAMS); corrected_v1 removes the dead surface
  instead of carrying parameters that silently do nothing;
- transcendentals are platform libm via numpy on binary64 (retained per
  NUMERICS; any replacement re-versions the profile).

Empty in-scope sets produce exactly 0.0 for every metric.
"""
from __future__ import annotations

import numpy as np

from .errors import ParameterDomainError
from .graph import validate_cutoff


def _ordered_sum(values) -> float:
    """Left-to-right binary64 accumulation (declared corrected order)."""
    acc = 0.0
    for v in values:
        acc = acc + float(v)     # one binary64 RN add per step
    return acc


def _plateau_clamp(t: float) -> float:
    """Declared clamp: keep t only when strictly positive; -0.0 -> +0.0."""
    return t if t > 0.0 else 0.0


def corrected_reach_gravity_knn(
    d_distance,
    d_weights,
    cutoff,
    gravity_beta,
    gravity_plateau,
    gravity_logistic_midpoint,
    gravity_growth_rate,
    knn_weights,
    knn_decay,
):
    """Corrected per-origin accessibility metrics (see module docstring).

    Parameters follow the legacy kernel's model: ``d_distance`` are the
    (already position-adjusted) destination distances, ``d_weights``
    the destination weights; decay selects the KNN decay family
    ("exponential", "logistic", or anything else = undecay).
    Returns (reach, gravity_exponential, gravity_logistic, knn_access)
    as Python binary64 floats.
    """
    cutoff = validate_cutoff(cutoff)
    distance = np.asarray(d_distance, dtype=np.float64)
    weights = np.asarray(d_weights, dtype=np.float64)
    if distance.shape != weights.shape:
        raise ValueError(
            f"d_distance shape {distance.shape} != d_weights shape "
            f"{weights.shape}")
    knn_coefficients = np.asarray(knn_weights, dtype=np.float64)

    beta = float(gravity_beta)
    plateau = float(gravity_plateau)
    midpoint = float(gravity_logistic_midpoint)
    growth = float(gravity_growth_rate)

    # Declared parameter domains (BUG-TEA-NEGATIVE-DECAY-OVERFLOW
    # corrected rule): decay coefficients are finite and nonnegative --
    # a negative beta/growth makes exp() overflow to inf and silently
    # poisons the accessibility surface; corrected_v1 refuses instead.
    # Plateau/midpoint are finite offsets of either sign.
    for name, value, nonneg in (
        ("gravity_beta", beta, True),
        ("gravity_plateau", plateau, False),
        ("gravity_logistic_midpoint", midpoint, False),
        ("gravity_growth_rate", growth, True),
    ):
        if value != value or value in (float("inf"), float("-inf")):
            raise ParameterDomainError(f"{name} is not finite: {value!r}")
        if nonneg and value < 0.0:
            raise ParameterDomainError(
                f"{name}={value!r} < 0 (decay coefficients are nonnegative)")

    in_scope = np.where(distance <= cutoff)[0]   # closed radius
    sel_distance = distance[in_scope]
    sel_weights = weights[in_scope]

    # reach: ordered binary64 sum (float64 output -- corrected rule).
    reach = _ordered_sum(sel_weights)

    # gravity (exponential decay family), per destination in index order:
    # term = w * exp(-beta * clamp(d - plateau))
    acc = 0.0
    for j in range(sel_distance.shape[0]):
        t = _plateau_clamp(float(sel_distance[j]) - plateau)
        decay = float(np.exp(-beta * t))
        acc = acc + float(sel_weights[j]) * decay
    gravity_exponential = acc

    # gravity (logistic decay family):
    # u = (d - plateau) - midpoint; term = w * (1 - 1 / (1 + exp(-g*u)))
    acc = 0.0
    for j in range(sel_distance.shape[0]):
        u = (float(sel_distance[j]) - plateau) - midpoint
        decay = 1.0 - 1.0 / (1.0 + float(np.exp(-growth * u)))
        acc = acc + float(sel_weights[j]) * decay
    gravity_logistic = acc

    # KNN: stable sort (ties by ascending destination index), k nearest.
    k = min(sel_distance.shape[0], knn_coefficients.shape[0])
    if k == 0:
        knn_access = 0.0
    else:
        order = np.argsort(sel_distance, kind="stable")
        d_sorted = sel_distance[order]
        w_sorted = sel_weights[order]
        c_sorted = knn_coefficients[:k]
        acc = 0.0
        for j in range(k):
            a = float(c_sorted[j]) * float(w_sorted[j])
            if knn_decay == "exponential":
                t = _plateau_clamp(float(d_sorted[j]) - plateau)
                b = float(np.exp(-beta * t))
                term = a * b
            elif knn_decay == "logistic":
                u = (float(d_sorted[j]) - plateau) - midpoint
                b = 1.0 - 1.0 / (1.0 + float(np.exp(-growth * u)))
                term = a * b
            else:
                term = a
            acc = acc + term
        knn_access = acc

    return reach, gravity_exponential, gravity_logistic, knn_access


__all__ = [
    "corrected_reach_gravity_knn",
    "_ordered_sum",
    "_plateau_clamp",
]
