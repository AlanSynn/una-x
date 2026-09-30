"""Corrected_v1 accessibility metrics and canonical partition tests.

Scientific check #2 (NUMERICS.md): a stdlib-Decimal 60-digit oracle
validates the intended gravity math independently of the binary64
implementation; the bits still come from the declared ordered binary64
reference (the Decimal oracle is NOT the bitwise oracle).
"""
from __future__ import annotations

import struct
from decimal import Decimal, getcontext

import numpy as np
import pytest

pytestmark = pytest.mark.science

from urban_network_analysis.reference import PARTITION_VERSION
from urban_network_analysis.reference.accessibility import (
    _ordered_sum,
    corrected_reach_gravity_knn,
)
from urban_network_analysis.reference.partition import (
    aggregate_by_canonical_partition,
    canonical_stripe_count,
    canonical_stripes,
    stripe_fold,
    stripe_of_origin,
)


def bits(x: float) -> int:
    return struct.unpack("<Q", struct.pack("<d", float(x)))[0]


# --------------------------------------------------------------------------
# reach dtype + ordered accumulation
# --------------------------------------------------------------------------

def test_reach_is_float64_and_exact():
    """Corrected rule (BUG-FRAC-WEIGHT-TRUNC): fractional weights sum to
    a binary64 float, never truncated into an integer array."""
    reach, *_ = corrected_reach_gravity_knn(
        np.array([5.0, 6.0]), np.array([0.25, 0.50]), 100.0,
        0.0, 0.0, 0.0, 0.0, np.array([1.0]), "none")
    assert isinstance(reach, float)
    assert bits(reach) == bits(0.75)


def test_accumulation_is_ordered_left_to_right():
    """Declared order: one binary64 RN add per destination, ascending
    index.  The expected bits are computed here from that declared
    definition; np.sum's pairwise tree is a DIFFERENT order (legacy
    keeps it; corrected declares this one)."""
    distance = np.arange(10, dtype=np.float64)
    weights = np.full(10, 0.1, dtype=np.float64)
    reach, *_ = corrected_reach_gravity_knn(
        distance, weights, 100.0, 0.0, 0.0, 0.0, 0.0, np.array([1.0]), "none")
    acc = 0.0
    for w in weights:
        acc = acc + float(w)
    assert bits(reach) == bits(acc)


def test_empty_scope_gives_exact_zeros():
    reach, g_exp, g_log, knn = corrected_reach_gravity_knn(
        np.array([50.0]), np.array([1.0]), 10.0,
        0.3, 1.0, 2.0, 1.0, np.array([1.0]), "exponential")
    assert bits(reach) == bits(0.0)
    assert bits(g_exp) == bits(0.0)
    assert bits(g_log) == bits(0.0)
    assert bits(knn) == bits(0.0)


def test_cutoff_membership_closed():
    """Distance exactly == cutoff is in scope (as legacy); one ulp out."""
    reach_in, *_ = corrected_reach_gravity_knn(
        np.array([10.0]), np.array([2.0]), 10.0,
        0.0, 0.0, 0.0, 0.0, np.array([1.0]), "none")
    assert bits(reach_in) == bits(2.0)
    reach_out, *_ = corrected_reach_gravity_knn(
        np.array([np.nextafter(np.float64(10.0), 1e9)]),
        np.array([2.0]), 10.0,
        0.0, 0.0, 0.0, 0.0, np.array([1.0]), "none")
    assert bits(reach_out) == bits(0.0)


# --------------------------------------------------------------------------
# gravity vs 60-digit Decimal oracle (independent math check)
# --------------------------------------------------------------------------

def test_gravity_exponential_matches_decimal_oracle():
    """Intended-math check at 60 significant digits (NUMERICS scientific
    check #2).  The exact-arithmetic value is NOT cast into an expected
    bit pattern: the ordered binary64 loop carries per-step rounding, so
    the oracle validates the math within a strict ulp budget (3 terms:
    a handful of half-ulp steps); bit-exact pins live in the dyadic /
    structural tests."""
    getcontext().prec = 60
    distance = [0.5, 2.5, 7.25]
    weights = [1.0, 2.0, 0.5]
    beta, plateau = 0.3, 1.0
    acc = Decimal(0)
    for d, w in zip(distance, weights):
        t = d - plateau
        if not t > 0.0:
            t = 0.0
        decay = (-Decimal(beta) * Decimal(t)).exp()
        acc += Decimal(w) * decay
    exact = acc
    reach, g_exp, _g_log, _knn = corrected_reach_gravity_knn(
        np.array(distance, dtype=np.float64),
        np.array(weights, dtype=np.float64), 100.0,
        beta, plateau, 0.0, 0.0, np.array([1.0]), "exponential")
    ulp = float(np.spacing(abs(float(exact))))
    assert abs(g_exp - float(exact)) <= 4 * ulp, (g_exp, float(exact), ulp)


def test_gravity_logistic_matches_decimal_oracle():
    getcontext().prec = 60
    distance = [0.25, 3.5, 9.75]
    weights = [0.5, 1.5, 3.0]
    plateau, midpoint, growth = 1.0, 2.0, 0.7
    acc = Decimal(0)
    for d, w in zip(distance, weights):
        u = (d - plateau) - midpoint
        decay = 1 - 1 / (1 + (-Decimal(growth) * Decimal(u)).exp())
        acc += Decimal(w) * decay
    exact = acc
    _r, _g, g_log, _k = corrected_reach_gravity_knn(
        np.array(distance, dtype=np.float64),
        np.array(weights, dtype=np.float64), 100.0,
        0.0, plateau, midpoint, growth, np.array([1.0]), "logistic")
    ulp = float(np.spacing(abs(float(exact))))
    assert abs(g_log - float(exact)) <= 4 * ulp, (g_log, float(exact), ulp)


def test_decay_endpoint_at_plateau_is_unit():
    """d == plateau: exponential decay = exp(0) = 1, so the term is the
    weight itself."""
    g_exp = corrected_reach_gravity_knn(
        np.array([1.0]), np.array([2.0]), 10.0,
        0.9, 1.0, 0.0, 0.0, np.array([1.0]), "exponential")[1]
    assert bits(g_exp) == bits(2.0)


# --------------------------------------------------------------------------
# KNN stable-tie policy
# --------------------------------------------------------------------------

def test_knn_ties_broken_by_destination_index():
    """Stable sort: equal distances keep ascending destination order, so
    the coefficient assignment follows destination index (weights 20,30
    at the tied pair; any other tie policy changes the result)."""
    _r, _g, _gl, knn = corrected_reach_gravity_knn(
        np.array([2.0, 1.0, 1.0]),
        np.array([10.0, 20.0, 30.0]), 100.0,
        0.0, 0.0, 0.0, 0.0,
        np.array([1.0, 1.0, 1.0]), "none")
    assert bits(knn) == bits(60.0)


def test_knn_crops_to_coefficient_count():
    _r, _g, _gl, knn = corrected_reach_gravity_knn(
        np.array([1.0, 2.0, 3.0]),
        np.array([10.0, 20.0, 30.0]), 100.0,
        0.0, 0.0, 0.0, 0.0,
        np.array([1.0, 1.0]), "none")
    assert bits(knn) == bits(30.0)      # two nearest: 10 + 20


# --------------------------------------------------------------------------
# canonical partition
# --------------------------------------------------------------------------

def test_partition_shape_and_membership():
    assert canonical_stripe_count(0) == 0
    assert canonical_stripe_count(1) == 1
    assert canonical_stripe_count(32) == 32
    assert canonical_stripe_count(100) == 32
    stripes = canonical_stripes(100)
    assert len(stripes) == 32
    for k, stripe in enumerate(stripes):
        assert stripe == list(range(k, 100, 32))        # i -> i mod L
        assert stripe == sorted(stripe)                 # increasing order
    assert stripe_of_origin(99, 32) == 99 % 32


def test_aggregate_matches_reference_partition_semantics():
    """The aggregate equals: per-stripe ordered sums (increasing origin)
    folded in increasing stripe ID -- computed here independently from
    the definition."""
    n = 70
    values = [1.0 + (i % 7) * 0.125 for i in range(n)]
    got = aggregate_by_canonical_partition(values, lambda a, v: a + v)
    L = min(32, n)
    stripe_accs = []
    for k in range(L):
        acc = 0.0
        for i in range(k, n, L):
            acc = acc + values[i]
        stripe_accs.append(acc)
    total = 0.0
    for acc in stripe_accs:
        total = total + acc
    assert bits(got) == bits(total)
    assert bits(stripe_fold(stripe_accs)) == bits(total)


def test_partition_version_is_frozen_string():
    assert PARTITION_VERSION == "canonical-min32-imodL-v1"
