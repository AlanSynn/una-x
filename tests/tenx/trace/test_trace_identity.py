"""C11 identity gates: the layered TraceID is sensitive to exactly the
search-defining inputs (campaign una_10x).

One-ULP cutoff flips, dtype changes, seed/cost bit flips and epoch
rollover changes all miss.  Metric parameters and destination METRIC
weights are excluded BY CONSTRUCTION (MetricID owns them): a build
with different metric-only inputs yields the SAME identity and the
same packed bytes.  A handle is not a durable cache identity: two
independently built records with equal identity are equal bytewise,
never by pointer.
"""
import dataclasses

import numpy as np
import pytest

from conftest import (admitted_case, bits, build, packed_equal)


def test_cutoff_ulp_flips_identity(tr):
    case = admitted_case("base")
    r25 = build(tr, case)
    ulp = np.nextafter(25.0, np.inf)
    case_ulp = dict(case)
    case_ulp["cutoff"] = ulp
    r_ulp = build(tr, case_ulp)
    assert r25.trace_id != r_ulp.trace_id
    assert r_ulp.complete and r25.complete


def test_cutoff_dtype_flips_identity_and_replay_refuses(tr):
    case = admitted_case("base")
    r_f64 = build(tr, case)
    case_f32 = dict(case)
    case_f32["cutoff"] = np.float32(25.0)
    r_f32 = build(tr, case_f32)
    assert r_f64.trace_id != r_f32.trace_id
    rp = tr.replay_trace(
        r_f64, case["dwt"], 1.5, 2.0, 3.0, 0.8, "exponential",
        np.array([1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]),
        np.float32(25.0))
    assert rp.route == "trace_replay_refused"
    assert rp.refusal == "cutoff_mismatch_with_trace_identity"


def test_seed_and_cost_bit_flips_miss(tr):
    """A 1-ULP flip in any recorded input bit produces a different
    identity (mutation law)."""
    base = admitted_case("base")
    r0 = build(tr, base)
    variant = dict(base)
    variant["ow"] = base["ow"].copy()
    variant["ow"][0, 0] = np.nextafter(base["ow"][0, 0], np.inf)
    r1 = build(tr, variant)
    assert r0.trace_id != r1.trace_id
    variant2 = dict(base)
    variant2["dw"] = base["dw"].copy()
    variant2["dw"][0, 0] = np.nextafter(base["dw"][0, 0], np.inf)
    r2 = build(tr, variant2)
    assert r0.trace_id != r2.trace_id


def test_metric_params_excluded_from_identity(tr):
    """Metric parameters and destination METRIC weights are excluded
    from identity BY CONSTRUCTION: no such parameter exists on the
    identity or build signatures; replay exposes every one of them and
    one trace replays two different metric bundles successfully."""
    import inspect
    forbidden = ("gravity_beta", "gravity_plateau",
                 "gravity_logistic_midpoint", "gravity_growth_rate",
                 "knn_decay", "knn_weights", "d_weights")
    id_params = list(inspect.signature(tr.compute_trace_id).parameters)
    build_params = list(inspect.signature(tr.build_trace).parameters)
    for name in forbidden:
        assert name not in id_params, name
    # build_trace carries d_weights for ADMISSION only (the S11 gate);
    # the six fold-only parameters are not build inputs at all.
    metric_only = forbidden[:-1]
    for name in metric_only:
        assert name not in build_params, name
    replay_params = list(
        inspect.signature(tr.replay_trace).parameters)
    for name in forbidden:
        assert name in replay_params, name
    case = admitted_case("base")
    rec = build(tr, case)
    knn = np.array([1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1])
    rp_a = tr.replay_trace(rec, case["dwt"], 1.5, 2.0, 3.0, 0.8,
                           "exponential", knn, case["cutoff"])
    rp_b = tr.replay_trace(rec, case["dwt"], 2.5, 0.5, 1.25, 0.4,
                           "logistic", np.full(10, 0.3),
                           case["cutoff"])
    assert rp_a.route == "trace_replay" and rp_b.route == "trace_replay"
    # reach is cutoff-driven only: identical across metric bundles.
    assert bits(rp_a.reach) == bits(rp_b.reach)


def test_epoch_rollover_flips_identity(tr):
    case = admitted_case("base")
    a = build(tr, case)
    b = tr.build_trace(
        case["o"], case["ow"], case["ptr"], case["vec"], case["wts"],
        case["flag"], case["d"], case["dw"], case["dwt"],
        case["cutoff"], epoch_rollover=4)
    assert a.trace_id != b.trace_id
    assert b.complete


def test_handle_is_not_identity(tr):
    """Two independently built records with the same identity are
    equal bytewise on the packed arrays; object identity and pointer
    addresses are irrelevant."""
    case = admitted_case("base")
    a = build(tr, case)
    b = build(tr, case)
    assert a is not b
    assert a.trace_id == b.trace_id
    assert packed_equal(a, b)
    assert np.asarray(a.indptr).__array_interface__["data"][0] != \
        np.asarray(b.indptr).__array_interface__["data"][0]


def test_frozen_record_and_readonly_arrays(tr):
    """The record is frozen and its packed arrays are read-only:
    in-place mutation of the residency payload is impossible."""
    case = admitted_case("base")
    rec = build(tr, case)
    with pytest.raises(dataclasses.FrozenInstanceError):
        rec.trace_id = "tampered"
    with pytest.raises(ValueError):
        np.asarray(rec.destination_ids)[0] = 123
    with pytest.raises(ValueError):
        np.asarray(rec.distance_bits)[0] = -1.0
    with pytest.raises(ValueError):
        np.asarray(rec.indptr)[0] = 5
