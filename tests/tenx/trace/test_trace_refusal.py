"""C11 refusal gates: the trace route serves exactly the S11-admitted
domain (campaign una_10x).

The admission gate is the S11 module's own (imported, not copied): on
every refused battery case the trace build must produce an INCOMPLETE
record whose refusal reason equals the S11 driver's refusal reason,
with empty packed arrays and no published trace status — and the
unchanged S11/public path remains bitwise the baseline's.  No invalid
input becomes successful because a trace exists.
"""
import dataclasses

import numpy as np
import pytest

from conftest import (CASE_PARAMS, RESULT_FIELDS, admitted_case, bits,
                      build, build_case, case_kwargs, driver_args,
                      replay, result_equal)

EXPECTED_REFUSAL = {n: "values" for n in CASE_PARAMS}
EXPECTED_REFUSAL["weights_f32"] = "a1_guard"


@pytest.mark.parametrize("name", CASE_PARAMS)
def test_refused_battery_cases_never_publish_a_trace(
        name, tr, sp, baseline):
    """The battery's native domain refuses (designed -1.0
    destination-terminal weight; float32 variant); the trace build
    refuses with the SAME reason the S11 driver refuses, and S11's
    fallback stays bitwise the baseline's."""
    case = build_case(3 + CASE_PARAMS.index(name), **case_kwargs(name))
    args = driver_args(case)
    ref = baseline(*args)
    s11 = sp.sparse_scope_access(*args)
    assert s11.route == sp.ROUTE_FALLBACK
    assert s11.refusal == EXPECTED_REFUSAL[name]
    assert result_equal(s11, ref)
    rec = build(tr, case)
    assert rec.complete is False
    assert rec.route == "compact_fallback"
    assert rec.refusal == s11.refusal
    assert np.asarray(rec.indptr).shape[0] == rec.origin_count + 1
    assert np.asarray(rec.destination_ids).shape[0] == 0
    assert np.asarray(rec.distance_bits).shape[0] == 0
    rp = replay(tr, rec, case)
    assert rp.route == "trace_replay_refused"
    assert rp.folds_done == 0


def test_invalid_metric_input_refused_despite_existing_trace(tr):
    """No invalid input becomes successful because a trace exists:
    NaN/negative destination metric weights, wrong destination count
    and layout errors are all refused at replay time."""
    case = admitted_case("base")
    rec = build(tr, case)
    good = replay(tr, rec, case)
    assert good.route == "trace_replay"
    knn = np.array([1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1])

    def try_replay(d_weights, cutoff=None):
        return tr.replay_trace(rec, d_weights, 1.5, 2.0, 3.0, 0.8,
                               "exponential", knn,
                               case["cutoff"] if cutoff is None
                               else cutoff)

    nan_w = case["dwt"].copy()
    nan_w[0] = np.nan
    r = try_replay(nan_w)
    assert r.route == "trace_replay_refused"
    assert r.refusal == "dest_weights_value_domain"
    neg_w = case["dwt"].copy()
    neg_w[0] = -0.5
    r = try_replay(neg_w)
    assert r.refusal == "dest_weights_value_domain"
    r = try_replay(case["dwt"][:-1])
    assert r.refusal == "dest_count_mismatch"
    r = try_replay(list(case["dwt"]))
    assert r.refusal == "dest_weights_layout"
    r = try_replay(case["dwt"].reshape(1, -1))
    assert r.refusal == "dest_weights_layout"
    r = try_replay(case["dwt"], cutoff=np.float32(25.0))
    assert r.refusal == "cutoff_mismatch_with_trace_identity"
    r = try_replay(case["dwt"], cutoff=np.nextafter(25.0, np.inf))
    assert r.refusal == "cutoff_mismatch_with_trace_identity"


def test_not_a_record_and_schema_mismatch_refused(tr):
    case = admitted_case("base")
    rec = build(tr, case)
    r = tr.replay_trace(
        object(), case["dwt"], 1.5, 2.0, 3.0, 0.8, "exponential",
        np.array([1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]),
        case["cutoff"])
    assert r.route == "trace_replay_refused"
    assert r.refusal == "not_a_trace_record"
    forged = dataclasses.replace(rec, schema_version="someone-elses")
    r = tr.replay_trace(
        forged, case["dwt"], 1.5, 2.0, 3.0, 0.8, "exponential",
        np.array([1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]),
        case["cutoff"])
    assert r.refusal == "trace_schema_mismatch"
