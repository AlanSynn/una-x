"""P03M: MetricID v2 identity laws (campaign una_10x post_p00,
designs/04 "Test collisions BEFORE benchmarking").

Every test binds a REAL admitted C11 case and the LIVE replay binding:
the identity is exercised over the same argument tuples the compiled
fold consumes, never over synthetic dicts.  The C31 v1 cohort key is
used as the COLLISION WITNESS: where its dtype-erasing coercion
collapses two genuinely different jobs into one key, MetricID v2 must
distinguish them (and refuse what it cannot encode losslessly).
"""
import struct

import numpy as np
import pytest

from conftest import (BUNDLE, METRIC_PARAMS, admitted_case, bits,
                      replay_binding)


# --------------------------------------------------------------------------
# Version law: versioned IDs without invalidating the trace producer
# --------------------------------------------------------------------------
def test_schema_versions_are_pinned(mid, tr):
    assert mid.METRIC_ID_SCHEMA_VERSION == "una-10x-metricid-v2-1"
    assert mid.METRIC_CLOSURE_VERSION == 1
    # The trace producer schema is UNCHANGED (designs/04: version
    # metric IDs without invalidating an unchanged valid trace
    # producer).
    assert tr.TRACE_SCHEMA_VERSION == "una-10x-c11-trace-1"


def test_id_stable_and_trace_producer_untouched(mid, tr):
    """Same binding -> same ID; metric variations of ONE trace keep the
    trace identity byte-identical while their metric IDs differ."""
    case = admitted_case("base")
    rec = tck_build(tr, case)
    kw = replay_binding(case)
    id_a = mid.compute_metric_id(rec, *kw)
    id_b = mid.compute_metric_id(rec, *kw)
    assert id_a == id_b

    kw_beta = replay_binding(case, bundle=(2.5, BUNDLE[1], BUNDLE[2],
                                           BUNDLE[3]))
    id_c = mid.compute_metric_id(rec, *kw_beta)
    assert id_c != id_a
    # The embedded trace identity is the same bytes for both jobs.
    fields_a = mid.metric_binding_fields(rec, *kw)
    fields_c = mid.metric_binding_fields(rec, *kw_beta)
    ta = dict(fields_a)["trace_id"]
    tc = dict(fields_c)["trace_id"]
    assert ta == tc == rec.trace_id


def tck_build(trm, case):
    return tck.build(trm, case)


from conftest import tck  # noqa: E402  (battery build helper)


# --------------------------------------------------------------------------
# v1 collisions v2 must distinguish (E8 closure law)
# --------------------------------------------------------------------------
def _v1_witness_key(trm, cutoff, knn_weights, d_weights, trace_id="t"):
    """The C31 v1 cohort key construction, verbatim coercion, over the
    SAME canonical encoder the cohort route uses."""
    from urban_network_analysis.cache import identity as _ckid
    return _ckid.stage_key("una:cohort:o_access:v1", {
        "trace_id": trace_id,
        "gravity_beta": 1.5,
        "gravity_plateau": 2.0,
        "gravity_logistic_midpoint": 3.0,
        "gravity_decay_constant": 2.4,
        "knn_decay": "exponential",
        "knn_weights": np.asarray(knn_weights, dtype=np.float64),
        "search_radius": float(cutoff),
        "d_weights": np.asarray(d_weights, dtype=np.float64),
    })


def test_int_float_cutoff_collapses_in_v1_not_v2(mid, tr):
    case = admitted_case("base")
    rec = tck_build(tr, case)
    kw_f = replay_binding(case)
    kw_i = replay_binding(case, cutoff=25)
    v1_f = _v1_witness_key(tr, kw_f[-1], METRIC_PARAMS[4], case["dwt"])
    v1_i = _v1_witness_key(tr, kw_i[-1], METRIC_PARAMS[4], case["dwt"])
    # The dtype-erasing float() coercion COLLAPSES int 25 and float
    # 25.0 into one reporting key...
    assert v1_f == v1_i
    # ...while the typed identity distinguishes them.
    assert mid.compute_metric_id(rec, *kw_f) != \
        mid.compute_metric_id(rec, *kw_i)


def test_weight_dtype_collapse_in_v1_not_v2(mid, tr):
    # float32-EXACT values: the f32 array carries the same VALUES as
    # its f64 twin, so the v1 float64 coercion genuinely collapses two
    # differently-typed replay inputs into one key.
    rep = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 0.25, 0.75, 1.25, 1.75,
                    2.25], dtype=np.float64)
    assert (rep.astype(np.float32).astype(np.float64) == rep).all()
    case = admitted_case("base")
    rec = tck_build(tr, case)
    kw_f64 = replay_binding(case, knn_weights=rep)
    kw_f32 = replay_binding(case, knn_weights=rep.astype(np.float32))
    v1_f64 = _v1_witness_key(tr, case["cutoff"], rep, case["dwt"])
    v1_f32 = _v1_witness_key(tr, case["cutoff"], rep.astype(np.float32),
                             case["dwt"])
    # The dtype-erasing np.asarray(..., float64) coercion COLLAPSES
    # equal-value float32/float64 knn weights into one reporting key...
    assert v1_f64 == v1_f32
    # ...while the typed identity distinguishes them.
    assert mid.compute_metric_id(rec, *kw_f64) != \
        mid.compute_metric_id(rec, *kw_f32)


def test_weight_dtype_distinguishable_when_values_genuinely_differ(
        mid, tr):
    """Same collapse law for the destination metric weights: identical
    VALUES in float32 vs float64 are one v1 key, two v2 identities."""
    rep = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 0.25, 0.75, 1.25, 1.75,
                    2.25, 0.125, 3.0], dtype=np.float64)
    assert (rep.astype(np.float32).astype(np.float64) == rep).all()
    case = admitted_case("base")
    rec = tck_build(tr, case)
    kw_f64 = replay_binding(case, d_weights=rep)
    kw_f32 = replay_binding(case, d_weights=rep.astype(np.float32))
    v1_f64 = _v1_witness_key(tr, case["cutoff"], METRIC_PARAMS[4], rep)
    v1_f32 = _v1_witness_key(tr, case["cutoff"], METRIC_PARAMS[4],
                             rep.astype(np.float32))
    assert v1_f64 == v1_f32
    assert mid.compute_metric_id(rec, *kw_f64) != \
        mid.compute_metric_id(rec, *kw_f32)


# --------------------------------------------------------------------------
# Scalar specialization: exact type AND bits
# --------------------------------------------------------------------------
def test_scalar_negative_zero_and_ulp_distinct(mid, tr):
    case = admitted_case("base")
    rec = tck_build(tr, case)
    ids = {name: mid.compute_metric_id(
        rec, *replay_binding(case, bundle=(beta, BUNDLE[1], BUNDLE[2],
                                           BUNDLE[3])))
        for name, beta in [("plus", 1.5), ("minus0", -0.0),
                           ("ulp", np.nextafter(1.5, 2.0))]}
    assert len(set(ids.values())) == 3


def test_scalar_nan_payloads_distinct_no_normalization(mid, tr):
    case = admitted_case("base")
    rec = tck_build(tr, case)
    qnan = struct.unpack("<d", struct.pack("<Q", 0x7FF8000000000000))[0]
    qnan1 = struct.unpack("<d", struct.pack("<Q", 0x7FF8000000000001))[0]
    id_a = mid.compute_metric_id(
        rec, *replay_binding(case, bundle=(qnan, BUNDLE[1], BUNDLE[2],
                                           BUNDLE[3])))
    id_b = mid.compute_metric_id(
        rec, *replay_binding(case, bundle=(qnan1, BUNDLE[1], BUNDLE[2],
                                           BUNDLE[3])))
    assert id_a != id_b


def test_scalar_types_distinguished_at_equal_value(mid, tr):
    """int 25 vs float 25.0 cutoff, np.float64 vs python float beta:
    equal VALUES, different typed identities."""
    case = admitted_case("base")
    rec = tck_build(tr, case)
    id_int = mid.compute_metric_id(rec, *replay_binding(case, cutoff=25))
    id_float = mid.compute_metric_id(rec, *replay_binding(case))
    assert id_int != id_float
    id_pyfloat = mid.compute_metric_id(rec, *replay_binding(case))
    id_npfloat = mid.compute_metric_id(
        rec, *replay_binding(case, bundle=(np.float64(BUNDLE[0]),
                                           BUNDLE[1], BUNDLE[2],
                                           BUNDLE[3])))
    # np.float64(1.5) and python 1.5 share IEEE-754 binary64 semantics
    # AND dtype: one identity (this is their EXACT type, not a cast).
    assert id_pyfloat == id_npfloat


# --------------------------------------------------------------------------
# Array specialization: dtype-only / byte-order-only / layout-only
# --------------------------------------------------------------------------
def test_array_dtype_only_change_distinct(mid, tr):
    case = admitted_case("base")
    rec = tck_build(tr, case)
    d64 = np.asarray(case["dwt"], np.float64)
    d32 = d64.astype(np.float32)
    id64 = mid.compute_metric_id(rec, *replay_binding(case, d_weights=d64))
    id32 = mid.compute_metric_id(rec, *replay_binding(case, d_weights=d32))
    assert id64 != id32


def test_array_byteorder_only_change_distinct(mid, tr):
    case = admitted_case("base")
    rec = tck_build(tr, case)
    dle = np.asarray(case["dwt"], np.float64)
    dbe = dle.astype(dle.dtype.newbyteorder(">"))
    assert np.array_equal(dle, dbe)          # identical values
    id_le = mid.compute_metric_id(rec, *replay_binding(case, d_weights=dle))
    id_be = mid.compute_metric_id(rec, *replay_binding(case, d_weights=dbe))
    assert id_le != id_be


def test_array_layout_only_change_distinct(mid, tr):
    case = admitted_case("base")
    rec = tck_build(tr, case)
    base = np.concatenate([case["dwt"], case["dwt"]])
    d_view = base[::2]                       # same values, strided
    assert not d_view.flags.c_contiguous
    d_cont = np.ascontiguousarray(d_view)
    id_view = mid.compute_metric_id(
        rec, *replay_binding(case, d_weights=d_view))
    id_cont = mid.compute_metric_id(
        rec, *replay_binding(case, d_weights=d_cont))
    assert id_view != id_cont


def test_array_writability_distinct(mid, tr):
    case = admitted_case("base")
    rec = tck_build(tr, case)
    d = np.asarray(case["dwt"], np.float64).copy()
    d_ro = d.copy()
    d_ro.flags.writeable = False
    id_rw = mid.compute_metric_id(rec, *replay_binding(case, d_weights=d))
    id_ro = mid.compute_metric_id(rec, *replay_binding(case,
                                                       d_weights=d_ro))
    assert id_rw != id_ro


# --------------------------------------------------------------------------
# Refusals: no convenience coercion (designs/04 enumeration law)
# --------------------------------------------------------------------------
def test_refusal_complex_weights(mid, tr):
    case = admitted_case("base")
    rec = tck_build(tr, case)
    with pytest.raises(mid.MetricIdentityError):
        mid.compute_metric_id(
            rec, *replay_binding(case,
                                 d_weights=case["dwt"].astype(np.complex128)))


def test_refusal_string_scalar(mid, tr):
    case = admitted_case("base")
    rec = tck_build(tr, case)
    with pytest.raises(mid.MetricIdentityError):
        mid.compute_metric_id(
            rec, *replay_binding(case, bundle=("1.5", BUNDLE[1],
                                               BUNDLE[2], BUNDLE[3])))


def test_refusal_oversize_python_int(mid, tr):
    case = admitted_case("base")
    rec = tck_build(tr, case)
    with pytest.raises(ValueError):
        mid.compute_metric_id(
            rec, *replay_binding(case, cutoff=1 << 70))


def test_refusal_object_array(mid, tr):
    case = admitted_case("base")
    rec = tck_build(tr, case)
    obj = np.array([float(x) for x in case["dwt"]], dtype=object)
    with pytest.raises(mid.MetricIdentityError):
        mid.compute_metric_id(rec, *replay_binding(case, d_weights=obj))


def test_refusal_non_ndarray_weights(mid, tr):
    case = admitted_case("base")
    rec = tck_build(tr, case)
    with pytest.raises(mid.MetricIdentityError):
        mid.compute_metric_id(
            rec, *replay_binding(case, d_weights=list(case["dwt"])))


def test_refusal_trace_schema_guard(mid, tr):
    import dataclasses
    case = admitted_case("base")
    rec = tck_build(tr, case)
    forged = dataclasses.replace(rec, schema_version="not-the-trace-schema")
    with pytest.raises(ValueError):
        kw = replay_binding(case)
        mid.compute_metric_id(forged, *kw)


def test_float16_knn_weights_encodable_and_distinct(mid, tr):
    """f2 is in the enumerated float support: encodable losslessly and
    distinct from the f32/f64 closures of the same values."""
    case = admitted_case("base")
    rec = tck_build(tr, case)
    kw_f16 = METRIC_PARAMS[4].astype(np.float16)
    kw_f32 = METRIC_PARAMS[4].astype(np.float32)
    id16 = mid.compute_metric_id(rec, *replay_binding(case,
                                                      knn_weights=kw_f16))
    id32 = mid.compute_metric_id(rec, *replay_binding(case,
                                                      knn_weights=kw_f32))
    assert id16 != id32


# --------------------------------------------------------------------------
# Cohort transport surface
# --------------------------------------------------------------------------
def test_cohort_delta_exposes_metric_id():
    """The route's per-row slice carries the v2 identity field."""
    from urban_network_analysis.batch.cohort import CohortSession
    s = CohortSession("/tmp/una10x-p03m-nonexistent-root", 1 << 26)
    d = s.delta(s.mark())
    assert "metric_id" in d and d["metric_id"] is None
