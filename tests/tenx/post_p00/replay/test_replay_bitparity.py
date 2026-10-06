"""P03C L1: compiled outer replay is bitwise the pre-change replay
AND the original integrated engine (campaign una_10x post_p00,
designs/03: "compare bits against BOTH current replay and original
integrated engine. ... Run parity before timing.").

Three arms meet at every case: (1) the FROZEN pre-P03C Python
``replay_trace`` (byte-faithful copy at commit a7281c8, loaded from its
bytes — the "current replay" arm), (2) the LIVE compiled-driver
``replay_trace``, and (3) the original integrated engine
(``integrated_scope_access``, unchanged) wherever its driver admits
the dtype domain.  No allclose: every output is compared by bytes.
"""
import numpy as np
import pytest

from conftest import (CASE_PARAMS, BUNDLE, METRIC_PARAMS,
                      admitted_case, assert_three_way, bits, driver_args,
                      first_difference, replay_legacy, replay_live,
                      result_equal)


@pytest.mark.parametrize("name", CASE_PARAMS)
def test_compiled_replay_bitwise_legacy_and_integrated(
        name, tr, baseline, legacy):
    if name == "weights_f32":
        pytest.skip("float32 variant refuses on dtype; the trace route "
                    "serves the S11-admitted domain (refusal suite)")
    case = admitted_case(name)
    rec = tck_build(tr, case)
    old = replay_legacy(rec, case)
    new = replay_live(tr, rec, case)
    args = driver_args(case)
    ref = baseline(*args)
    assert_three_way(new, old, ref)
    assert new.folds_done == rec.origin_count


def tck_build(trm, case):
    from conftest import tck
    return tck.build(trm, case)


@pytest.mark.parametrize("decay", ["exponential", "logistic", "other"])
def test_decay_branches_three_way(tr, baseline, legacy, decay):
    """All three KNN decay branches of the fold, three arms bitwise."""
    case = admitted_case("base")
    rec = tck_build(tr, case)
    old = replay_legacy(rec, case, knn_decay=decay)
    new = replay_live(tr, rec, case, knn_decay=decay)
    args = list(driver_args(case, knn_decay=decay))
    ref = baseline(*args)
    assert_three_way(new, old, ref)


def test_metric_variations_three_way(tr, baseline, legacy):
    """Fold parameter variations replay bitwise from one trace on the
    compiled path (the trace identity does not embed MetricID)."""
    case = admitted_case("base")
    rec = tck_build(tr, case)
    variations = [
        dict(gravity_beta=2.5),
        dict(gravity_plateau=0.5),
        dict(gravity_logistic_midpoint=1.25),
        dict(gravity_growth_rate=0.4),
    ]
    for kw in variations:
        args = list(driver_args(case))
        args[9] = kw.get("gravity_beta", BUNDLE[0])
        args[10] = kw.get("gravity_plateau", BUNDLE[1])
        args[11] = kw.get("gravity_logistic_midpoint", BUNDLE[2])
        args[12] = kw.get("gravity_growth_rate", BUNDLE[3])
        ref = baseline(*args)
        new = tr.replay_trace(rec, case["dwt"], kw.get(
            "gravity_beta", BUNDLE[0]), kw.get(
            "gravity_plateau", BUNDLE[1]), kw.get(
            "gravity_logistic_midpoint", BUNDLE[2]), kw.get(
            "gravity_growth_rate", BUNDLE[3]), "exponential",
            METRIC_PARAMS[4], case["cutoff"])
        old = legacy_replay_kw(rec, case, kw)
        assert_three_way(new, old, ref)


def legacy_replay_kw(rec, case, kw):
    from conftest import legacy_replay_trace
    return legacy_replay_trace(
        rec, case["dwt"], kw.get("gravity_beta", BUNDLE[0]), kw.get(
            "gravity_plateau", BUNDLE[1]), kw.get(
            "gravity_logistic_midpoint", BUNDLE[2]), kw.get(
            "gravity_growth_rate", BUNDLE[3]), "exponential",
        METRIC_PARAMS[4], case["cutoff"])


def test_medium_ring_three_way(tr, baseline, legacy):
    """The C11 medium ring (40 origins, beyond the toy battery):
    compiled replay bitwise against frozen legacy AND the integrated
    engine."""
    rng = np.random.default_rng(23)
    v = 400
    src = np.concatenate([np.arange(v, dtype=np.int64),
                          rng.integers(0, v, 300).astype(np.int64)])
    vec = np.concatenate([((np.arange(v) + 1) % v).astype(np.int64),
                          rng.integers(0, v, 300).astype(np.int64)])
    order = np.argsort(src, kind="stable")
    src, vec = src[order], vec[order]
    ptr = np.zeros(v + 1, dtype=np.int64)
    ptr[1:] = np.cumsum(np.bincount(src, minlength=v))
    wts = rng.uniform(0.1, 1.0, vec.shape[0])
    flag = np.ones(vec.shape[0], dtype=bool)
    origins = np.arange(40, dtype=np.int64)
    o = np.stack([origins, (origins + 11) % v], axis=1).astype(np.int64)
    ow = np.full((40, 2), 0.1)
    d_count = 100
    d = np.stack([rng.integers(0, v, d_count),
                  rng.integers(0, v, d_count)], axis=1).astype(np.int64)
    dw = rng.uniform(0.0, 0.5, (d_count, 2))
    dwt = rng.uniform(0.25, 1.0, d_count)
    case = dict(o=o, ow=ow, ptr=ptr, vec=vec, wts=wts, flag=flag,
                d=d, dw=dw, dwt=dwt, cutoff=8.0)
    rec = tck_build(tr, case)
    assert rec.complete and rec.origin_count == 40
    old = replay_legacy(rec, case)
    new = replay_live(tr, rec, case)
    ref = baseline(*driver_args(case))
    assert_three_way(new, old, ref)


@pytest.mark.parametrize("dtype", [np.float32, np.int64])
def test_weight_dtype_specializations_compiled_vs_legacy(
        tr, legacy, dtype):
    """float32/int64 destination METRIC weights: both arms accept and
    specialize identically (bitwise); the integrated driver arm is out
    of its admitted dtype domain here, so the frozen pre-change bytes
    are the reference."""
    case = admitted_case("base")
    rec = tck_build(tr, case)
    w = case["dwt"].astype(dtype)
    assert w.shape == case["dwt"].shape
    old = replay_legacy(rec, case, d_weights=w)
    new = replay_live(tr, rec, case, d_weights=w)
    assert new.route == "trace_replay" and old.route == "trace_replay"
    assert_three_way(new, old)
    # float32 weights genuinely change the fold's bits vs float64 (the
    # specialization is real, not a silent coercion to one dtype).
    w64 = replay_live(tr, rec, case)
    different = any(bits(getattr(w64, f)) != bits(getattr(new, f))
                    for f in ["reach", "gravity_exponential",
                              "gravity_logistic", "knn_access"])
    assert different, dtype


def test_int_weights_reach_stays_int_domain(tr, legacy):
    """int64 weights: reach (a plain weight SUM in the fold) returns
    int-domain values; the compiled driver must not widen it to float
    (designs/03: "Do not change reach int output into float")."""
    case = admitted_case("base")
    rec = tck_build(tr, case)
    w = case["dwt"].astype(np.int64)
    new = replay_live(tr, rec, case, d_weights=w)
    old = replay_legacy(rec, case, d_weights=w)
    assert_three_way(new, old)
    assert new.reach.dtype == np.dtype(rec.reach_dtype)


def test_repeated_replay_deterministic(tr):
    """Two compiled replays of the same trace are bitwise identical
    (no cross-call state, no fast-math jitter)."""
    case = admitted_case("base")
    rec = tck_build(tr, case)
    a = replay_live(tr, rec, case)
    b = replay_live(tr, rec, case)
    assert first_difference(a, b) is None
    for f in ["reach", "gravity_exponential", "gravity_logistic",
              "knn_access"]:
        assert getattr(a, f) is not getattr(b, f)
