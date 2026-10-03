"""L1: compiled sparse kernel vs actual compiled baseline (S11 gate).

The reference is `integrated_scope_access` — the pinned compiled driver
(a1 guard, a3/a1 private routes, engine fold) — run TWICE first to
record true variability (VALIDATION oracle discipline).  Every case
compares all four outputs bitwise; route attribution plus counters
prove which path executed and that work actually shrank (dossier 03:
counters, not entry).  The fold-prefix sweep pins dossier 03's
"direct actual-fold equivalence": the engine's own fold on the full
destination vector vs the same fold on the candidate prefix must agree
bitwise for every retained count (the np.where gather normalizes the
argsort input length; ties included).
"""
import threading
import time

import numpy as np
import pytest

from conftest import (CASE_PARAMS, admitted_case, bit_equal_result, bits,
                      build_case, case_kwargs, driver_args)

RESULT_FIELDS = ("reach", "gravity_exponential", "gravity_logistic",
                 "knn_access")


@pytest.mark.parametrize("name", CASE_PARAMS)
def test_battery_admitted_variants(name, sp, baseline):
    if name == "weights_f32":
        pytest.skip("float32 variant refuses on dtype; covered by the "
                    "fallback test")
    case = admitted_case(name)
    args = driver_args(case)
    b1 = baseline(*args)
    b2 = baseline(*args)
    assert all(bits(x) == bits(y) for x, y in zip(b1, b2)), "baseline jitter"
    res = sp.sparse_scope_access(*args)
    assert res.route == sp.ROUTE_SPARSE and res.refusal == ""
    assert bit_equal_result(res, b1)


EXPECTED_FALLBACK = {n: "values" for n in CASE_PARAMS}
EXPECTED_FALLBACK["weights_f32"] = "a1_guard"


@pytest.mark.parametrize("name", CASE_PARAMS)
def test_battery_original_domain_fallback(name, sp, baseline):
    """The battery's native -1.0 destination-terminal weight and the
    float32 variant both refuse; fallback stays bitwise."""
    case = build_case(3 + CASE_PARAMS.index(name), **case_kwargs(name))
    args = driver_args(case)
    ref = baseline(*args)
    res = sp.sparse_scope_access(*args)
    assert res.route == sp.ROUTE_FALLBACK
    assert res.refusal == EXPECTED_FALLBACK[name]
    assert bit_equal_result(res, ref)


@pytest.mark.parametrize("decay", ["exponential", "logistic", "other"])
def test_knn_decay_branches(decay, sp, baseline):
    case = admitted_case("base")
    args = driver_args(case, knn_decay=decay)
    ref = baseline(*args)
    res = sp.sparse_scope_access(*args)
    assert res.route == sp.ROUTE_SPARSE
    assert bit_equal_result(res, ref)


def _fold_positional(dist, wts, cutoff, decay, knn_weights):
    # 13 positional args in the engine fold's own parameter order
    from urban_network_analysis.Engines.AccessibilityWElevation import (
        reach_gravity_knn_access)
    return reach_gravity_knn_access(
        dist, wts, cutoff, 1.5, 2.0, 3.0, 2.0, knn_weights, 1.5,
        decay, 3.0, 0.8, 0.8)


@pytest.mark.parametrize("kept", [0, 1, 2, 3, 4, 7, 8, 9, 15, 16, 17, 31,
                                  32, 33, 64, 65, 257])
def test_fold_prefix_equivalence(kept):
    rng = np.random.default_rng(7)
    d_count = 300
    dist = np.concatenate([rng.uniform(0.0, 1.0, kept),
                           rng.uniform(5.0, 6.0, d_count - kept)])
    wts = rng.uniform(0.25, 2.0, d_count)
    knn_weights = np.linspace(1.0, 0.1, 10)
    a = _fold_positional(dist, wts, 2.0, "exponential", knn_weights)
    b = _fold_positional(
        np.ascontiguousarray(dist[:kept]),
        np.ascontiguousarray(wts[:kept]), 2.0, "exponential", knn_weights)
    assert all(bits(x) == bits(y) for x, y in zip(a, b))


def test_fold_prefix_tie_permutation():
    """Tied distances with unequal weights: the gather normalizes the
    argsort input, so the tie permutation (default argsort contract)
    must match full vs prefix bitwise."""
    rng = np.random.default_rng(8)
    kept = 12
    tied = np.array([0.5, 0.5, 0.5, 0.25, 0.25, 0.75, 0.1, 0.1, 0.9, 0.9,
                     0.9, 0.2])
    dist = np.concatenate([tied, rng.uniform(5.0, 6.0, 288 - kept)])
    wts = rng.uniform(0.25, 2.0, 288)
    knn_weights = np.linspace(1.0, 0.1, 10)
    a = _fold_positional(dist, wts, 2.0, "logistic", knn_weights)
    b = _fold_positional(np.ascontiguousarray(dist[:kept]),
                         np.ascontiguousarray(wts[:kept]), 2.0, "logistic",
                         knn_weights)
    assert all(bits(x) == bits(y) for x, y in zip(a, b))


def test_two_concurrent_calls(sp, baseline):
    """Per-call arenas share no global state: two threads at the barrier
    produce bitwise-serial results on the sparse route."""
    case = admitted_case("base")
    args = driver_args(case)
    ref = baseline(*args)
    out = {}
    barrier = threading.Barrier(2)

    def go(i):
        barrier.wait()
        out[i] = sp.sparse_scope_access(*args)

    threads = [threading.Thread(target=go, args=(i,)) for i in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    for i in (0, 1):
        assert out[i].route == sp.ROUTE_SPARSE
        assert bit_equal_result(out[i], ref)


def test_work_reduction_counters_large(sp, baseline):
    """Larger ring graph: counters prove the work actually shrank
    (touched nodes << dense initialization, candidates <= O*D); rough
    wall time recorded as evidence, NOT a performance claim."""
    rng = np.random.default_rng(11)
    v = 2000
    vec = np.concatenate([((np.arange(v) + 1) % v).astype(np.int64),
                          rng.integers(0, v, 1500).astype(np.int64)])
    src = np.concatenate([np.arange(v, dtype=np.int64),
                          rng.integers(0, v, 1500).astype(np.int64)])
    order = np.argsort(src, kind="stable")
    vec = np.ascontiguousarray(vec[order])
    src = src[order]
    ptr = np.zeros(v + 1, dtype=np.int64)
    ptr[1:] = np.cumsum(np.bincount(src, minlength=v))
    wts = rng.uniform(0.1, 1.0, vec.shape[0])
    flag = np.ones(vec.shape[0], dtype=bool)
    origins = np.arange(200, dtype=np.int64)
    o = np.stack([origins, (origins + 7) % v], axis=1).astype(np.int64)
    ow = np.full((200, 2), 0.1)
    d_count = 500
    d = np.stack([rng.integers(0, v, d_count),
                  rng.integers(0, v, d_count)], axis=1).astype(np.int64)
    dw = rng.uniform(0.0, 0.5, (d_count, 2))
    dwt = rng.uniform(0.25, 1.0, d_count)
    args = (o, ow, ptr, vec, wts, flag, d, dw, dwt) + (
        1.5, 2.0, 3.0, 0.8, "exponential",
        np.array([1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1])) \
        + (1.5,)
    ref = baseline(*args)
    res = sp.sparse_scope_access(*args)
    assert res.route == sp.ROUTE_SPARSE
    assert bit_equal_result(res, ref)
    assert res.counters.touched_total < 200 * v
    assert res.counters.candidates_total <= 200 * d_count
    t0 = time.perf_counter()
    sp.sparse_scope_access(*args)
    t_sparse = time.perf_counter() - t0
    t0 = time.perf_counter()
    baseline(*args)
    t_base = time.perf_counter() - t0
    print(f"\nwork counters: touched={res.counters.touched_total} "
          f"candidates={res.counters.candidates_total} "
          f"stamps_zeroed={res.counters.stamps_zeroed} "
          f"dense_init_avoided={res.counters.dense_init_avoided}")
    print(f"timing evidence (no claim): sparse {t_sparse:.3f}s vs "
          f"baseline {t_base:.3f}s")


def test_route_attribution_distinguishes(sp, baseline):
    """Guard refusal and sparse execution are distinguishable in
    evidence (dossier 03) via route + refusal + zeroed counters."""
    case = admitted_case("base")
    args = driver_args(case)
    ok = sp.sparse_scope_access(*args)
    bad = build_case(3, nan_seed=True)
    refused = sp.sparse_scope_access(*driver_args(bad))
    assert ok.route != refused.route
    assert ok.refusal == "" and refused.refusal == "values"
    assert ok.counters.origins > 0 and refused.counters.origins == 0
    assert refused.counters.touched_total == 0
    assert refused.counters.stamps_zeroed == 0
