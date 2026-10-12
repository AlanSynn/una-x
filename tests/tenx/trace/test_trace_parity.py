"""C11 L1: trace/replay vs the full actual per-job results (campaign
una_10x).

Three independent references meet at every battery case: (1) the
pinned compiled baseline driven twice (VALIDATION oracle discipline),
(2) the S11 sparse kernel (the strongest current-main implementation),
and (3) the preserved packet reference chain loaded from its immutable
bytes.  The trace's packed slices must equal the preserved chain's
ascending candidate ids and evaluated candidate distance bits per
origin; the replay's four outputs must be bitwise the baseline's and
S11's; one trace must serve every metric variation of its query.
"""
import numpy as np
import pytest

from conftest import (CASE_PARAMS, METRIC_PARAMS, RESULT_FIELDS,
                      admitted_case, bits, build, build_case,
                      case_kwargs, driver_args, packed_equal,
                      reference_candidates, replay, result_equal)


@pytest.mark.parametrize("name", CASE_PARAMS)
def test_replay_matches_baseline_and_s11(name, tr, sp, baseline):
    if name == "weights_f32":
        pytest.skip("float32 variant refuses on dtype; the trace route "
                    "serves the S11-admitted domain (refusal suite)")
    case = admitted_case(name)
    args = driver_args(case)
    b1 = baseline(*args)
    b2 = baseline(*args)
    assert all(bits(x) == bits(y) for x, y in zip(b1, b2)), \
        "baseline jitter"
    s11 = sp.sparse_scope_access(*args)
    assert s11.route == sp.ROUTE_SPARSE
    rec = build(tr, case)
    assert rec.complete and rec.route == tr.ROUTE_TRACE
    rp = replay(tr, rec, case)
    assert rp.route == "trace_replay" and rp.refusal == ""
    assert result_equal(rp, b1)
    assert result_equal(rp, s11)
    assert rp.folds_done == rec.origin_count


@pytest.mark.parametrize("name", CASE_PARAMS)
def test_trace_slices_match_preserved_chain(name, tr, probe):
    if name == "weights_f32":
        pytest.skip("float32 variant refuses on dtype")
    case = admitted_case(name)
    rec = build(tr, case)
    assert rec.complete
    refs = reference_candidates(probe, case)
    indptr = np.asarray(rec.indptr)
    ids_all = np.asarray(rec.destination_ids)
    dist_all = np.asarray(rec.distance_bits)
    assert indptr.shape[0] == rec.origin_count + 1
    assert int(indptr[0]) == 0
    for origin in range(rec.origin_count):
        sl = slice(int(indptr[origin]), int(indptr[origin + 1]))
        ref_ids, ref_dist = refs[origin]
        got_ids = ids_all[sl]
        got_dist = dist_all[sl]
        assert got_ids.tobytes() == ref_ids.tobytes(), \
            (name, origin, "candidate id order/content")
        assert got_dist.tobytes() == ref_dist.tobytes(), \
            (name, origin, "candidate distance bits")


def test_slices_strictly_ascending_and_in_range(tr):
    """Dossier 04: every slice is strictly ascending ORIGINAL
    destination ids within [0, dest_count)."""
    case = admitted_case("same_terminal")
    rec = build(tr, case)
    indptr = np.asarray(rec.indptr)
    ids_all = np.asarray(rec.destination_ids)
    for origin in range(rec.origin_count):
        sl = slice(int(indptr[origin]), int(indptr[origin + 1]))
        c = ids_all[sl]
        if c.shape[0] > 1:
            assert int(np.diff(c).min()) > 0
    assert ids_all.shape[0] == 0 or int(ids_all.min()) >= 0
    assert ids_all.shape[0] == 0 \
        or int(ids_all.max()) < rec.dest_count


def test_metric_variations_replay_from_one_trace(tr, sp, baseline):
    """One trace serves every metric variation of the same query: the
    fold parameters and the destination METRIC weights are not trace
    identity; each replay is bitwise the corresponding full run."""
    case = admitted_case("base")
    rec = build(tr, case)
    assert rec.complete
    variations = [
        dict(gravity_beta=2.5),
        dict(gravity_plateau=0.5),
        dict(gravity_logistic_midpoint=1.25),
        dict(gravity_growth_rate=0.4),
        dict(knn_decay="logistic"),
        dict(knn_decay="other"),
    ]
    for kw in variations:
        args = driver_args(case, knn_decay=kw.get("knn_decay",
                                                  "exponential"))
        args = args[:9] + (
            kw.get("gravity_beta", 1.5),
            kw.get("gravity_plateau", 2.0),
            kw.get("gravity_logistic_midpoint", 3.0),
            kw.get("gravity_growth_rate", 0.8),
        ) + args[13:]
        ref = baseline(*args)
        rp = tr.replay_trace(
            rec, case["dwt"], kw.get("gravity_beta", 1.5),
            kw.get("gravity_plateau", 2.0),
            kw.get("gravity_logistic_midpoint", 3.0),
            kw.get("gravity_growth_rate", 0.8),
            kw.get("knn_decay", "exponential"), METRIC_PARAMS[4],
            case["cutoff"])
        assert result_equal(rp, ref), kw


def test_destination_metric_weights_replay(tr, sp, baseline):
    """Changed destination METRIC weights: same trace (TraceID does not
    embed MetricID), fresh replay bitwise-equal to the full run with
    those weights."""
    case = admitted_case("base")
    rec = build(tr, case)
    d_weights_b = case["dwt"]
    d_weights_c = np.array([1.7, 0.2, 0.9, 1.1, 0.05, 0.6])
    assert d_weights_c.shape == d_weights_b.shape
    args = list(driver_args(case))
    args[8] = d_weights_c
    ref = baseline(*args)
    rp = replay(tr, rec, case, d_weights=d_weights_c)
    assert result_equal(rp, ref)
    args[8] = d_weights_b
    ref_b = baseline(*args)
    rp_b = replay(tr, rec, case, d_weights=d_weights_b)
    assert result_equal(rp_b, ref_b)
    assert not result_equal(rp, ref_b) or np.array_equal(
        d_weights_b, d_weights_c)


def test_build_repeat_determinism(tr):
    """Two builds of the same query produce the same trace_id and
    bytewise-identical packed arrays (exact-repeat determinism)."""
    case = admitted_case("same_terminal")
    r1 = build(tr, case)
    r2 = build(tr, case)
    assert r1.trace_id == r2.trace_id
    assert packed_equal(r1, r2)
    assert r1.complete and r2.complete


def test_cohort_rows_are_distinct(tr):
    """Distinct parameter jobs carry distinct traces and identities."""
    a = build(tr, admitted_case("base"))
    b = build(tr, admitted_case("neg_seed"))
    c = build(tr, admitted_case("zero_cutoff"))
    assert len({a.trace_id, b.trace_id, c.trace_id}) == 3
    assert not packed_equal(a, b) and not packed_equal(a, c)


def test_fresh_output_isolation(tr):
    """Each replay receives fresh correctly typed arrays; mutating one
    replay's outputs affects nothing else; the trace's packed arrays
    are unchanged by replaying."""
    case = admitted_case("base")
    rec = build(tr, case)
    before = (np.asarray(rec.indptr).tobytes(),
              np.asarray(rec.destination_ids).tobytes(),
              np.asarray(rec.distance_bits).tobytes())
    r1 = replay(tr, rec, case)
    r2 = replay(tr, rec, case)
    saved = {f: bits(getattr(r2, f)) for f in RESULT_FIELDS}
    for f in RESULT_FIELDS:
        x = getattr(r1, f)
        assert x is not getattr(r2, f)
        x[0] = -1.5
    r3 = replay(tr, rec, case)
    for f in RESULT_FIELDS:
        assert bits(getattr(r3, f)) == saved[f], f
    after = (np.asarray(rec.indptr).tobytes(),
             np.asarray(rec.destination_ids).tobytes(),
             np.asarray(rec.distance_bits).tobytes())
    assert before == after


def test_medium_ring_graph_slices_and_replay(tr, sp, baseline):
    """Beyond the toy battery: a medium ring graph with random extra
    arcs — trace/replay stays bitwise against S11 and the baseline,
    slices ascending, and the trace carries honest counters."""
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
    args = driver_args(case)
    ref = baseline(*args)
    s11 = sp.sparse_scope_access(*args)
    assert s11.route == sp.ROUTE_SPARSE
    rec = build(tr, case)
    assert rec.complete and rec.route == tr.ROUTE_TRACE
    rp = replay(tr, rec, case)
    assert result_equal(rp, ref)
    assert result_equal(rp, s11)
    indptr = np.asarray(rec.indptr)
    ids_all = np.asarray(rec.destination_ids)
    for origin in range(rec.origin_count):
        sl = slice(int(indptr[origin]), int(indptr[origin + 1]))
        c = ids_all[sl]
        if c.shape[0] > 1:
            assert int(np.diff(c).min()) > 0
    assert rec.counters.candidates_total <= 40 * d_count
