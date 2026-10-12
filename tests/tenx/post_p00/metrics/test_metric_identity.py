"""P03M typed MetricID v2 + admitted partial metric reuse (campaign
una_10x).

Laws under test (dossier 04):

* Identity is a dependency proof, not a parameter name: a field id
  changes when a CONSUMED input changes one bit (dtype-only, layout-
  only, writability-only, alignment-only, 0/-0, NaN payload,
  int-vs-float) and is INVARIANT under inputs the fold never reads
  on that branch — asserted here by the full sensitivity matrix.
* The cohort v1 ``result_key`` float64-coerces weights and stays a
  REPORTING key (C31-NOTE-1 / errata E8): the collision it collapses
  is demonstrated, and v2 is proven to separate it.
* The admitted ge-only split is bit-identical to the full compiled
  fold — never allclose — across decay branches, metric dtypes,
  int cutoffs, boundary slices and corrupt records (identical
  refusals); only a bit-identical split is admitted at all.
* The metric field store is bounded, immutable inside, fresh mutable
  outside, runtime-only (detach law); the process registry is
  bounded; the public route engages through RunBatch and stays
  bitwise vs the serial route.
"""
import hashlib
import json
import struct

import numpy as np
import pytest

import urban_network_analysis.Engines._trace_access_scratch as tr
from urban_network_analysis.Engines import _metric_identity as mrid
from urban_network_analysis.batch.metric_reuse import MetricFieldStore
from urban_network_analysis.cache import identity as _ckid

DECAYS = ("exponential", "logistic", "linear")

# The suite metric bundle (mirrors METRIC_PARAMS so the split parity
# below compares against the exact folds the replay suite pins).
BETA, PLATEAU, MIDPOINT, GROWTH = 1.5, 2.0, 3.0, 0.8


# ----------------------------------------------------------------------
# encoding laws


def test_scalar_encoding_laws():
    """Type tag + raw bytes: 0/-0, NaN payloads, int-vs-float, bool
    separation, dtype-width separation; str/complex/None/extended
    precision are refused, never coerced."""
    assert mrid.scalar_part(0.0) != mrid.scalar_part(-0.0)
    nan_a = mrid.scalar_part(float("nan"))
    nan_b = mrid.scalar_part(
        struct.unpack("<d", struct.pack("<Q",
                                        0x7FF8000000000001))[0])
    assert nan_a != nan_b                       # payload bits ride
    assert mrid.scalar_part(1) != mrid.scalar_part(1.0)
    assert mrid.scalar_part(1) != mrid.scalar_part(np.int64(1))
    assert mrid.scalar_part(1) != mrid.scalar_part(True)
    assert mrid.scalar_part(np.float32(1.5)) != \
        mrid.scalar_part(np.float64(1.5))
    assert mrid.scalar_part(1.5) == mrid.scalar_part(np.float64(1.5))
    assert mrid.scalar_part(np.int64(-2)) == \
        mrid.scalar_part(np.int64(-2))
    for bad in ("1.5", complex(1, 2), None, b"1.5", object()):
        with pytest.raises(mrid.EncodingRefused):
            mrid.scalar_part(bad)
    if np.dtype(np.longdouble).itemsize > 8:    # x86_64: float128
        with pytest.raises(mrid.EncodingRefused):
            mrid.scalar_part(np.longdouble(1.5))
    with pytest.raises(mrid.EncodingRefused):
        mrid.scalar_part(1 << 64)               # outside int64 domain


def test_array_encoding_laws():
    """The weight vector encodes raw dtype / shape / layout flags /
    writability / 8-byte alignment class / payload bytes — every
    single-bit class change is visible in the encoding."""
    a = np.array([1.0, 2.0, 3.0])
    base = mrid.array_part(a)
    # dtype-only
    assert mrid.array_part(a.astype("<f4")) != base
    # byteorder-only (same values, opposite byte order)
    assert mrid.array_part(
        a.astype(a.dtype.newbyteorder(">")))["dtype"] != base["dtype"]
    # layout-only: same logical bytes, reversed iteration
    rev = np.array([3.0, 2.0, 1.0])[::-1]
    assert rev.tobytes() == a.tobytes()
    assert mrid.array_part(rev) != base
    assert mrid.array_part(rev)["c_contiguous"] is False
    # writability-only
    ro = a.view()
    ro.setflags(write=False)
    assert mrid.array_part(ro) != base
    assert mrid.array_part(ro)["writeable"] is False
    # alignment-only: same bytes/dtype/layout/writability, ptr % 8 == 1
    back = np.zeros(a.nbytes + 1, dtype=np.uint8)
    back[1:] = a.view(np.uint8)
    mis = back[1:].view(a.dtype)
    assert mis.tobytes() == a.tobytes()
    assert mis.flags.c_contiguous and mis.flags.writeable
    assert mrid.array_part(mis) != base
    assert mrid.array_part(mis)["aligned8"] is False
    # shape and content
    assert mrid.array_part(np.array([1.0, 2.0])) != base
    assert mrid.array_part(np.array([1.0, 2.0, 4.0])) != base
    # refusals
    with pytest.raises(mrid.EncodingRefused):
        mrid.array_part(a.reshape(1, 3))
    with pytest.raises(mrid.EncodingRefused):
        mrid.array_part([1.0, 2.0, 3.0])
    # weights_digest is the canonical one-sha form of array_part
    want = hashlib.sha256(json.dumps(
        base, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    assert mrid.weights_digest(a) == want
    assert mrid.weights_digest(a) != mrid.weights_digest(np.ones(3))


# ----------------------------------------------------------------------
# dependency matrix


def _base_args():
    return dict(
        trace_id="tid-1", trace_schema="una-10x-c11-trace-1",
        cutoff=6.0, weights_sha="w" * 64, reach_dtype="int64",
        metric_dtype="<f8", beta=BETA, plateau=PLATEAU,
        midpoint=MIDPOINT, growth_rate=GROWTH, knn_decay="logistic",
        knn_weights_part=mrid.array_part(np.array([1.0, 1.0, 0.5])))


_PERTURB = {
    "trace_id": "tid-2",
    "trace_schema": "una-10x-c11-trace-2",
    "cutoff": 7.0,
    "weights_sha": "x" * 64,
    "reach_dtype": "int32",
    "metric_dtype": "<f4",
    "beta": 2.5,
    "plateau": 0.5,
    "midpoint": 4.0,
    "growth_rate": 0.9,
    "knn_decay": "exponential",      # linear-based runs use "linear"
    "knn_weights_part": mrid.array_part(np.array([1.0, 1.0, 0.75])),
}

# sensitive = the closure metric_field_id actually encodes per kind
# (each entry cites the consuming fold line in _metric_identity);
# everything else MUST leave the id untouched.
_SENSITIVE = {
    "reach": {"trace_id", "trace_schema", "cutoff", "weights_sha",
              "reach_dtype", "metric_dtype"},
    "gravity_exponential": {"trace_id", "trace_schema", "cutoff",
                            "weights_sha", "reach_dtype",
                            "metric_dtype", "beta", "plateau"},
    "gravity_logistic": {"trace_id", "trace_schema", "cutoff",
                         "weights_sha", "reach_dtype", "metric_dtype",
                         "plateau", "midpoint", "growth_rate"},
    "knn_access": {"trace_id", "trace_schema", "cutoff", "weights_sha",
                   "reach_dtype", "metric_dtype", "knn_decay",
                   "knn_weights_part", "plateau", "midpoint",
                   "growth_rate"},        # logistic branch base
    "group": {"trace_id", "trace_schema", "cutoff", "weights_sha",
              "reach_dtype", "metric_dtype", "beta", "plateau",
              "midpoint", "growth_rate", "knn_decay",
              "knn_weights_part"},
}
_KNOWN = ("trace_id", "trace_schema", "cutoff", "weights_sha",
          "reach_dtype", "metric_dtype", "beta", "plateau", "midpoint",
          "growth_rate", "knn_decay", "knn_weights_part")


def test_dependency_matrix_sensitivity_and_invariance():
    """For every admitted kind: a consumed input's single change moves
    the id, every unconsumed input's change does NOT (deterministic on
    repeat)."""
    args = _base_args()
    fid0 = mrid.metric_field_id("knn_access", **args)
    assert fid0 == mrid.metric_field_id("knn_access", **args)
    for kind, sensitive in _SENSITIVE.items():
        ref = mrid.metric_field_id(kind, **args)
        for name in _KNOWN:
            pert = dict(args)
            pert[name] = _PERTURB[name]
            got = mrid.metric_field_id(kind, **pert)
            if name in sensitive:
                assert got != ref, (kind, name, "should differ")
            else:
                assert got == ref, (kind, name, "must not differ")


def test_knn_decay_branch_matrix():
    """Branch-conditioned closure of knn_access: exponential consumes
    beta+plateau (fold line 104), logistic plateau+midpoint+growth
    (line 106), any other branch no decay scalar (line 108)."""
    for decay in ("exponential", "logistic", "linear"):
        args = _base_args()
        args["knn_decay"] = decay
        ref = mrid.metric_field_id("knn_access", **args)
        if decay == "exponential":       # line 104: beta + plateau
            consumed = {"beta", "plateau"}
        elif decay == "logistic":        # line 106: plateau/mid/growth
            consumed = {"plateau", "midpoint", "growth_rate"}
        else:                            # line 108: no decay scalar
            consumed = set()
        for name in ("beta", "plateau", "midpoint", "growth_rate"):
            pert = dict(args)
            pert[name] = _PERTURB[name]
            changed = mrid.metric_field_id("knn_access", **pert) != ref
            if name in consumed:
                assert changed, (decay, name, "consumed -> must differ")
            else:
                assert not changed, (decay, name,
                                     "not consumed -> must not differ")


def test_v1_reporting_key_collides_v2_differs():
    """C31-NOTE-1 (errata E8): the cohort v1 result composition
    float64-coerces knn_weights/d_weights/scalars, so float32-vs-
    float64 bindings COLLIDE there; v2 (the only reuse identity)
    separates them by dtype bits."""
    knw32 = np.array([1.0, 1.0, 0.5], dtype=np.float32)
    knw64 = knw32.astype(np.float64)          # same values

    def v1(knw):
        return _ckid.stage_key("una:cohort:o_access:v1", {
            "trace_id": "tid",
            "gravity_beta": float(np.float32(BETA)),
            "gravity_plateau": float(PLATEAU),
            "gravity_logistic_midpoint": float(MIDPOINT),
            "gravity_decay_constant": float(GROWTH * MIDPOINT),
            "knn_decay": "logistic",
            "knn_weights": np.asarray(knw, dtype=np.float64),
            "search_radius": float(90.0),
            "d_weights": np.asarray(np.ones(6, np.float32),
                                    dtype=np.float64),
        })

    assert v1(knw32) == v1(knw64)             # the v1 hole, proven
    args32 = _base_args()
    args32["knn_weights_part"] = mrid.array_part(knw32)
    args64 = _base_args()
    args64["knn_weights_part"] = mrid.array_part(knw64)
    assert mrid.metric_field_id("group", **args32) != \
        mrid.metric_field_id("group", **args64)
    # scalar dtype separation too: a float32 beta coerced by v1
    args_b32 = _base_args()
    args_b32["beta"] = np.float32(BETA)
    args_b64 = _base_args()
    args_b64["beta"] = BETA
    assert float(np.float32(BETA)) == BETA
    assert mrid.metric_field_id("gravity_exponential", **args_b32) != \
        mrid.metric_field_id("gravity_exponential", **args_b64)
    # and the unknown kind is refused, not silently keyed
    with pytest.raises(mrid.EncodingRefused):
        mrid.metric_field_id("gravity", **args64)


# ----------------------------------------------------------------------
# admitted ge-only split: bit parity vs the full fold


def test_split_admission_flag_and_kernel_identity():
    """The split is module-private, flagged admitted, and compiled with
    the declared reduction plan (kernel fastmath, driver serial nogil
    without its own fastmath)."""
    assert tr._REPLAY_GE_SPLIT_ADMITTED is True
    kto = tr._ge_split_kernel.targetoptions
    assert kto.get("nogil") is True and kto.get("parallel") is False
    assert kto.get("fastmath") is True
    dto = tr._replay_ge_split_compiled.targetoptions
    assert dto.get("nogil") is True and dto.get("parallel") is False
    assert not dto.get("fastmath", False)


def test_split_bitparity_full_battery(tr, tch):
    """ge-split == full fold BY BITS (compiled reference AND preserved
    python-loop reference) across decay branches, a float32 metric
    record, an int-cutoff record, boundary slices, an empty trace and
    a zero-origin record; every non-ge field stays EMPTY (never
    fabricated); refusals are identical to the whole-row path."""
    for decay in DECAYS:
        case = tch.ring_case(v=120, extra=60, origins=17, d_count=40,
                             cutoff=6.0, seed=23)
        rec = tch.build(tr, case)
        dwt = case["dwt"]
        rge = tr.replay_trace_ge_split(rec, dwt, BETA, PLATEAU,
                                       case["cutoff"])
        assert rge.route == "trace_replay_ge_split"
        assert rge.refusal == ""
        assert all(np.asarray(getattr(rge, f)).shape[0] == 0
                   for f in ("reach", "gravity_logistic", "knn_access"))
        for compiled in (True, False):
            tr._REPLAY_COMPILED = compiled
            try:
                res = tch.replay(tr, rec, case, knn_decay=decay)
            finally:
                tr._REPLAY_COMPILED = True
            assert rge.gravity_exponential.tobytes() == \
                res.gravity_exponential.tobytes(), (decay, compiled)
        assert rge.folds_done == rec.origin_count
    assert tr._REPLAY_COMPILED is True

    # hand-packed surfaces over one base record
    base = tch.ring_case(v=64, extra=30, origins=5, d_count=24,
                         cutoff=5.0, seed=11)
    rec0 = tch.build(tr, base)
    dwt = base["dwt"]
    lengths = [0, 1, 2, 3, 4, 5, 8, 16, 17, 6]
    indptr = np.zeros(len(lengths) + 1, dtype=np.int64)
    indptr[1:] = np.cumsum(lengths)
    nnz = int(indptr[-1])
    rng = np.random.default_rng(7)
    ids = np.zeros(nnz, dtype=np.int64)
    for i, ln in enumerate(lengths):
        lo, hi = int(indptr[i]), int(indptr[i + 1])
        ids[lo:hi] = np.sort(rng.choice(rec0.dest_count, ln,
                                        replace=False))
    dist = np.round(rng.uniform(0.0, 10.0, nnz), 3)
    if lengths[2] >= 2:
        dist[int(indptr[2]):int(indptr[2]) + 2] = 3.5
    first = int(indptr[0]) if lengths[0] else int(indptr[1])
    dist[first] = 5.0

    def both(rec):
        rge = tr.replay_trace_ge_split(rec, dwt, BETA, PLATEAU,
                                       rec.cutoff_value)
        res = tr.replay_trace(rec, dwt, BETA, PLATEAU, MIDPOINT,
                              GROWTH, "logistic",
                              tch.METRIC_PARAMS[4],
                              rec.cutoff_value)
        return rge, res

    for cutoff in (5.0, np.nextafter(5.0, 1.0), 0.0):
        rec = tch.direct_record(rec0, indptr, ids, dist, cutoff)
        rge, res = both(rec)
        assert rge.gravity_exponential.tobytes() == \
            res.gravity_exponential.tobytes(), cutoff
    rec32 = tch.direct_record(rec0, indptr, ids, dist, 5.0,
                              metric_dtype="float32")
    rge, res = both(rec32)
    assert rge.gravity_exponential.dtype == np.float32
    assert rge.gravity_exponential.tobytes() == \
        res.gravity_exponential.tobytes()
    # int-cutoff identity: split accepts only the record's own dtype
    case_i = tch.ring_case(v=90, extra=40, origins=9, d_count=25,
                           cutoff=7, seed=31)
    rec_i = tch.build(tr, case_i)
    assert rec_i.cutoff_dtype == "int64"
    rge = tr.replay_trace_ge_split(rec_i, case_i["dwt"], BETA, PLATEAU,
                                   7)
    res = tr.replay_trace(rec_i, case_i["dwt"], BETA, PLATEAU,
                          MIDPOINT, GROWTH, "logistic",
                          tch.METRIC_PARAMS[4], 7)
    assert rge.gravity_exponential.tobytes() == \
        res.gravity_exponential.tobytes()
    bad = tr.replay_trace_ge_split(rec_i, case_i["dwt"], BETA, PLATEAU,
                                   7.0)
    assert bad.route == "trace_replay_refused"
    assert bad.refusal == "cutoff_mismatch_with_trace_identity"
    # empty trace and zero-origin record
    zero_indptr = np.zeros_like(indptr)       # all-empty: nnz == 0
    recz = tch.direct_record(rec0, zero_indptr, np.zeros(0, np.int64),
                             np.zeros(0, np.float64), 5.0)
    rge, res = both(recz)
    assert rge.gravity_exponential.tobytes() == \
        res.gravity_exponential.tobytes()
    assert rge.folds_done == recz.origin_count
    rec_zo = tch.direct_record(rec0, np.zeros(1, np.int64),
                               np.zeros(0, np.int64),
                               np.zeros(0, np.float64), 5.0)
    rge, res = both(rec_zo)
    assert rge.gravity_exponential.shape[0] == 0
    assert rge.folds_done == 0


def test_split_refusals_identical_to_whole_row(tr, tch):
    """Corrupt records (slice order, digest, range, dtype, structure)
    are refused with the SAME reason, the SAME empty outputs and
    folds_done 0 on both the split and the whole-row path."""
    base = tch.ring_case(v=64, extra=30, origins=5, d_count=40,
                         cutoff=5.0, seed=13)
    rec0 = tch.build(tr, base)
    dwt = base["dwt"]
    rng = np.random.default_rng(9)
    lengths = [3, 4, 5, 2, 6]
    indptr = np.zeros(len(lengths) + 1, dtype=np.int64)
    indptr[1:] = np.cumsum(lengths)
    nnz = int(indptr[-1])
    ids = np.zeros(nnz, dtype=np.int64)
    for i, ln in enumerate(lengths):
        lo, hi = int(indptr[i]), int(indptr[i + 1])
        ids[lo:hi] = np.sort(rng.choice(rec0.dest_count, ln,
                                        replace=False))
    dist = np.abs(rng.uniform(0.5, 9.0, nnz))
    # (a) descending ids inside one slice
    bad_ids = ids.copy()
    lo, hi = int(indptr[1]), int(indptr[2])
    bad_ids[lo:hi] = np.sort(bad_ids[lo:hi])[::-1]
    rec = tch.direct_record(rec0, indptr, bad_ids, dist, 5.0)
    rge = tr.replay_trace_ge_split(rec, dwt, BETA, PLATEAU, 5.0)
    res = tr.replay_trace(rec, dwt, BETA, PLATEAU, MIDPOINT, GROWTH,
                          "logistic", tch.METRIC_PARAMS[4], 5.0)
    assert rge.refusal == res.refusal == "corrupt_trace_slice_order"
    assert rge.route == res.route == "trace_replay_refused"
    assert rge.folds_done == res.folds_done == 0
    assert rge.gravity_exponential.shape[0] == 0
    # (b) wrong content digest
    rec = dataclasses_replace_digest(rec0, indptr, ids, dist)
    rge = tr.replay_trace_ge_split(rec, dwt, BETA, PLATEAU, 5.0)
    assert rge.refusal == "corrupt_trace_content_digest"
    # (c) out-of-range destination id
    bad = ids.copy()
    bad[nnz // 2] = rec0.dest_count
    rec = tch.direct_record(rec0, indptr, bad, dist, 5.0)
    rge = tr.replay_trace_ge_split(rec, dwt, BETA, PLATEAU, 5.0)
    assert rge.refusal == "corrupt_trace_destination_ids"
    # (d) non-int64 ids and (e) negative slice count -> structure
    rec = tch.direct_record(rec0, indptr, ids.astype(np.int32), dist,
                            5.0)
    rge = tr.replay_trace_ge_split(rec, dwt, BETA, PLATEAU, 5.0)
    assert rge.refusal == "corrupt_trace_structure"
    badp = indptr.copy()
    badp[-1] = badp[-2] - 1
    rec = tch.direct_record(rec0, badp, ids[:-1], dist[:-1], 5.0)
    rge = tr.replay_trace_ge_split(rec, dwt, BETA, PLATEAU, 5.0)
    assert rge.refusal == "corrupt_trace_structure"
    # (f) not a record / incomplete / weight domain
    rge = tr.replay_trace_ge_split(object(), dwt, BETA, PLATEAU, 5.0)
    assert rge.refusal == "not_a_trace_record"
    rge = tr.replay_trace_ge_split(rec0, dwt.reshape(-1, 1), BETA,
                                   PLATEAU, 5.0)
    assert rge.refusal == "dest_weights_layout"
    rge = tr.replay_trace_ge_split(
        rec0, np.append(dwt, -1.0), BETA, PLATEAU, 5.0)
    assert rge.refusal == "dest_count_mismatch"
    neg = dwt.copy()
    neg[0] = -1.0
    rge = tr.replay_trace_ge_split(rec0, neg, BETA, PLATEAU, 5.0)
    assert rge.refusal == "dest_weights_value_domain"


def dataclasses_replace_digest(rec0, indptr, ids, dist):
    """A record over a valid pack whose digest was clobbered."""
    import dataclasses
    return dataclasses.replace(rec0, indptr=indptr,
                               destination_ids=ids,
                               distance_bits=dist,
                               content_digest="0" * 64,
                               origin_count=int(indptr.shape[0]) - 1)


# ----------------------------------------------------------------------
# metric field store + registry laws


def test_metric_store_laws():
    """Immutable inside / fresh writable outside / LRU byte-bounded /
    nofit / detach law / counters, with no correctness surface."""
    store = MetricFieldStore(cap_bytes=1024)
    assert store.get("missing") is None
    assert store.counters["misses"] == 1
    a = np.arange(5, dtype=np.int64)
    assert store.put("id1", a) is True
    assert store.counters["produced"] == 1 and len(store) == 1
    g1 = store.get("id1")
    assert g1.tobytes() == a.tobytes() and g1.flags.writeable
    assert store.counters["hits"] == 1
    assert store.counters["copied_bytes"] == int(g1.nbytes)
    g1[0] += 12345                       # a served copy is owned
    g2 = store.get("id1")
    assert g2.tobytes() == a.tobytes()   # ... and never poisons
    assert g2 is not g1 and g2 is not store._entries["id1"]
    # put of the same id replaces, no eviction
    assert store.put("id1", a + 1) is True
    assert len(store) == 1 and store.counters["evictions"] == 0
    assert store.get("id1").tobytes() == (a + 1).tobytes()
    # LRU byte bound: 4 x 40 bytes in a 150-byte cap
    small = MetricFieldStore(cap_bytes=150)
    for i, kid in enumerate(("k1", "k2", "k3")):
        assert small.put(kid, np.full(5, i, np.int64)) is True
    warm = small.get("k1")               # k1 becomes most-recent
    assert warm is not None
    assert small.put("k4", np.zeros(5, np.int64)) is True
    assert len(small) == 3 and small.counters["evictions"] == 1
    assert small.get("k2") is None       # k2 was least-recent
    assert small.get("k1") is not None   # k1 survived by recency
    # nofit: bigger than the whole cap is never stored
    tiny = MetricFieldStore(cap_bytes=10)
    assert tiny.put("big", np.zeros(5)) is False
    assert tiny.counters["nofit"] == 1 and len(tiny) == 0
    # detach law
    with pytest.raises(TypeError):
        MetricFieldStore.__getstate__(store)


def test_metric_registry_bounded():
    """The process-level registry keeps at most _METRIC_STORE_MAX=4
    stores, evicts the oldest root, hands the SAME store back per root
    within the bound, and its stores carry the detach law too."""
    import tempfile

    from urban_network_analysis.batch import cohort as cohort_mod
    assert cohort_mod._METRIC_STORE_MAX == 4
    before_evictions = cohort_mod._METRIC_STORES_EVICTED[0]
    roots = [tempfile.mkdtemp(prefix="_p03m_reg_")
             for _ in range(cohort_mod._METRIC_STORE_MAX + 1)]
    got = {}
    for i, root in enumerate(roots):
        s = cohort_mod._metric_store_for(root, 1 << 20)
        assert isinstance(s, MetricFieldStore)
        assert s.cap_bytes == 1 << 20
        got[root] = s
        if i < cohort_mod._METRIC_STORE_MAX:
            assert cohort_mod._metric_store_for(root, 1 << 20) is s
    assert len(cohort_mod._METRIC_STORES) <= \
        cohort_mod._METRIC_STORE_MAX
    assert cohort_mod._METRIC_STORES_EVICTED[0] > before_evictions
    assert roots[0] not in cohort_mod._METRIC_STORES   # oldest gone
    assert roots[-1] in cohort_mod._METRIC_STORES      # newest kept
    with pytest.raises(TypeError):
        got[roots[-1]].__getstate__()


# ----------------------------------------------------------------------
# public route engagement


def test_public_route_engagement(make_batch, tmp_path):
    """Two same-radius / different-beta rows through the PUBLIC
    RunBatch: row 1 produces the trio from one full replay; row 2 hits
    the trio and runs the admitted ge-split; the parallel tree is
    bitwise vs the serial route (no tolerance anywhere)."""
    from conftest import tree_bytes

    def extra(_i, settings):
        settings.search_radius = 90      # one trace identity
        settings.gravity_beta = 0.5 if _i == 0 else 0.9

    ser = make_batch(2, out_folder=tmp_path / "ser", extra=extra)
    ser.RunBatch("accessibility")

    par = make_batch(2, out_folder=tmp_path / "par", extra=extra)
    par.RunBatch("accessibility", parallel=True, workers=1)

    extra_block = getattr(par.batch_report, "extra", None) or {}
    cs = extra_block.get("cohort")
    assert cs is not None, "cohort block missing"
    assert cs["rows"] == 2
    assert cs["refusals"] == {}, cs["refusals"]
    assert cs["distinct_trace_identities"] == 1
    assert cs["builds"] == 1
    assert cs["replays"] == 1
    assert cs["metric_field_hits"] == 3
    assert cs["metric_split_folds"] == 1
    assert cs["metric_field_produced"] == 3
    assert cs["metric_copied_bytes"] > 0
    assert cs["metric_fallbacks"] == 0
    assert cs["metric_errors"] == 0
    assert cs["folds_done"] > 0
    # the trio ids are beta-invariant and identical across both rows
    per_row = [st.get("metric_ids") or {}
               for _rid, st in sorted((cs.get("per_row") or {}).items())]
    assert len(per_row) == 2 and all(per_row)
    for kind in ("reach", "gravity_logistic", "knn_access"):
        assert per_row[0][kind] == per_row[1][kind], kind
    ser_tree = tree_bytes(tmp_path / "ser", skip_hidden=True)
    par_tree = tree_bytes(tmp_path / "par", skip_hidden=True)
    assert ser_tree == par_tree, sorted(
        k for k in set(ser_tree) | set(par_tree)
        if ser_tree.get(k) != par_tree.get(k))[:10]
