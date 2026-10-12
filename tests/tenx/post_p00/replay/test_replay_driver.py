"""P03C compiled outer replay: bit parity + law battery (campaign
una_10x).

Every case compares the private compiled whole-row driver against
BOTH references the dossier names — the preserved pre-P03C Python
origin loop AND the original integrated engine driver — by raw output
bits.  There is no allclose anywhere in this suite.  The compiled
switch is a module-private seam: tests flip it in-process only
(monkeypatch), never through any public selector, and it is restored
(True) after every use.
"""
import dataclasses

import numpy as np

DECAYS = ("exponential", "logistic", "linear")


def _run_both(tr, tch, monkeypatch, rec, case, decay="exponential",
              d_weights=None, cut=None):
    """Replay the same record through both row loops (compiled last so
    the restored default is the one that ran last)."""
    out = {}
    for compiled in (True, False):
        monkeypatch.setattr(tr, "_REPLAY_COMPILED", compiled)
        out[compiled] = tch.replay(tr, rec, case, knn_decay=decay,
                                   d_weights=d_weights, cutoff=cut)
    monkeypatch.setattr(tr, "_REPLAY_COMPILED", True)
    return out[True], out[False]


def _replay_direct(tr, tch, monkeypatch, rec, dwt, decay="exponential"):
    """replay_trace over a hand-packed record with the suite metric
    bundle (METRIC_PARAMS, never retyped) and the record's own cutoff
    identity."""
    cutoff = rec.cutoff_value
    out = {}
    for compiled in (True, False):
        monkeypatch.setattr(tr, "_REPLAY_COMPILED", compiled)
        out[compiled] = tr.replay_trace(
            rec, dwt, *tch.METRIC_PARAMS[:4], decay,
            tch.METRIC_PARAMS[4], cutoff)
    monkeypatch.setattr(tr, "_REPLAY_COMPILED", True)
    return out[True], out[False]


def test_three_way_ring_battery(tr, baseline, tch, monkeypatch):
    """Ring battery: compiled == python reference == integrated engine,
    bitwise, across all decay branches; the fold count equals all
    requested origins."""
    for decay in DECAYS:
        for seed in (23, 24):
            case = tch.ring_case(v=120, extra=60, origins=17,
                                 d_count=40, cutoff=6.0, seed=seed)
            args = tch.driver_args(case, knn_decay=decay)
            b1 = baseline(*args)
            b2 = baseline(*args)
            assert all(tch.bits(x) == tch.bits(y)
                       for x, y in zip(b1, b2)), "baseline jitter"
            rec = tch.build(tr, case)
            assert rec.complete and rec.route == tr.ROUTE_TRACE
            rp_c, rp_p = _run_both(tr, tch, monkeypatch, rec, case,
                                   decay=decay)
            assert tch.outbits(rp_c) == tch.outbits(rp_p)
            assert tch.result_equal(rp_c, b1)
            oc = rec.origin_count
            assert getattr(rp_c, tch.FOLD_FIELD) == oc
            assert getattr(rp_p, tch.FOLD_FIELD) == oc
    assert tr._REPLAY_COMPILED is True


def test_int_cutoff_identity(tr, baseline, tch, monkeypatch):
    """The typed cutoff identity survives the compiled caller: an int
    cutoff specializes int64, replays bitwise against both references,
    and a float cutoff against the same record refuses by identity."""
    case = tch.ring_case(v=90, extra=40, origins=9, d_count=25,
                         cutoff=7, seed=31)
    rec = tch.build(tr, case)
    assert rec.cutoff_dtype == "int64"
    args = tch.driver_args(case)
    b1 = baseline(*args)
    rp_c, rp_p = _run_both(tr, tch, monkeypatch, rec, case)
    assert tch.outbits(rp_c) == tch.outbits(rp_p)
    assert tch.result_equal(rp_c, b1)
    # float cutoff: same value bits, different dtype identity -> refuse
    out = {}
    for compiled in (True, False):
        monkeypatch.setattr(tr, "_REPLAY_COMPILED", compiled)
        out[compiled] = tch.replay(tr, rec, case, cutoff=7.0)
    monkeypatch.setattr(tr, "_REPLAY_COMPILED", True)
    for rp in out.values():
        assert rp.route == "trace_replay_refused"
        assert rp.refusal == "cutoff_mismatch_with_trace_identity"
    assert tch.outbits(out[True]) == tch.outbits(out[False])


def test_direct_boundary_battery(tr, tch, monkeypatch):
    """Hand-packed records: slice lengths 0..17 (0 and 17 included),
    duplicate distance bits (argsort tie order), one retained distance
    exactly at the cutoff boundary, nextafter and zero cutoffs, a
    float32 metric record (downcast rounding) and an all-empty trace —
    compiled == python reference on every surface."""
    base = tch.ring_case(v=64, extra=30, origins=5, d_count=24,
                         cutoff=5.0, seed=11)
    rec0 = tch.build(tr, base)
    dest_count = rec0.dest_count
    dwt = base["dwt"]
    lengths = [0, 1, 2, 3, 4, 5, 8, 16, 17, 6]
    indptr = np.zeros(len(lengths) + 1, dtype=np.int64)
    indptr[1:] = np.cumsum(lengths)
    nnz = int(indptr[-1])
    rng = np.random.default_rng(7)
    # producer contract: strictly ascending ids per slice
    ids = np.zeros(nnz, dtype=np.int64)
    for i, ln in enumerate(lengths):
        lo, hi = int(indptr[i]), int(indptr[i + 1])
        ids[lo:hi] = np.sort(rng.choice(dest_count, ln, replace=False))
    dist = np.round(rng.uniform(0.0, 10.0, nnz), 3)
    if lengths[2] >= 2:                    # duplicate bits: tie order
        dist[int(indptr[2]):int(indptr[2]) + 2] = 3.5
    first = int(indptr[0]) if lengths[0] else int(indptr[1])
    dist[first] = 5.0                      # the <= boundary, retained
    for cutoff in (5.0, np.nextafter(5.0, 1.0), 0.0):
        rec = tch.direct_record(rec0, indptr, ids, dist, cutoff)
        rp_c, rp_p = _replay_direct(tr, tch, monkeypatch, rec, dwt)
        assert tch.outbits(rp_c) == tch.outbits(rp_p)
        assert getattr(rp_c, tch.FOLD_FIELD) == len(lengths)
    rec32 = tch.direct_record(rec0, indptr, ids, dist, 5.0,
                              metric_dtype="float32")
    rp_c, rp_p = _replay_direct(tr, tch, monkeypatch, rec32, dwt)
    assert tch.outbits(rp_c) == tch.outbits(rp_p)
    assert rp_c.gravity_exponential.dtype == np.float32
    # all-empty trace: folds still equal all requested origins
    zero_ids = np.zeros(0, np.int64)
    zero_dist = np.zeros(0, np.float64)
    zero_indptr = np.zeros_like(indptr)   # indptr[-1] == nnz == 0
    recz = tch.direct_record(rec0, zero_indptr, zero_ids, zero_dist,
                             5.0)
    rp_c, rp_p = _replay_direct(tr, tch, monkeypatch, recz, dwt)
    assert tch.outbits(rp_c) == tch.outbits(rp_p)
    assert getattr(rp_c, tch.FOLD_FIELD) == len(lengths)
    assert all(np.count_nonzero(getattr(rp_c, f)) == 0
               for f in tch.RESULT_FIELDS)


def _descending_ids(rng, dest_count, lengths, which):
    """Packed layout with DISTINCT ids in DESCENDING order inside one
    slice (structure gates pass; the slice-order check must catch)."""
    indptr = np.zeros(len(lengths) + 1, dtype=np.int64)
    indptr[1:] = np.cumsum(lengths)
    nnz = int(indptr[-1])
    ids = np.zeros(nnz, dtype=np.int64)
    for i, ln in enumerate(lengths):
        lo, hi = int(indptr[i]), int(indptr[i + 1])
        ids[lo:hi] = np.sort(rng.choice(dest_count, ln, replace=False))
    lo = int(indptr[which])
    hi = int(indptr[which + 1])
    if hi - lo >= 2:
        ids[lo:hi] = np.sort(ids[lo:hi])[::-1]  # descending, distinct
    return indptr, ids


def test_slice_order_corruption_refused_identically(
        tr, tch, monkeypatch):
    base = tch.ring_case(v=64, extra=30, origins=5, d_count=40,
                         cutoff=5.0, seed=13)
    rec0 = tch.build(tr, base)
    dwt = base["dwt"]
    rng = np.random.default_rng(9)
    lengths = [3, 4, 5, 2, 6]
    for which in (0, len(lengths) // 2, len(lengths) - 1):
        if lengths[which] < 2:
            continue
        indptr, ids = _descending_ids(rng, rec0.dest_count, lengths,
                                      which)
        dist = np.abs(rng.uniform(0.5, 9.0, int(indptr[-1])))
        rec = tch.direct_record(rec0, indptr, ids, dist, 5.0)
        rp_c, rp_p = _replay_direct(tr, tch, monkeypatch, rec, dwt)
        assert tch.outbits(rp_c) == tch.outbits(rp_p)
        assert rp_c.route == "trace_replay_refused"
        assert rp_c.refusal == "corrupt_trace_slice_order"
        assert getattr(rp_c, tch.FOLD_FIELD) == 0
        assert all(getattr(rp_c, f).shape[0] == 0
                   for f in tch.RESULT_FIELDS)


def test_gates_not_skipped_on_driver_path(tr, tch, monkeypatch):
    """Digest, range and structure gates run on EVERY replay call on
    the compiled path — address reuse is not validation."""
    base = tch.ring_case(v=64, extra=30, origins=5, d_count=40,
                         cutoff=5.0, seed=17)
    rec0 = tch.build(tr, base)
    dwt = base["dwt"]
    rng = np.random.default_rng(11)
    lengths = [0, 2, 4, 1, 8]
    indptr = np.zeros(len(lengths) + 1, dtype=np.int64)
    indptr[1:] = np.cumsum(lengths)
    nnz = int(indptr[-1])
    ids = np.zeros(nnz, dtype=np.int64)
    for i, ln in enumerate(lengths):
        lo, hi = int(indptr[i]), int(indptr[i + 1])
        ids[lo:hi] = np.sort(rng.choice(rec0.dest_count, ln,
                                        replace=False))
    dist = np.abs(rng.uniform(0.5, 9.0, nnz))

    def pack(i_, ids_, dist_):
        return tch.direct_record(rec0, i_, ids_, dist_, 5.0)

    # (a) wrong digest on an otherwise valid record
    rec = pack(indptr, ids, dist)
    rec = dataclasses.replace(rec, content_digest="0" * 64)
    rp_c, rp_p = _replay_direct(tr, tch, monkeypatch, rec, dwt)
    assert tch.outbits(rp_c) == tch.outbits(rp_p)
    assert rp_c.refusal == "corrupt_trace_content_digest"
    # (b) destination id == dest_count (out of range)
    bad = ids.copy()
    if nnz:
        bad[nnz // 2] = rec0.dest_count
    rec = pack(indptr, bad, dist)
    rp_c, rp_p = _replay_direct(tr, tch, monkeypatch, rec, dwt)
    assert rp_c.refusal == "corrupt_trace_destination_ids"
    assert tch.outbits(rp_c) == tch.outbits(rp_p)
    # (c) non-int64 ids dtype -> structure gate (before the driver)
    rec = pack(indptr, ids.astype(np.int32), dist)
    rp_c, rp_p = _replay_direct(tr, tch, monkeypatch, rec, dwt)
    assert rp_c.refusal == "corrupt_trace_structure"
    assert tch.outbits(rp_c) == tch.outbits(rp_p)
    # (d) non-monotonic indptr (negative count) -> structure gate
    badp = indptr.copy()
    badp[-1] = badp[-2] - 1
    rec = pack(badp, ids[:-1] if nnz else ids, dist[:-1] if nnz else dist)
    rp_c, rp_p = _replay_direct(tr, tch, monkeypatch, rec, dwt)
    assert rp_c.refusal == "corrupt_trace_structure"
    assert tch.outbits(rp_c) == tch.outbits(rp_p)


def test_output_freshness_and_repeat(tr, tch, monkeypatch):
    """Fresh output arrays per replay: a mutated returned array never
    poisons the next call; repeated replays of one record are bitwise
    identical (digest re-verified each call)."""
    case = tch.ring_case(v=90, extra=40, origins=11, d_count=25,
                         cutoff=6.0, seed=19)
    rec = tch.build(tr, case)
    rp1 = tch.replay(tr, rec, case)
    bits1 = tch.outbits(rp1)
    rp2 = tch.replay(tr, rec, case)
    assert tch.outbits(rp2) == bits1
    rp1.reach[0] += 1     # reach rides the int64 reach_dtype record law
    rp3 = tch.replay(tr, rec, case)
    assert tch.outbits(rp3) == bits1
    # compiled path: same laws
    rp_c, rp_p = _run_both(tr, tch, monkeypatch, rec, case)
    assert tch.outbits(rp_c) == tch.outbits(rp_p) == bits1


def test_zero_origin_record(tr, tch, monkeypatch):
    """An origin-less record replays to empty outputs with fold count
    0 on both paths."""
    base = tch.ring_case(v=64, extra=30, origins=5, d_count=24,
                         cutoff=5.0, seed=23)
    rec0 = tch.build(tr, base)
    dwt = base["dwt"]
    indptr = np.zeros(1, dtype=np.int64)
    rec = tch.direct_record(rec0, indptr, np.zeros(0, np.int64),
                            np.zeros(0, np.float64), 5.0)
    rp_c, rp_p = _replay_direct(tr, tch, monkeypatch, rec, dwt)
    assert tch.outbits(rp_c) == tch.outbits(rp_p)
    assert rp_c.route == "trace_replay"
    assert getattr(rp_c, tch.FOLD_FIELD) == 0
    assert all(getattr(rp_c, f).shape[0] == 0
               for f in tch.RESULT_FIELDS)


def test_driver_shape_is_serial_nogil(tr, tch):
    """The private driver is compiled serial/nogil with no fastmath of
    its own (all arithmetic lives in the original full fold), the
    python reference stays importable beside it, and the module
    default is the compiled path."""
    assert tr._REPLAY_COMPILED is True
    to = tr._replay_rows_compiled.targetoptions
    assert to.get("nogil") is True
    assert to.get("parallel") is False
    assert not to.get("fastmath", False)
    assert callable(tr._replay_rows_python)
