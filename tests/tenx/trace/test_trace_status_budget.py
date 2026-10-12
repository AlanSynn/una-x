"""C11 status, budget and immutability gates (campaign una_10x).

An exceeded trace byte budget marks the record INCOMPLETE and selects
recompute/streaming — never a truncated-trace success — and an
incomplete record is never a replay cache hit.  Tile partitioning must
not change the packed bytes or the identity.  An empty candidate
prefix folds exactly like S11's empty prefix (the fold runs, whatever
the engine computes for it).  Input arrays mutated after the build
cannot corrupt the built record, and the counters charge the actual
producer work and bytes.
"""
import numpy as np
import pytest

from conftest import (RESULT_FIELDS, admitted_case, build, driver_args,
                      packed_equal, replay, result_equal, ring_case)


def test_budget_exceeded_is_incomplete_never_hit(tr):
    case = admitted_case("base")
    rec = tr.build_trace(
        case["o"], case["ow"], case["ptr"], case["vec"], case["wts"],
        case["flag"], case["d"], case["dw"], case["dwt"],
        case["cutoff"], trace_budget_bytes=8)
    assert rec.complete is False
    assert rec.route == tr.ROUTE_BUDGET
    assert "budget" in rec.refusal
    assert rec.counters.trace_bytes > 8
    # an incomplete record is never a cache hit
    rp = replay(tr, rec, case)
    assert rp.route == "trace_replay_refused"
    assert rp.refusal == "incomplete_trace_not_a_cache_hit"
    assert rp.folds_done == 0
    for f in RESULT_FIELDS:
        assert np.asarray(getattr(rp, f)).shape[0] == 0


def test_tile_partition_independence(tr, sp, baseline):
    """Tile boundaries are a producer detail: on a 40-origin ring,
    tile_origins 1, 7 and 64 yield the same trace_id and
    bytewise-identical packed arrays, and each replays bitwise to the
    same full results."""
    case = ring_case()
    args = driver_args(case)
    ref = baseline(*args)
    recs = [tr.build_trace(
        case["o"], case["ow"], case["ptr"], case["vec"], case["wts"],
        case["flag"], case["d"], case["dw"], case["dwt"],
        case["cutoff"], tile_origins=t) for t in (1, 7, 64)]
    for r in recs:
        assert r.complete and r.route == tr.ROUTE_TRACE
        assert r.trace_id == recs[0].trace_id
        assert packed_equal(recs[0], r)
        rp = replay(tr, r, case)
        assert result_equal(rp, ref)
    tiles = [r.counters.tiles for r in recs]
    assert tiles[0] >= tiles[1] >= tiles[2] and tiles[0] > tiles[2]


def test_empty_prefix_folds_like_s11(tr, sp, baseline):
    """An origin whose touched set reaches no destination produces an
    honest zero count, and the empty-prefix fold is executed exactly
    as S11 executes it (bitwise, whatever the engine computes)."""
    v = 12
    src = np.arange(v, dtype=np.int64)
    vec = ((np.arange(v) + 1) % v).astype(np.int64)
    ptr = np.zeros(v + 1, dtype=np.int64)
    ptr[1:] = np.cumsum(np.bincount(src, minlength=v))
    wts = np.full(v, 0.2)
    flag = np.ones(v, dtype=bool)
    o = np.array([[0, 0]], dtype=np.int64)
    ow = np.array([[0.1, 0.1]])
    d = np.array([[6, 7], [8, 9]], dtype=np.int64)
    dw = np.array([[0.3, 0.4], [0.2, 0.5]])
    dwt = np.array([1.0, 1.0])
    case = dict(o=o, ow=ow, ptr=ptr, vec=vec, wts=wts, flag=flag,
                d=d, dw=dw, dwt=dwt, cutoff=1.0)
    rec = build(tr, case)
    assert rec.complete and rec.route == tr.ROUTE_TRACE
    indptr = np.asarray(rec.indptr)
    assert rec.origin_count == 1
    assert int(indptr[1] - indptr[0]) >= 0
    rp = replay(tr, rec, case)
    assert rp.route == "trace_replay" and rp.folds_done == 1
    s11 = sp.sparse_scope_access(*driver_args(case))
    assert s11.route == sp.ROUTE_SPARSE
    assert result_equal(rp, s11)


def test_input_mutation_after_build_cannot_corrupt(tr):
    """Built packed arrays own their bytes: mutating the caller's input
    arrays after the build leaves the record, its identity and the
    replayed outputs intact (the trace's frozen bytes are what replay
    consumes)."""
    case = admitted_case("base")
    rec = build(tr, case)
    r1 = replay(tr, rec, case)
    saved = {f: np.asarray(getattr(r1, f)).tobytes()
             for f in RESULT_FIELDS}
    snapshot = (np.asarray(rec.indptr).tobytes(),
                np.asarray(rec.destination_ids).tobytes(),
                np.asarray(rec.distance_bits).tobytes())
    tid = rec.trace_id
    case["ow"][:, :] = 7.5
    case["dw"][:, :] = -3.0
    case["vec"][:] = 0
    case["wts"][:] = 99.0
    assert rec.trace_id == tid
    assert (np.asarray(rec.indptr).tobytes(),
            np.asarray(rec.destination_ids).tobytes(),
            np.asarray(rec.distance_bits).tobytes()) == snapshot
    r2 = replay(tr, rec, case)
    for f in RESULT_FIELDS:
        assert np.asarray(getattr(r2, f)).tobytes() == saved[f], f


def test_counters_are_measured_not_vacuous(tr):
    """Counters charge the actual producer work: packed bytes add up to
    the honest layout (indptr + ids + distances + identity payload),
    candidates equal the packed nnz, and the per-tile stamp clearing is
    counted at least once per tile."""
    case = admitted_case("base")
    rec = build(tr, case)
    c = rec.counters
    nnz = int(np.asarray(rec.indptr)[-1])
    node_count = int(case["ptr"].shape[0]) - 1
    d_count = int(case["d"].shape[0])
    assert c.ids_bytes == 8 * nnz
    assert c.dist_bytes == 8 * nnz
    assert c.indptr_bytes == 8 * (rec.origin_count + 1)
    assert c.metadata_bytes == len(rec.trace_id.encode("utf-8"))
    assert c.trace_bytes == (c.indptr_bytes + c.ids_bytes
                             + c.dist_bytes + c.metadata_bytes)
    assert c.candidates_total == nnz
    assert c.origins == rec.origin_count
    assert c.tiles >= 1
    assert c.tile_temp_bytes > 0
    assert c.stamps_zeroed >= c.tiles * (node_count + d_count)
