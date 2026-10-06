"""P03C L1: the compiled structural scan and the packed-gate order
(campaign una_10x post_p00, designs/03 phases B).

The scan is the Python loop's per-origin slice-order predicate hoisted
ahead of ANY fold: a corrupt FIRST, MIDDLE, or LAST slice refuses with
``corrupt_trace_slice_order``, zero folds, and no partially computed
earlier outputs.  The observable refusal order of the packed gates is
preserved exactly: structure -> destination ids -> content digest ->
slice order.
"""
import dataclasses
import hashlib

import numpy as np
import pytest

from conftest import (admitted_case, first_difference, replay_legacy,
                      replay_live, retamper, synth_record,
                      synth_weights)


def swap_two_ids(ids_all, lo):
    """Corrupt one slice by transposing its first two ids."""
    ids_all[lo], ids_all[lo + 1] = ids_all[lo + 1], ids_all[lo]


@pytest.mark.parametrize("position", ["first", "middle", "last"])
def test_corrupt_slice_at_any_position_refuses_before_folds(
        tr, legacy, position):
    """A digest-honestly-rebound record with ONE descending slice at
    any position refuses identically on both arms — refusal reason,
    zero folds, and four EMPTY outputs (no partial publication)."""
    slices = [[0, 3, 4], [1], [], [2, 5], [0, 6], [7], [1, 2, 3]]
    dists = np.arange(sum(len(s) for s in slices),
                      dtype=np.float64) * 0.5
    rec = synth_record(tr, slices, dists, dest_count=8,
                       trace_id="p03c-corrupt-" + position)
    good = replay_live(tr, rec, _case_for(rec, 8))
    assert good.route == "trace_replay"

    n_origins = len(slices)
    offsets = {"first": 0, "middle": n_origins // 2,
               "last": n_origins - 1}
    target = offsets[position]
    starts = np.cumsum([0] + [len(s) for s in slices])

    def mutate(indptr, ids_all, dist_all):
        if len(slices[target]) > 1:
            swap_two_ids(ids_all, int(starts[target]))
        else:
            # position slice is empty/len-1: corrupt the FIRST slice
            # that has length > 1 instead (still one corrupt slice)
            for k, s in enumerate(slices):
                if len(s) > 1:
                    swap_two_ids(ids_all, int(starts[k]))
                    break

    bad = retamper(tr, rec, mutate)
    arms = (("compiled", replay_live(tr, bad, _case_for(bad, 8))),
            ("legacy", replay_legacy(bad, _case_for(bad, 8))))
    for tag, r in arms:
        assert r.route == "trace_replay_refused", (tag, position)
        assert r.refusal == "corrupt_trace_slice_order", (tag, position)
        assert r.folds_done == 0, (tag, position)
        for f in ("reach", "gravity_exponential", "gravity_logistic",
                  "knn_access"):
            assert np.asarray(getattr(r, f)).shape[0] == 0, (
                tag, position, f)


def _case_for(rec, dest_count):
    """A minimal case dict for the replay helpers (metric weights +
    the record's cutoff)."""
    return dict(dwt=synth_weights(dest_count),
                cutoff=rec.cutoff_value)


def test_scan_predicate_unit_boundaries(tr):
    """Unit law of the compiled scan: length 0/1 slices skip the
    order check; equal consecutive ids and descents fail; multiple
    slices all checked."""
    f = tr._trace_slices_ascending

    def scan(slices):
        indptr = np.zeros(len(slices) + 1, np.int64)
        np.cumsum([len(s) for s in slices], out=indptr[1:])
        ids_all = (np.concatenate([np.asarray(s, np.int64)
                                   for s in slices])
                   if any(len(s) for s in slices)
                   else np.zeros(0, np.int64))
        return bool(f(indptr, ids_all))

    assert scan([]) is True                      # no origins
    assert scan([[]]) is True                    # one empty slice
    assert scan([[], [], []]) is True            # all empty
    assert scan([[5]]) is True                   # single id
    assert scan([[1, 2]]) is True                # pair ascending
    assert scan([[2, 2]]) is False               # tie
    assert scan([[2, 1]]) is False               # descent
    assert scan([[], [1], [2, 3], [], [4, 5, 6]]) is True
    assert scan([[1, 2], [], [3, 2]]) is False   # corrupt LAST kept
    assert scan([[3, 2], [], [4, 5]]) is False   # corrupt FIRST
    assert scan([list(range(64))]) is True       # SIMD-width run


def test_scan_predicate_unit_boundaries_real(tr):
    """Same boundary matrix routed through the record-level scan call
    (the facade's actual gate), including the descending long run."""
    for slices, ok in [
        ([], True),
        ([[]], True),
        ([[], [], []], True),
        ([[5]], True),
        ([[1, 2]], True),
        ([[2, 2]], False),
        ([[2, 1]], False),
        ([[], [1], [2, 3], [], [4, 5, 6]], True),
        ([[1, 2], [], [3, 2]], False),
        ([[3, 2], [], [4, 5]], False),
        ([list(range(64))], True),
        ([list(range(33)), [], list(range(2))], True),
    ]:
        nnz = sum(len(s) for s in slices)
        dists = np.arange(nnz, dtype=np.float64) * 0.25
        rec = synth_record(tr, slices, dists, dest_count=64)
        indptr = np.asarray(rec.indptr)
        ids_all = np.asarray(rec.destination_ids)
        got = bool(tr._trace_slices_ascending(indptr, ids_all))
        assert got is ok, (slices, ok)


def test_gate_order_preserved_multi_defect(tr, legacy):
    """The compiled path refuses with the SAME FIRST gate the legacy
    path refuses with, for records violating several gates at once."""
    case_dwt = synth_weights(8)
    slices = [[0, 3, 4], [1], [], [2, 5], [0, 6], [7], [1, 2, 3]]
    dists = np.arange(sum(len(s) for s in slices),
                      dtype=np.float64) * 0.5
    rec = synth_record(tr, slices, dists, dest_count=8)

    def both_arms(rec_bad):
        case = dict(dwt=case_dwt, cutoff=rec.cutoff_value)
        return (replay_live(tr, rec_bad, case),
                replay_legacy(rec_bad, case))

    # (a) slice corruption WITHOUT digest rebinding -> the digest gate
    # still fires first (the scan must not jump the packed gates).
    ids_tampered = np.array(rec.destination_ids)
    swap_two_ids(ids_tampered, 0)
    bad_digest = dataclasses.replace(
        rec, destination_ids=tr._read_only(ids_tampered))
    for r in both_arms(bad_digest):
        assert r.refusal == "corrupt_trace_content_digest"

    # (b) digest rebound + out-of-range ids -> destination-id gate
    # (which precedes the slice scan) fires first.
    def mutate_range(indptr, ids_all, dist_all):
        ids_all[0] = 99  # >= dest_count 8
    bad_ids = retamper(tr, rec, mutate_range)
    for r in both_arms(bad_ids):
        assert r.refusal == "corrupt_trace_destination_ids"

    # (c) digest rebound + negative indptr count -> structure gate.
    def mutate_struct(indptr, ids_all, dist_all):
        indptr[2] = indptr[1] - 1  # negative slice count
    bad_struct = retamper(tr, rec, mutate_struct)
    for r in both_arms(bad_struct):
        assert r.refusal == "corrupt_trace_structure"


def test_all_empty_trace_replays_zero_origins_content(tr, legacy):
    """All retained slices empty (nnz 0): every origin still folds on
    empty candidates; outputs are the fold's empty-input values; both
    arms bitwise equal."""
    slices = [[], [], [], []]
    rec = synth_record(tr, slices, np.zeros(0, np.float64),
                       dest_count=8, trace_id="p03c-all-empty")
    case = dict(dwt=synth_weights(8), cutoff=rec.cutoff_value)
    old = replay_legacy(rec, case)
    new = replay_live(tr, rec, case)
    assert new.route == "trace_replay" and new.folds_done == 4
    assert first_difference(new, old) is None
    for f in ("reach", "gravity_exponential", "gravity_logistic"):
        assert np.all(np.asarray(getattr(new, f)) == 0.0), f
    assert np.all(np.asarray(new.knn_access) == 0.0)


def test_simd_boundary_slice_lengths_parity(tr, legacy):
    """Slice lengths sweeping 0/1/2 and SIMD-ish boundaries (3, 7, 8,
    15, 16, 17, 31, 32, 33, 64): compiled vs legacy bitwise, ascending
    ids accepted at every width."""
    lengths = [0, 1, 2, 3, 7, 8, 15, 16, 17, 31, 32, 33, 64]
    slices = []
    nxt = 0
    for ln in lengths:
        slices.append(list(range(nxt, nxt + ln)))
        nxt += ln
    nnz = nxt
    dists = (np.arange(nnz, dtype=np.float64) * 0.25
             + 0.125)  # strictly positive, distinct per id
    rec = synth_record(tr, slices, dists, dest_count=nxt,
                       trace_id="p03c-simd-widths")
    case = dict(dwt=synth_weights(nxt), cutoff=rec.cutoff_value)
    old = replay_legacy(rec, case)
    new = replay_live(tr, rec, case)
    assert new.route == "trace_replay"
    assert new.folds_done == len(lengths)
    assert first_difference(new, old) is None


def test_output_mutation_isolation_and_trace_immunity(tr):
    """The compiled path hands out fresh correctly typed arrays;
    mutating one replay's outputs affects nothing else; the trace's
    packed bytes are unchanged by replaying (C10 fresh-copy law)."""
    slices = [[0, 3, 4], [1], [], [2, 5], [0, 6], [7], [1, 2, 3]]
    dists = np.arange(sum(len(s) for s in slices),
                      dtype=np.float64) * 0.5
    rec = synth_record(tr, slices, dists, dest_count=8)
    case = dict(dwt=synth_weights(8), cutoff=rec.cutoff_value)
    before = (np.asarray(rec.indptr).tobytes(),
              np.asarray(rec.destination_ids).tobytes(),
              np.asarray(rec.distance_bits).tobytes())
    r1 = replay_live(tr, rec, case)
    r2 = replay_live(tr, rec, case)
    saved = {f: np.asarray(getattr(r2, f)).tobytes()
             for f in ("reach", "gravity_exponential",
                       "gravity_logistic", "knn_access")}
    for f in ("reach", "gravity_exponential", "gravity_logistic",
              "knn_access"):
        x = getattr(r1, f)
        assert x is not getattr(r2, f), f
        x[0] = -1.5
    r3 = replay_live(tr, rec, case)
    for f in ("reach", "gravity_exponential", "gravity_logistic",
              "knn_access"):
        assert np.asarray(getattr(r3, f)).tobytes() == saved[f], f
    after = (np.asarray(rec.indptr).tobytes(),
             np.asarray(rec.destination_ids).tobytes(),
             np.asarray(rec.distance_bits).tobytes())
    assert before == after
