"""C11-NOTE-4 pins (carried to C20, campaign una_10x).

The COMPLETE record carries a content digest bound at build: sha256
over the packed bytes in layout order.  ``replay_trace`` verifies it
before any fold, so a forged or bit-flipped retained record is
refused (``corrupt_trace_content_digest``) — never silently replayed
as a cache hit.  Structural forgeries that recompute the digest (the
stronger adversary) remain pinned by the C10-MINOR-1 gates; here the
NAIVE adversary (stale digest) is pinned.
"""
import dataclasses
import hashlib

import numpy as np

from conftest import build, ring_case

KNN = np.array([1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1])


def _fields(rec):
    return {f.name: getattr(rec, f.name)
            for f in dataclasses.fields(rec)}


def _packed_digest(indptr, ids, dist):
    return hashlib.sha256(
        np.asarray(indptr).tobytes() + np.asarray(ids).tobytes()
        + np.asarray(dist).tobytes()).hexdigest()


def _replay(tr, rec, case):
    return tr.replay_trace(
        rec, case["dwt"], 1.5, 2.0, 3.0, 0.8, "exponential", KNN,
        case["cutoff"])


def test_content_digest_bound_at_build(tr):
    """The digest is the packed bytes' sha256 in layout order; an
    INCOMPLETE record carries the empty digest."""
    case = ring_case()
    rec = build(tr, case)
    assert rec.complete
    assert rec.content_digest == _packed_digest(
        rec.indptr, rec.destination_ids, rec.distance_bits)
    assert len(rec.content_digest) == 64
    incomplete = tr.build_trace(
        case["o"], case["ow"], case["ptr"], case["vec"], case["wts"],
        case["flag"], case["d"], case["dw"], case["dwt"],
        case["cutoff"], trace_budget_bytes=1)
    assert not incomplete.complete
    assert incomplete.content_digest == ""


def test_stale_digest_refused_before_any_fold(tr):
    """Bit-flipped retained bytes under a STALE digest refuse with
    ``corrupt_trace_content_digest`` — the digest gate fires before
    the fold loop (whole-record content check first)."""
    case = ring_case()
    rec = build(tr, case)

    fields = _fields(rec)
    ids = np.asarray(rec.destination_ids).copy()
    ids[len(ids) // 2] ^= 1            # a single flipped id bit
    fields["destination_ids"] = ids    # digest deliberately STALE
    r = _replay(tr, tr.TraceRecord(**fields), case)
    assert r.route == "trace_replay_refused"
    assert r.refusal == "corrupt_trace_content_digest"

    fields = _fields(rec)
    dist = np.asarray(rec.distance_bits).copy()
    dist[0] = np.nextafter(dist[0], np.float64(1e300))  # 1-ULP flip
    fields["distance_bits"] = dist     # digest deliberately STALE
    r = _replay(tr, tr.TraceRecord(**fields), case)
    assert r.refusal == "corrupt_trace_content_digest"


def test_digest_consistent_forger_still_structure_refused(tr):
    """Recomputing the digest does NOT admit a forged record: the
    structural gates still refuse (id flip under a fresh digest hits
    an order or range refusal, never a fold)."""
    case = ring_case()
    rec = build(tr, case)
    fields = _fields(rec)
    ids = np.asarray(rec.destination_ids).copy()
    ids[len(ids) // 2] ^= 1
    fields["destination_ids"] = ids
    fields["content_digest"] = _packed_digest(
        fields["indptr"], ids, fields["distance_bits"])
    r = _replay(tr, tr.TraceRecord(**fields), case)
    assert r.route == "trace_replay_refused"
    assert r.refusal in ("corrupt_trace_slice_order",
                         "corrupt_trace_destination_ids")
    assert r.folds_done == 0
