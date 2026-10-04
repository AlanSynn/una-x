"""C20 repair pins (campaign una_10x) — the independent review's
adversary classes from c20_review.json, each refused at the layer the
reviewer probed.  These are committed regression pins for the repair
round, not new laws: the laws are the module docstrings'.

- C20-MINOR-1: the dossier-04 layout law is ENFORCED at publish and
  at load (non-decreasing indptr, ids ascending within every slice);
  load re-derives the content address from the entry's own identity
  (a meta swapped under another trace's key refuses).
- C20-MINOR-2: every malformed meta raises ResidencyError — a caller
  following the documented catch-and-recompute discipline never sees
  a foreign KeyError/ValueError.
- C20-MINOR-3: a content address IS the identity — republishing
  DIFFERENT packed bytes under an existing key refuses (resident tier
  and encoded store tier alike); identical bytes republish cleanly.
"""
import gc
import json

import numpy as np

from conftest import meta_for, packed_triple

from urban_network_analysis.cache.trace_residency import (
    ResidencyError, TraceResidency, packed_digest)

MEG = 1 << 20


def _stored_digest(payload):
    return packed_digest(payload["indptr.bin"],
                         payload["destination_ids.bin"],
                         payload["distance_bits.bin"])


def test_non_monotonic_indptr_refused_at_publish(residency):
    ip, ids, dist = packed_triple(8)
    bad_ip = np.array([0, 4, 2, 6], np.int64)  # slice bounds decrease
    residency.counters["publish_rejections"] = 0
    try:
        residency.publish(bad_ip, ids[:6], dist[:6],
                          meta_for(bad_ip, ids[:6], trace_id="t-nm"))
        raise AssertionError("non-monotonic indptr accepted")
    except ResidencyError as e:
        assert "layout violation" in str(e)
    assert residency.counters["publish_rejections"] == 1
    assert residency.counters["publishes"] == 0


def test_unsorted_slice_ids_refused_at_publish(residency):
    ip = np.array([0, 2, 5], np.int64)
    ids = np.array([3, 1, 2, 4, 5], np.int64)   # first slice descends
    dist = np.arange(5.0)
    try:
        residency.publish(ip, ids, dist,
                          meta_for(ip, ids, trace_id="t-us"))
        raise AssertionError("unsorted slice ids accepted")
    except ResidencyError as e:
        assert "ascending" in str(e)


def test_digest_consistent_structural_forgery_refused_at_load(
        disk_store, producer_sha):
    """The rv2-C adversary: recompute the content digest over
    STRUCTURALLY forged bytes (sorted order destroyed) and re-put
    under the original key.  The digest passes; the layout law at
    load refuses — never a silent ResidentLease."""
    res1 = TraceResidency(disk_store, max_resident_bytes=MEG,
                          producer_sha256=producer_sha)
    ip, ids, dist = packed_triple(30)
    key = res1.publish(ip, ids, dist, meta_for(ip, ids, trace_id="t-f"))
    res1.close()
    hit = disk_store.get(key)
    assert hit is not None
    payload = dict(hit.payload)
    forged_ids = np.frombuffer(payload["destination_ids.bin"],
                               dtype=np.int64).copy()
    # forge a descent INSIDE a slice that provably holds >= 2 ids
    # (the single cut leaves one slice with half the nnz): swapping
    # the first two ids of a one-element slice would be structurally
    # VALID and correctly accepted
    wide = int(np.argmax(np.diff(ip)))  # slice i spans [ip[i], ip[i+1])
    a, b = int(ip[wide]), int(ip[wide + 1])
    assert b - a >= 2
    forged_ids[a], forged_ids[a + 1] = forged_ids[a + 1], forged_ids[a]
    payload["destination_ids.bin"] = forged_ids.tobytes()
    # the stronger adversary: the forgery's OWN honest digest
    meta = json.loads(payload["meta.json"])
    meta["content_digest"] = _stored_digest(payload)
    payload["meta.json"] = json.dumps(meta).encode("utf-8")
    disk_store.put(key, payload)
    res2 = TraceResidency(disk_store, max_resident_bytes=MEG,
                          producer_sha256=producer_sha)
    try:
        res2.acquire(key)
        raise AssertionError("digest-consistent forgery replayed")
    except ResidencyError as e:
        assert "layout violation" in str(e)


def test_meta_swapped_under_foreign_key_refused_at_load(
        disk_store, producer_sha):
    """The rv3-F adversary: another trace's VERBATIM meta stored
    under a foreign content address.  The digest and layout pass;
    the key re-derivation refuses — the key must BIND the entry's
    identity."""
    res1 = TraceResidency(disk_store, max_resident_bytes=MEG,
                          producer_sha256=producer_sha)
    ip, ids, dist = packed_triple(12)
    key_a = res1.publish(ip, ids, dist,
                         meta_for(ip, ids, trace_id="t-a"))
    key_b = res1.publish(ip, ids, dist,
                         meta_for(ip, ids, trace_id="t-b"))
    assert key_a != key_b
    res1.close()
    disk_store.put(key_a, dict(disk_store.get(key_b).payload))
    res2 = TraceResidency(disk_store, max_resident_bytes=MEG,
                          producer_sha256=producer_sha)
    try:
        res2.acquire(key_a)
        raise AssertionError("foreign-key entry replayed")
    except ResidencyError as e:
        assert "key does not bind" in str(e)


def test_missing_content_digest_is_residency_error(
        disk_store, producer_sha):
    """The rv3-D escape: meta without content_digest refused inside
    the ResidencyError contract — never a KeyError."""
    res1 = TraceResidency(disk_store, max_resident_bytes=MEG,
                          producer_sha256=producer_sha)
    ip, ids, dist = packed_triple(12)
    key = res1.publish(ip, ids, dist, meta_for(ip, ids, trace_id="t-d"))
    res1.close()
    hit = disk_store.get(key)
    payload = dict(hit.payload)
    meta = json.loads(payload["meta.json"])
    meta.pop("content_digest")
    payload["meta.json"] = json.dumps(meta).encode("utf-8")
    disk_store.put(key, payload)
    res2 = TraceResidency(disk_store, max_resident_bytes=MEG,
                          producer_sha256=producer_sha)
    try:
        res2.acquire(key)
        raise AssertionError("digest-less meta replayed")
    except KeyError as e:
        raise AssertionError(f"foreign KeyError escaped: {e!r}")
    except ResidencyError:
        pass


def test_non_integer_counts_are_residency_error(
        disk_store, producer_sha):
    """The rv3-E escape: non-int origin_count refused inside the
    ResidencyError contract — never a ValueError."""
    res1 = TraceResidency(disk_store, max_resident_bytes=MEG,
                          producer_sha256=producer_sha)
    ip, ids, dist = packed_triple(12)
    key = res1.publish(ip, ids, dist, meta_for(ip, ids, trace_id="t-c"))
    res1.close()
    hit = disk_store.get(key)
    payload = dict(hit.payload)
    meta = json.loads(payload["meta.json"])
    meta["origin_count"] = "two"
    payload["meta.json"] = json.dumps(meta).encode("utf-8")
    disk_store.put(key, payload)
    res2 = TraceResidency(disk_store, max_resident_bytes=MEG,
                          producer_sha256=producer_sha)
    try:
        res2.acquire(key)
        raise AssertionError("non-int count replayed")
    except ValueError as e:
        raise AssertionError(f"foreign ValueError escaped: {e!r}")
    except ResidencyError:
        pass


def test_republish_same_key_different_bytes_refused(residency,
                                                    disk_store):
    """The rv4 divergence, both tiers: publishing different packed
    bytes under an existing content address refuses and leaves the
    stored bytes EXACTLY as they were."""
    ip, ids, dist = packed_triple(20)
    key = residency.publish(ip, ids, dist,
                            meta_for(ip, ids, trace_id="t-r"))
    before = dict(disk_store.get(key).payload)
    residency.counters["publish_rejections"] = 0
    try:
        residency.publish(ip, ids, dist + 1.0,
                          meta_for(ip, ids, trace_id="t-r"))
        raise AssertionError("divergent republish accepted")
    except ResidencyError as e:
        assert "identity violation" in str(e)
    assert residency.counters["publish_rejections"] == 1
    after = dict(disk_store.get(key).payload)
    assert after["distance_bits.bin"] == before["distance_bits.bin"]
    assert after["indptr.bin"] == before["indptr.bin"]
    assert after["destination_ids.bin"] == before["destination_ids.bin"]


def test_republish_identical_bytes_accepted(residency):
    """Idempotent republish (same content address, same bytes) is the
    normal cold-start race outcome: accepted, one entry, no
    divergence."""
    ip, ids, dist = packed_triple(20)
    meta = meta_for(ip, ids, trace_id="t-same")
    key1 = residency.publish(ip, ids, dist, meta)
    key2 = residency.publish(ip, ids, dist, meta)
    assert key1 == key2
    assert residency.counters["publishes"] == 2
    assert len(residency._entries) == 1


def test_republish_under_live_lease_refuses_divergence(residency):
    """A pinned lease must never observe its entry change: divergent
    bytes under a pinned key refuse (the one-line repair the review
    demanded, exercised with the pin held)."""
    ip, ids, dist = packed_triple(20)
    key = residency.publish(ip, ids, dist,
                            meta_for(ip, ids, trace_id="t-pin"))
    lease = residency.acquire(key)
    try:
        assert residency._entries[key]["pinned"] == 1
        try:
            residency.publish(ip, ids, dist + 2.0,
                              meta_for(ip, ids, trace_id="t-pin"))
            raise AssertionError("divergent republish under lease")
        except ResidencyError:
            pass
        aip, aid, adist = lease.arrays()
        assert adist.tobytes() == dist.tobytes()
    finally:
        lease.release()
        gc.collect()


def test_republish_encoded_tier_edges_fresh_residency(
        residency, disk_store, producer_sha):
    """C20-RV-2 (the C20 re-verify residual): BOTH edges of the
    divergence law at the encoded store tier alone — a FRESH residency
    over the same store holds no resident copy.  Divergent bytes refuse
    with the stored bytes exactly as they were; the SAME bytes
    republish cleanly (the idempotent cold-start race must never be
    refused by a future over-strict repair)."""
    ip, ids, dist = packed_triple(24)
    meta = meta_for(ip, ids, trace_id="t-enc")
    key = residency.publish(ip, ids, dist, meta)
    res2 = TraceResidency(disk_store, max_resident_bytes=MEG,
                          producer_sha256=producer_sha)
    assert key not in res2._entries            # encoded tier only
    before = dict(disk_store.get(key).payload)
    res2.counters["publish_rejections"] = 0
    try:
        res2.publish(ip, ids, dist + 3.0,
                     meta_for(ip, ids, trace_id="t-enc"))
        raise AssertionError("encoded-tier divergent republish accepted")
    except ResidencyError as e:
        assert "identity violation" in str(e)
    assert res2.counters["publish_rejections"] == 1
    after = dict(disk_store.get(key).payload)
    for name in ("indptr.bin", "destination_ids.bin", "distance_bits.bin"):
        assert after[name] == before[name]
    # acceptance edge: identical bytes republish cleanly, one entry
    key2 = res2.publish(ip, ids, dist, meta_for(ip, ids, trace_id="t-enc"))
    assert key2 == key
    assert res2.counters["publishes"] == 1
