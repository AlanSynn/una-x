"""C20 corruption / owner-death / reset / resume laws (campaign
una_10x).  Load verifies IDENTITY, not presence: schema, producer
code, packed layout and the content digest (C11-NOTE-4) — a forged,
partial, stale-code or bit-flipped entry raises ResidencyError and is
never a silent hit.  Resume is a NEW residency instance over the SAME
store, re-verifying every identity on fault-in."""
import gc
import json

import numpy as np

from conftest import meta_for, packed_triple

from urban_network_analysis.cache.trace_residency import (
    RESIDENT_SCHEMA, ResidencyError, TraceResidency, packed_digest)

MEG = 1 << 20


def _payload(ip, ids, dist, meta):
    return {
        "indptr.bin": np.ascontiguousarray(ip).tobytes(),
        "destination_ids.bin": np.ascontiguousarray(ids).tobytes(),
        "distance_bits.bin": np.ascontiguousarray(dist).tobytes(),
        "meta.json": json.dumps(meta, sort_keys=True,
                                separators=(",", ":")).encode("utf-8"),
    }


def _full_meta(producer_sha, **over):
    ip, ids, dist = packed_triple(10)
    meta = meta_for(ip, ids)
    meta.update({
        "residency_schema": RESIDENT_SCHEMA,
        "producer_sha256": producer_sha,
        "content_digest": packed_digest(
            np.ascontiguousarray(ip).tobytes(),
            np.ascontiguousarray(ids).tobytes(),
            np.ascontiguousarray(dist).tobytes()),
        "origin_count": int(ip.shape[0] - 1),
        "nnz": int(ids.shape[0]),
    })
    meta.update(over)
    return meta


def test_resume_over_same_store_faults_in_and_verifies(
        disk_store, producer_sha):
    """Resume = a NEW residency instance over the SAME store: the
    entry faults in from disk and is re-verified (identity + digest)
    — never trusted because 'it was published once'."""
    res1 = TraceResidency(disk_store, max_resident_bytes=MEG,
                          producer_sha256=producer_sha)
    ip, ids, dist = packed_triple(40)
    key = res1.publish(ip, ids, dist, meta_for(ip, ids))
    res1.close()
    res2 = TraceResidency(disk_store, max_resident_bytes=MEG,
                          producer_sha256=producer_sha)
    assert res2.resident_bytes() == 0
    lease = res2.acquire(key)
    assert lease is not None and res2.counters["fault_ins"] == 1
    aip, aids, adist = lease.arrays()
    assert aip.tobytes() == ip.tobytes()
    assert aids.tobytes() == ids.tobytes()
    assert adist.tobytes() == dist.tobytes()
    lease.release()


def test_bit_flip_in_stored_bytes_refused_at_fault_in(
        disk_store, producer_sha):
    """The dossier-06 attack: a bit-flipped entry that the store's
    checksum accepted (the forger re-checksummed) must fail the
    content digest at decode — never replayed."""
    res1 = TraceResidency(disk_store, max_resident_bytes=MEG,
                          producer_sha256=producer_sha)
    ip, ids, dist = packed_triple(30)
    key = res1.publish(ip, ids, dist, meta_for(ip, ids))
    res1.close()
    hit = disk_store.get(key)
    assert hit is not None
    forged = dict(hit.payload)
    dist2 = np.frombuffer(forged["distance_bits.bin"],
                          dtype=np.float64).copy()
    dist2[0] = np.nextafter(dist2[0], np.float64(1e300))
    forged["distance_bits.bin"] = dist2.tobytes()  # digest now stale
    disk_store.put(key, forged)  # re-checksummed forgery, same key
    res2 = TraceResidency(disk_store, max_resident_bytes=MEG,
                          producer_sha256=producer_sha)
    try:
        res2.acquire(key)
        raise AssertionError("bit-flipped entry replayed")
    except ResidencyError as e:
        assert "content digest" in str(e)


def test_partial_entry_incomplete_identity_refused(
        disk_store, producer_sha):
    res1 = TraceResidency(disk_store, max_resident_bytes=MEG,
                          producer_sha256=producer_sha)
    ip, ids, dist = packed_triple(8)
    key = res1.publish(ip, ids, dist, meta_for(ip, ids))
    res1.close()
    hit = disk_store.get(key)
    meta = json.loads(hit.payload["meta.json"].decode("utf-8"))
    del meta["cutoff_bits"]  # strip one identity field
    disk_store.put(key, _payload(ip, ids, dist, meta))
    res2 = TraceResidency(disk_store, max_resident_bytes=MEG,
                          producer_sha256=producer_sha)
    try:
        res2.acquire(key)
        raise AssertionError("partial identity replayed")
    except ResidencyError as e:
        assert "identity" in str(e)


def test_producer_code_mismatch_refused(disk_store, producer_sha):
    """An entry recorded under a stale producer digest refuses at
    decode — schema/profile/code mismatch is never a hit."""
    res1 = TraceResidency(disk_store, max_resident_bytes=MEG,
                          producer_sha256=producer_sha)
    ip, ids, dist = packed_triple(8)
    key = res1.publish(ip, ids, dist, meta_for(ip, ids))
    res1.close()
    disk_store.put(key, _payload(ip, ids, dist,
                                 _full_meta(producer_sha,
                                            producer_sha256="0" * 64)))
    res2 = TraceResidency(disk_store, max_resident_bytes=MEG,
                          producer_sha256=producer_sha)
    try:
        res2.acquire(key)
        raise AssertionError("stale-code entry replayed")
    except ResidencyError as e:
        assert "producer code" in str(e)


def test_schema_mismatch_refused(disk_store, producer_sha):
    res1 = TraceResidency(disk_store, max_resident_bytes=MEG,
                          producer_sha256=producer_sha)
    ip, ids, dist = packed_triple(8)
    key = res1.publish(ip, ids, dist, meta_for(ip, ids))
    res1.close()
    disk_store.put(key, _payload(
        ip, ids, dist,
        _full_meta(producer_sha, residency_schema="other")))
    res2 = TraceResidency(disk_store, max_resident_bytes=MEG,
                          producer_sha256=producer_sha)
    try:
        res2.acquire(key)
        raise AssertionError("foreign-schema entry replayed")
    except ResidencyError as e:
        assert "schema" in str(e)


def test_undecodable_payload_refused(disk_store, producer_sha):
    res1 = TraceResidency(disk_store, max_resident_bytes=MEG,
                          producer_sha256=producer_sha)
    ip, ids, dist = packed_triple(8)
    key = res1.publish(ip, ids, dist, meta_for(ip, ids))
    res1.close()
    disk_store.put(key, {"indptr.bin": b"\x00" * 8,
                         "destination_ids.bin": b"\x01",
                         "distance_bits.bin": b"\x02",
                         "meta.json": b"{not json"})
    res2 = TraceResidency(disk_store, max_resident_bytes=MEG,
                          producer_sha256=producer_sha)
    try:
        res2.acquire(key)
        raise AssertionError("garbage payload replayed")
    except ResidencyError as e:
        assert "undecodable" in str(e)


def test_verify_resident_fault_injection(disk_store, producer_sha):
    """The documented test hook simulates a failing verification: a
    resident entry that stops matching its bound digest RAISES; an
    absent key never verifies."""
    res = TraceResidency(disk_store, max_resident_bytes=MEG,
                         producer_sha256=producer_sha)
    ip, ids, dist = packed_triple(10)
    key = res.publish(ip, ids, dist, meta_for(ip, ids))
    res.verify_resident(key)  # healthy: passes
    res.fault = "verify"
    try:
        res.verify_resident(key)
        raise AssertionError("corrupt verification passed")
    except ResidencyError:
        pass
    res.fault = None
    res.verify_resident(key)
    try:
        res.verify_resident("e" * 64)
        raise AssertionError("absent key verified")
    except ResidencyError as e:
        assert "not resident" in str(e)


def test_reset_store_then_miss(disk_store, producer_sha):
    """Store reset removes the durable tier: the residency reports a
    MISS (the caller recomputes) — never a stale hit."""
    res1 = TraceResidency(disk_store, max_resident_bytes=MEG,
                          producer_sha256=producer_sha)
    ip, ids, dist = packed_triple(10)
    key = res1.publish(ip, ids, dist, meta_for(ip, ids))
    res1.close()
    disk_store.clear()
    res2 = TraceResidency(disk_store, max_resident_bytes=MEG,
                          producer_sha256=producer_sha)
    assert res2.acquire(key) is None
    assert res2.counters["store_misses"] == 1


def test_oversized_entry_and_owner_death(disk_store, producer_sha):
    """A stream lease hands back identical verified bytes; owner death
    at arbitrary GC time releases cleanly (no resident state to
    corrupt) and later admissions still work."""
    res = TraceResidency(disk_store, max_resident_bytes=MEG,
                         producer_sha256=producer_sha)
    ip, ids, dist = packed_triple(80_000)  # ~1.25 MB > whole budget
    key = res.publish(ip, ids, dist, meta_for(ip, ids))
    lease = res.acquire(key)  # oversized -> stream lease
    aip, aids, _ = lease.arrays()
    assert aids.tobytes() == ids.tobytes()
    del lease
    gc.collect()
    ip2, ids2, dist2 = packed_triple(40)
    k2 = res.publish(ip2, ids2, dist2, meta_for(ip2, ids2))
    lease2 = res.acquire(k2)
    assert lease2 is not None
    lease2.release()
