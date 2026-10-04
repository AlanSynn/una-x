"""C20 lease laws (campaign una_10x): fresh public access, exactly-
once release, weakref reap, read-only resident storage."""
import gc

import numpy as np

from conftest import meta_for, packed_triple

from urban_network_analysis.cache.trace_residency import (
    ResidencyError, packed_digest)


def test_published_bytes_round_trip(residency):
    ip, ids, dist = packed_triple(50)
    key = residency.publish(ip, ids, dist, meta_for(ip, ids))
    with residency.acquire(key) as lease:
        aip, aids, adist = lease.arrays()
        assert aip.tobytes() == ip.tobytes()
        assert aids.tobytes() == ids.tobytes()
        assert adist.tobytes() == dist.tobytes()
    assert residency.counters["publishes"] == 1


def test_public_access_is_fresh_never_aliased(residency):
    """Callers mutate their copies; the resident arrays (and the
    round trip) are untouched — the "caller edits an exposed array"
    invalidation hole cannot exist."""
    ip, ids, dist = packed_triple(30)
    key = residency.publish(ip, ids, dist, meta_for(ip, ids))
    lease = residency.acquire(key)
    a = lease.arrays()
    a[0][:] = -1
    a[1][0] = -(1 << 40)
    a[2][0] = np.nan
    b = lease.arrays()
    assert (b[0] == ip).all() and (b[1] == ids).all() \
        and (b[2] == dist).all()
    residency.verify_resident(key)  # resident digest still holds
    entry = residency._entries[key]
    assert all(not arr.flags.writeable for arr in entry["arrays"])
    lease.release()


def test_release_exactly_once_and_use_after_release(residency):
    ip, ids, dist = packed_triple(10)
    key = residency.publish(ip, ids, dist, meta_for(ip, ids))
    lease = residency.acquire(key)
    assert residency._entries[key]["pinned"] == 1
    lease.release()
    assert residency._entries[key]["pinned"] == 0
    lease.release()  # idempotent: exactly-once semantics
    assert residency._entries[key]["pinned"] == 0
    try:
        lease.arrays()
        raise AssertionError("use after release must raise")
    except ResidencyError:
        pass


def test_owner_death_reaps_the_pin(residency):
    ip, ids, dist = packed_triple(20)
    key = residency.publish(ip, ids, dist, meta_for(ip, ids))
    lease = residency.acquire(key)
    assert residency._entries[key]["pinned"] == 1
    del lease
    gc.collect()
    assert residency._entries[key]["pinned"] == 0
    assert residency.counters["leases_ended"] == 1


def test_digest_bound_at_publish_rejects_disagreement(residency):
    """The C11-NOTE-4 law at the residency boundary: a record whose
    bound digest disagrees with its packed bytes is rejected — never
    published, never resident."""
    ip, ids, dist = packed_triple(15)
    good = packed_digest(ip.tobytes(), ids.tobytes(), dist.tobytes())
    residency.counters["publish_rejections"] = 0
    try:
        residency.publish(ip, ids, dist + 1.0, meta_for(ip, ids),
                          bound_digest=good)
        raise AssertionError("forged publish accepted")
    except ResidencyError:
        pass
    assert residency.counters["publish_rejections"] == 1
    assert residency.counters["publishes"] == 0


def test_incomplete_record_never_publishable(residency):
    """The C11 gate law at the residency boundary: the empty-digest
    INCOMPLETE layout (zero nnz) publishes only as an EMPTY trace —
    an empty triple is valid layout; an arbitrary meta without the
    identity fields is rejected."""
    ip = np.zeros(1, np.int64)
    ids = np.zeros(0, np.int64)
    dist = np.zeros(0, np.float64)
    try:
        residency.publish(ip, ids, dist, {"trace_id": "x"})
        raise AssertionError("incomplete identity meta accepted")
    except ResidencyError as e:
        assert "incomplete" in str(e)


def test_disabled_store_cannot_back_residency(producer_sha, tmp_path):
    """Cache disabled keeps the original route: constructing a
    residency over a disabled store is refused — no identity or
    serialization work exists to do."""
    from urban_network_analysis.Execution import CacheOptions
    from urban_network_analysis.cache.store import ContentStore
    from urban_network_analysis.cache.trace_residency import (
        TraceResidency)
    store = ContentStore(CacheOptions(mode="off",
                                      directory=str(tmp_path / "off")))
    try:
        TraceResidency(store, max_resident_bytes=1 << 20,
                       producer_sha256=producer_sha)
        raise AssertionError("disabled store accepted")
    except ValueError:
        pass


def test_quiet_counters_are_not_engagement(residency):
    """Dossier 06: a hit counter alone is not engagement.  The
    residency exposes publish/acquire/decode/eviction counters and
    they must all move on real work only."""
    ip, ids, dist = packed_triple(12)
    before = dict(residency.counters)
    miss = residency.acquire("f" * 64)
    assert miss is None
    assert residency.counters["store_misses"] == \
        before["store_misses"] + 1
    key = residency.publish(ip, ids, dist, meta_for(ip, ids))
    assert residency.counters["publishes"] == before["publishes"] + 1
    residency.acquire(key).release()
    assert residency.counters["resident_hits"] == 1  # publish resident
