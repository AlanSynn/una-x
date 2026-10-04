"""C20 shared-quota laws (campaign una_10x): resident decoded arenas
and the encoded store's memory tier charge TOGETHER; pinned entries
are never evicted; unpinned entries evict LRU; oversized entries
stream back without residency."""
import numpy as np

from conftest import meta_for, packed_triple

from urban_network_analysis.cache.trace_residency import (
    packed_digest)


def test_shared_quota_charges_both_tiers(mem_store, producer_sha):
    """RESOURCES.md: the decoded arenas and the encoded cache can
    never each claim the same quota — one allowance charges both."""
    from urban_network_analysis.cache.trace_residency import (
        TraceResidency)
    res = TraceResidency(mem_store, max_resident_bytes=1 << 20,
                         producer_sha256=producer_sha)
    ip, ids, dist = packed_triple(400)
    key = res.publish(ip, ids, dist, meta_for(ip, ids))
    assert res.charged_bytes() >= res.resident_bytes() > 0
    # acquire the STORE copy too (memory mode: an encoded mirror
    # enters the store's own memory tier)
    with res.acquire(key):
        pass
    assert res.charged_bytes() == res.resident_bytes() + \
        int(mem_store.stats()["memory_bytes"])


def test_budget_bounds_residency_and_accounts_both_tiers(
        disk_store, producer_sha):
    """The residency's own allowance bounds its DECODED side after
    every operation; the encoded tier is charged in the accounting
    (charged == resident + store memory — no hidden double claim) and
    bounded by the STORE's own options (its contract).  Every key
    still resolves through the durable store tier after eviction."""
    from urban_network_analysis.cache.trace_residency import (
        TraceResidency)
    budget = 40_000
    res = TraceResidency(disk_store, max_resident_bytes=budget,
                         producer_sha256=producer_sha)
    keys = []
    for i in range(12):
        ip, ids, dist = packed_triple(600 + i, seed=i + 1)
        keys.append(res.publish(ip, ids, dist, meta_for(
            ip, ids, trace_id="tid-%d" % i)))
        assert res.resident_bytes() <= budget
        assert res.charged_bytes() == res.resident_bytes() + \
            int(disk_store.stats()["memory_bytes"])
    assert res.counters["evictions"] > 0, \
        "an always-full budget must evict"
    hits = 0
    for k in keys[-3:]:
        lease = res.acquire(k)
        assert lease is not None
        lease.arrays()
        lease.release()
        hits += 1
        assert res.resident_bytes() <= budget
    assert hits == 3


def test_pinned_entries_are_never_evicted(residency):
    """Eviction mid-read is impossible: a pinned entry survives any
    admission; its arrays stay usable for the lease's whole life."""
    ip, ids, dist = packed_triple(100)
    key = residency.publish(ip, ids, dist, meta_for(ip, ids))
    lease = residency.acquire(key)
    before = dict(residency.counters)
    for i in range(6):
        big_ip, big_ids, big_dist = packed_triple(50_000, seed=i + 9)
        residency.publish(big_ip, big_ids, big_dist, meta_for(
            big_ip, big_ids, trace_id="tid-big-%d" % i),
            bound_digest=packed_digest(big_ip.tobytes(),
                                       big_ids.tobytes(),
                                       big_dist.tobytes()))
    assert residency._entries.get(key) is not None, \
        "pinned entry was evicted"
    assert residency.counters["evictions"] >= before["evictions"]
    aip, aids, adist = lease.arrays()
    assert aids.tobytes() == ids.tobytes()
    residency.verify_resident(key)
    lease.release()


def test_unpinned_eviction_is_lru(residency):
    """The least-recently-USED unpinned entry leaves first; a recent
    touch protects an entry that younger entries do not need to
    displace."""
    keep = []
    for i in range(4):
        ip, ids, dist = packed_triple(12_000, seed=i + 1)  # ~192 KB
        keep.append(residency.publish(ip, ids, dist, meta_for(
            ip, ids, trace_id="tid-lru-%d" % i)))
    # touch keep[0]: keep[1] is now the least recently used
    residency.acquire(keep[0]).release()
    ip, ids, dist = packed_triple(20_000, seed=77)  # ~320 KB new
    residency.publish(ip, ids, dist, meta_for(ip, ids,
                                              trace_id="tid-new"))
    entry_keys = set(residency._entries)
    assert keep[1] not in entry_keys, "LRU entry survived"
    assert keep[0] in entry_keys, "recently-used entry was evicted"
    assert keep[2] in entry_keys and keep[3] in entry_keys, \
        "eviction evicted entries the new admission did not require"


def test_oversized_entry_streams_without_residency(disk_store,
                                                   producer_sha):
    """An entry bigger than the WHOLE allowance is still usable: it
    publishes store-only, faults in as a stream lease (fresh copies
    of the verified decode), and never enters the resident dict."""
    from urban_network_analysis.cache.trace_residency import (
        TraceResidency)
    res = TraceResidency(disk_store, max_resident_bytes=1 << 16,
                         producer_sha256=producer_sha)
    ip, ids, dist = packed_triple(30_000)  # ~360 KB > 64 KB
    key = res.publish(ip, ids, dist, meta_for(ip, ids))
    assert res.counters["capacity_writes"] == 1
    assert res.resident_bytes() == 0
    lease = res.acquire(key)
    assert lease is not None
    aip, aids, adist = lease.arrays()
    assert aids.tobytes() == ids.tobytes()
    assert adist.tobytes() == dist.tobytes()
    assert res.resident_bytes() == 0  # still not resident
    lease.release()
    res.close()


def test_close_drops_unpinned_keeps_pinned(residency):
    ip, ids, dist = packed_triple(25)
    k1 = residency.publish(ip, ids, dist,
                           meta_for(ip, ids, trace_id="tid-a"))
    ip2, ids2, dist2 = packed_triple(25, seed=2)
    k2 = residency.publish(ip2, ids2, dist2,
                           meta_for(ip2, ids2, trace_id="tid-b"))
    pinned = residency.acquire(k1)
    dropped = residency.close()
    assert dropped == 1
    assert k2 not in residency._entries
    assert k1 in residency._entries
    pinned.arrays()  # the pinned entry stays fully usable
    pinned.release()
