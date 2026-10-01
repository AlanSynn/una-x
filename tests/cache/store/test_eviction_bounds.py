"""Bounded residency: memory-tier policy, LRU eviction (memory + disk),
eviction under use, clear() — dossier 06 'no unbounded resident data'."""
from __future__ import annotations

import os

import pytest

from urban_network_analysis.Execution import CacheOptions
from urban_network_analysis.cache import ContentStore

KB = 1000


def _gen_dir(store, key):
    return os.path.join(str(store.options.directory), "store", key[:2], key)


def _age(store, key, ns):
    """Pin a generation dir's LRU clock explicitly (deterministic order)."""
    os.utime(_gen_dir(store, key), ns=(ns, ns))


@pytest.mark.cache_store
def test_memory_tier_inert_without_explicit_bound(make_store):
    store = make_store(mode="memory")
    assert store.stats()["memory_enabled"] is False
    store.put("a" * 64, {"x": b"payload"})
    assert store.get("a" * 64) is None  # inert: always a miss, always compute
    s = store.stats()
    assert s["memory_entries"] == 0 and s["memory_bytes"] == 0
    assert s["hit_reasons"]["miss"] == 1


@pytest.mark.cache_store
def test_memory_mode_leaves_no_disk_residue(make_store, cache_dir):
    """mode='memory' caches in memory ONLY: the cache directory is never
    even created — no store/ tree, no locks/, no manifest files on disk."""
    store = make_store(mode="memory", max_memory_bytes=1 << 20)
    store.put("a" * 64, {"x": b"payload"})
    hit = store.get("a" * 64)
    assert hit is not None and hit.hit == "memory"
    assert not cache_dir.exists()


@pytest.mark.cache_store
def test_disk_mode_memory_layer_optional(make_store):
    """mode='disk' without a memory bound: pure disk store — reads hit disk,
    nothing is retained resident."""
    store = make_store(mode="disk")
    store.put("a" * 64, {"x": b"p"})
    assert store.get("a" * 64).hit == "disk"
    assert store.get("a" * 64).hit == "disk"  # stays disk: no memory tier
    s = store.stats()
    assert s["memory_enabled"] is False and s["memory_entries"] == 0


@pytest.mark.cache_store
def test_memory_lru_eviction_order_pure_tier(make_store):
    """Pure LRU order on the memory tier (memory mode — no disk reads to
    confound use-order): the least-recently-USED key is evicted."""
    store = make_store(mode="memory", max_memory_bytes=250)
    ka, kb, kc = "a" * 64, "b" * 64, "c" * 64
    store.put(ka, {"x": b"x" * 100})
    store.put(kb, {"x": b"y" * 100})
    store.get(ka)                       # touch A → LRU order: B, A
    store.put(kc, {"x": b"z" * 100})    # 300 > 250 → evict B (LRU)
    s = store.stats()
    assert s["evicted_memory"] == 1
    assert s["memory_entries"] == 2 and s["memory_bytes"] == 200
    assert store.get(ka).hit == "memory"   # recently used: retained
    assert store.get(kb) is None           # evicted
    assert store.get(kc).hit == "memory"


@pytest.mark.cache_store
def test_disk_tier_survives_memory_eviction(make_store):
    """In disk mode, a key evicted from the (small) memory tier is still
    served — bitwise — from disk."""
    store = make_store(mode="disk", max_memory_bytes=250)
    ka, kb = "a" * 64, "b" * 64
    store.put(ka, {"x": b"x" * 100})
    store.put(kb, {"x": b"y" * 100})    # both fit: 200 ≤ 250
    store.put("c" * 64, {"x": b"z" * 100})   # 300 > 250 → evict LRU (ka)
    assert store.stats()["evicted_memory"] == 1
    assert store.get(ka).hit == "disk"  # memory-evicted, disk-served
    assert store.get(ka).payload["x"] == b"x" * 100


@pytest.mark.cache_store
def test_memory_never_exceeds_bound(make_store):
    store = make_store(mode="disk", max_memory_bytes=500)
    for i in range(20):
        store.put(f"{i:064x}", {"x": b"q" * 100})
        store.get(f"{i:064x}")
    s = store.stats()
    assert s["memory_bytes"] <= 500
    assert s["memory_entries"] <= 5
    assert s["evicted_memory"] >= 15
    # and eviction is transparent: everything is still readable from disk
    for i in range(20):
        assert store.get(f"{i:064x}").payload["x"] == b"q" * 100


@pytest.mark.cache_store
def test_entry_larger_than_whole_tier_never_resident(make_store):
    store = make_store(mode="disk", max_memory_bytes=100)
    k = "e" * 64
    store.put(k, {"x": b"big" * 500})   # 1500 bytes >> 100-byte tier
    hit = store.get(k)
    assert hit.hit == "disk"            # served, but never cached resident
    assert store.stats()["memory_entries"] == 0


@pytest.mark.cache_store
def test_oversized_put_evicts_nothing_else(make_store):
    """Review NOTE-7: 'oversized entry never resident' must mean exactly
    that — the size guard refuses it WITHOUT draining the tier (mutating
    the guard away still passes drain-style tests; this one doesn't)."""
    store = make_store(mode="disk", max_memory_bytes=250)
    ka, kb = "a" * 64, "b" * 64
    store.put(ka, {"x": b"x" * 100})
    store.put(kb, {"x": b"y" * 100})         # tier: [ka, kb] = 200 ≤ 250
    store.put("c" * 64, {"x": b"z" * 1000})  # oversized: refused wholesale
    s = store.stats()
    assert s["evicted_memory"] == 0          # nobody was evicted to admit it
    assert store.get(ka).hit == "memory"     # both prior entries retained
    assert store.get(kb).hit == "memory"
    assert store.get("c" * 64).payload["x"] == b"z" * 1000   # served from disk
    assert s["memory_entries"] == 2


@pytest.mark.cache_store
def test_memory_mode_also_bounded_and_lru(make_store):
    store = make_store(mode="memory", max_memory_bytes=150)
    ka, kb, kc = "a" * 64, "b" * 64, "c" * 64
    store.put(ka, {"x": b"x" * 100})    # 100
    store.put(kb, {"x": b"y" * 100})    # 200 > 150 → evict A
    store.put(kc, {"x": b"z" * 100})    # 200 > 150 → evict B
    s = store.stats()
    assert s["evicted_memory"] == 2 and s["memory_bytes"] == 100
    assert store.get(ka) is None        # evicted: memory mode has no disk
    assert store.get(kb) is None
    assert store.get(kc).hit == "memory"


@pytest.mark.cache_store
def test_disk_quota_evicts_oldest(make_store):
    store = make_store(mode="disk", max_disk_bytes=4000)
    k1, k2, k3 = "1" * 64, "2" * 64, "3" * 64
    store.put(k1, {"x": b"a" * KB})
    store.put(k2, {"x": b"b" * KB})
    store.put(k3, {"x": b"c" * KB})
    _age(store, k1, 1_000_000)          # oldest
    _age(store, k2, 2_000_000)
    _age(store, k3, 3_000_000)          # newest
    store.put("4" * 64, {"x": b"d" * KB})  # over quota → evict k1 (LRU)
    s = store.stats()
    assert s["evicted_disk"] == 1
    assert s["disk_bytes"] <= 4000
    assert not os.path.isdir(_gen_dir(store, k1))
    assert os.path.isdir(_gen_dir(store, k2))
    assert os.path.isdir(_gen_dir(store, k3))
    assert store.get(k1) is None
    assert store.get(k2).payload["x"] == b"b" * KB


@pytest.mark.cache_store
def test_recent_read_is_protected_from_disk_eviction(make_store):
    """The LRU touch after a verified read moves a key's generation dir to
    the eviction order's end — being read recently protects it."""
    store = make_store(mode="disk", max_disk_bytes=4000)
    k1, k2, k3 = "1" * 64, "2" * 64, "3" * 64
    store.put(k1, {"x": b"a" * KB})
    store.put(k2, {"x": b"b" * KB})
    store.put(k3, {"x": b"c" * KB})
    _age(store, k1, 1_000_000)
    _age(store, k2, 2_000_000)
    _age(store, k3, 3_000_000)
    store.get(k1)                        # read → touched → now newest
    store.put("4" * 64, {"x": b"d" * KB})
    # oldest is now k2 (k1 was touched after k2's write)
    assert os.path.isdir(_gen_dir(store, k1))
    assert not os.path.isdir(_gen_dir(store, k2))
    assert store.get(k1).payload["x"] == b"a" * KB
    assert store.stats()["evicted_disk"] == 1


@pytest.mark.cache_store
def test_eviction_under_use_is_safe(make_store):
    """A returned hit is fresh bytes: eviction/clear afterwards cannot pull
    the rug from under in-flight work."""
    store = make_store(mode="disk", max_memory_bytes=1 << 20)
    k = "f" * 64
    store.put(k, {"x": b"precious-inflight-bytes"})
    hit = store.get(k)
    store.clear()                        # everything gone, mid-use
    assert hit.payload["x"] == b"precious-inflight-bytes"  # copy stands
    assert store.get(k) is None          # but the store is empty
    s = store.stats()
    assert s["entries"] == 0 and s["memory_entries"] == 0


@pytest.mark.cache_store
def test_disk_eviction_under_use_is_safe(make_store):
    store = make_store(mode="disk", max_disk_bytes=1250)  # fits one entry
    ka, kb = "a" * 64, "b" * 64
    store.put(ka, {"x": b"a" * 1000})    # ~1.2 KB with manifest: quota full
    hit = store.get(ka)
    store.put(kb, {"x": b"b" * 1000})    # forces eviction of ka on disk
    assert hit.payload["x"] == b"a" * 1000       # in-flight copy unaffected
    assert store.get(ka) is None                 # evicted from disk AND mem
    assert store.get(kb).payload["x"] == b"b" * 1000


@pytest.mark.cache_store
def test_clear_removes_everything_including_quarantine(make_store):
    store = make_store(mode="disk")
    root = str(store.options.directory)
    store.put("a" * 64, {"x": b"1"})
    store.put("b" * 64, {"x": b"2"})
    # park a damaged entry in quarantine
    k = "c" * 64
    store.put(k, {"x": b"3"})
    with open(os.path.join(store._gen_dir(k), "payload", "x"), "wb") as f:
        f.write(b"damaged!")
    store.get(k)
    assert store.stats()["quarantined"] == 1
    assert os.listdir(os.path.join(root, "quarantine"))
    store.clear()
    s = store.stats()
    assert s["entries"] == 0 and s["disk_bytes"] == 0
    assert s["memory_entries"] == 0 and s["memory_bytes"] == 0
    assert store.get("a" * 64) is None
    assert os.listdir(os.path.join(root, "quarantine")) == []
    # and the store still works after clear
    store.put("a" * 64, {"x": b"fresh"})
    assert store.get("a" * 64).payload["x"] == b"fresh"


@pytest.mark.cache_store
def test_disabled_and_memoryless_stores_hold_nothing_resident(tmp_path):
    off = ContentStore(CacheOptions(mode="off",
                                    directory=str(tmp_path / "off")))
    off.put("a" * 64, {"x": b"x" * 10 * KB})
    s = off.stats()
    assert s["memory_bytes"] == 0 and s["entries"] == 0
