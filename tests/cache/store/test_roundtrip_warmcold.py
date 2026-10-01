"""Bitwise round-trips, fresh-copy discipline, warm/cold/off equivalence."""
from __future__ import annotations

import hashlib

import pytest

from urban_network_analysis.Execution import CacheOptions
from urban_network_analysis.cache import ContentStore

K1 = "c" * 64
K2 = "d" * 64
BLOB = bytes(range(256)) * 17 + b"una-x cache store"


def _produce() -> bytes:
    """Stand-in for expensive scientific work: deterministic bytes."""
    return BLOB


@pytest.mark.cache_store
def test_roundtrip_is_bitwise(store):
    gen = store.put(K1, {"out": _produce(), "aux": b"\x00\xff\x00"})
    assert gen == 1
    hit = store.get(K1)
    assert hit is not None
    assert hit.hit == "disk"
    assert hit.generation == 1
    assert hit.payload["out"] == _produce()          # == is bitwise for bytes
    assert hashlib.sha256(hit.payload["out"]).hexdigest() \
        == hashlib.sha256(_produce()).hexdigest()
    assert hit.payload["aux"] == b"\x00\xff\x00"


@pytest.mark.cache_store
def test_hit_payloads_are_fresh_copies(store):
    store.put(K1, {"out": _produce()})
    h1 = store.get(K1)
    h2 = store.get(K1)
    assert h1.payload is not h2.payload          # separate dicts
    h1.payload["out"] = b"CLOBBERED"
    h1.payload["injected"] = b"new key"
    assert store.get(K1).payload["out"] == _produce()   # store unaffected
    assert "injected" not in store.get(K1).payload
    assert h2.payload["out"] == _produce()              # sibling hit unaffected


@pytest.mark.cache_store
def test_put_write_through_populates_the_memory_tier(make_store):
    """A put writes through into a configured memory tier: the very next
    get is a memory hit carrying the same bits."""
    store = make_store(mode="disk", max_memory_bytes=1 << 20)
    store.put(K1, {"out": _produce()})
    hit = store.get(K1)
    assert hit.hit == "memory"
    assert hit.payload["out"] == _produce()


@pytest.mark.cache_store
def test_disk_hit_promotes_into_the_memory_tier_bitwise(tmp_path):
    """Disk → memory promotion: a fresh reader over an existing disk store
    serves the first read from disk and identical bits from memory after."""
    d = str(tmp_path / "una-cache")
    writer = ContentStore(CacheOptions(mode="disk", directory=d))
    writer.put(K1, {"out": _produce()})
    reader = ContentStore(CacheOptions(
        mode="disk", directory=d, max_memory_bytes=1 << 20))
    first = reader.get(K1)
    assert first.hit == "disk"
    second = reader.get(K1)
    assert second.hit == "memory"
    assert second.payload["out"] == _produce()
    assert second.generation == first.generation
    assert reader.stats()["hit_reasons"]["memory"] == 1


@pytest.mark.cache_store
def test_cold_warm_off_bitwise_equivalence(tmp_path):
    """Dossier 06 acceptance: the same key delivers the same bits cold
    (first compute), warm (fresh store over the same directory), and with
    the cache off (fresh compute every time)."""
    key = hashlib.sha256(BLOB).hexdigest()
    d = str(tmp_path / "una-cache")

    cold_store = ContentStore(CacheOptions(mode="disk", directory=d))
    assert cold_store.get(key) is None                    # cold miss
    cold_store.put(key, {"out": _produce()},
                   metadata={"profile": "una_legacy", "stage": "flow"})
    cold = cold_store.get(key).payload["out"]

    warm_store = ContentStore(CacheOptions(mode="disk", directory=d))
    warm_hit = warm_store.get(key)
    assert warm_hit is not None and warm_hit.hit == "disk"
    assert warm_hit.payload["out"] == cold                # bitwise
    assert warm_hit.generation == 1
    assert dict(warm_hit.metadata)["profile"] == "una_legacy"

    off_store = ContentStore(CacheOptions(mode="off", directory=d))
    assert off_store.get(key) is None                     # off always computes
    assert off_store.put(key, {"out": _produce()}) == 0   # and stores nothing
    fresh = _produce()
    assert fresh == cold                                  # compute == cached


@pytest.mark.cache_store
def test_warm_reopen_with_stricter_verification_still_exact(tmp_path):
    key = hashlib.sha256(BLOB).hexdigest()
    d = str(tmp_path / "una-cache")
    ContentStore(CacheOptions(
        mode="disk", directory=d, verification="on_read")
    ).put(key, {"out": BLOB})
    strict = ContentStore(CacheOptions(
        mode="disk", directory=d, verification="always", max_memory_bytes=None))
    hit = strict.get(key)
    assert hit is not None and hit.payload["out"] == BLOB
    assert strict.stats()["quarantined"] == 0


@pytest.mark.cache_store
def test_same_content_same_key_two_stores(tmp_path):
    """Content addressing: identical bytes → identical key → identical
    payload, across independent store instances/directories."""
    k = hashlib.sha256(BLOB).hexdigest()
    hits = []
    for i in range(2):
        s = ContentStore(CacheOptions(
            mode="disk", directory=str(tmp_path / f"cache-{i}")))
        s.put(k, {"out": BLOB})
        hits.append(s.get(k).payload["out"])
    assert hits[0] == hits[1] == BLOB


@pytest.mark.cache_store
def test_different_content_different_keys(store):
    b1, b2 = b"result variant A", b"result variant B"
    k1, k2 = hashlib.sha256(b1).hexdigest(), hashlib.sha256(b2).hexdigest()
    assert k1 != k2
    store.put(k1, {"out": b1})
    store.put(k2, {"out": b2})
    assert store.get(k1).payload["out"] == b1
    assert store.get(k2).payload["out"] == b2
    assert store.stats()["entries"] == 2


@pytest.mark.cache_store
def test_reput_is_visible_and_bitwise(store):
    """A re-put swaps generations; readers afterwards see the new bytes —
    never a mix, never a partial file (publication is a rename, so once
    put() returns, every later read lands on the new generation)."""
    store.put(K1, {"out": b"generation-one-payload"})
    store.put(K1, {"out": b"generation-two-payload"})
    for _ in range(3):
        hit = store.get(K1)
        assert hit.generation == 2
        assert hit.payload["out"] == b"generation-two-payload"
