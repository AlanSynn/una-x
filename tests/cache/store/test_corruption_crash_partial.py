"""Corruption, partial writes, crash-mid-write, require() semantics.

Dossier 06 acceptance: a crash mid-write never creates a hit; a corrupted
entry is a quarantined miss (never partially-finite data); require()
distinguishes absent from damaged and never hides evidence.
"""
from __future__ import annotations

import json
import os

import pytest

from urban_network_analysis.Execution import CacheOptions
from urban_network_analysis.cache import CacheCorruption, ContentStore

K1 = "e" * 64
GOOD = b"verified-good-payload"


def _payload_path(store, key=K1, name="out"):
    return os.path.join(store._gen_dir(key), "payload", name)


def _manifest_path(store, key=K1):
    return os.path.join(store._gen_dir(key), "manifest.json")


def _put_good(store, key=K1):
    store.put(key, {"out": GOOD})


@pytest.fixture
def store_off(tmp_path):
    return ContentStore(CacheOptions(
        mode="off", directory=str(tmp_path / "unused")))


@pytest.mark.cache_store
def test_truncated_payload_is_quarantined_miss(store):
    _put_good(store)
    with open(_payload_path(store), "rb") as f:
        data = f.read()
    with open(_payload_path(store), "wb") as f:
        f.write(data[: len(data) // 2])  # truncated on disk
    assert store.get(K1) is None                    # miss — never partial
    s = store.stats()
    assert s["quarantined"] == 1
    assert s["misses"] == 1
    assert not os.path.isdir(store._gen_dir(K1))    # removed from the store…
    assert os.listdir(os.path.join(str(store.options.directory),
                                   "quarantine"))   # …and kept for diagnosis


@pytest.mark.cache_store
def test_bitflip_detected_on_read(store):
    _put_good(store)
    p = _payload_path(store)
    with open(p, "rb") as f:
        data = bytearray(f.read())
    data[0] ^= 0xFF                                  # silent bit rot
    with open(p, "wb") as f:
        f.write(bytes(data))
    assert store.get(K1) is None
    assert store.stats()["quarantined"] == 1


@pytest.mark.cache_store
def test_verification_never_documents_its_trust(store):
    """verification='never' is an explicit caller decision: sizes are still
    enforced, checksums are not re-run.  The policy is observable, not
    hidden — the same damage under the default on_read policy quarantines."""
    trust = ContentStore(CacheOptions(
        mode="disk", directory=str(store.options.directory),
        verification="never"))
    _put_good(trust)
    with open(_payload_path(trust), "wb") as f:
        f.write(b"t" * len(GOOD))  # same length: only checksums could tell
    hit = trust.get(K1)
    assert hit is not None and hit.payload["out"] == b"t" * len(GOOD)
    assert trust.stats()["quarantined"] == 0
    strict = ContentStore(CacheOptions(
        mode="disk", directory=str(store.options.directory)))
    assert strict.get(K1) is None
    assert strict.stats()["quarantined"] == 1


def _rewrite_manifest(store, mutate):
    mp = _manifest_path(store)
    with open(mp, "r", encoding="utf-8") as f:
        m = json.load(f)
    mutate(m)
    with open(mp, "w", encoding="utf-8") as f:
        json.dump(m, f)


@pytest.mark.cache_store
@pytest.mark.parametrize("damage", ["garbage-json", "wrong-key",
                                    "wrong-schema", "generation-not-int",
                                    "payload-entry-missing"])
def test_manifest_damage_variants_are_misses(store, damage):
    _put_good(store)
    mp = _manifest_path(store)
    if damage == "garbage-json":
        with open(mp, "w", encoding="utf-8") as f:
            f.write("{not json")
    elif damage == "wrong-key":
        _rewrite_manifest(store, lambda m: m.update(key="f" * 64))
    elif damage == "wrong-schema":
        _rewrite_manifest(store, lambda m: m.update(schema_version=999))
    elif damage == "generation-not-int":
        _rewrite_manifest(store,
                          lambda m: m.update(generation="not-a-number"))
    elif damage == "payload-entry-missing":
        _rewrite_manifest(store, lambda m: m["payload"].append(
            {"name": "phantom", "sha256": "0" * 64, "bytes": 4}))
    assert store.get(K1) is None
    assert store.stats()["quarantined"] == 1


@pytest.mark.cache_store
def test_manifest_is_not_a_dict(store):
    _put_good(store)
    with open(_manifest_path(store), "w", encoding="utf-8") as f:
        f.write('["a", "list"]')
    assert store.get(K1) is None
    assert store.stats()["quarantined"] == 1


@pytest.mark.cache_store
def test_staging_leftovers_never_create_hits_and_never_block(store):
    """A crash mid-put leaves tmp/<key>-<pid>-<n>/ without a manifest; the
    store must see a miss and keep working (fresh staging names never
    collide with the wreckage)."""
    staged = os.path.join(str(store.options.directory), "tmp",
                          f"{K1}-999-1")
    os.makedirs(os.path.join(staged, "payload"))
    with open(os.path.join(staged, "payload", "out"), "wb") as f:
        f.write(b"half-written")
    # no manifest.json in the staged dir: publication never happened
    assert store.get(K1) is None
    gen = store.put(K1, {"out": GOOD})
    assert gen == 1
    assert store.get(K1).payload["out"] == GOOD
    store.clear()
    assert not os.path.exists(staged)


@pytest.mark.cache_store
def test_crash_before_publish_leaves_no_hit(store, monkeypatch):
    """Kill the publication rename in-flight: the staged tree must never
    become readable, and the store must recover for the next writer."""
    import urban_network_analysis.cache.store as store_mod

    def crashing_rename(src, dst, *a, **kw):
        raise OSError(5, "Simulated crash between staging and publish")
    monkeypatch.setattr(store_mod.os, "rename", crashing_rename)
    with pytest.raises(OSError, match="Simulated crash"):
        store.put(K1, {"out": GOOD})
    monkeypatch.undo()
    assert store.get(K1) is None          # crash mid-write is a miss
    assert store.stats()["entries"] == 0
    # recovery: the next (real) put succeeds and hits
    assert store.put(K1, {"out": GOOD}) == 1
    assert store.get(K1).payload["out"] == GOOD


@pytest.mark.cache_store
def test_crash_during_payload_write_leaves_no_hit(store, monkeypatch):
    """Kill the process 'during' a payload file write (open succeeds, the
    staged file never completes): nothing readable is published."""
    import urban_network_analysis.cache.store as store_mod

    real_open = open

    def crashing_open(path, mode="r", *a, **kw):
        if str(path).endswith(os.path.join("payload", "out")) \
                and "w" in mode and os.sep + "tmp" + os.sep in str(path):
            raise OSError(5, "Simulated crash mid payload write")
        return real_open(path, mode, *a, **kw)
    # raising=False: 'open' is a builtin, not a module attribute — setting a
    # module-global of that name shadows the builtin inside store.py only.
    monkeypatch.setattr(store_mod, "open", crashing_open, raising=False)
    with pytest.raises(OSError, match="mid payload"):
        store.put(K1, {"out": GOOD})
    monkeypatch.undo()
    assert store.get(K1) is None
    assert store.stats()["entries"] == 0


@pytest.mark.cache_store
def test_require_absent_is_keyerror_damaged_is_corruption(store):
    with pytest.raises(KeyError):
        store.require(K1)                 # absent: a different failure
    _put_good(store)
    hit = store.require(K1)
    assert hit.payload["out"] == GOOD
    with open(_payload_path(store), "wb") as f:
        f.write(b"XXXXXXXXXXXXXXXXXX")
    with pytest.raises(CacheCorruption):
        store.require(K1)
    # require() keeps evidence in place (no quarantine side effect):
    assert store.stats()["quarantined"] == 0
    assert os.path.exists(_payload_path(store))
    assert os.path.exists(_manifest_path(store))


@pytest.mark.cache_store
def test_require_on_disabled_store_raises(store_off):
    with pytest.raises(CacheCorruption, match="disabled"):
        store_off.require("a" * 64)


@pytest.mark.cache_store
@pytest.mark.parametrize("damage", [
    "entry-is-string", "entry-missing-bytes", "entry-missing-sha256",
    "name-is-int", "name-traversal", "metadata-not-mapping",
    "generation-missing", "payload-empty-list",
])
def test_structurally_corrupt_manifests_are_misses_not_raises(store, damage):
    """Every field the read path trusts is validated in _read_manifest
    (review MINOR-1): structural damage is ordinary corruption — get()
    misses + quarantines, require() raises CacheCorruption, and no
    manifest-declared name can traverse (the read path applies the same
    name guard as the write path)."""
    _put_good(store)
    mp = _manifest_path(store)
    if damage == "entry-is-string":
        _rewrite_manifest(store, lambda m: m.update(payload=["nope"]))
    elif damage == "entry-missing-bytes":
        _rewrite_manifest(store, lambda m: m["payload"][0].pop("bytes"))
    elif damage == "entry-missing-sha256":
        _rewrite_manifest(store, lambda m: m["payload"][0].pop("sha256"))
    elif damage == "name-is-int":
        _rewrite_manifest(
            store, lambda m: m["payload"][0].update(name=123))
    elif damage == "name-traversal":
        _rewrite_manifest(store, lambda m: m["payload"][0].update(
            name="../../escaped"))
    elif damage == "metadata-not-mapping":
        _rewrite_manifest(store, lambda m: m.update(metadata=[1, 2, 3]))
    elif damage == "generation-missing":
        _rewrite_manifest(store, lambda m: m.pop("generation"))
    elif damage == "payload-empty-list":
        _rewrite_manifest(store, lambda m: m.update(payload=[]))
    assert store.get(K1) is None                     # miss, never a raise
    assert store.stats()["quarantined"] == 1         # …quarantined as damage


@pytest.mark.cache_store
def test_racing_reader_never_quarantines_the_new_generation(store,
                                                            monkeypatch):
    """Deterministic replay of review probe C: a reader that read the OLD
    manifest and is descheduled while a writer re-puts (different payload)
    must observe a plain MISS — it must NOT quarantine the freshly
    published generation (quarantine only when the damage is still
    current)."""
    import io
    import urban_network_analysis.cache.store as store_mod

    store.put(K1, {"out": b"A" * 10})
    real_open = open
    state = {"reput_done": False}

    def racing_open(path, mode="r", *a, **kw):
        if "r" in mode and "w" not in mode \
                and str(path).endswith(os.path.join("payload", "out")) \
                and os.sep + "store" + os.sep in str(path) \
                and not state["reput_done"]:
            state["reput_done"] = True   # reader was descheduled here;
            store.put(K1, {"out": b"B" * 6})  # writer republishes gen 2
            raise OSError(2, "old payload renamed away mid-read")
        return real_open(path, mode, *a, **kw)

    monkeypatch.setattr(store_mod, "open", racing_open, raising=False)
    hit = store.get(K1)
    monkeypatch.undo()
    assert hit is None                                  # plain miss
    assert store.stats()["quarantined"] == 0            # good data untouched
    again = store.get(K1)                               # gen 2 readable
    assert again is not None and again.generation == 2
    assert again.payload["out"] == b"B" * 6


@pytest.mark.cache_store
def test_stale_tmp_and_quarantine_debris_is_swept(store):
    """Crash debris (review probe F1: a real os._exit(9) between the retire
    and publish renames leaks the full retired generation into tmp/) ages
    out — unbounded tmp//quarantine growth would break bounded-data
    discipline at the store/ boundary.  Fresh debris is never touched."""
    import time as _time
    root = str(store.options.directory)
    old_tmp = os.path.join(root, "tmp", f"{K1}-999-1")
    os.makedirs(os.path.join(old_tmp, "payload"))
    with open(os.path.join(old_tmp, MANIFEST), "w", encoding="utf-8") as f:
        f.write("{}")
    old_quar = os.path.join(root, "quarantine", "deadbeef-1-2")
    os.makedirs(old_quar)
    with open(os.path.join(old_quar, "x"), "w", encoding="utf-8") as f:
        f.write("damaged")
    stale = _time.time() - 7 * 3600
    os.utime(old_tmp, (stale, stale))
    os.utime(old_quar, (stale, stale))
    fresh_tmp = os.path.join(root, "tmp", f"{K1}-999-2")
    os.makedirs(fresh_tmp)                       # a live writer's staging
    store.put("a" * 64, {"x": b"trigger"})       # put() sweeps stale first
    assert not os.path.exists(old_tmp)
    assert not os.path.exists(old_quar)
    assert os.path.isdir(fresh_tmp)              # live staging survives
    import shutil as _sh
    _sh.rmtree(fresh_tmp)


MANIFEST = "manifest.json"


@pytest.mark.cache_store
def test_memory_tier_corruption_detected_under_always(make_store):
    """verification='always' re-checksums even memory hits; a tampered
    resident entry falls back to the (still-good) disk copy."""
    store = make_store(mode="disk", max_memory_bytes=1 << 20,
                       verification="always")
    store.put(K1, {"out": GOOD})
    assert store.get(K1).hit == "memory"  # put wrote through into the tier
    # tamper with the resident copy (white-box: simulate memory bit rot)
    with store._mem_lock:
        size, blob, manifest, csums = store._mem[K1]
        store._mem[K1] = (size, {"out": b"rotten"}, manifest, csums)
    hit = store.get(K1)
    assert hit.hit == "disk"                       # fell back to disk
    assert hit.payload["out"] == GOOD
    assert store.stats()["quarantined"] == 1       # rot was accounted
    # the good disk copy was re-cached; the rotten one is gone:
    again = store.get(K1)
    assert again.hit == "memory"
    assert again.payload["out"] == GOOD
