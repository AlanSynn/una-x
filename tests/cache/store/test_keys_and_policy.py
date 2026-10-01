"""Keys, key policy, non-executable value policy, layout, generations."""
from __future__ import annotations

import hashlib
import json
import os

import pytest

from urban_network_analysis.Execution import CacheOptions
from urban_network_analysis.cache import CacheCorruption, ContentStore

K1 = "a" * 64
K2 = "b" * 64
BAD_KEYS = [
    "A" * 64,                       # uppercase (not hashlib hexdigest form)
    "a" * 63,                       # too short
    "a" * 65,                       # too long
    "g" * 64,                       # not hex
    "a" * 32 + "/../../etc/passwd", # traversal-shaped (and not hex)
    "../" * 16,                     # traversal
    "",                             # empty
    None,                           # not a string
    12345,                          # not a string
    b"a" * 64,                      # bytes, not str
]


@pytest.mark.cache_store
@pytest.mark.parametrize("bad", BAD_KEYS)
def test_invalid_keys_rejected_everywhere(store, bad):
    """One validator gates get/put/require/singleflight/inspect — and it is
    the path-traversal guard: anything not a 64-char lowercase hex digest
    cannot name a filesystem path under the store root."""
    with pytest.raises(ValueError):
        store.put(bad, {"x": b"1"})
    with pytest.raises(ValueError):
        store.get(bad)
    with pytest.raises(ValueError):
        store.inspect(bad)
    import contextlib
    with pytest.raises(ValueError):
        with store.singleflight(bad):
            pass


@pytest.mark.cache_store
def test_path_and_mtime_derived_keys_are_unrepresentable(store):
    """Dossier 06: identity from a path or an mtime is not expressible —
    such a 'key' fails the digest validator, so the old failure mode
    (mtime bumped → silently different data under the same key) cannot be
    constructed against this store."""
    bogus = "network.geojson_mtime_20260101"  # what a path+mtime key looks like
    with pytest.raises(ValueError, match="mtime"):
        store.put(bogus, {"x": b"1"})


@pytest.mark.cache_store
def test_content_addressed_keys_round_trip(store):
    blob = b"city network payload \x00\x01"
    key = hashlib.sha256(blob).hexdigest()
    gen = store.put(key, {"out": blob})
    assert gen == 1
    hit = store.get(key)
    assert hit is not None and hit.payload["out"] == blob  # bitwise
    assert hit.hit == "disk"


@pytest.mark.cache_store
def test_non_cache_options_rejected():
    with pytest.raises(TypeError, match="CacheOptions"):
        ContentStore({"mode": "disk"})


@pytest.mark.cache_store
def test_contract_defaults_are_off_and_create_nothing(tmp_path):
    """CacheOptions() defaults to mode='off': inert store, and NOT EVEN THE
    CACHE DIRECTORY is created (a disabled cache must leave no residue)."""
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    store = ContentStore(CacheOptions(directory=str(cwd / ".una-cache")))
    assert store.disabled
    assert store.get("a" * 64) is None
    assert store.put("a" * 64, {"x": b"1"}) == 0
    assert not (cwd / ".una-cache").exists()
    with store.singleflight("a" * 64) as owner:
        assert owner is True  # compute always stands on a disabled store


@pytest.mark.cache_store
@pytest.mark.parametrize("name", ["a/b", "..", ".", "", "a\\b", "dir/x"])
def test_payload_name_validation(store, name):
    with pytest.raises(ValueError, match="payload name"):
        store.put(K1, {name: b"1"})


@pytest.mark.cache_store
def test_non_bytes_payload_rejected_no_pickle(store):
    """The non-executable guarantee at the door: only raw bytes enter the
    store, so nothing deserialized from it can ever execute (no pickle)."""
    with pytest.raises(TypeError, match="pickled"):
        store.put(K1, {"obj": {"arbitrary": "python object"}})
    with pytest.raises(TypeError):
        store.put(K1, {"obj": [1, 2, 3]})
    assert store.get(K1) is None  # nothing half-entered


@pytest.mark.cache_store
def test_buffer_types_accepted_and_normalized(store):
    store.put(K1, {"b": bytearray(b"abc"), "m": memoryview(b"def")})
    hit = store.get(K1)
    assert hit.payload["b"] == b"abc"
    assert isinstance(hit.payload["b"], bytes)
    assert hit.payload["m"] == b"def"


@pytest.mark.cache_store
def test_empty_payload_rejected(store):
    with pytest.raises(ValueError, match="at least one"):
        store.put(K1, {})


@pytest.mark.cache_store
def test_unknown_verification_mode_rejected(store):
    with pytest.raises(ValueError, match="verification"):
        store.put(K1, {"x": b"1"}, verification="sometimes")


@pytest.mark.cache_store
def test_on_write_verification_reads_back_staged_bytes(store, monkeypatch):
    """on_write is a real check of the write path: the staged bytes are read
    back and compared against the caller's bytes.  A staging layer that
    silently corrupts (simulated: the readback returns flipped bytes) must
    abort the commit — and 'never' must genuinely skip the readback."""
    import io
    import urban_network_analysis.cache.store as store_mod

    real_open = open

    def tampering_open(path, mode="r", *a, **kw):
        if "r" in mode and "w" not in mode \
                and str(path).endswith(os.path.join("payload", "x")) \
                and os.sep + "tmp" + os.sep in str(path):
            with real_open(path, "rb") as f:
                data = f.read()
            return io.BytesIO(data[:-1] + bytes([data[-1] ^ 0xFF]))
        return real_open(path, mode, *a, **kw)

    strict = ContentStore(CacheOptions(
        mode="disk", directory=str(store.options.directory),
        verification="on_write"))
    monkeypatch.setattr(store_mod, "open", tampering_open, raising=False)
    with pytest.raises(CacheCorruption, match="write verification"):
        strict.put(K1, {"x": b"payload"})
    monkeypatch.undo()
    assert store.get(K1) is None  # nothing was committed

    # without tampering, on_write commits normally:
    assert strict.put(K1, {"x": b"payload"}) == 1
    assert store.get(K1).payload["x"] == b"payload"

    # 'never' skips the readback: the same tampering succeeds unnoticed —
    # the caller explicitly bought that tradeoff.
    trusting = ContentStore(CacheOptions(
        mode="disk", directory=str(store.options.directory),
        verification="never"))
    k2 = "f" * 64
    monkeypatch.setattr(store_mod, "open", tampering_open, raising=False)
    gen = trusting.put(k2, {"x": b"payload"})
    monkeypatch.undo()
    assert gen == 1


@pytest.mark.cache_store
def test_disk_layout_and_manifest_checksums(store):
    blob = b"0123456789abcdef" * 4
    store.put(K1, {"out": blob}, metadata={"profile": "una_legacy"})
    gen_dir = store._gen_dir(K1)
    assert gen_dir == os.path.join(str(store.options.directory),
                                   "store", K1[:2], K1)
    assert os.path.isdir(os.path.join(str(store.options.directory), "locks"))
    assert os.path.isdir(os.path.join(str(store.options.directory), "tmp"))
    assert os.path.isdir(os.path.join(str(store.options.directory),
                                      "quarantine"))
    with open(os.path.join(gen_dir, "manifest.json"), "rb") as f:
        manifest = json.loads(f.read().decode("utf-8"))
    assert manifest["schema_version"] == 1
    assert manifest["key"] == K1
    assert manifest["generation"] == 1
    assert manifest["metadata"] == {"profile": "una_legacy"}
    assert manifest["payload"] == [
        {"name": "out", "sha256": hashlib.sha256(blob).hexdigest(),
         "bytes": len(blob)}]
    with open(os.path.join(gen_dir, "payload", "out"), "rb") as f:
        assert f.read() == blob  # what is on disk is exactly what was put


@pytest.mark.cache_store
def test_no_unexpected_files_in_store_tree(store):
    """The store contains only manifests and payload files — nothing that
    a deserializer could be pointed at."""
    blob = b"payload-bytes"
    store.put(K1, {"out": blob, "aux": b"x"})
    root = os.path.join(str(store.options.directory), "store")
    found = []
    for base, _dirs, files in os.walk(root):
        for n in files:
            found.append(n)
    assert sorted(found) == ["aux", "manifest.json", "out"]


@pytest.mark.cache_store
def test_generation_increments_and_swaps(store):
    g1 = store.put(K1, {"x": b"first"})
    g2 = store.put(K1, {"x": b"second"})
    assert g1 == 1 and g2 == 2
    hit = store.get(K1)
    assert hit.generation == 2
    assert hit.payload["x"] == b"second"  # the new generation, bitwise
    assert store.get(K2) is None  # other keys untouched


@pytest.mark.cache_store
def test_metadata_roundtrip(store):
    md = {"semantic_profile": "corrected_v1", "stage": "paths",
          "fingerprint": "f" * 64}
    store.put(K1, {"x": b"1"}, metadata=md)
    assert dict(store.get(K1).metadata) == md
    assert dict(store.inspect(K1)["metadata"]) == md


@pytest.mark.cache_store
def test_inspect_is_raw_and_never_quarantines(store):
    store.put(K1, {"x": b"good"})
    # damage the payload behind the store's back
    p = os.path.join(store._gen_dir(K1), "payload", "x")
    with open(p, "wb") as f:
        f.write(b"tampered")
    raw = store.inspect(K1)
    assert raw["key"] == K1
    assert raw["files"] == {"x": len(b"tampered")}  # raw view, no verdict
    assert raw["on_disk_bytes"] > 0
    assert store.stats()["quarantined"] == 0  # inspect verifies nothing


@pytest.mark.cache_store
def test_schema_version_option_is_honored(tmp_path):
    """CacheOptions.schema_version is real (review MINOR-3): a store
    configured for version 2 writes version-2 manifests and reads them; a
    manifest from another schema version is ordinary corruption (miss +
    quarantine); nonsensical values are rejected loudly, not ignored."""
    d = str(tmp_path / "una-cache")
    v2 = ContentStore(CacheOptions(mode="disk", directory=d,
                                   schema_version=2))
    k = "a" * 64
    assert v2.put(k, {"x": b"v2-entry"}) == 1
    import json as _json
    with open(os.path.join(v2._gen_dir(k), "manifest.json"),
              encoding="utf-8") as f:
        assert _json.load(f)["schema_version"] == 2
    assert v2.get(k).payload["x"] == b"v2-entry"

    # a v1 manifest under the v2 store: not our schema -> miss + quarantine
    with open(os.path.join(v2._gen_dir(k), "manifest.json"),
              encoding="utf-8") as f:
        m = _json.load(f)
    m["schema_version"] = 1
    with open(os.path.join(v2._gen_dir(k), "manifest.json"), "w",
              encoding="utf-8") as f:
        _json.dump(m, f)
    assert v2.get(k) is None
    assert v2.stats()["quarantined"] == 1

    with pytest.raises(ValueError, match="schema_version"):
        ContentStore(CacheOptions(mode="disk", directory=str(tmp_path / "x"),
                                  schema_version=0))


@pytest.mark.cache_store
def test_oversized_memory_mode_put_reports_not_stored(tmp_path):
    """Review NOTE-2: an entry larger than the whole tier is never resident,
    and memory-mode put() says so — generation 0, not a number that implies
    the entry exists."""
    store = ContentStore(CacheOptions(
        mode="memory", directory=str(tmp_path / "una-cache"),
        max_memory_bytes=100))
    gen = store.put("a" * 64, {"x": b"z" * 1000})
    assert gen == 0
    assert store.get("a" * 64) is None
    s = store.stats()
    assert s["entries"] == 0 and s["memory_entries"] == 0


@pytest.mark.cache_store
def test_stats_entries_counts_resident_tier_in_memory_mode(tmp_path):
    store = ContentStore(CacheOptions(
        mode="memory", directory=str(tmp_path / "una-cache"),
        max_memory_bytes=10_000))
    store.put("a" * 64, {"x": b"1"})
    store.put("b" * 64, {"x": b"2"})
    assert store.stats()["entries"] == 2  # review NOTE-2, second half


@pytest.mark.cache_store
def test_stats_shape_and_fresh_counters(store):
    s = store.stats()
    assert s["mode"] == "disk"
    assert s["hits"] == 0 and s["misses"] == 0
    assert s["hit_reasons"] == {"memory": 0, "disk": 0, "miss": 0}
    assert s["entries"] == 0 and s["disk_bytes"] == 0
    assert s["memory_enabled"] is False  # no positive bound configured
    store.put(K1, {"x": b"1"})
    store.get(K1)
    store.get(K2)
    s = store.stats()
    assert s["hits"] == 1 and s["misses"] == 1
    assert s["hit_reasons"] == {"memory": 0, "disk": 1, "miss": 1}
    assert s["entries"] == 1 and s["disk_bytes"] > 0
