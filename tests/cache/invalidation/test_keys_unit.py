"""Unit-level injectivity and encoding discipline of the stage identity.

The canonical encoder must be injective on every value class that can enter
a key (including -0.0 vs 0.0 and NaN), refuse everything it cannot
represent, and embed the environment fingerprint that the profile and the
compiler stack vary on.
"""
from __future__ import annotations

import numpy as np
import pytest

from urban_network_analysis.cache.identity import (
    IDENTITY_SCHEMA, arrays_digest, compute_env, file_digest, geometry_env,
    stage_key)
from urban_network_analysis.cache.stages import decode_arrays, encode_arrays

pytestmark = pytest.mark.cache_graph


def _k(**deps):
    return stage_key("test:stage", deps)


@pytest.mark.cache_graph
def test_negzero_and_zero_get_different_keys():
    assert _k(a=0.0) != _k(a=-0.0)


@pytest.mark.cache_graph
def test_nan_payloads_get_one_canonical_key():
    # All NaNs are one canonical tag: identity is by "is NaN", not payload.
    assert _k(a=float("nan")) == _k(a=np.float64(np.nan))


@pytest.mark.cache_graph
def test_infinities_are_distinguished():
    keys = {_k(a=float("inf")), _k(a=float("-inf")), _k(a=1e308)}
    assert len(keys) == 3


@pytest.mark.cache_graph
def test_int_vs_float_distinguished():
    assert _k(a=1) != _k(a=1.0)


@pytest.mark.cache_graph
def test_bool_not_confused_with_int():
    assert _k(a=True) != _k(a=1)
    assert _k(a=False) != _k(a=0)


@pytest.mark.cache_graph
def test_mapping_order_is_irrelevant():
    assert _k(a=1, b=2) == _k(b=2, a=1)


@pytest.mark.cache_graph
def test_ndarray_dtype_shape_and_content_sensitivity():
    base = np.array([1.0, 2.0, 3.0])
    assert _k(a=base) == _k(a=np.array([1.0, 2.0, 3.0]))
    assert _k(a=base) != _k(a=base + 1)            # content
    assert _k(a=base) != _k(a=base.reshape(1, 3))  # shape
    assert _k(a=base) != _k(a=base.astype(np.float32))  # dtype
    assert _k(a=np.asfortranarray(base.reshape(1, 3))) == \
        _k(a=np.ascontiguousarray(base.reshape(1, 3)))  # layout-insensitive


@pytest.mark.cache_graph
def test_unrepresentable_types_raise():
    with pytest.raises(TypeError):
        _k(a=object())
    with pytest.raises(TypeError):
        _k(a={1, 2})
    with pytest.raises(TypeError):
        _k(a={1: "int keys not allowed"})


@pytest.mark.cache_graph
def test_schema_bump_changes_every_key():
    k1 = stage_key("s", {"a": 1}, schema=IDENTITY_SCHEMA)
    k2 = stage_key("s", {"a": 1}, schema=IDENTITY_SCHEMA + 1)
    assert k1 != k2


@pytest.mark.cache_graph
def test_stage_name_in_the_key():
    assert stage_key("s1", {"a": 1}) != stage_key("s2", {"a": 1})


@pytest.mark.cache_graph
def test_arrays_digest_none_vs_absent():
    assert arrays_digest({"x": None}) == arrays_digest({"x": None})
    assert arrays_digest({"x": None}) != arrays_digest({"x": np.array([1.0])})
    assert arrays_digest({"x": np.array([1.0])}) == \
        arrays_digest({"x": np.array([1.0])})


@pytest.mark.cache_graph
def test_file_digest_bytes_only():
    assert file_digest(b"abc") == file_digest(b"abc")
    assert file_digest(b"abc") != file_digest(b"abd")


@pytest.mark.cache_graph
def test_environment_fingerprints_carry_profile_and_compiler_stack():
    geo_a = geometry_env("una_legacy", "1.2.3")
    geo_b = geometry_env("corrected_v1", "1.2.3")
    assert geo_a["profile"] == "una_legacy"
    assert geo_a != geo_b
    comp = compute_env("una_legacy", "1.2.3")
    assert comp["kind"] == "compute"
    assert "numba" in comp["libs"] and "llvmlite" in comp["libs"]
    # keys embedding the two envs must differ
    assert _k(env=geo_a) != _k(env=geo_b)
    assert _k(env=geo_a) != _k(env=comp)


@pytest.mark.cache_graph
def test_npy_encoding_roundtrip_is_exact_and_fresh():
    arrays = {
        "f64": np.array([0.0, -0.0, np.inf, np.nan, -1.5e-300]),
        "i64": np.array([-3, 0, 2 ** 62], dtype=np.int64),
        "str": np.array(["pid_001", "pid_002"], dtype="<U7"),
    }
    blobs = encode_arrays(arrays)
    back = decode_arrays(blobs)
    for name, original in arrays.items():
        assert back[name].dtype == original.dtype
        assert back[name].tobytes() == original.tobytes()  # bitwise
        assert back[name] is not original                  # fresh object


@pytest.mark.cache_graph
def test_npy_encoding_refuses_object_and_void():
    with pytest.raises(TypeError):
        encode_arrays({"bad": np.array([{"a": 1}], dtype=object)})
    with pytest.raises(TypeError):
        encode_arrays({"bad": np.zeros(2, dtype=[("x", "f8")])})


@pytest.mark.cache_graph
def test_npy_encoding_skips_none_entries():
    blobs = encode_arrays({"keep": np.array([1.0]), "drop": None})
    assert set(blobs) == {"keep"}


@pytest.mark.cache_graph
def test_mixed_dimension_geometry_is_not_cacheable():
    import shapely
    from urban_network_analysis.Topology import _geoms_to_wkb_payload
    mixed = np.array([
        shapely.LineString([(0, 0), (1, 1)]),
        shapely.LineString([(0, 0, 0), (1, 1, 1)]),
    ])
    wkb, lens, coord_dim = _geoms_to_wkb_payload(mixed, "test")
    assert wkb is None and lens is None and coord_dim is None


@pytest.mark.cache_graph
def test_wkb_payload_roundtrip_preserves_dimensions():
    import shapely
    from urban_network_analysis.Topology import (
        _geoms_from_wkb_payload, _geoms_to_wkb_payload)
    for z in (False, True):
        geoms = np.array([
            shapely.LineString([(0, 0, 5), (3, 0, 7)] if z
                               else [(0, 0), (3, 0)]),
            shapely.Point(1, 2, 3) if z else shapely.Point(1, 2),
        ])
        wkb, lens, coord_dim = _geoms_to_wkb_payload(geoms, "test")
        assert coord_dim == (3 if z else 2)
        back = _geoms_from_wkb_payload(wkb, lens)
        assert back.shape == geoms.shape  # EVERY geometry recovered
        assert np.array_equal(shapely.get_type_id(back),
                              shapely.get_type_id(geoms))
        assert np.array_equal(
            shapely.get_coordinates(back, include_z=True),
            shapely.get_coordinates(geoms, include_z=True),
            equal_nan=True)
        # re-encode is byte-stable
        wkb2, lens2, _ = _geoms_to_wkb_payload(back, "test")
        assert wkb2.tobytes() == wkb.tobytes() and np.array_equal(lens2, lens)


@pytest.mark.cache_graph
def test_compute_once_under_threads(tmp_path):
    """Concurrent identical keys compute once and both callers see the
    same committed generation (single-flight at the stage facade)."""
    import threading
    from urban_network_analysis.Execution import CacheOptions
    from urban_network_analysis.cache.stages import StageCache
    from urban_network_analysis.cache.store import ContentStore

    cache = StageCache(ContentStore(CacheOptions(
        mode="disk", directory=str(tmp_path / "sf-cache"))))
    calls = {"n": 0}
    lock = threading.Lock()
    key = _k(worker="compute-once")

    def produce():
        with lock:
            calls["n"] += 1
        return {"v": np.array([1.0, 2.0])}, {"stage": "t"}

    barrier = threading.Barrier(4)
    results = []

    def worker():
        barrier.wait()
        arrays, _meta, hit = cache.arrays_once(key, produce)
        results.append((arrays["v"][0], hit is None))

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert calls["n"] == 1
    assert len(results) == 4
    producers = sum(1 for _, ran_produce in results if ran_produce)
    assert producers == 1  # exactly one caller ran the producer
