"""P02 increment A: Add*-state equivalence between the produced path and
the existing stage-cache restore path.

These fixtures are the evidence the errata demands before the topology
serializers may be reused by a prepared layer ("existence != full state
coverage"): the restored state must match the produced state over EXACTLY
the attribute surfaces the engine census (prepared_consumer_audit.md §4)
consumes — arrays bitwise (shape AND dtype), geometry via WKB bytes, CRS
via identity, and the derived spatial index by query parity.

Also pins the uid-dtype guard observable (Topology.py:448-462): an
unsupported uid dtype attaches the identity and SKIPS the store — a later
run misses again ("miss never wrong hit") and still reproduces the result.
"""
from __future__ import annotations

import numpy as np
import pytest

from urban_network_analysis.Topology import Topology


@pytest.fixture(autouse=True)
def _prepared_off(monkeypatch):
    """This module certifies the STAGE-cache restore path.  The prepared
    layer (increment B) short-circuits BEFORE stage lookups when enabled,
    which would starve the served-hit observables below — pin it off for
    this module's stage-cache evidence."""
    monkeypatch.setenv("UNA_PREPARED", "0")


LAYERS = ("network", "origins", "destinations")

NETWORK_ARRAYS = (
    "lengths", "weights", "node_points", "start_nodes", "end_nodes",
    "degrees", "z", "start_degrees", "end_degrees", "min_degrees",
    "max_degrees",
)
AP_ARRAYS = ("edge_start_node", "edge_end_node", "weight_to_start",
             "weight_to_end", "node_weight")


def _arr_eq(a, b):
    if a is None or b is None:
        return a is None and b is None
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    if a.dtype.kind == "f":
        return np.array_equal(a, b, equal_nan=True)
    return np.array_equal(a, b)


def _network_surface(topo):
    net = topo.network
    surf = {k: getattr(net, k) for k in NETWORK_ARRAYS}
    surf["geometry_wkb"] = [g.wkb for g in net.geometry]
    surf["crs_projjson"] = topo.crs.to_json() if topo.crs is not None else None
    surf["stage_key"] = getattr(net, "_stage_key", None)
    return surf


def _ap_surface(pt):
    surf = {k: getattr(pt, k, None) for k in AP_ARRAYS}
    surf["geometry_wkb"] = ([g.wkb for g in pt.geometry]
                            if pt.geometry is not None else None)
    surf["crs"] = getattr(pt, "crs", None)
    surf["stage_key"] = getattr(pt, "_stage_key", None)
    return surf


def _assert_surface_eq(prod, rest, keys):
    for k in keys:
        if k == "geometry_wkb":
            assert prod[k] == rest[k], k
        elif k == "crs":
            assert (prod[k] is None) == (rest[k] is None), k
            if prod[k] is not None:
                assert prod[k] == rest[k], k
        elif k == "stage_key":
            assert prod[k] and prod[k] == rest[k], k
        else:
            assert _arr_eq(prod[k], rest[k]), k


def test_network_state_produced_vs_restored(smoke_fixture, tmp_path,
                                            make_prep_settings, stage_stats):
    """AddNetwork cache miss (produce) vs second run on the same cache root
    (restore): every engine-consumed surface identical, identity identical,
    and the spatial index answers queries identically."""
    cache_dir = str(tmp_path / "cache")
    s1 = make_prep_settings(smoke_fixture, cache_dir=cache_dir)
    t1 = Topology()
    t1.AddNetwork(s1)
    produced = _network_surface(t1)
    assert produced["stage_key"]

    s2 = make_prep_settings(smoke_fixture, cache_dir=cache_dir)
    before = stage_stats(s2)
    t2 = Topology()
    t2.AddNetwork(s2)
    hits_delta = stage_stats(s2)["hits"] - before["hits"]
    assert hits_delta >= 1, "second AddNetwork must be a served stage hit"

    restored = _network_surface(t2)
    _assert_surface_eq(produced, restored, list(NETWORK_ARRAYS) + [
        "geometry_wkb", "crs_projjson", "stage_key"])

    # Derived state parity: the sindex is rebuilt, not stored — the restore
    # is valid only if it answers identically (contract restore_validity).
    probe = t1.network.geometry.iloc[0].interpolate(0.5)
    a = t1.network.edge_sindex.nearest(probe, return_all=False)
    b = t2.network.edge_sindex.nearest(probe, return_all=False)
    assert np.array_equal(np.asarray(a), np.asarray(b))


def test_accesspoints_state_produced_vs_restored(smoke_fixture, tmp_path,
                                                 make_prep_settings,
                                                 stage_stats):
    """AddOrigins/AddDestinations produce vs restore: snap arrays, weights,
    geometry, CRS and identity all identical, both roles served from the
    stage cache on the second run.  Array dtypes and row order are pinned
    inside _arr_eq (shape AND dtype equality before comparison)."""
    cache_dir = str(tmp_path / "cache")
    s1 = make_prep_settings(smoke_fixture, cache_dir=cache_dir)
    t1 = Topology()
    t1.AddNetwork(s1)  # snap needs the network's sindex
    t1.AddOrigins(s1)
    t1.AddDestinations(s1)
    prod_o, prod_d = _ap_surface(t1.origins), _ap_surface(t1.destinations)
    assert prod_o["stage_key"] and prod_d["stage_key"]

    s2 = make_prep_settings(smoke_fixture, cache_dir=cache_dir)
    before = stage_stats(s2)
    t2 = Topology()
    t2.AddNetwork(s2)
    t2.AddOrigins(s2)
    t2.AddDestinations(s2)
    hits_delta = stage_stats(s2)["hits"] - before["hits"]
    assert hits_delta >= 2, "both access-point layers must be served hits"

    rest_o, rest_d = _ap_surface(t2.origins), _ap_surface(t2.destinations)
    _assert_surface_eq(prod_o, rest_o,
                       list(AP_ARRAYS) + ["geometry_wkb", "crs",
                                          "stage_key"])
    _assert_surface_eq(prod_d, rest_d,
                       list(AP_ARRAYS) + ["geometry_wkb", "crs",
                                          "stage_key"])


def test_uid_dtype_guard_skips_store(tmp_path, smoke_fixture,
                                     make_prep_settings, stage_stats,
                                     smoke_layers):
    """uid dtype kind outside {i,f,u,b,U} attaches the stage key and skips
    the store: the second run misses again (never a stale/wrong hit) and
    the result still reproduces exactly.  An int uid DOES hit."""
    gpd = pytest.importorskip("geopandas")
    data = tmp_path / "data_uid"
    data.mkdir(parents=True)
    smoke_layers["network"].to_file(data / "network.geojson",
                                    driver="GeoJSON")
    results = {}
    n = len(smoke_layers["origins"])
    for kind, col in (
            ("int", np.arange(n, dtype=np.int64)),
            ("str", np.array([f"o{i:02d}" for i in range(n)],
                             dtype=object))):
        gdf = smoke_layers["origins"].copy()
        gdf["uid"] = col
        gdf.to_file(data / f"origins_{kind}.geojson", driver="GeoJSON")
        cache_dir = str(tmp_path / f"cache_{kind}")
        common = dict(cache_dir=cache_dir,
                      origins=f"origins_{kind}.geojson",
                      origin_uid_column="uid")
        s1 = make_prep_settings(data, **common)
        t1 = Topology()
        t1.AddNetwork(s1)
        t1.AddOrigins(s1)
        results[kind] = _ap_surface(t1.origins)

        s2 = make_prep_settings(data, **common)
        t2 = Topology()
        t2.AddNetwork(s2)  # network stage may hit — outside the window
        before = stage_stats(s2)  # window covers AddOrigins ONLY
        t2.AddOrigins(s2)
        hits_delta = stage_stats(s2)["hits"] - before["hits"]
        results[f"{kind}_hits_delta"] = hits_delta
        results[f"{kind}_restore"] = _ap_surface(t2.origins)

    assert results["int_hits_delta"] >= 1, \
        "supported uid dtype must be served from the stage cache"
    assert results["str_hits_delta"] == 0, \
        "unsupported uid dtype must skip the store (miss never wrong hit)"
    _assert_surface_eq(results["int"], results["int_restore"],
                       list(AP_ARRAYS) + ["geometry_wkb", "crs",
                                          "stage_key"])
    _assert_surface_eq(results["str"], results["str_restore"],
                       list(AP_ARRAYS) + ["geometry_wkb", "crs",
                                          "stage_key"])
