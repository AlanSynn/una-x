"""P02 increment B: worker-local prepared reuse across repeated rows.

Fixtures (contract increments.B acceptance):
- PP02  same preparation, different output/beta paths -> served hits,
        bitwise-identical surfaces, FRESH mutable objects per row
        (nothing aliased with the lease or the previous row)
- PP03  file bytes change under a preserved mtime -> different identity,
        MISS (never a stale hit), results match a fresh produce
- PP04  producer-read param changes invalidate EXACTLY the dependent
        layers (per-dependency identity closure: an origins-only change
        leaves the network layer served)
plus the chronology/validation-on-hit law, the UNA_PREPARED kill switch,
the .geojson-first admission boundary, and coexistence with the stage
cache.
"""
from __future__ import annotations

import os
import shutil

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import LineString, Polygon

import urban_network_analysis.batch.prepared as prepared
from urban_network_analysis.Topology import Topology

NETWORK_ARRAYS = (
    "lengths", "weights", "node_points", "start_nodes", "end_nodes",
    "degrees", "z", "start_degrees", "end_degrees", "min_degrees",
    "max_degrees",
)
AP_ARRAYS = ("edge_start_node", "edge_end_node", "weight_to_start",
             "weight_to_end", "node_weight", "nearest_edge_id")


@pytest.fixture(autouse=True)
def _isolated_prepared():
    """The prepared table is process-global: isolate every test."""
    prepared.reset()
    yield
    prepared.reset()


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
    surf["crs_projjson"] = (topo.crs.to_json()
                            if topo.crs is not None else None)
    surf["stage_key"] = getattr(net, "_stage_key", None)
    return surf


def _ap_surface(pt):
    surf = {k: getattr(pt, k, None) for k in AP_ARRAYS}
    surf["geometry_wkb"] = [g.wkb for g in pt.geometry]
    surf["uid_values"] = (list(pt.uid) if pt.uid is not None else None)
    surf["stage_key"] = getattr(pt, "_stage_key", None)
    return surf


def _assert_surface_eq(a, b, *, require_stage_key=True):
    for k, v in a.items():
        if k == "geometry_wkb":
            assert v == b[k], k
        elif k == "crs_projjson":
            assert v == b[k], k
        elif k == "stage_key":
            if require_stage_key:
                # Prepared-engaged rows record identity in every path.
                assert v and v == b[k], k
            else:
                # Pure produced rows (prepared off/unadmitted) may leave
                # identity unset — it must simply agree between rows.
                assert v == b[k], k
        elif isinstance(v, np.ndarray) or v is None or isinstance(b[k], np.ndarray):
            assert _arr_eq(v, b[k]), k
        else:
            assert v == b[k], k


def _run_row(settings):
    t = Topology()
    t.AddNetwork(settings)
    t.AddOrigins(settings)
    t.AddDestinations(settings)
    return t


def test_pp02_worker_local_reuse_identical_and_fresh(smoke_fixture,
                                                     tmp_path,
                                                     make_prep_settings):
    """Three rows over the same inputs: row 1 produces, rows 2-3 are
    served.  Row 3 varies a NON-producer-read setting (output folder) —
    identity must not move.  Every served surface is bitwise identical
    and every mutable object is fresh."""
    s1 = make_prep_settings(smoke_fixture)
    t1 = _run_row(s1)
    st1 = prepared.stats()
    assert st1["topology"]["publishes"] == 1
    assert st1["snap"]["publishes"] == 2
    assert st1["topology"]["hits"] == 0 and st1["snap"]["hits"] == 0
    surf1 = (_network_surface(t1), _ap_surface(t1.origins),
             _ap_surface(t1.destinations))

    s2 = make_prep_settings(smoke_fixture)
    t2 = _run_row(s2)
    st2 = prepared.stats()
    assert st2["topology"]["hits"] == 1, "row 2 network must be served"
    assert st2["snap"]["hits"] == 2, "row 2 snaps must be served"
    surf2 = (_network_surface(t2), _ap_surface(t2.origins),
             _ap_surface(t2.destinations))
    _assert_surface_eq(surf1[0], surf2[0])
    _assert_surface_eq(surf1[1], surf2[1])
    _assert_surface_eq(surf1[2], surf2[2])

    # Fresh mutable outputs: no aliasing with row 1 or with lease state.
    topo_bundle = next(b for b in prepared._table.values()
                       if "n_nodes" in b.payload)
    assert t2.network.geometry is not t1.network.geometry
    assert t2.network.weights is not t1.network.weights
    assert t2.network.weights is not topo_bundle.payload["weights"]
    assert t2.network.geometry is not topo_bundle.payload["geometry"]
    assert t2.origins.geometry is not t1.origins.geometry
    assert t2.origins.node_weight is not t1.origins.node_weight
    # Produced semantics pinned: BuildAccessPoints keeps `raw = gdf` as a
    # local — the attribute is never populated on any path.
    assert t1.destinations.raw is None and t2.destinations.raw is None

    # PP02 variation: a different output path (not a producer-read param)
    # rides the same identity.
    s3 = make_prep_settings(
        smoke_fixture, output_folder=str(tmp_path / "out_beta"))
    t3 = _run_row(s3)
    st3 = prepared.stats()
    assert st3["topology"]["hits"] == 2 and st3["snap"]["hits"] == 4
    _assert_surface_eq(surf1[0], _network_surface(t3))


def test_pp03_bytes_change_same_mtime_never_stale(smoke_fixture, tmp_path,
                                                  make_prep_settings):
    """Rewrite the network file's BYTES while preserving its mtime: the
    identity must move (content, not path/mtime), the row must MISS, and
    results must equal a fresh produce on the new bytes."""
    d = tmp_path / "data"
    shutil.copytree(smoke_fixture, d)
    net_file = d / "network.geojson"
    st = net_file.stat()

    s1 = make_prep_settings(str(d))
    t1 = _run_row(s1)
    st1 = prepared.stats()
    assert st1["topology"]["publishes"] == 1

    gdf = gpd.read_file(net_file, engine="pyogrio", use_arrow=True)
    gdf.geometry = gdf.geometry.translate(xoff=0.001)
    gdf.to_file(net_file, driver="GeoJSON")
    os.utime(net_file, (st.st_atime, st.st_mtime))
    assert net_file.stat().st_mtime == st.st_mtime, "trap must hold: mtime preserved"

    # Changed bytes under the same path+mtime: MISS, no stale hit.
    s2 = make_prep_settings(str(d))
    t2 = _run_row(s2)
    st2 = prepared.stats()
    assert st2["topology"]["hits"] - st1["topology"]["hits"] == 0, \
        "changed bytes must never serve the old lease"
    assert st2["topology"]["misses"] - st1["topology"]["misses"] == 1
    assert st2["topology"]["publishes"] - st1["topology"]["publishes"] == 1
    assert st2["snap"]["hits"] - st1["snap"]["hits"] == 0, \
        "snaps must chain the new parent, not the old lease"
    assert ([g.wkb for g in t1.network.geometry]
            != [g.wkb for g in t2.network.geometry]), \
        "row 2 must reflect the new bytes (translated geometry)"
    assert (t1.network._stage_key != t2.network._stage_key), \
        "identity must move with the bytes"

    # A third row over the NEW bytes is served — the new generation.
    s3 = make_prep_settings(str(d))
    t3 = _run_row(s3)
    st3 = prepared.stats()
    assert st3["topology"]["hits"] - st2["topology"]["hits"] == 1
    assert st3["snap"]["hits"] - st2["snap"]["hits"] == 2
    _assert_surface_eq(_network_surface(t2), _network_surface(t3))


def test_pp04_dependency_invalidation_exact(smoke_fixture,
                                            make_prep_settings):
    """Per-dependency closure: an origins-only producer-read change
    re-keys the snap layers while the network layer stays served; a
    network param change re-keys everything."""
    base = make_prep_settings(smoke_fixture)
    t1 = _run_row(base)
    st1 = prepared.stats()
    assert st1["topology"]["publishes"] == 1
    assert st1["snap"]["publishes"] == 2

    # (a) origins weight column change -> the ORIGINS layer re-keys while
    # the network stays served AND the destinations layer (whose params
    # did not change) also stays served: exact per-dependency invalidation.
    s_o = make_prep_settings(smoke_fixture, origin_weight_column="NoSuchCol")
    t_o = _run_row(s_o)
    st_o = prepared.stats()
    assert st_o["topology"]["hits"] - st1["topology"]["hits"] == 1, \
        "network layer must stay served"
    assert st_o["snap"]["publishes"] - st1["snap"]["publishes"] == 1, \
        "origins must re-key"
    assert st_o["snap"]["hits"] - st1["snap"]["hits"] == 1, \
        "destinations must stay served"
    _assert_surface_eq(_network_surface(t1), _network_surface(t_o))
    _assert_surface_eq(_ap_surface(t1.destinations),
                       _ap_surface(t_o.destinations))

    # (b) network precision change -> network re-keys, snaps chain the new
    # parent and re-key too (both).
    s_p = make_prep_settings(smoke_fixture, network_precision=7)
    t_p = _run_row(s_p)
    st_p = prepared.stats()
    assert st_p["topology"]["hits"] - st_o["topology"]["hits"] == 0, \
        "new precision must miss"
    assert st_p["topology"]["publishes"] - st_o["topology"]["publishes"] == 1
    assert st_p["snap"]["publishes"] - st_o["snap"]["publishes"] == 2, \
        "snaps must chain the new parent"
    assert st_p["snap"]["hits"] - st_o["snap"]["hits"] == 0

    # (c) uid column change -> that layer's params differ; origins re-keys,
    # destinations rides (and the Series-uid path is exercised end to end).
    s_u = make_prep_settings(smoke_fixture, origin_uid_column="OID")
    t_u = _run_row(s_u)
    st_u = prepared.stats()
    assert st_u["topology"]["hits"] - st_p["topology"]["hits"] == 1, \
        "network still served"
    assert st_u["snap"]["publishes"] - st_p["snap"]["publishes"] == 1, \
        "origins re-key"
    assert st_u["snap"]["hits"] - st_p["snap"]["hits"] == 1, \
        "destinations still served"
    assert t_u.origins.uid is not None


def test_validation_on_hit_raises_like_produced(smoke_fixture, tmp_path,
                                                make_prep_settings):
    """Chronology law: a prepared hit re-runs geometry validation — a
    corrupted payload fails closed with the produced message."""
    s1 = make_prep_settings(smoke_fixture)
    t1 = _run_row(s1)

    # Corrupt the published network bundle's input frame (whole-column
    # assignment: chained iloc assignment is a no-op under CoW).
    topo_bundle = next(b for b in prepared._table.values()
                       if "n_nodes" in b.payload)
    src = topo_bundle.payload["input_gdf"]
    geoms = list(src.geometry)
    geoms[0] = Polygon([(0, 0), (1, 0), (1, 1)])
    bad = src.copy()
    bad.geometry = geoms
    assert (bad.geometry.geom_type == "Polygon").any(), "corruption must take"
    topo_bundle.payload["input_gdf"] = bad

    # Produced reference: the same invalid content via the real path.
    d = tmp_path / "poly"
    d.mkdir()
    poly = gpd.GeoDataFrame(
        geometry=[Polygon([(0, 0), (1, 0), (1, 1)])], crs=t1.crs)
    poly.to_file(d / "network.geojson", driver="GeoJSON")
    shutil.copy(os.path.join(str(smoke_fixture), "origins.geojson"),
                d / "origins.geojson")
    shutil.copy(os.path.join(str(smoke_fixture), "destinations.geojson"),
                d / "destinations.geojson")
    s_bad = make_prep_settings(str(d))
    with pytest.raises(ValueError) as prod_exc:
        Topology().AddNetwork(s_bad)

    # Hit path: identical validation, identical message.
    s2 = make_prep_settings(smoke_fixture)
    with pytest.raises(ValueError) as hit_exc:
        Topology().AddNetwork(s2)
    assert str(hit_exc.value) == str(prod_exc.value)

    # Point layers: same law on a corrupted snap bundle.  (The produced
    # path never populates pt.raw, so snap payloads carry no input frame —
    # corrupt the geometry surface the restore validates over.)
    prepared.reset()
    t = _run_row(make_prep_settings(smoke_fixture))
    snap_bundle = next(b for b in prepared._table.values()
                       if "n_points" in b.payload)
    assert snap_bundle.payload["geometry"] is not None
    bad_geoms = list(snap_bundle.payload["geometry"])
    bad_geoms[0] = LineString([(0, 0), (1, 1)])
    src_pts = snap_bundle.payload["geometry"]
    bad_pts = gpd.GeoSeries(bad_geoms, index=src_pts.index, crs=src_pts.crs)
    assert (bad_pts.geom_type == "LineString").any(), "corruption must take"
    snap_bundle.payload["geometry"] = bad_pts
    # The consuming topology must have a loaded network first (the
    # produced path would crash the same way without one).
    s_f = make_prep_settings(smoke_fixture)
    t_f = Topology()
    t_f.AddNetwork(s_f)
    with pytest.raises(ValueError, match="must contain Point geometries"):
        t_f.AddOrigins(s_f)


def test_log_chronology_preserved_on_hits(smoke_fixture, tmp_path,
                                          make_prep_settings):
    """Content-derived log lines (here: the missing-weight-column
    fallback warning) replay on served rows in the produced order."""
    s1 = make_prep_settings(smoke_fixture, origin_weight_column="NoSuchCol")
    t1 = _run_row(s1)

    def build_lines(log_list, *events):
        return [(e["event"], e["details"]) for e in log_list
                if e["event"] in events]

    prod_lines = build_lines(t1.logger.log_list, "Build AccessPoints",
                             "Origins added")
    # The warning actually fired on the produced row.
    assert any("falling back to unit weights" in d for _, d in prod_lines)

    s2 = make_prep_settings(smoke_fixture, origin_weight_column="NoSuchCol")
    t2 = _run_row(s2)
    assert prepared.stats()["snap"]["hits"] >= 1
    hit_lines = build_lines(t2.logger.log_list, "Build AccessPoints",
                            "Origins added")
    # log_list records produce-only v=3 detail lines too, so compare the
    # load-bearing content lines and their ORDER: the built-line fires,
    # then the same warning text replays, then the Origins-added line.
    def extract(lines, needle):
        return [i for i, (_, d) in enumerate(lines) if needle in d]

    built = [i for i, (_, d) in enumerate(hit_lines)
             if "Access points are built from source" in d
             and "origins.geojson" in d]
    warn = extract(hit_lines, "falling back to unit weights")
    added = extract(hit_lines, "Origins added from source")
    assert len(built) == 1 and len(warn) == 1 and len(added) == 1
    assert built[0] < warn[0] < added[0], "chronology must match produced"
    prod_warn = [d for _, d in prod_lines
                 if "falling back to unit weights" in d]
    prod_added = [d for _, d in prod_lines if "Origins added from source" in d]
    assert hit_lines[warn[0]][1] == prod_warn[0]
    assert hit_lines[added[0]][1] == prod_added[0]


def test_kill_switch_disables_engagement(smoke_fixture, make_prep_settings,
                                         monkeypatch):
    """UNA_PREPARED=0: zero engagement, produced-path results unchanged."""
    monkeypatch.setenv("UNA_PREPARED", "0")
    t1 = _run_row(make_prep_settings(smoke_fixture))
    t2 = _run_row(make_prep_settings(smoke_fixture))
    st = prepared.stats()
    assert all(v == 0 for counters in st.values() for v in counters.values())
    _assert_surface_eq(_network_surface(t1), _network_surface(t2),
                       require_stage_key=False)


def test_unadmitted_formats_fall_through(format_dir, make_prep_settings):
    """.geojson-first engagement: feather/parquet rows never touch the
    prepared layer and reproduce through the produced path."""
    d = str(format_dir("feather"))
    s = make_prep_settings(d, network="network.feather",
                           origins="origins.feather",
                           destinations="destinations.feather")
    t1 = _run_row(s)
    t2 = _run_row(s)
    st = prepared.stats()
    assert all(v == 0 for counters in st.values() for v in counters.values())
    _assert_surface_eq(_network_surface(t1), _network_surface(t2),
                       require_stage_key=False)
    _assert_surface_eq(_ap_surface(t1.origins), _ap_surface(t2.origins),
                       require_stage_key=False)


def test_coexistence_with_stage_cache(smoke_fixture, tmp_path,
                                      make_prep_settings, stage_stats):
    """Prepared reuse composes with the stage cache: row 2 is served by
    the prepared layer (short-circuiting the stage lookup) over identical
    stage identities."""
    cache_dir = str(tmp_path / "cache")
    s1 = make_prep_settings(smoke_fixture, cache_dir=cache_dir)
    t1 = _run_row(s1)
    st1 = prepared.stats()
    assert st1["topology"]["publishes"] == 1
    assert st1["snap"]["publishes"] == 2
    # Stage cache recorded the produces too.
    before = stage_stats(s1)

    s2 = make_prep_settings(smoke_fixture, cache_dir=cache_dir)
    t2 = _run_row(s2)
    st2 = prepared.stats()
    assert st2["topology"]["hits"] == 1 and st2["snap"]["hits"] == 2
    # Prepared short-circuits: no additional stage-cache activity.
    after = stage_stats(s2)
    assert after["hits"] == before["hits"]
    assert after["misses"] == before["misses"]
    _assert_surface_eq(_network_surface(t1), _network_surface(t2))
    _assert_surface_eq(_ap_surface(t1.origins), _ap_surface(t2.origins))
