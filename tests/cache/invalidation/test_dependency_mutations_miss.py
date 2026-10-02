"""Every numerical dependency mutation must miss, at stage granularity.

Run 1 seeds the cache; run 2 applies ONE dependency mutation and must show
the exact expected (hits, misses) over the five stored stages — upstream
stages whose closures are untouched legitimately hit, everything downstream
of the mutation misses.  Run 2's artifacts must additionally equal a fully
uncached run of the same mutated settings (a miss recomputes correctly; a
hit was only served where the closure is provably unchanged).

Expected (hits, misses) by mutated dependency:

    network bytes / network_precision / weight column / semantic profile
        -> (0, 5)  everything downstream of the topology stage
    origins / destinations / obstacles bytes, origin uid column
        -> (3, 2)  that layer's snap stage + the result stage
    search_radius, gravity_beta, gravity_plateau, gravity_logistic_midpoint,
    knn_weights, knn_decay, elevation flag, elevation_penalty,
    obstacle snap_to, obstacle direction column
        -> (4, 1)  result stage only (engine-key inputs)
"""
from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import pytest

from tests.cache.invalidation.conftest import (
    make_settings, run_access, hash_out, stats_snapshot, stats_delta)

pytestmark = pytest.mark.cache_graph


# ------------------------------------------------------------ mutations

def _nudge_first_coord(fname, vertex=0):
    def _m(data_dir):
        p = Path(data_dir) / fname
        doc = json.loads(p.read_text())
        doc["features"][0]["geometry"]["coordinates"][vertex][0] += 1.0
        p.write_text(json.dumps(doc))
    return _m


def _nudge_first_point(fname):
    def _m(data_dir):
        p = Path(data_dir) / fname
        doc = json.loads(p.read_text())
        doc["features"][0]["geometry"]["coordinates"][0] += 1.0
        p.write_text(json.dumps(doc))
    return _m


def _bump_obstacle_penalties(data_dir):
    p = Path(data_dir) / "obstacles.geojson"
    gdf = gpd.read_file(p)
    gdf["penalty"] = gdf["penalty"].to_numpy() + 7.25
    gdf.to_file(p, driver="GeoJSON")


def _add_penalty2_column(data_dir):
    p = Path(data_dir) / "obstacles.geojson"
    gdf = gpd.read_file(p)
    gdf["penalty2"] = gdf["penalty"].to_numpy() * 0.5 + 1.0
    gdf.to_file(p, driver="GeoJSON")


# ---------------------------------------------------------------- harness

def _two_runs(tmp_path, workload, cache_dir, mutate=None, **settings_over):
    s1 = make_settings(workload, tmp_path / "out1", cache_mode="disk",
                       cache_dir=cache_dir)
    run_access(s1)
    if mutate is not None:
        mutate(workload)
    s2 = make_settings(workload, tmp_path / "out2", cache_mode="disk",
                       cache_dir=cache_dir, **settings_over)
    before = stats_snapshot(s2)
    run_access(s2)
    hits, misses = stats_delta(s2, before)
    s3 = make_settings(workload, tmp_path / "out3", **settings_over)
    run_access(s3)
    return hits, misses, hash_out(s2), hash_out(s3)


def _assert(hits, misses, h2, h3, expect_hits, expect_misses):
    assert (hits, misses) == (expect_hits, expect_misses), (
        f"expected ({expect_hits}, {expect_misses}), got ({hits}, {misses})")
    assert h2 == h3, "cached run artifacts differ from the uncached run"


# ------------------------------------------------------------ file bytes

@pytest.mark.cache_graph
def test_network_bytes_mutate_misses_all_five(tmp_path, workload, cache_dir):
    h, m, h2, h3 = _two_runs(
        tmp_path, workload, cache_dir, mutate=_nudge_first_coord("network.geojson"))
    _assert(h, m, h2, h3, 0, 5)


@pytest.mark.cache_graph
def test_origins_bytes_mutate_misses_snap_and_result(tmp_path, workload, cache_dir):
    h, m, h2, h3 = _two_runs(
        tmp_path, workload, cache_dir, mutate=_nudge_first_point("origins.geojson"))
    _assert(h, m, h2, h3, 3, 2)


@pytest.mark.cache_graph
def test_destinations_bytes_mutate_misses_snap_and_result(tmp_path, workload, cache_dir):
    h, m, h2, h3 = _two_runs(
        tmp_path, workload, cache_dir,
        mutate=_nudge_first_point("destinations.geojson"))
    _assert(h, m, h2, h3, 3, 2)


@pytest.mark.cache_graph
def test_obstacle_bytes_mutate_misses_snap_and_result(tmp_path, workload, cache_dir):
    h, m, h2, h3 = _two_runs(
        tmp_path, workload, cache_dir, mutate=_bump_obstacle_penalties)
    _assert(h, m, h2, h3, 3, 2)


# ---------------------------------------------------------- topology keys

@pytest.mark.cache_graph
def test_network_precision_mutate_misses_all_five(tmp_path, workload, cache_dir):
    h, m, h2, h3 = _two_runs(tmp_path, workload, cache_dir,
                             network_precision=2)
    _assert(h, m, h2, h3, 0, 5)


@pytest.mark.cache_graph
def test_weight_column_mutate_misses_all_five(tmp_path, workload, cache_dir):
    h, m, h2, h3 = _two_runs(tmp_path, workload, cache_dir,
                             network_weight_column="cost")
    _assert(h, m, h2, h3, 0, 5)


@pytest.mark.cache_graph
def test_semantic_profile_mutate_misses_all_five(tmp_path, workload, cache_dir):
    h, m, h2, h3 = _two_runs(
        tmp_path, workload, cache_dir,
        **{"execution.semantic_profile": "madina_legacy"})
    _assert(h, m, h2, h3, 0, 5)


@pytest.mark.cache_graph
def test_origin_uid_column_mutate_misses_snap_and_result(tmp_path, workload,
                                                         cache_dir):
    h, m, h2, h3 = _two_runs(tmp_path, workload, cache_dir,
                             origin_uid_column="uid")
    _assert(h, m, h2, h3, 3, 2)


# ------------------------------------------------------- result-key params

@pytest.mark.cache_graph
@pytest.mark.parametrize("over", [
    {"search_radius": 500.0},
    {"gravity_beta": 0.002},
    {"gravity_plateau": 10.0},
    {"gravity_logistic_midpoint": 400.0},
    {"knn_weights": [1.0, 0.5]},
    {"knn_decay": "exponential"},
    {"elevation_penalty": 2.0},
    {"elevation": False},
])
def test_engine_param_mutate_misses_result_only(tmp_path, workload, cache_dir,
                                                over):
    h, m, h2, h3 = _two_runs(tmp_path, workload, cache_dir, **over)
    _assert(h, m, h2, h3, 4, 1)


# ------------------------------------------------ obstacle attach-time state

@pytest.mark.cache_graph
def test_obstacle_snap_to_mutate_misses_result_only(tmp_path, workload,
                                                    cache_dir):
    h, m, h2, h3 = _two_runs(tmp_path, workload, cache_dir,
                             obstacle_points_snap_to="node")
    _assert(h, m, h2, h3, 4, 1)


@pytest.mark.cache_graph
def test_obstacle_direction_column_mutate_misses_result_only(tmp_path, workload,
                                                             cache_dir):
    h, m, h2, h3 = _two_runs(tmp_path, workload, cache_dir,
                             obstacle_points_direction_column="direction")
    _assert(h, m, h2, h3, 4, 1)


@pytest.mark.cache_graph
def test_obstacle_penalty_column_name_mutate_misses_snap_and_result(
        tmp_path, workload, cache_dir):
    h, m, h2, h3 = _two_runs(
        tmp_path, workload, cache_dir, mutate=_add_penalty2_column,
        obstacle_points_penalty_column="penalty2")
    _assert(h, m, h2, h3, 3, 2)
