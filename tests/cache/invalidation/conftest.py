"""CACHE_GRAPH invalidation suite fixtures.

Builds a miniature but complete analysis workload (grid network with
elevation, weighted origins/destinations, obstacle points) in tmp_path and
drives the REAL public entry point (``UNA.RunAccessibility``) with per-test
Settings, so every invalidation/reuse claim is made against the same code
path a user runs.

Stage inventory for one RunAccessibility row on this workload (AWE engine,
obstacles loaded) — the exact (hits, misses) expectations in the tests pin
the CACHE_GRAPH v1 stage set:

    1. una:topology:v1     (network decode + BuildTopology)
    2. una:snap:v1         (origins)
    3. una:snap:v1         (destinations)
    4. una:snap:v1         (obstacles)
    5. una:awe:o_access:v1 (accessibility results)
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[3]
for _p in (str(REPO), str(REPO / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import geopandas as gpd  # noqa: E402
import shapely  # noqa: E402

from urban_network_analysis import UNA  # noqa: E402
from urban_network_analysis.Settings import Settings  # noqa: E402
from urban_network_analysis.cache.stages import stage_cache_for  # noqa: E402


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "cache_graph: stage-DAG content-addressed caching integration "
        "(dossier 06 numerical key closure, CACHE_GRAPH)")


# ----------------------------------------------------------------- workload

def _z(x, y):
    return 10.0 * np.sin(x / 300.0) + 7.0 * np.cos(y / 250.0)


def build_workload(data_dir: Path, *, cols=8, rows=6, spacing=90.0,
                   n_origins=10, n_dests=12, n_obstacles=3, seed=7):
    """Deterministic miniature workload; identical args -> identical bytes."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    # --- network: orthogonal grid, 3-D coordinates, a cost column ---
    lines, costs = [], []
    def node(c, r):
        x, y = c * spacing, r * spacing
        return shapely.Point(x, y, _z(x, y))

    def add_line(p, q):
        lines.append(shapely.LineString([p, q]))
        costs.append(round(p.distance(q) * (1.0 + 0.1 * float(np.sin(
            p.x * 0.013 + p.y * 0.017))), 3))

    for r in range(rows):
        for c in range(cols):
            if c + 1 < cols:
                add_line(node(c, r), node(c + 1, r))
            if r + 1 < rows:
                add_line(node(c, r), node(c, r + 1))
    net = gpd.GeoDataFrame(
        {"cost": costs, "geometry": lines}, crs="EPSG:32616")
    net.to_file(data_dir / "network.geojson", driver="GeoJSON")

    # --- access points: interpolations along random edges ---
    def points(n, weight_lo, weight_hi, name):
        geoms, weights, uids = [], [], []
        for i in range(n):
            e = lines[int(rng.integers(len(lines)))]
            f = float(rng.uniform(0.25, 0.75))
            geoms.append(shapely.LineString([e.coords[0], e.coords[-1]])
                         .interpolate(f))
            weights.append(round(float(rng.uniform(weight_lo, weight_hi)), 3))
            uids.append(f"{name}_{i:03d}")
        gdf = gpd.GeoDataFrame(
            {"weight": weights, "uid": uids, "geometry": geoms},
            crs="EPSG:32616")
        return gdf

    points(n_origins, 0.5, 2.0, "org").to_file(
        data_dir / "origins.geojson", driver="GeoJSON")
    points(n_dests, 0.5, 2.0, "dst").to_file(
        data_dir / "destinations.geojson", driver="GeoJSON")

    # --- obstacles: penalty + direction columns ---
    obs_geoms, pen, direc = [], [], ["AB", "BA", "both", "AB"]
    for i in range(n_obstacles):
        e = lines[int(rng.integers(len(lines)))]
        obs_geoms.append(e.interpolate(float(rng.uniform(0.3, 0.7))))
        pen.append([5.0, 12.5, 2.0][i % 3])
    gpd.GeoDataFrame(
        {"penalty": pen[:n_obstacles], "direction": direc[:n_obstacles],
         "uid": [f"obs_{i:02d}" for i in range(n_obstacles)],
         "geometry": obs_geoms}, crs="EPSG:32616").to_file(
        data_dir / "obstacles.geojson", driver="GeoJSON")
    return data_dir


# ----------------------------------------------------------------- settings

def make_settings(data_dir, out_dir, *, cache_mode="off", cache_dir=None,
                  **over):
    s = Settings()
    s.data_folder = str(data_dir)
    s.output_folder = str(out_dir)
    s.output_file_name = "Results"
    s.network_file = "network.geojson"
    s.origins_file = "origins.geojson"
    s.destinations_file = "destinations.geojson"
    s.obstacle_points_file = "obstacles.geojson"
    s.origin_weight_column = "weight"
    s.destination_weight_column = "weight"
    s.search_radius = 350.0
    s.elevation = True
    s.elevation_penalty = 4.0
    s.network_weight_column = "Geometric"
    s.network_precision = 3
    s.knn_weights = [1.0, 1.0, 0.5]
    s.knn_decay = "logistic"
    s.gravity_beta = 0.001
    s.gravity_plateau = 0.0
    s.gravity_logistic_midpoint = 500.0
    s.calculate_reach = True
    s.calculate_exponential_gravity = True
    s.calculate_logistic_gravity = True
    s.calculate_knn_access = True
    s.output_geojson = True
    s.output_feather = True
    s.output_csv = True
    s.output_wStamp = False
    s.csv_delimiter = ","
    s.progressbar = False
    s.logger_verbosity = 0
    for k, v in over.items():
        if k.startswith("execution."):
            s._apply_execution_key(k, v)
        else:
            setattr(s, k, v)
    if cache_dir is not None:
        s._apply_execution_key("execution.cache.mode", cache_mode)
        s._apply_execution_key("execution.cache.directory", cache_dir)
    return s


# --------------------------------------------------------------------- runs

def run_access(s):
    una = UNA(verbosity=0)
    una.settings = s
    una.RunAccessibility()
    return una


def hash_out(s):
    """Sorted (relative path, sha256) over everything the run published."""
    root = Path(s.output_folder)
    return [(str(p.relative_to(root)), hashlib.sha256(p.read_bytes()).hexdigest())
            for p in sorted(root.rglob("*")) if p.is_file()]


def stats_snapshot(s):
    st = stage_cache_for(s.execution.cache).stats()
    return {"hits": st["hits"], "misses": st["misses"]}


def stats_delta(s, before):
    st = stage_cache_for(s.execution.cache).stats()
    return (st["hits"] - before["hits"], st["misses"] - before["misses"])


# ----------------------------------------------------------------- fixtures

@pytest.fixture
def workload(tmp_path):
    data = tmp_path / "data"
    build_workload(data)
    return data


@pytest.fixture
def cache_dir(tmp_path):
    return str(tmp_path / "una-cache")


@pytest.fixture
def seeded(workload, cache_dir, tmp_path):
    """One cold cached run; returns (settings, artifact-hash) of that run."""
    s = make_settings(workload, tmp_path / "out_seed", cache_mode="disk",
                      cache_dir=cache_dir)
    run_access(s)
    return s, hash_out(s)
