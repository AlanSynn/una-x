"""Deterministic genuine-GIS workload fixtures for the UNA CPU campaign.

W1 "small genuine": tiny street-like network with elevation, origins,
destinations and obstacles; exercises loading, CRS, snapping, accessibility,
gravity metrics, all export formats and a flow run whose gravity cap is
derived from an accessibility pass.

W3 "medium proxy": a larger grid used for profiling and CPU-concurrency
selection. PROXY qualification only — the real production workload is
unavailable (WORKLOADS.md W4).

Every file written here is byte-identical across runs on the same
geopandas/shapely versions; the manifest records SHA-256 hashes.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
from shapely.geometry import LineString, Point

CRS = "EPSG:32616"


def _grid_network(gx: int, gy: int, spacing: float, jitter: float, seed: int,
                  with_z: bool):
    """Street-like grid: jittered lattice nodes, edges to right/up neighbors,
    some diagonals. Deterministic under seed."""
    rng = np.random.default_rng(seed)
    node_xy = {}
    k = 0
    for j in range(gy):
        for i in range(gx):
            x = i * spacing + (rng.random() - 0.5) * jitter
            y = j * spacing + (rng.random() - 0.5) * jitter
            z = 0.0
            if with_z:
                # gentle deterministic hills (meters)
                z = 6.0 * np.sin(0.0015 * x) + 4.0 * np.cos(0.0011 * y)
            node_xy[(i, j)] = (x, y, z)
            k += 1
    edges = []
    for j in range(gy):
        for i in range(gx):
            if i < gx - 1:
                edges.append(((i, j), (i + 1, j)))
            if j < gy - 1:
                edges.append(((i, j), (i, j + 1)))
            # sparse deterministic diagonals to create asymmetric topology
            if i < gx - 1 and j < gy - 1 and rng.random() < 0.12:
                edges.append(((i, j), (i + 1, j + 1)))
    geoms, wcol = [], []
    for (a, b) in edges:
        pa = node_xy[a]
        pb = node_xy[b]
        if with_z:
            geoms.append(LineString([(pa[0], pa[1], pa[2]),
                                     (pb[0], pb[1], pb[2])]))
        else:
            geoms.append(LineString([(pa[0], pa[1]), (pb[0], pb[1])]))
        wcol.append(abs(pa[0] - pb[0]) + abs(pa[1] - pb[1]))
    gdf = gpd.GeoDataFrame(
        {"edge_id": np.arange(len(geoms), dtype=np.int64),
         "length_check": np.asarray(wcol, dtype=np.float64)},
        geometry=geoms, crs=CRS)
    return gdf, node_xy


def _points_on_network(gdf, node_xy, n_points, seed, weight_col="weight"):
    """Deterministic points snapped to (already on) random edges at random
    normalized positions. Column `weight` exercises custom-weight loading."""
    rng = np.random.default_rng(seed)
    geoms, weights, pids = [], [], []
    edge_rows = gdf.geometry.values
    for p in range(n_points):
        e = int(rng.integers(0, len(edge_rows)))
        line = edge_rows[e]
        f = 0.15 + 0.7 * rng.random()
        pt = line.interpolate(f, normalized=True)
        geoms.append(Point(pt.x, pt.y))
        weights.append(0.25 + 2.0 * rng.random())
        pids.append(p)
    return gpd.GeoDataFrame(
        {"point_id": np.asarray(pids, dtype=np.int64),
         weight_col: np.asarray(weights, dtype=np.float64)},
        geometry=geoms, crs=CRS)


def _obstacles(node_xy, n, seed):
    rng = np.random.default_rng(seed)
    keys = sorted(node_xy)
    picks = rng.choice(len(keys), size=n, replace=False)
    geoms, pen = [], []
    for idx in picks:
        x, y, _z = node_xy[keys[int(idx)]]
        geoms.append(Point(x, y))
        pen.append(float(2.0 + 8.0 * rng.random()))
    return gpd.GeoDataFrame({"penalty": np.asarray(pen, dtype=np.float64)},
                            geometry=geoms, crs=CRS)


def generate_w1(data_dir: Path):
    """Small genuine fixture: 10x8 grid, elevation, 14 origins, 26
    destinations, 4 obstacles."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    net, node_xy = _grid_network(10, 8, 90.0, 12.0, seed=11, with_z=True)
    origins = _points_on_network(net, node_xy, 14, seed=22)
    dests = _points_on_network(net, node_xy, 26, seed=33)
    obstacles = _obstacles(node_xy, 4, seed=44)
    net.to_file(data_dir / "network.geojson", driver="GeoJSON")
    origins.to_file(data_dir / "origins.geojson", driver="GeoJSON")
    dests.to_file(data_dir / "destinations.geojson", driver="GeoJSON")
    obstacles.to_file(data_dir / "obstacles.geojson", driver="GeoJSON")
    return {
        "name": "W1_small_genuine",
        "class": "genuine-small",
        "network": "network.geojson",
        "origins": "origins.geojson",
        "destinations": "destinations.geojson",
        "obstacles": "obstacles.geojson",
        "nodes_est": 10 * 8,
        "edges_est": len(net),
        "origins_n": 14,
        "destinations_n": 26,
    }


def generate_w3(data_dir: Path):
    """Medium proxy fixture: 92x80 grid, elevation, 2560 origins, 2560
    destinations, 8 obstacles. Sized so a single accessibility job runs in
    the ~2-8 s band on the laptop budget — large enough that CSR/search/
    export contributions are separable, small enough for repeated runs.
    (Recalibrated 2026-09-25 from an initial 46x40/320/640 draft whose
    0.37 s job wall left selection measurements noise-dominated; change
    made before any L3/L4 measurement was recorded. W1 is untouched.)"""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    net, node_xy = _grid_network(92, 80, 55.0, 9.0, seed=101, with_z=True)
    origins = _points_on_network(net, node_xy, 2560, seed=202)
    dests = _points_on_network(net, node_xy, 2560, seed=303)
    obstacles = _obstacles(node_xy, 8, seed=404)
    net.to_file(data_dir / "network.geojson", driver="GeoJSON")
    origins.to_file(data_dir / "origins.geojson", driver="GeoJSON")
    dests.to_file(data_dir / "destinations.geojson", driver="GeoJSON")
    obstacles.to_file(data_dir / "obstacles.geojson", driver="GeoJSON")
    return {
        "name": "W3_medium_proxy",
        "class": "proxy-medium",
        "network": "network.geojson",
        "origins": "origins.geojson",
        "destinations": "destinations.geojson",
        "obstacles": "obstacles.geojson",
        "nodes_est": 92 * 80,
        "edges_est": len(net),
        "origins_n": 2560,
        "destinations_n": 2560,
    }


def hash_dir(root: Path):
    """SHA-256 over sorted relative paths + file contents."""
    root = Path(root)
    entries = []
    for p in sorted(root.rglob("*")):
        if p.is_file():
            rel = str(p.relative_to(root))
            h = hashlib.sha256(p.read_bytes()).hexdigest()
            entries.append({"path": rel, "sha256": h,
                            "bytes": p.stat().st_size})
    return entries


def manifest_for(data_dir: Path, spec: dict):
    files = {}
    for key in ("network", "origins", "destinations", "obstacles"):
        p = Path(data_dir) / spec[key]
        files[key] = {"file": spec[key],
                      "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
                      "bytes": p.stat().st_size}
    out = dict(spec)
    out["files"] = files
    return out


SETTINGS_COMMON = {
    "network_weight_column": "Geometric",
    "network_precision": 3,
    "origin_weight_column": "weight",
    "destination_weight_column": "weight",
    "search_radius": None,          # per workload
    "elevation": True,
    "elevation_penalty": 4,
    "turns": False,
    "knn_weights": (1.0, 1.0, 0.5),
    "knn_decay": "logistic",
    "gravity_beta": 0.001,
    "gravity_plateau": 0,
    "gravity_logistic_midpoint": 500,
    "calculate_reach": True,
    "calculate_exponential_gravity": True,
    "calculate_logistic_gravity": True,
    "calculate_knn_access": True,
    "output_geojson": True,
    "output_feather": True,
    "output_csv": True,
    "output_wStamp": True,
    "csv_delimiter": ",",
    "progressbar": False,
    "logger_verbosity": 0,
}


def settings_for(spec_name: str, data_dir: Path, output_root: Path,
                 analysis: str = "accessibility"):
    """Frozen Settings values per workload, as plain python types."""
    if spec_name == "W1_small_genuine":
        s = dict(SETTINGS_COMMON)
        s.update(search_radius=350)
    elif spec_name == "W3_medium_proxy":
        s = dict(SETTINGS_COMMON)
        s.update(search_radius=1200)
    else:
        raise ValueError(spec_name)
    spec = {"W1_small_genuine": generate_w1, "W3_medium_proxy": generate_w3}
    s.update(
        data_folder=str(data_dir),
        output_folder=str(output_root),
        output_file_name="Results",
        network_file="network.geojson",
        origins_file="origins.geojson",
        destinations_file="destinations.geojson",
        obstacle_points_file="obstacles.geojson",
    )
    if analysis == "flow":
        s.update(
            flow_engine="aggregate_flow",
            flow_decay=True,
            flow_decay_method="gravity_cap",
            flow_gravity_cap="p95",
            flow_destination_weights=True,
            flow_origin_weights=True,
        )
    return s
