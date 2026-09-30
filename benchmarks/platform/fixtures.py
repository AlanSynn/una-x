"""Deterministic smoke fixtures and manifest factory for the platform
harness (HARNESS one-real-job positive control).

The 3x3 grid is the campaign's L2 smoke workload: 100-unit edges, two
origins, one destination, all CRS:3857.  Everything is written into a
campaign-owned directory; nothing is committed.
"""
from __future__ import annotations

import json
from pathlib import Path

from .manifest import freeze_manifest

CAMPAIGN = "una-platform-2026-09"

_EDGES = []
for row in range(4):
    for col in range(3):
        x0, y0 = col * 100.0, row * 100.0
        if col < 2:
            _EDGES.append((x0, y0, x0 + 100.0, y0))
        if row < 3:
            _EDGES.append((x0, y0, x0, y0 + 100.0))

ORIGINS = [(5.0, 5.0, 10), (205.0, 205.0, 20)]
DESTINATIONS = [(205.0, 5.0, 30)]

SMOKE_SETTINGS_PATCH = {
    "name": "smoke_grid",
    "network_file": "network.geojson",
    "origins_file": "origins.geojson",
    "destinations_file": "destinations.geojson",
    "network_weight_column": "Geometric",
    "origin_weight_column": "Count",
    "destination_weight_column": "Count",
    "search_radius": 1000,
    "output_wStamp": False,
    "output_file_name": "Results",
    "output_geojson": True,
    "output_feather": True,
    "output_csv": False,
    "output_shp": False,
}


def _point_feature(x, y, props):
    return {"type": "Feature",
            "geometry": {"type": "Point", "coordinates": [x, y]},
            "properties": props}


def _line_feature(x0, y0, x1, y1):
    return {"type": "Feature",
            "geometry": {"type": "LineString",
                         "coordinates": [[x0, y0], [x1, y1]]},
            "properties": {"Geometric": 100.0, "Length": 100.0}}


def _geojson(features, crs_name="EPSG:3857"):
    return {
        "type": "FeatureCollection",
        "crs": {"type": "name", "properties": {"name": crs_name}},
        "features": features,
    }


def build_smoke_fixture(root: str | Path) -> dict:
    """Write the deterministic 3x3-grid fixture; return {name: path}."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    paths = {}
    paths["network"] = root / "network.geojson"
    paths["network"].write_text(json.dumps(
        _geojson([_line_feature(*e) for e in _EDGES])) + "\n",
        encoding="utf-8")
    paths["origins"] = root / "origins.geojson"
    paths["origins"].write_text(json.dumps(_geojson([
        _point_feature(x, y, {"Count": c, "OID": f"O{i}"})
        for i, (x, y, c) in enumerate(ORIGINS)])) + "\n", encoding="utf-8")
    paths["destinations"] = root / "destinations.geojson"
    paths["destinations"].write_text(json.dumps(_geojson([
        _point_feature(x, y, {"Count": c, "DID": f"D{i}"})
        for i, (x, y, c) in enumerate(DESTINATIONS)])) + "\n",
        encoding="utf-8")
    return paths


def smoke_manifest(out_path: str | Path, fixture_dir: str | Path, *,
                   analysis: str = "accessibility",
                   backend_requested: str = "reference",
                   expected_identity: dict | None = None,
                   extra_settings: dict | None = None,
                   output_obligations: dict | None = None,
                   resource_policy: dict | None = None) -> dict:
    """Freeze the smoke workload manifest (input hashes computed now)."""
    fixture_dir = Path(fixture_dir)
    settings = dict(SMOKE_SETTINGS_PATCH)
    settings["data_folder"] = str(fixture_dir)
    settings["output_folder"] = str(fixture_dir / "out")
    if extra_settings:
        settings.update(extra_settings)
    workload = {
        "class": "L2_smoke_grid_3x3",
        "inputs": {
            "network": str(fixture_dir / "network.geojson"),
            "origins": str(fixture_dir / "origins.geojson"),
            "destinations": str(fixture_dir / "destinations.geojson"),
        },
    }
    obligations = output_obligations if output_obligations is not None else {
        "files": {
            "Results.geojson": {"format": "geojson"},
            "Results.feather": {"format": "feather"},
        },
    }
    return freeze_manifest(
        out_path, campaign=CAMPAIGN, profile="una_legacy",
        model={"analysis": analysis, "engine": "auto"},
        backend_requested=backend_requested, workload=workload,
        settings_patch=settings, output_obligations=obligations,
        resource_policy=resource_policy or {"max_workers": 1,
                                            "queue_depth": 2,
                                            "memory_limit_bytes": None,
                                            "threads_per_worker": 1},
        expected_identity=expected_identity or {},
        base_dir=str(fixture_dir))
