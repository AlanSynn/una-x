"""L2 error-path probe: exercises documented invalid-input failures and
prints their exception types/messages as JSON for baseline-vs-candidate
comparison. Executed with PYTHONPATH bound to one arm's source tree."""
from __future__ import annotations

import json
import os
import sys


def probe(data_dir, out_dir):
    results = {}
    from urban_network_analysis import UNA

    # 1: missing network file
    s_kwargs = dict(data_folder=str(data_dir), network_file="does_not_exist.geojson",
                    origins_file="origins.geojson",
                    destinations_file="destinations.geojson",
                    output_folder=str(out_dir), progressbar=False,
                    logger_verbosity=0)
    una = UNA(verbosity=0)
    _apply(una, s_kwargs)
    try:
        una.RunAccessibility()
        results["missing_network"] = {"raised": False}
    except Exception as exc:  # noqa: BLE001
        results["missing_network"] = {"raised": True,
                                      "type": type(exc).__name__,
                                      "message": str(exc)}

    # 2: destination CRS mismatch
    import geopandas as gpd
    from pathlib import Path
    bad = Path(out_dir) / "bad_crs_destinations.geojson"
    gdf = gpd.read_file(Path(data_dir) / "destinations.geojson")
    gdf = gdf.set_crs("EPSG:4326", allow_override=True)
    gdf.to_file(bad, driver="GeoJSON")
    una2 = UNA(verbosity=0)
    _apply(una2, dict(s_kwargs, network_file="network.geojson",
                      destinations_file=bad.name))
    try:
        una2.RunAccessibility()
        results["crs_mismatch"] = {"raised": False}
    except Exception as exc:  # noqa: BLE001
        results["crs_mismatch"] = {"raised": True,
                                   "type": type(exc).__name__,
                                   "message": str(exc)}

    # 3: polygon network rejected
    from shapely.geometry import Polygon
    poly = gpd.GeoDataFrame({"a": [1]}, geometry=[Polygon([(0, 0), (1, 0), (1, 1)])],
                            crs="EPSG:32616")
    badnet = Path(out_dir) / "polygon_network.geojson"
    poly.to_file(badnet, driver="GeoJSON")
    una3 = UNA(verbosity=0)
    _apply(una3, dict(s_kwargs, network_file=badnet.name))
    try:
        una3.RunAccessibility()
        results["polygon_network"] = {"raised": False}
    except Exception as exc:  # noqa: BLE001
        results["polygon_network"] = {"raised": True,
                                      "type": type(exc).__name__,
                                      "message": str(exc)}
    return results


def _apply(una, kwargs):
    import numpy as np
    for k, v in kwargs.items():
        setattr(una.settings, k, v)
    una.settings.knn_weights = np.asarray(una.settings.knn_weights, dtype=np.float64)


if __name__ == "__main__":
    out = probe(sys.argv[1], sys.argv[2])
    print("@@PROBE@@" + json.dumps(out))
