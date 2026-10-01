"""MADINA_WORKFLOWS scoping probe 1 (fixture half): write the file-based
data_folder for the pairing workflows. Mirrors the FLOW-suite grid
(streets 8-edge loop+diagonal, 3 origins, 2 destinations, EPSG:3857,
weight column 'length'/'weight') as GeoJSON files plus pairings.csv.
Run: python probe_wf1_fixture.py <data_folder> [--two-origins]
"""
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import LineString, Point

data_folder = Path(sys.argv[1])
two_origins = "--two-origins" in sys.argv
data_folder.mkdir(parents=True, exist_ok=True)

streets = [
    ("F-A", 100.0, LineString([(-100, 0), (0, 0)])),
    ("A-B", 100.0, LineString([(0, 0), (100, 0)])),
    ("B-C", 100.0, LineString([(100, 0), (100, 100)])),
    ("C-D", 100.0, LineString([(100, 100), (0, 100)])),
    ("D-A", 100.0, LineString([(0, 100), (0, 0)])),
    ("A-C", float(np.hypot(100, 100)), LineString([(0, 0), (100, 100)])),
    ("D-E", 50.0, LineString([(0, 100), (0, 150)])),
    ("C-G", 100.0, LineString([(100, 100), (200, 100)])),
]
gpd.GeoDataFrame(
    {"id": [r[0] for r in streets], "length": [r[1] for r in streets]},
    geometry=[r[2] for r in streets], crs="EPSG:3857",
).to_file(data_folder / "streets.geojson", driver="GeoJSON", engine="pyogrio")

origin_pts = [Point((-50, 0)), Point((50, 0)), Point((100, 50))]
origin_w = [1.0, 2.0, 1.0]
if two_origins:
    origin_pts, origin_w = origin_pts[:2], origin_w[:2]
gpd.GeoDataFrame(
    {"weight": origin_w}, geometry=origin_pts, crs="EPSG:3857",
).to_file(data_folder / "origins.geojson", driver="GeoJSON", engine="pyogrio")

gpd.GeoDataFrame(
    {"weight": [2.0, 3.0]},
    geometry=[Point((50, 100)), Point((0, 150))], crs="EPSG:3857",
).to_file(data_folder / "destinations.geojson", driver="GeoJSON",
          engine="pyogrio")

pairings = pd.DataFrame([{
    "Flow_Name": "huff_flow",
    "Origin_Name": "origins",
    "Origin_File": "origins.geojson",
    "Origin_Weight": "Count",
    "Destination_Name": "destinations",
    "Destination_File": "destinations.geojson",
    "Destination_Weight": "Count",
    "Network_File": "streets.geojson",
    "Network_Cost": "Geometric",
    "Turn_Penalty": 0,
    "Turn_Threshold": 45,
    "Turns": False,
    "Radius": 250,
    "Detour": 1.0,
    "Decay": False,
    "Decay_Mode": "exponent",
    "Beta": 0.003,
    "Elastic_Weights": False,
    "KNN_Weight": None,
    "Plateau": 0,
    "Closest_destination": True,
}])
pairings.to_csv(data_folder / "pairings.csv", index=False)
print("fixture written:", sorted(p.name for p in data_folder.iterdir()))
