"""MADINA_WORKFLOWS scoping probe 3: upstream KNN_accessibility
end-to-end (reference arm) + ValueError probes for both workflows.
Run: probe_wf3_knn.py <repo_root>
"""
import json
import os
import shutil
import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString, Point

REPO = Path(sys.argv[1])
sys.path.insert(0, str(REPO / ".refs" / "madina_ref" / "src"))
os.environ["USE_PYGEOS"] = "0"

from madina.una.workflows import KNN_accessibility, betweenness_flow_simulation  # noqa: E402

BASE = Path("/tmp/wf_probe3")


def make_fixture(folder: Path, rows: list[dict], fname: str):
    folder.mkdir(parents=True, exist_ok=True)
    streets = [
        ("F-A", 100.0, LineString([(-100, 0), (0, 0)])),
        ("A-B", 100.0, LineString([(0, 0), (100, 0)])),
        ("B-C", 100.0, LineString([(100, 0), (100, 100)])),
        ("C-D", 100.0, LineString([(100, 100), (0, 100)])),
        ("D-A", 100.0, LineString([(0, 100), (0, 0)])),
        ("A-C", 141.4213562373095, LineString([(0, 0), (100, 100)])),
        ("D-E", 50.0, LineString([(0, 100), (0, 150)])),
        ("C-G", 100.0, LineString([(100, 100), (200, 100)])),
    ]
    gpd.GeoDataFrame(
        {"id": [r[0] for r in streets], "length": [r[1] for r in streets]},
        geometry=[r[2] for r in streets], crs="EPSG:3857",
    ).to_file(folder / "streets.geojson", driver="GeoJSON", engine="pyogrio")
    gpd.GeoDataFrame(
        {"weight": [1.0, 2.0, 1.0]},
        geometry=[Point((-50, 0)), Point((50, 0)), Point((100, 50))],
        crs="EPSG:3857",
    ).to_file(folder / "origins.geojson", driver="GeoJSON", engine="pyogrio")
    gpd.GeoDataFrame(
        {"weight": [2.0, 3.0]},
        geometry=[Point((50, 100)), Point((0, 150))], crs="EPSG:3857",
    ).to_file(folder / "destinations.geojson", driver="GeoJSON",
              engine="pyogrio")
    pd.DataFrame(rows).to_csv(folder / fname, index=False)


base_row = {
    "Flow_Name": "knn_a",
    "Origin_Name": "origins", "Origin_File": "origins.geojson",
    "Origin_Weight": "Count",
    "Destination_Name": "destinations",
    "Destination_File": "destinations.geojson",
    "Destination_Weight": "Count",
    "Network_File": "streets.geojson", "Network_Cost": "Geometric",
    "Turn_Penalty": 0, "Turn_Threshold": 45, "Turns": False,
    "Radius": 250, "Beta": 0.003, "KNN_Weight": "[0.5,0.5]", "Plateau": 0,
}
rows2 = [dict(base_row, Flow_Name="knn_a"),
         dict(base_row, Flow_Name="knn_b", Radius=200)]

out = {"runs": {}}


def list_tree(p: Path):
    return {str(f.relative_to(p)): f.stat().st_size
            for f in sorted(p.rglob("*")) if f.is_file()}


# --- run 1: two pairings, default pairing.csv name, num_cores=1 --------
d, o = BASE / "knn" / "data", BASE / "knn" / "out"
shutil.rmtree(BASE / "knn", ignore_errors=True)
make_fixture(d, rows2, "pairing.csv")
err = None
try:
    KNN_accessibility(city_name="probe_city", data_folder=str(d),
                      output_folder=str(o), num_cores=1)
except Exception as exc:  # noqa: BLE001
    err = f"{type(exc).__name__}: {exc}"
run1 = {"error": err, "files": list_tree(o) if o.is_dir() else None}
rec = o / "origin_record.csv"
if rec.is_file():
    import hashlib
    run1["origin_record_md5"] = hashlib.md5(rec.read_bytes()).hexdigest()
    df = pd.read_csv(rec)
    run1["origin_cols"] = list(df.columns)
    run1["origin_rows"] = df[[c for c in df.columns
                              if c != "geometry"]].to_dict("records")
out["runs"]["knn_two_pairings"] = run1

# --- run 2: ValueError probes -----------------------------------------
probes = {}
try:
    betweenness_flow_simulation()
except Exception as exc:  # noqa: BLE001
    probes["flow_noargs"] = f"{type(exc).__name__}: {exc}"
try:
    KNN_accessibility()
except Exception as exc:  # noqa: BLE001
    probes["knn_city_none"] = f"{type(exc).__name__}: {exc}"
try:
    betweenness_flow_simulation(data_folder=str(BASE / "nope"))
except Exception as exc:  # noqa: BLE001
    probes["flow_missing_data_folder"] = f"{type(exc).__name__}: {exc}"
out["valueerrors"] = probes

print("PROBE3_JSON<<<")
print(json.dumps(out, indent=1, default=str))
print(">>>")
