"""MADINA_WORKFLOWS scoping probe 2: isolate the betweenness.py:858
UnboundLocalError trigger at the workflow level.
Variants (all reference arm, legacy venv):
  A: pairing Turn_Penalty=0, workflow num_cores=1
  B: pairing Turn_Penalty=30, workflow num_cores=8 (default)
  C: probe-1 repeat (Turn_Penalty=0, nc=8) twice -> pydeck html md5
     determinism + betweenness_record.csv md5 determinism
Run: probe_wf2.py <repo_root>
"""
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString, Point

REPO = Path(sys.argv[1])
REF_SRC = REPO / ".refs" / "madina_ref" / "src"
BASE = Path("/tmp/wf_probe2")

import os  # noqa: E402
sys.path.insert(0, str(REF_SRC))
os.environ["USE_PYGEOS"] = "0"


def make_fixture(folder: Path, turn_penalty: float, num_origins: int = 3):
    folder.mkdir(parents=True, exist_ok=True)
    streets = [
        ("F-A", 100.0, LineString([(-100, 0), (0, 0)])),
        ("A-B", 100.0, LineString([(0, 0), (100, 0)])),
        ("B-C", 100.0, LineString([(100, 0), (100, 100)])),
        ("C-D", 100.0, LineString([(100, 100), (0, 100)])),
        ("D-A", 100.0, LineString([(0, 100), (0, 0)])),
        ("A-C", float(np_hypot()), LineString([(0, 0), (100, 100)])),
        ("D-E", 50.0, LineString([(0, 100), (0, 150)])),
        ("C-G", 100.0, LineString([(100, 100), (200, 100)])),
    ]
    gpd.GeoDataFrame(
        {"id": [r[0] for r in streets], "length": [r[1] for r in streets]},
        geometry=[r[2] for r in streets], crs="EPSG:3857",
    ).to_file(folder / "streets.geojson", driver="GeoJSON", engine="pyogrio")
    pts = [Point((-50, 0)), Point((50, 0)), Point((100, 50))][:num_origins]
    gpd.GeoDataFrame(
        {"weight": [1.0, 2.0, 1.0][:num_origins]}, geometry=pts,
        crs="EPSG:3857",
    ).to_file(folder / "origins.geojson", driver="GeoJSON", engine="pyogrio")
    gpd.GeoDataFrame(
        {"weight": [2.0, 3.0]},
        geometry=[Point((50, 100)), Point((0, 150))], crs="EPSG:3857",
    ).to_file(folder / "destinations.geojson", driver="GeoJSON",
              engine="pyogrio")
    pd.DataFrame([{
        "Flow_Name": "huff_flow",
        "Origin_Name": "origins", "Origin_File": "origins.geojson",
        "Origin_Weight": "Count",
        "Destination_Name": "destinations",
        "Destination_File": "destinations.geojson",
        "Destination_Weight": "Count",
        "Network_File": "streets.geojson", "Network_Cost": "Geometric",
        "Turn_Penalty": turn_penalty, "Turn_Threshold": 45, "Turns": False,
        "Radius": 250, "Detour": 1.0, "Decay": False,
        "Decay_Mode": "exponent", "Beta": 0.003, "Elastic_Weights": False,
        "KNN_Weight": None, "Plateau": 0, "Closest_destination": True,
    }]).to_csv(folder / "pairings.csv", index=False)


def np_hypot():
    import math  # noqa: PLC0415
    return math.hypot(100, 100)


def run(tag, data, out, num_cores=None):
    from madina.una.workflows import betweenness_flow_simulation  # noqa: PLC0415
    kw = {} if num_cores is None else {"num_cores": num_cores}
    err = None
    try:
        betweenness_flow_simulation(data_folder=str(data),
                                    output_folder=str(out), **kw)
    except Exception as exc:  # noqa: BLE001
        err = f"{type(exc).__name__}: {exc}"
    rec = out / "betweenness_record.csv"
    h = hashlib.md5(rec.read_bytes()).hexdigest() if rec.is_file() else None
    htmls = {
        p.name: hashlib.md5(p.read_bytes()).hexdigest()
        for p in sorted(out.rglob("*.html"))}
    csv_txt = rec.read_text() if rec.is_file() else ""
    return {
        "tag": tag, "error": err, "record_md5": h,
        "html_md5s": htmls,
        "record_head": csv_txt.splitlines()[:3],
        "betweenness_col": [
            ln.split(",") for ln in csv_txt.splitlines()[1:6]],
    }


results = []
for tag, tp, nc in [("A_tp0_nc1", 0, 1), ("B_tp30_nc8", 30, None)]:
    d, o = BASE / tag / "data", BASE / tag / "out"
    shutil.rmtree(BASE / tag, ignore_errors=True)
    make_fixture(d, tp)
    results.append(run(tag, d, o, num_cores=nc))

# C: probe-1 config twice for determinism
for rep in (1, 2):
    d, o = BASE / f"C{rep}" / "data", BASE / f"C{rep}" / "out"
    shutil.rmtree(BASE / f"C{rep}", ignore_errors=True)
    make_fixture(d, 0)
    results.append(run(f"C{rep}_tp0_nc8", d, o))

print("PROBE2_JSON<<<")
print(json.dumps(results, indent=1))
print(">>>")
