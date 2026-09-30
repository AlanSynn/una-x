"""Read-only bridge to the pinned Madina tree + single-site causal
interventions (FAILURES / dossier 03 "mechanism minimization").

An intervention loads the pinned module SOURCE, applies exactly one
documented textual correction (the community PR's fix, or the reported
defect's inverse), and execs it into a fresh module registered in
sys.modules under the original name BEFORE the madina package import.
The tree on disk is never modified; "deliberate restoration" is simply
running the attempt without the intervention.
"""
from __future__ import annotations

import hashlib
import sys
import types

FIXTURE_DIR = "/tmp/una_incidents/fx"


def ensure_fixture() -> dict:
    """Deterministic 3x3-grid fixture (100-unit edges, EPSG:3857), two
    origins, two destinations; byte-stable across runs.  Returns the
    dataset identity recorded in every attempt envelope."""
    import json
    from pathlib import Path

    root = Path(FIXTURE_DIR)
    root.mkdir(parents=True, exist_ok=True)

    def _geojson(features):
        return {"type": "FeatureCollection",
                "crs": {"type": "name", "properties": {"name": "EPSG:3857"}},
                "features": features}

    edges = []
    for row in range(4):
        for col in range(3):
            x0, y0 = col * 100.0, row * 100.0
            if col < 2:
                edges.append([[x0, y0], [x0 + 100.0, y0]])
            if row < 3:
                edges.append([[x0, y0], [x0, y0 + 100.0]])
    streets = _geojson([
        {"type": "Feature",
         "geometry": {"type": "LineString", "coordinates": e},
         "properties": {"Geometric": 100.0}} for e in edges])
    origins = _geojson([
        {"type": "Feature", "geometry": {"type": "Point",
                                         "coordinates": [5.0, 5.0]},
         "properties": {"OID": "O1"}},
        {"type": "Feature", "geometry": {"type": "Point",
                                         "coordinates": [205.0, 205.0]},
         "properties": {"OID": "O2"}}])
    destinations = _geojson([
        {"type": "Feature", "geometry": {"type": "Point",
                                         "coordinates": [205.0, 5.0]},
         "properties": {"DID": "D1"}},
        {"type": "Feature", "geometry": {"type": "Point",
                                         "coordinates": [5.0, 205.0]},
         "properties": {"DID": "D2"}}])
    import hashlib
    shas = {}
    for name, obj in (("streets", streets), ("origins", origins),
                      ("destinations", destinations)):
        blob = (json.dumps(obj, sort_keys=True) + "\n").encode("utf-8")
        path = root / f"{name}.geojson"
        path.write_bytes(blob)
        shas[name] = hashlib.sha256(blob).hexdigest()
    return {"dir": str(root), "crs": "EPSG:3857",
            "sha256": shas,
            "census": {"street_edges": len(edges),
                       "origins": 2, "destinations": 2}}

# The exact pinned source line behind issue #8 / PR #10 (GeoPandas >= 1.0
# GeometryArray lost `.data`).  Community correction: np.array-based
# extraction, as in unmerged PR #10.
I1_GEOMARRAY_DATA = (
    "point_xy = GeoPandaExtractor(geometry_gdf.geometry.values.data)",
    "point_xy = GeoPandaExtractor(np.asarray(geometry_gdf.geometry.values))",
)

# The exact pinned calls behind issue #12 / PR #13 part 1: pd.Series
# `fastpath` was removed in pandas 3.0.  Every pinned occurrence carries a
# trailing comma (some followed by a newline), so the token below covers
# all of them; removing the comma leaves valid argument lists.
I2_FASTPATH_TOKEN = "fastpath=True,"
I2_FASTPATH_REPLACEMENT = ""

# Issue #12 / PR #13 part 2: np.array_split turns a GeoDataFrame into a
# numpy object array under numpy >= 2.0; split by position instead.
I3_ARRAYSPLIT = (
    "splitted_origins = np.array_split(origins, num_procs)",
    "splitted_origins = [origins.iloc[s] for s in __import__('numpy')"
    ".array_split(np.arange(len(origins)), num_procs)]",
)

def ensure_fixture_redundant() -> dict:
    """Deterministic fixture for the redundant-edge default probe (PR #9):
    the 3x3 grid plus TWO edges joining the same node pair (0,0)-(100,100):
    a straight diagonal (weight ~141.42) and a bent polyline through an
    interior vertex (weight ~156.21) -- a redundant pair with different
    weights, so discard-vs-split behavior is observable in the edge set."""
    import json
    from pathlib import Path

    root = Path(FIXTURE_DIR)
    root.mkdir(parents=True, exist_ok=True)
    coords = []
    for row in range(4):
        for col in range(3):
            x0, y0 = col * 100.0, row * 100.0
            if col < 2:
                coords.append([[x0, y0], [x0 + 100.0, y0]])
            if row < 3:
                coords.append([[x0, y0], [x0, y0 + 100.0]])
    # the redundant pair: same endpoints, different weights
    coords.append([[0.0, 0.0], [100.0, 100.0]])                  # ~141.42
    coords.append([[0.0, 0.0], [20.0, 90.0], [100.0, 100.0]])    # ~172.82
    streets = {"type": "FeatureCollection",
               "crs": {"type": "name", "properties": {"name": "EPSG:3857"}},
               "features": [
                   {"type": "Feature",
                    "geometry": {"type": "LineString", "coordinates": e},
                    "properties": {"Geometric": 100.0}} for e in coords]}
    blob = (json.dumps(streets, sort_keys=True) + "\n").encode("utf-8")
    path = root / "streets_redundant.geojson"
    path.write_bytes(blob)
    return {"file": str(path), "sha256": hashlib.sha256(blob).hexdigest(),
            "features": len(streets["features"])}


# module relpath -> interventions touching it (bridge applies ALL of a
# module's named interventions in ONE source pass, so stacking I1+I2 keeps
# both corrections)
_MODULE_OF = {
    "I1_geomarray_data": "madina/zonal/network_utils.py",
    "I2_fastpath": "madina/zonal/network_utils.py",
    "I3_arraysplit": "madina/una/betweenness.py",
}


def module_source(madina_src: str, relpath: str) -> str:
    with open(f"{madina_src}/{relpath}", "r", encoding="utf-8") as fh:
        return fh.read()


def source_sha256(madina_src: str, relpath: str) -> str:
    return hashlib.sha256(
        module_source(madina_src, relpath).encode("utf-8")).hexdigest()


def _load_patched(madina_src: str, relpath: str, modname: str,
                  replacements: list[tuple[str, str]]):
    src = module_source(madina_src, relpath)
    applied = []
    for old, new in replacements:
        count = src.count(old)
        if count == 0:
            raise AssertionError(
                f"intervention site not found in {relpath}: {old!r}")
        src = src.replace(old, new)
        applied.append({"module": relpath, "site": old,
                        "replacement": new, "count": count})
    mod = types.ModuleType(modname)
    mod.__file__ = f"{madina_src}/{relpath} (in-memory intervention copy)"
    code = compile(src, f"<intervention:{relpath}>", "exec")
    exec(code, mod.__dict__)
    sys.modules[modname] = mod
    return applied


def bridge_import(madina_src: str, interventions: list[str] | None = None):
    """Import madina with optional named interventions pre-registered.

    Returns (apply_log, shas) for the attempt envelope.  Must be called
    before any other madina import in the process.
    """
    shas = {"zonal/network_utils.py": source_sha256(
        madina_src, "madina/zonal/network_utils.py"),
        "una/betweenness.py": source_sha256(madina_src,
                                            "madina/una/betweenness.py")}
    applied = []
    want = interventions or []
    # path first: exec'ing the betweenness copy imports the madina package
    if madina_src not in sys.path:
        sys.path.insert(0, madina_src)
    by_module: dict[str, list[tuple[str, str]]] = {}
    for name in want:
        by_module.setdefault(_MODULE_OF[name], []).append(
            {"I1_geomarray_data": I1_GEOMARRAY_DATA,
             "I2_fastpath": (I2_FASTPATH_TOKEN, I2_FASTPATH_REPLACEMENT),
             "I3_arraysplit": I3_ARRAYSPLIT}[name])
    # network_utils is imported by the zonal package chain, betweenness by
    # una.tools; register every patched copy BEFORE the package import
    rel2mod = {"madina/zonal/network_utils.py": "madina.zonal.network_utils",
               "madina/una/betweenness.py": "madina.una.betweenness"}
    for rel, reps in by_module.items():
        applied += _load_patched(madina_src, rel, rel2mod[rel], reps)
    return applied, shas


def build_project(madina_src: str, interventions: list[str] | None = None,
                  node_snapping_tolerance: float = 0.0,
                  streets_file: str | None = None,
                  redundant_edge_treatment: str | None = None):
    """Tiny 3x3-grid project through the documented recipe
    (load_layer x3 -> create_street_network).  Returns (zonal, info)."""
    applied, shas = bridge_import(madina_src, interventions)
    from madina.zonal.zonal import Zonal
    z = Zonal()
    z.load_layer("streets", streets_file or f"{FIXTURE_DIR}/streets.geojson")
    z.load_layer("origins", f"{FIXTURE_DIR}/origins.geojson")
    z.load_layer("destinations", f"{FIXTURE_DIR}/destinations.geojson")
    kwargs = {"source_layer": "streets",
              "node_snapping_tolerance": node_snapping_tolerance,
              "weight_attribute": None}
    if redundant_edge_treatment is not None:
        kwargs["redundant_edge_treatment"] = redundant_edge_treatment
    z.create_street_network(**kwargs)
    info = {
        "interventions_applied": applied,
        "pinned_source_sha256": shas,
        "nodes": int(z.network.nodes.shape[0]),
        "edges": int(z.network.edges.shape[0]),
        "weights": sorted(float(w) for w in z.network.edges["weight"]),
        "fixture_crs": "EPSG:3857",
        "fixture_dir": FIXTURE_DIR,
    }
    return z, info
