"""Parity scenario runner (TOPOLOGY): build a deterministic scenario
through either the compat facade or the bridged pinned reference, then
dump a BITWISE-strict state digest.

Both arms run under the SAME interpreter (the dependency-bridged
reference venv) with an identically seeded stdlib RNG, so any difference
in the digest is attributable to the code under test, not to the
environment.  Floats are compared as IEEE-754 bit patterns; geometries
as WKB hashes; dtypes/categories/order as recorded.

Usage:
    python _scenario.py --arm {facade,reference} --scenario NAME
                        --out DIGEST.json --seed 42 [--sabotage NAME]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import sys
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


# ----------------------------------------------------------------------
# deterministic fixtures (byte-stable; same recipe as tests/incidents)
# ----------------------------------------------------------------------

def _fc(features):
    return {"type": "FeatureCollection",
            "crs": {"type": "name", "properties": {"name": "EPSG:3857"}},
            "features": features}


def _line(coords, props=None):
    return {"type": "Feature",
            "geometry": {"type": "LineString", "coordinates": coords},
            "properties": props or {"Geometric": 100.0}}


def _point(xy, props):
    return {"type": "Feature",
            "geometry": {"type": "Point", "coordinates": list(xy)},
            "properties": props}


def _grid_edges():
    edges = []
    for row in range(4):
        for col in range(3):
            x0, y0 = col * 100.0, row * 100.0
            if col < 2:
                edges.append([[x0, y0], [x0 + 100.0, y0]])
            if row < 3:
                edges.append([[x0, y0], [x0, y0 + 100.0]])
    return edges


def write_fixtures(root: str, variant: str) -> dict:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    paths = {}

    if variant == "grid":
        feats = [_line(e) for e in _grid_edges()]
    elif variant == "redundant":
        coords = _grid_edges()
        coords.append([[0.0, 0.0], [100.0, 100.0]])                 # ~141.42
        coords.append([[0.0, 0.0], [20.0, 90.0], [100.0, 100.0]])   # ~172.82
        feats = [_line(e) for e in coords]
    elif variant == "snap30":  # endpoints 30 units off the grid nodes
        feats = [_line(e) for e in _grid_edges()]
        feats.append(_line([[200.0, 30.0], [230.0, 0.0]]))
    elif variant == "prep":  # Polygon + LineStrings, ALL with Z coordinates
        # (pinned prepare_geometry only strips Z when has_z.any(); a purely
        # 2D layer skips the transform, a MIXED layer crashes _to_2d — that
        # mixed crash is pinned behavior, covered by error:prep_mixed_z.
        # MultiLineString cannot be used here at all: GeoPandaExtractor
        # crashes on multi-part geometries — error:csn_multilinestring.)
        feats = [
            {"type": "Feature",
             "geometry": {"type": "Polygon",
                          "coordinates": [[[0, 0, 0], [100, 0, 0], [100, 100, 0],
                                           [0, 100, 0], [0, 0, 0]]]},
             "properties": {"Geometric": 100.0}},
            {"type": "Feature",
             "geometry": {"type": "LineString",
                          "coordinates": [[0, 0, 1.5], [0, 200, 1.5]]},
             "properties": {"Geometric": 100.0}},
            {"type": "Feature",
             "geometry": {"type": "LineString",
                          "coordinates": [[100, 0, 2.5], [100, 200, 2.5]]},
             "properties": {"Geometric": 100.0}},
            {"type": "Feature",
             "geometry": {"type": "LineString",
                          "coordinates": [[200, 0, 7.5], [200, 100, 3.25]]},
             "properties": {"Geometric": 100.0}},
        ]
    elif variant == "mls_mixed":  # MLS + LineString: pinned GeoPandaExtractor crash
        # prepare_geometry only unwraps MLS when ALL rows are MLS, so the
        # multi-part geometry reaches GeoPandaExtractor and .coords raises
        feats = [
            {"type": "Feature",
             "geometry": {"type": "MultiLineString",
                          "coordinates": [[[0, 0], [0, 200]],
                                          [[100, 0], [100, 200]]]},
             "properties": {"Geometric": 100.0}},
            {"type": "Feature",
             "geometry": {"type": "LineString",
                          "coordinates": [[200, 0], [200, 100]]},
             "properties": {"Geometric": 100.0}},
        ]
    elif variant == "mls_pure":  # ALL-MLS: pinned takes geoms[0] (first part only)
        feats = [
            {"type": "Feature",
             "geometry": {"type": "MultiLineString",
                          "coordinates": [[[0, 0], [0, 200]],
                                          [[100, 0], [100, 200]]]},
             "properties": {"Geometric": 100.0}},
            {"type": "Feature",
             "geometry": {"type": "MultiLineString",
                          "coordinates": [[[150, 0], [150, 100], [150, 200]],
                                          [[250, 0], [250, 200]]]},
             "properties": {"Geometric": 100.0}},
        ]
    elif variant == "prep_mixed":  # 2D features + one 3D feature
        feats = [
            {"type": "Feature",
             "geometry": {"type": "Polygon",
                          "coordinates": [[[0, 0], [100, 0], [100, 100], [0, 100], [0, 0]]]},
             "properties": {"Geometric": 100.0}},
            {"type": "Feature",
             "geometry": {"type": "LineString",
                          "coordinates": [[200, 0, 7.5], [200, 100, 3.25]]},
             "properties": {"Geometric": 100.0}},
        ]
    elif variant == "od":
        feats = [_line(e) for e in _grid_edges()]
    else:
        raise ValueError(variant)

    paths["streets"] = str(root / "streets.geojson")
    Path(paths["streets"]).write_text(json.dumps(_fc(feats)))

    od = variant == "od"
    origins = [_point((5.0, 5.0), {"OID": "O1"}),
               _point((205.0, 205.0), {"OID": "O2"})]
    dests = [_point((205.0, 5.0), {"DID": "D1"}),
             _point((5.0, 205.0), {"DID": "D2"})]
    if od:  # an origin and a destination sitting ON distinct grid nodes,
        # plus a coincident pair on the same node (same-edge/OD mechanism)
        origins.append(_point((100.0, 0.0), {"OID": "O3"}))
        origins.append(_point((100.0, 0.0), {"OID": "O4"}))
        dests.append(_point((100.0, 200.0), {"DID": "D3"}))
    paths["origins"] = str(root / "origins.geojson")
    Path(paths["origins"]).write_text(json.dumps(_fc(origins)))
    paths["destinations"] = str(root / "destinations.geojson")
    Path(paths["destinations"]).write_text(json.dumps(_fc(dests)))
    return paths


def _weighted_streets(root: str) -> str:
    # includes a zero-weight feature (pinned max(w, 0.01) branch) and
    # fractional weights
    feats = [_line(e, {"cost": 100.0}) for e in _grid_edges()[:10]]
    feats.append(_line([[200, 0], [200, 100]], {"cost": 0.0}))
    feats.append(_line([[300, 0], [300, 100]], {"cost": 12.5}))
    p = Path(root) / "streets_w.geojson"
    p.write_text(json.dumps(_fc(feats)))
    return str(p)


def _crs4326_streets(root: str) -> str:
    # same grid, geographic CRS (pinned does NOT project; parity keeps that)
    feats = [{"type": "Feature",
              "geometry": {"type": "LineString",
                           "coordinates": [[x / 111320.0, y / 111320.0],
                                           [(x + dx) / 111320.0,
                                            (y + dy) / 111320.0]],
                           },
              "properties": {"Geometric": 100.0}}
             for (x, y, dx, dy) in
             [(0, 0, 100, 0), (100, 0, 100, 0), (200, 0, 0, 100),
              (0, 100, 100, 0), (100, 100, 100, 0), (200, 100, 0, 100),
              (0, 200, 100, 0), (100, 200, 100, 0), (200, 200, 0, 100),
              (0, 0, 0, 100), (100, 0, 0, 100), (200, 0, 0, 100),
              (0, 100, 0, 100), (100, 100, 0, 100), (200, 100, 0, 100),
              (0, 200, 0, 100), (100, 200, 0, 100)]]
    p = Path(root) / "streets_4326.geojson"
    p.write_text(json.dumps(
        {"type": "FeatureCollection",
         "crs": {"type": "name", "properties": {"name": "EPSG:4326"}},
         "features": feats}))
    return str(p)


# ----------------------------------------------------------------------
# arm import shim
# ----------------------------------------------------------------------

def import_zonal(arm: str):
    if arm == "facade":
        sys.path.insert(0, str(REPO / "src"))
        from urban_network_analysis.compat.madina.zonal import Zonal
    elif arm == "reference":
        sys.path.insert(0, str(REPO / ".refs" / "madina_ref" / "src"))
        from madina.zonal.zonal import Zonal
    else:
        raise ValueError(arm)
    return Zonal


# ----------------------------------------------------------------------
# bitwise digest
# ----------------------------------------------------------------------

def _f(v) -> str:
    import struct
    return struct.pack("<d", float(v)).hex()


def _geom_sha(geom) -> str:
    return hashlib.sha256(geom.wkb).hexdigest()


def _gdf_digest(gdf) -> dict:
    cols = {}
    for col in gdf.columns:
        s = gdf[col]
        if col == "geometry":
            cols[col] = {"dtype": str(s.dtype),
                         "values": [_geom_sha(g) for g in s]}
        elif str(s.dtype) == "category":
            cols[col] = {"dtype": str(s.dtype),
                         "categories": [str(c) for c in s.cat.categories],
                         "values": [str(v) for v in s]}
        else:
            import pandas as pd
            import numpy as np
            if pd.api.types.is_float_dtype(s):
                vals = [None if v != v else _f(v) for v in s]  # NaN-stable
            elif pd.api.types.is_integer_dtype(s):
                vals = [int(v) for v in s]
            elif pd.api.types.is_bool_dtype(s):
                vals = [bool(v) for v in s]
            else:
                vals = [str(v) for v in s]
            cols[col] = {"dtype": str(s.dtype), "values": vals}
    return {"columns": list(gdf.columns),
            "dtypes": {c: str(gdf[c].dtype) for c in gdf.columns},
            "index_name": gdf.index.name,
            "index": [int(i) for i in gdf.index],
            "crs": str(gdf.crs),
            "columns_digest": cols}


def _graph_digest(graph) -> dict:
    import networkx as nx  # noqa: F401  (attribute access only)
    edges = sorted(
        (int(u), int(v), _f(d["weight"]), int(d.get("id", -1)))
        for u, v, d in graph.edges(data=True))
    nodes = sorted((int(n), str(d.get("type")))
                   for n, d in graph.nodes(data=True))
    return {"edges": edges, "nodes": nodes,
            "added_nodes": [int(n) for n in graph.graph.get("added_nodes", [])]}


def zonal_digest(z) -> dict:
    digest = {
        "geo_center": ([None, None] if z.geo_center == (None, None)
                       else [_f(z.geo_center[0]), _f(z.geo_center[1])]),
        "layer_order": list(z.layers.layers),
        "layers": {
            name: {"crs": str(z.layers[name].crs),
                   "show": bool(z.layers[name].show),
                   "file_path": str(z.layers[name].file_path),
                   "default_color": z.layers[name].default_color,
                   "gdf": _gdf_digest(z.layers[name].gdf)}
            for name in z.layers.layers},
    }
    if z.network is not None:
        digest["network"] = {
            "weight_attribute": z.network.weight_attribute,
            "edge_source_layer": z.network.edge_source_layer,
            "turn_threshold_degree": (_f(z.network.turn_threshold_degree)
                                      if z.network.turn_threshold_degree is not None
                                      else None),
            "turn_penalty_amount": (_f(z.network.turn_penalty_amount)
                                    if z.network.turn_penalty_amount is not None
                                    else None),
            "street_node_ids": sorted(int(i) for i in
                                      (z.network.street_node_ids or ())),
            "nodes": _gdf_digest(z.network.nodes),
            "edges": _gdf_digest(z.network.edges),
            "light_graph": (_graph_digest(z.network.light_graph)
                            if z.network.light_graph else None),
            "d_graph": (_graph_digest(z.network.d_graph)
                        if z.network.d_graph else None),
            "od_graph": (_graph_digest(z.network.od_graph)
                         if z.network.od_graph else None),
        }
    return digest


# ----------------------------------------------------------------------
# scenarios
# ----------------------------------------------------------------------

def build_od(z, paths, weight=None):
    z.insert_node(layer_name="origins", label="origin",
                  weight_attribute=weight)
    z.insert_node(layer_name="destinations", label="destination",
                  weight_attribute=weight)


def scenario_grid_basic(Zonal, root):
    paths = write_fixtures(root, "grid")
    random.seed(20260930)
    z = Zonal()
    z.load_layer("streets", paths["streets"])
    z.load_layer("origins", paths["origins"])
    z.load_layer("destinations", paths["destinations"])
    z.create_street_network(source_layer="streets",
                            node_snapping_tolerance=0.0,
                            weight_attribute=None)
    build_od(z, paths)
    z.create_graph()
    return z


def scenario_redundant(Zonal, root, treatment):
    paths = write_fixtures(root, "redundant")
    random.seed(20260930)
    z = Zonal()
    z.load_layer("streets", paths["streets"])
    z.load_layer("origins", paths["origins"])
    z.load_layer("destinations", paths["destinations"])
    z.create_street_network(source_layer="streets",
                            node_snapping_tolerance=0.0,
                            redundant_edge_treatment=treatment)
    build_od(z, paths)
    z.create_graph()
    return z


def scenario_tolerance(Zonal, root):
    paths = write_fixtures(root, "snap30")
    random.seed(20260930)
    z = Zonal()
    z.load_layer("streets", paths["streets"])
    z.create_street_network(source_layer="streets",
                            node_snapping_tolerance=30.0,
                            weight_attribute=None)
    return z


def scenario_custom_weight(Zonal, root):
    paths = write_fixtures(root, "grid")
    sw = _weighted_streets(root)
    random.seed(20260930)
    z = Zonal()
    z.load_layer("streets", sw)
    z.load_layer("origins", paths["origins"])
    z.load_layer("destinations", paths["destinations"])
    z.create_street_network(source_layer="streets",
                            node_snapping_tolerance=0.0,
                            weight_attribute="cost")
    build_od(z, paths)
    z.create_graph()
    return z


def scenario_prep(Zonal, root):
    paths = write_fixtures(root, "prep")
    random.seed(20260930)
    z = Zonal()
    z.load_layer("streets", paths["streets"])
    z.create_street_network(source_layer="streets",
                            node_snapping_tolerance=0.0)
    return z


def scenario_mls_pure(Zonal, root):
    # pinned prepare_geometry unwraps an all-MLS layer by taking the FIRST
    # part of every multi-part geometry (silent truncation quirk, kept)
    paths = write_fixtures(root, "mls_pure")
    random.seed(20260930)
    z = Zonal()
    z.load_layer("streets", paths["streets"])
    z.create_street_network(source_layer="streets",
                            node_snapping_tolerance=0.0)
    return z


def scenario_layer_order(Zonal, root):
    paths = write_fixtures(root, "grid")
    random.seed(20260930)
    z = Zonal()
    z.load_layer("streets", paths["streets"])
    z.load_layer("origins", paths["origins"])
    z.load_layer("destinations", paths["destinations"])
    z.load_layer("extra", paths["origins"], pos=1)
    z.load_layer("top", paths["destinations"], first=True)
    z.load_layer("before_streets", paths["origins"], before="streets")
    z.load_layer("after_streets", paths["destinations"], after="streets")
    # Layers dunders
    _ = z.layers["streets"], z.layers[0], str(z.layers)
    assert "streets" in z.layers
    _ = [name for name in z.layers]
    _ = z.layers.get_layer_at_pos(2), z.layers.pos_at_label("streets")
    z.layers["streets"].gdf = z.layers["streets"].gdf  # __setitem__ str path
    return z


def scenario_clear_reinsert(Zonal, root):
    paths = write_fixtures(root, "grid")
    random.seed(20260930)
    z = Zonal()
    z.load_layer("streets", paths["streets"])
    z.load_layer("origins", paths["origins"])
    z.load_layer("destinations", paths["destinations"])
    z.create_street_network(source_layer="streets",
                            node_snapping_tolerance=0.0)
    build_od(z, paths)
    z.create_graph()
    z.clear_nodes()
    build_od(z, paths)          # repeated calls on the same Zonal
    z.create_graph()
    return z


def scenario_coincident_od(Zonal, root):
    paths = write_fixtures(root, "od")
    random.seed(20260930)
    z = Zonal()
    z.load_layer("streets", paths["streets"])
    z.load_layer("origins", paths["origins"])
    z.load_layer("destinations", paths["destinations"])
    z.create_street_network(source_layer="streets",
                            node_snapping_tolerance=0.0)
    build_od(z, paths)
    z.create_graph()
    return z


def scenario_gdf_input(Zonal, root):
    import geopandas as gpd
    paths = write_fixtures(root, "grid")
    random.seed(20260930)
    gdf = gpd.read_file(paths["streets"])
    z = Zonal()
    z.load_layer("streets", gdf)     # GeoDataFrame source keeps file_path marker
    z.create_street_network(source_layer="streets",
                            node_snapping_tolerance=0.0)
    return z


def scenario_crs_variant(Zonal, root):
    sw = _crs4326_streets(root)
    random.seed(20260930)
    z = Zonal()
    z.load_layer("streets", sw)
    z.create_street_network(source_layer="streets",
                            node_snapping_tolerance=0.0)
    return z


def scenario_repeat_network(Zonal, root):
    # create_street_network called twice on the same Zonal: second build
    # must fully replace network state
    paths = write_fixtures(root, "redundant")
    random.seed(20260930)
    z = Zonal()
    z.load_layer("streets", paths["streets"])
    z.load_layer("origins", paths["origins"])
    z.load_layer("destinations", paths["destinations"])
    z.create_street_network(source_layer="streets",
                            node_snapping_tolerance=0.0,
                            redundant_edge_treatment="keep")
    build_od(z, paths)
    z.create_street_network(source_layer="streets",
                            node_snapping_tolerance=0.0,
                            redundant_edge_treatment="split")
    build_od(z, paths)
    z.create_graph()
    return z


ERROR_SCENARIOS = {
    "load_layer_bad_name": lambda z, paths: z.load_layer(123, paths["streets"]),
    "load_layer_bad_source": lambda z, paths: z.load_layer("x", 3.14),
    "csn_unknown_layer": None,   # special-cased: no layer loaded at all
    "csn_bad_weight_attr": lambda z, paths: z.create_street_network(
        source_layer="streets", weight_attribute="missing"),
    "csn_negative_tolerance": lambda z, paths: z.create_street_network(
        source_layer="streets", node_snapping_tolerance=-1),
    "csn_bad_treatment": lambda z, paths: z.create_street_network(
        source_layer="streets", redundant_edge_treatment="trim"),
    "turn_threshold_range": lambda z, paths: z.set_turn_parameters(181, 30),
    "turn_penalty_negative": lambda z, paths: z.set_turn_parameters(45, -1),
    "insert_bad_label": lambda z, paths: z.insert_node("origins", label="poi"),
    "insert_unknown_layer": lambda z, paths: z.insert_node("nope", label="origin"),
    "create_graph_nonbool": lambda z, paths: z.create_graph(light_graph="yes"),
    "layers_bad_key": lambda z, paths: z.layers.__getitem__(3.5),
    "prep_mixed_z": None,        # special-cased: needs the mixed-Z fixture
}


def run_error_scenario(arm, name, root):
    Zonal = import_zonal(arm)
    random.seed(20260930)
    z = Zonal()
    try:
        if name == "csn_unknown_layer":
            pass  # no layer loaded: unknown-layer error path
        elif name == "prep_mixed_z":
            # pinned prepare_geometry crashes on a mixed 2D/3D layer
            # (_to_2d applied to 2D geometries) — registered quirk; the
            # crash fires inside load_layer itself
            paths = write_fixtures(root, "prep_mixed")
            z.load_layer("streets", paths["streets"])
            return {"raised": None}
        elif name == "csn_multilinestring":
            # pinned GeoPandaExtractor crashes on multi-part geometries
            # (mixed MLS layer skips prepare_geometry's all-MLS unwrap)
            paths = write_fixtures(root, "mls_mixed")
            z.load_layer("streets", paths["streets"])
            z.create_street_network(source_layer="streets",
                                    node_snapping_tolerance=0.0)
            return {"raised": None}
        else:
            paths = write_fixtures(root, "grid")
            z.load_layer("streets", paths["streets"])
            z.load_layer("origins", paths["origins"])
            z.load_layer("destinations", paths["destinations"])
            if (name.startswith("insert") or name == "create_graph_nonbool"
                    or name == "visualize_graph"):
                z.create_street_network(source_layer="streets",
                                        node_snapping_tolerance=0.0)
        if name == "csn_unknown_layer":
            z.create_street_network(source_layer="nope")
        elif name == "visualize_graph":
            z.network.visualize_graph()   # pinned unsupported (registered)
        else:
            ERROR_SCENARIOS[name](z, paths)
        return {"raised": None}
    except Exception as exc:
        return {"raised": type(exc).__name__, "message": str(exc)}


SABOTAGES = {
    # deliberate wrong variants; used ONLY to prove the comparator selects
    # differences (mutation kill), never in parity arms.  Each must yield
    # WELL-FORMED but bitwise-wrong output — a crash would evade the
    # comparator under test.
    "discard_longest": "_discard_redundant_edges keeps idxmax instead of idxmin",
    "swap_od_colors": "origin/destination categorical colors swapped",
    "bump_edge_weight": "first edge weight nudged by one ulp (bitwise delta)",
}


def apply_sabotage(name: str) -> None:
    import urban_network_analysis.compat.madina.zonal.network_utils as nu
    import urban_network_analysis.compat.madina.zonal.zonal as zz

    if name == "discard_longest":
        original = nu._discard_redundant_edges

        def longest(edge_gdf):
            import numpy as np
            import pandas as pd
            edge_stack = pd.DataFrame({
                'edge_id': np.concatenate([edge_gdf.index.values, edge_gdf.index.values]),
                'start': np.concatenate([edge_gdf['start'].values, edge_gdf['end'].values]),
                'end': np.concatenate([edge_gdf['end'].values, edge_gdf['start'].values]),
            })
            redundant = np.unique(edge_stack[edge_stack.duplicated(
                subset=['start', 'end'], keep=False)]['edge_id'].values)
            keep, queue = set({}), set(redundant)
            while queue:
                idx = queue.pop()
                s, e = edge_gdf.at[idx, 'start'], edge_gdf.at[idx, 'end']
                if s == e:
                    continue
                dups = edge_gdf[((edge_gdf['start'] == s) & (edge_gdf['end'] == e))
                                | ((edge_gdf['end'] == s) & (edge_gdf['start'] == e))]
                keep.add(dups["weight"].idxmax())   # SABOTAGE: longest kept
                queue -= set(dups.index.values)
            return edge_gdf.loc[list(set(edge_gdf.index.values)
                                     - (set(redundant) - keep))]

        nu._discard_redundant_edges = longest
        zz._discard_redundant_edges = longest
    elif name == "swap_od_colors":
        _orig = zz.color_gdf

        def swapped(*a, **k):
            if k.get("color_by_attribute") == "type":
                k["color"] = {'origin': [239, 89, 128],
                              'destination': [86, 5, 255],
                              'street_node': [125, 125, 125]}
            return _orig(*a, **k)

        zz.color_gdf = swapped
    elif name == "bump_edge_weight":
        import numpy as _np
        _orig = nu.node_edge_builder

        def bumped(*a, **k):
            nodes, edges = _orig(*a, **k)
            w = edges["weight"].to_numpy(dtype=_np.float64, copy=True)
            w[0] = _np.nextafter(w[0], _np.inf)   # exactly one ulp up
            edges["weight"] = w
            return nodes, edges

        nu.node_edge_builder = bumped
        zz.node_edge_builder = bumped
    else:
        raise ValueError(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["facade", "reference"], required=True)
    ap.add_argument("--scenario", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=20260930)
    ap.add_argument("--sabotage", default=None)
    args = ap.parse_args()

    random.seed(args.seed)

    # the layer digest records file_path, so the scenario root must be
    # IDENTICAL across arms (wiped per run); parity runs are sequential
    root = Path(f"/tmp/pgeom_fixed/{args.scenario.replace(':', '__')}")
    shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True, exist_ok=True)
    try:
        if args.scenario.startswith("error:"):
            out = run_error_scenario(args.arm, args.scenario.split(":", 1)[1],
                                     str(root))
        else:
            Zonal = import_zonal(args.arm)
            if args.sabotage:
                if args.arm != "facade":
                    raise SystemExit("sabotage applies to the facade arm only")
                apply_sabotage(args.sabotage)
            builder = {
                "grid_basic": scenario_grid_basic,
                "redundant_keep": lambda Z, r: scenario_redundant(Z, r, "keep"),
                "redundant_discard": lambda Z, r: scenario_redundant(Z, r, "discard"),
                "redundant_split": lambda Z, r: scenario_redundant(Z, r, "split"),
                "tolerance_snap": scenario_tolerance,
                "custom_weight": scenario_custom_weight,
                "prep_geometry": scenario_prep,
                "layer_order": scenario_layer_order,
                "clear_reinsert": scenario_clear_reinsert,
                "coincident_od": scenario_coincident_od,
                "gdf_input": scenario_gdf_input,
                "crs_variant": scenario_crs_variant,
                "repeat_network": scenario_repeat_network,
                "multilinestring_pure": scenario_mls_pure,
            }[args.scenario]
            z = builder(Zonal, str(root))
            out = zonal_digest(z)
    finally:
        shutil.rmtree(root, ignore_errors=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception:
        # record the failure INSIDE the artifact so both arms are comparable
        err = {"scenario_error": traceback.format_exc()}
        sys.stderr.write(err["scenario_error"])
        out = sys.argv[sys.argv.index("--out") + 1]
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(json.dumps(err, sort_keys=True))
        raise
