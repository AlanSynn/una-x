"""MADINA_ZONAL parity scenario runner: build a deterministic scenario
through either the compat facade or the bridged pinned reference, then
dump a BITWISE-strict state digest.

Same discipline as tests/platform_geometry (TOPOLOGY): both arms run
under the SAME interpreter (the dependency-bridged reference venv,
which carries pydeck) with an identically seeded stdlib RNG, so any
digest difference is attributable to the code under test, not the
environment.  Floats are compared as IEEE-754 bit patterns, geometries
as WKB hashes, dtypes/categories/order as recorded.

This runner owns the Zonal surface NOT exercised by the TOPOLOGY
geometry suite: layer styling/colors, describe() output, multi-node
insertion on one street edge (TOPOLOGY review handoff M4),
source_id/parent_street_id semantics, turn parameters, Layers
indexing/ordering API, create_map/Deck behavior, and the registered
pinned-quirk failures (Layer.set_style AttributeError,
Network.visualize_graph NotImplementedError).

Usage:
    python _zonal_scenario.py --arm {facade,reference} --scenario NAME
                              --out DIGEST.json --seed 42 [--sabotage NAME]
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import random
import struct
import sys
import traceback
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from shapely.geometry import LineString, Point

REPO = Path(__file__).resolve().parents[3]


def _f(v):
    """float -> IEEE-754 bit-pattern hex (bitwise-strict, NaN-safe)."""
    return struct.pack(">d", float(v)).hex()


def _wkb_hash(geom):
    return hashlib.sha256(shapely.to_wkb(geom, byte_order=1)).hexdigest()[:24]


def _series(s):
    if pd.api.types.is_float_dtype(s):
        return [_f(v) for v in s]
    if pd.api.types.is_bool_dtype(s):
        return [bool(v) for v in s]
    if pd.api.types.is_integer_dtype(s):
        return [int(v) for v in s]
    return [str(v) for v in s]


def gdf_digest(gdf):
    """Complete state digest of a (small) GeoDataFrame."""
    cols = {}
    for col in gdf.columns:
        if col == "geometry":
            cols["geometry"] = [_wkb_hash(g) for g in gdf.geometry]
        elif isinstance(gdf[col].dtype, pd.CategoricalDtype):
            cols[col] = {"categories": [str(c) for c in gdf[col].cat.categories],
                         "codes": [int(c) for c in gdf[col].cat.codes]}
        else:
            cols[col] = _series(gdf[col])
    return {
        "columns": [str(c) for c in gdf.columns],
        "dtypes": {str(c): str(t) for c, t in gdf.dtypes.items()},
        "index": _series(pd.Series(gdf.index)),
        "index_name": repr(gdf.index.name),
        "active_geometry_name": repr(gdf.geometry.name),
        "crs": str(gdf.crs),
        "values": cols,
    }


def _jsonable(v):
    """Bitwise-strict JSON-safe scalar: numpy integers/bools collapse to
    Python scalars (add_node_to_graph copies np.int64 node attrs);
    every float — Python or numpy — records its IEEE-754 bit pattern."""
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, (int, np.integer)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        return _f(v)
    return v


def graph_digest(G):
    nodes = {}
    for n in sorted(G.nodes()):
        nodes[str(n)] = {str(k): _jsonable(v)
                         for k, v in sorted(G.nodes[n].items())
                         if k != "geometry"}
    edges = {}
    for u, v, data in sorted(G.edges(data=True)):
        # nx.Graph cannot hold parallel edges, so endpoint pairs are
        # unique keys (review finding 6 removed an unreachable "#2"
        # duplicate-key branch)
        edges[f"{min(u, v)}|{max(u, v)}"] = {
            str(k): _jsonable(v)
            for k, v in sorted(data.items()) if k != "geometry"}
    return {"n_nodes": G.number_of_nodes(), "n_edges": G.number_of_edges(),
            "nodes": nodes, "edges": edges}


def _capture_stdout(fn, *args, **kwargs):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        fn(*args, **kwargs)
    return buf.getvalue().splitlines()


# ----------------------------------------------------------------------
# deterministic fixtures (byte-stable recipes; grid superset of TOPOLOGY's)
# ----------------------------------------------------------------------

def _streets_gdf():
    edges = [
        ((0.0, 0.0), (100.0, 0.0)), ((100.0, 0.0), (200.0, 0.0)),
        ((0.0, 0.0), (0.0, 100.0)), ((100.0, 0.0), (100.0, 100.0)),
        ((0.0, 100.0), (100.0, 100.0)), ((100.0, 100.0), (200.0, 100.0)),
        ((200.0, 0.0), (200.0, 100.0)),
    ]
    return gpd.GeoDataFrame(
        {"Geometric": [100.0] * len(edges),
         "weight_col": [50.0 + 10.0 * i for i in range(len(edges))]},
        geometry=[LineString([a, b]) for a, b in edges], crs="EPSG:3857")


def _redundant_gdf():
    """Grid plus a direct diagonal and a two-segment detour between the
    same endpoint pair (the TOPOLOGY redundant recipe) so that both
    'split' and 'discard' treatments have real work to do."""
    edges = [
        ((0.0, 0.0), (100.0, 0.0)), ((0.0, 0.0), (0.0, 100.0)),
        ((100.0, 0.0), (100.0, 100.0)), ((0.0, 100.0), (100.0, 100.0)),
        ((0.0, 0.0), (100.0, 100.0)),                 # diagonal ~141.42
        ((0.0, 0.0), (20.0, 90.0)),                   # detour in
        ((20.0, 90.0), (100.0, 100.0)),               # detour out
    ]
    return gpd.GeoDataFrame(
        {"Geometric": [100.0] * len(edges),
         "weight_col": [100.0 - 5.0 * i for i in range(len(edges))]},
        geometry=[LineString([a, b]) for a, b in edges], crs="EPSG:3857")


def _pts_on_edge_gdf():
    """Two points at different positions on street edge 0 (the y=0
    segment), one point on edge 3 (the x=100 vertical segment)."""
    return gpd.GeoDataFrame(
        {"demand": [3.0, 1.0, 2.0]},
        geometry=[Point(25.0, 0.0), Point(75.0, 5.0), Point(100.0, 40.0)],
        crs="EPSG:3857")


def _pts_cat_gdf():
    return gpd.GeoDataFrame(
        {"cat": ["a", "b", "a"], "demand": [3.0, 1.0, None]},
        geometry=[Point(10.0, 10.0), Point(190.0, 90.0), Point(60.0, 60.0)],
        crs="EPSG:3857")


def _new_zonal(mz):
    return mz.Zonal()


# ----------------------------------------------------------------------
# scenarios
# ----------------------------------------------------------------------

def scenario_style_colors(mz, seed):
    """Default layer colors (seeded RNG consumption), geo_center, and all
    four color_gdf methods incl. NaN handling."""
    random.seed(seed)
    z = _new_zonal(mz)
    z.load_layer("streets", _streets_gdf())
    z.load_layer("pts", _pts_cat_gdf())
    out = {
        "layer_defaults": {"streets": list(z.layers["streets"].default_color),
                           "pts": list(z.layers["pts"].default_color)},
        "geo_center": [_f(z.geo_center[0]), _f(z.geo_center[1])],
        "layer_show": {"streets": z.layers["streets"].show,
                       "pts": z.layers["pts"].show},
        "layer_crs": {"streets": str(z.layers["streets"].crs),
                      "pts": str(z.layers["pts"].crs)},
        "other_fields": {"streets": dict(z.layers["streets"].other_fields),
                         "pts": dict(z.layers["pts"].other_fields)},
    }

    gdf = z.layers["pts"].gdf.copy(deep=True)
    colored = mz.color_gdf(gdf.copy(deep=True))
    out["single_random"] = _series(colored["color"].astype(str))
    colored = mz.color_gdf(gdf.copy(deep=True), color=[1, 2, 3])
    out["single_explicit"] = _series(colored["color"].astype(str))
    colored = mz.color_gdf(gdf.copy(deep=True), color_by_attribute="cat")
    out["categorical_auto"] = _series(colored["color"].astype(str))
    # pinned quirk: an explicit color dict WITHOUT the '__other__' key
    # passes the auto-assignment branch entirely, so any value missing
    # from the dict raises KeyError at the lookup — pinned failure
    # parity (both arms must record the identical message).
    try:
        mz.color_gdf(gdf.copy(deep=True), color_by_attribute="cat",
                     color={"a": [10, 20, 30]})
        out["categorical_partial"] = "NO_ERROR"
    except Exception as exc:                          # noqa: BLE001
        out["categorical_partial"] = f"{type(exc).__name__}: {exc}"
    colored = mz.color_gdf(gdf.copy(deep=True), color_by_attribute="cat",
                           color={"__other__": [9, 9, 9],
                                  "a": [10, 20, 30]})
    out["categorical_full"] = _series(colored["color"].astype(str))
    colored = mz.color_gdf(gdf.copy(deep=True), color_by_attribute="demand",
                           color_method="gradient")
    out["gradient_with_nan"] = _series(colored["color"].astype(str))
    colored = mz.color_gdf(gdf.copy(deep=True), color_by_attribute="demand",
                           color_method="quantile")
    out["quantile_with_nan"] = _series(colored["color"].astype(str))
    return out


def scenario_describe_output(mz, seed):
    """stdout of Zonal.describe() in three states; pins the dead
    ``geo_center is None`` branch (geo_center is the tuple (None, None),
    never None, so the empty zonal still prints 'Geographic center')."""
    random.seed(seed)
    z = _new_zonal(mz)
    out = {"empty": _capture_stdout(z.describe)}
    z.load_layer("streets", _streets_gdf())
    z.load_layer("pts", _pts_on_edge_gdf())
    out["with_layers"] = _capture_stdout(z.describe)
    z.create_street_network("streets", weight_attribute="weight_col")
    z.insert_node("pts", label="origin", weight_attribute="demand")
    z.create_graph()
    out["with_network"] = _capture_stdout(z.describe)
    return out


def scenario_insert_multinode_edge(mz, seed):
    """TOPOLOGY review handoff M4: MULTIPLE inserted nodes land on ONE
    street edge.  Two origin points sit at different positions of street
    edge 0; one destination sits on edge 3.  Digests the full node/edge
    tables bitwise, per-edge insertion counts, both graphs, and the
    add_node_to_graph / remove_node_to_graph roundtrip."""
    random.seed(seed)
    z = _new_zonal(mz)
    z.load_layer("streets", _streets_gdf())
    z.load_layer("a_pts", _pts_on_edge_gdf())
    z.create_street_network("streets", weight_attribute="weight_col")
    z.insert_node("a_pts", label="origin", weight_attribute="demand")
    # destination insertion also carries a weight attribute (dossier:
    # insert_node weight_attribute applies to both labels)
    z.insert_node("a_pts", label="destination", weight_attribute="demand")
    n = z.network
    out = {
        "nodes": gdf_digest(n.nodes),
        "edges": gdf_digest(n.edges),
    }
    inserted = n.nodes[n.nodes["type"] != "street_node"]
    out["inserted_per_edge"] = {str(k): int(v) for k, v
                                in inserted["nearest_edge_id"].value_counts().items()}
    z.create_graph()
    out["light_graph"] = graph_digest(n.light_graph)
    out["d_graph"] = graph_digest(n.d_graph)

    # WorkingGraphScope-relevant primitives (SCIENCE discipline consumes
    # these): add/remove roundtrips.  The exact post-roundtrip state is
    # PINNED bitwise across arms — restoration is NOT assumed.  (A
    # street node carries the zero-filled placeholder columns from
    # insert_node's pre-fill, and networkx .copy() shares the
    # graph["added_nodes"] list, so the pinned outcome is whatever the
    # pinned reference does — including edge-set corruption; review
    # finding 1 replaced an earlier node-count-only pin that looked
    # like restoration.)
    G2 = n.d_graph.copy()
    street_id = sorted(n.street_node_ids)[0]
    n.add_node_to_graph(G2, street_id)
    out["roundtrip_street_node"] = {
        "after_add": graph_digest(G2),
    }
    n.remove_node_to_graph(G2, street_id)
    out["roundtrip_street_node"]["after_remove"] = graph_digest(G2)

    # same roundtrip on a REAL inserted node id (an origin/destination
    # the engines actually add and remove per origin)
    inserted_id = sorted(set(n.nodes.index) - set(n.street_node_ids))[0]
    G3 = n.d_graph.copy()
    n.add_node_to_graph(G3, inserted_id)
    out["roundtrip_inserted_node"] = {"after_add": graph_digest(G3),
                                      "node_id": int(inserted_id)}
    n.remove_node_to_graph(G3, inserted_id)
    out["roundtrip_inserted_node"]["after_remove"] = graph_digest(G3)

    # update_light_graph add/remove path (dossier-listed Network API)
    G4 = n.light_graph.copy()
    n.update_light_graph(G4, add_nodes=[inserted_id])
    out["update_light_graph_after_add"] = graph_digest(G4)
    n.update_light_graph(G4, remove_nodes=[inserted_id])
    out["update_light_graph_after_remove"] = graph_digest(G4)
    return out


def scenario_id_semantics(mz, seed):
    """source_id / parent_street_id preservation under 'split' and
    'discard', network replacement on repeated calls, weight-column
    change visibility, and the Layers indexing/ordering API."""
    random.seed(seed)
    out = {}
    for treatment in ("split", "discard"):
        z = _new_zonal(mz)
        z.load_layer("streets", _redundant_gdf())
        z.create_street_network("streets", weight_attribute="weight_col",
                                redundant_edge_treatment=treatment)
        n = z.network
        out[treatment] = {
            "edges": gdf_digest(n.edges[["length", "weight", "parent_street_id",
                                         "start", "end", "geometry"]]),
            "nodes": gdf_digest(n.nodes[["source_layer", "source_id", "type",
                                         "weight", "degree", "geometry"]]),
        }
        # repeated call on the same Zonal replaces the network; identical
        # seeded rebuild must reproduce the same digest
        z.create_street_network("streets", weight_attribute="weight_col",
                                redundant_edge_treatment=treatment)
        out[treatment + "_repeat_equal"] = (
            gdf_digest(z.network.edges[["length", "weight", "parent_street_id",
                                        "start", "end", "geometry"]])
            == out[treatment]["edges"])
        # changed layer weight column -> new network sees new weights
        z.layers["streets"].gdf["weight_col"] = (
            z.layers["streets"].gdf["weight_col"] + 7.5)
        z.create_street_network("streets", weight_attribute="weight_col",
                                redundant_edge_treatment=treatment)
        out[treatment + "_weight_shifted"] = _series(
            z.network.edges["weight"])

    # Layers API: ordering, indexing by label and position, iteration,
    # membership, pos_at_label, get_layer_at_pos, str()
    random.seed(seed)
    z = _new_zonal(mz)
    z.load_layer("l1", _streets_gdf())
    z.load_layer("l2", _pts_on_edge_gdf())
    z.load_layer("l3", _redundant_gdf())
    layers = z.layers
    out["layers_api"] = {
        "order_after_appends": list(layers.layers),
        "labels_by_pos": [layers.get_layer_at_pos(i).label for i in range(3)],
        "str_output": str(layers).splitlines(),
        "contains_label": "l2" in layers,
        "pos_of_l3": layers.pos_at_label("l3"),
        "iter_labels": [label for label in layers],
    }
    z.load_layer("l0", _pts_cat_gdf(), first=True)
    out["layers_api"]["order_after_first"] = list(z.layers.layers)
    z.load_layer("lm", _streets_gdf(), after="l1")
    out["layers_api"]["order_after_l1"] = list(z.layers.layers)
    z.load_layer("ln", _streets_gdf(), before="l3")
    out["layers_api"]["order_before_l3"] = list(z.layers.layers)
    z.load_layer("lp", _streets_gdf(), pos=2)
    out["layers_api"]["order_pos2"] = list(z.layers.layers)
    # real duplicate-label reload (review finding 3: recorded, not a
    # hardcoded string)
    try:
        z.load_layer("l1", _streets_gdf())
        out["duplicate_label_error"] = "NO_ERROR"
    except Exception as exc:                          # noqa: BLE001
        out["duplicate_label_error"] = f"{type(exc).__name__}: {exc}"

    # Layers.__setitem__: replace by label and by position (dossier
    # layer-indexing surface; the int path is covered nowhere else).
    # Pinned quirk: the int path writes the Layer OBJECT into the
    # label-order list (upstream bug preserved verbatim), so the order
    # representation coerces non-label entries to <Layer:label> marks.
    replacement = mz.Layer("l1", _streets_gdf(), True, "EPSG:3857", "gdf")
    layers["l1"] = replacement
    out["layers_api"]["setitem_str_label_is_new"] = (
        layers["l1"] is replacement)
    replacement2 = mz.Layer("l9", _streets_gdf(), True, "EPSG:3857", "gdf")
    layers[1] = replacement2
    out["layers_api"]["setitem_int"] = {
        "order": [x if isinstance(x, str) else f"<Layer:{x.label}>"
                  for x in layers.layers],
        "l9_reachable_by_label": layers["l9"] is replacement2,
        "l2_gone": "l2" not in layers.label_layers,
    }

    # Network.set_node_value (dossier-listed Network API)
    zn = _new_zonal(mz)
    zn.load_layer("streets", _streets_gdf())
    zn.create_street_network("streets", weight_attribute="weight_col")
    first_street = zn.network.nodes[
        zn.network.nodes["type"] == "street_node"].index[0]
    zn.network.set_node_value(first_street, "weight", 123.0)
    out["set_node_value"] = {
        "node": int(first_street),
        "weight": _f(zn.network.nodes.loc[first_street, "weight"]),
    }
    return out


def scenario_turn_params(mz, seed):
    """set_turn_parameters behavior, constructor-provided turn
    parameters, and the recorded validation failures."""
    random.seed(seed)
    z = _new_zonal(mz)
    z.load_layer("streets", _streets_gdf())
    z.create_street_network("streets", weight_attribute="weight_col",
                            turn_threshold_degree=60, turn_penalty_amount=7.5)
    out = {"constructor": [z.network.turn_threshold_degree,
                           z.network.turn_penalty_amount]}
    z.set_turn_parameters(90, 15)
    out["after_set"] = [z.network.turn_threshold_degree,
                        z.network.turn_penalty_amount]
    z.create_graph(light_graph=True, d_graph=True, od_graph=True)
    out["od_graph"] = graph_digest(z.network.od_graph)

    failures = {}
    for name, fn in [
        ("threshold_high", lambda: z.set_turn_parameters(181, 1)),
        ("threshold_low", lambda: z.set_turn_parameters(-1, 1)),
        ("penalty_negative", lambda: z.set_turn_parameters(45, -0.5)),
        ("threshold_type", lambda: z.set_turn_parameters("45", 1)),
        ("penalty_type", lambda: z.set_turn_parameters(45, None)),
        ("csn_bad_treatment", lambda: z.create_street_network(
            "streets", redundant_edge_treatment="merge")),
        ("csn_negative_tolerance", lambda: z.create_street_network(
            "streets", node_snapping_tolerance=-1)),
        ("csn_unknown_weight", lambda: z.create_street_network(
            "streets", weight_attribute="nope")),
        ("insert_bad_label", lambda: z.insert_node("streets", label="poi")),
        ("insert_unknown_layer", lambda: z.insert_node("nope", label="origin")),
        ("create_graph_nonbool", lambda: z.create_graph(light_graph=1)),
        ("layers_bad_key", lambda: z.layers[3.5]),
        ("load_bad_name", lambda: z.load_layer(7, _streets_gdf())),
        ("load_dup_label", lambda: z.load_layer("streets", _streets_gdf())),
    ]:
        try:
            fn()
            failures[name] = "NO_ERROR"
        except Exception as exc:                      # noqa: BLE001
            failures[name] = f"{type(exc).__name__}: {exc}"
    out["validation_failures"] = failures
    return out


def scenario_map_deck(mz, seed):
    """create_map: default visible-layer map, explicit layer_list map,
    saved HTML bytes, and the basemap variant.  pydeck is present in the
    reference venv interpreter (0.9.3), so both arms build real
    pydeck.Deck objects with identically seeded colors.

    Controlled presentation randomness: pydeck.Layer assigns each layer
    an ``id`` via ``uuid.uuid4()`` (pydeck/bindings/layer.py:78, module-
    attribute lookup).  Per dossier 01 the test RNG controls random
    presentation fields rather than dropping them, so uuid4 is replaced
    with a seeded deterministic generator before any deck is built."""
    import uuid as _uuid

    import pydeck as pdk

    rng = random.Random(seed ^ 0x5EEDD1C5)

    def _controlled_uuid4():
        return _uuid.UUID(int=rng.getrandbits(128))

    _uuid.uuid4 = _controlled_uuid4

    random.seed(seed)
    z = _new_zonal(mz)
    z.load_layer("streets", _streets_gdf())
    z.load_layer("pts", _pts_cat_gdf())

    out = {}
    deck = z.create_map()
    out["default_deck"] = {
        "is_deck": isinstance(deck, pdk.Deck),
        "json": deck.to_json(),
        "n_layers": len(deck.layers),
    }
    deck2 = z.create_map(layer_list=[{"layer": "pts"}])
    out["pts_only_deck"] = {
        "json": deck2.to_json(),
        "n_layers": len(deck2.layers),
    }
    deck3 = z.create_map(layer_list=[{"gdf": z.layers["pts"].gdf.copy(deep=True),
                                      "color_by_attribute": "cat"}],
                         save_as="map_pts.html", basemap=True)
    out["basemap_deck"] = {
        "json": deck3.to_json(),
        "saved_exists": Path("map_pts.html").is_file(),
        "saved_sha256": hashlib.sha256(
            Path("map_pts.html").read_bytes()).hexdigest(),
    }
    # radius/width/opacity/text branches of create_deckGL_map (dossier
    # map surface; the NaN demand row exercises the radius NaN filter
    # and the string cat column the astype text fallback)
    deck4 = z.create_map(layer_list=[{
        "gdf": z.layers["pts"].gdf.copy(deep=True),
        "radius": "demand", "radius_min": 3, "radius_max": 9,
        "width": "demand", "width_scale": 2.0, "opacity": 0.5,
        "text": "cat",
    }])
    out["styled_deck"] = {"json": deck4.to_json(),
                          "n_layers": len(deck4.layers)}
    return out


def scenario_quirks(mz, seed):
    """Registered pinned-quirk failures — the pinned failure behavior IS
    the parity behavior for the madina profile:
      - Layer.set_style raises AttributeError (BUG-LAYER-SET-STYLE-ATTRERROR:
        Layer.color_layer reads self.default_colors, never defined);
      - Network.visualize_graph raises NotImplementedError (documented
        unsupported stub, dossier 01 census rule);
      - Zonal[key] KeyError, Layers[bad-type] KeyError."""
    random.seed(seed)
    z = _new_zonal(mz)
    z.load_layer("streets", _streets_gdf())
    layer = z.layers["streets"]
    out = {}
    try:
        layer.set_style({"color": [1, 2, 3]})
        out["set_style"] = "NO_ERROR"
    except Exception as exc:                          # noqa: BLE001
        out["set_style"] = f"{type(exc).__name__}: {exc}"
    z.create_street_network("streets", weight_attribute="weight_col")
    try:
        z.network.visualize_graph()
        out["visualize_graph"] = "NO_ERROR"
    except Exception as exc:                          # noqa: BLE001
        out["visualize_graph"] = f"{type(exc).__name__}: {exc}"
    try:
        z["missing_layer"]
        out["zonal_bad_key"] = "NO_ERROR"
    except Exception as exc:                          # noqa: BLE001
        out["zonal_bad_key"] = f"{type(exc).__name__}: {exc}"
    try:
        z.layers[object()]
        out["layers_bad_type"] = "NO_ERROR"
    except Exception as exc:                          # noqa: BLE001
        out["layers_bad_type"] = f"{type(exc).__name__}: {exc}"
    return out


SCENARIOS = {
    "style_colors": scenario_style_colors,
    "describe_output": scenario_describe_output,
    "insert_multinode_edge": scenario_insert_multinode_edge,
    "id_semantics": scenario_id_semantics,
    "turn_params": scenario_turn_params,
    "map_deck": scenario_map_deck,
    "quirks": scenario_quirks,
}

# Comparator-sensitivity mutations: each drops/alters exactly one piece
# of recorded state.  An honest comparator MUST flag these digests as
# different from the honest arm (tests assert the difference).
SABOTAGES = {
    "drop_parent_col": ("id_semantics",
                        lambda d: d["split"]["edges"]["values"].pop("parent_street_id")),
    "drop_last_edge": ("insert_multinode_edge",
                       lambda d: d["edges"]["values"]["geometry"].pop()),
    "swap_default_color": ("style_colors",
                           lambda d: d["layer_defaults"].__setitem__(
                               "pts", [9, 9, 9])),
    "drop_describe_line": ("describe_output",
                           lambda d: d["with_layers"].pop()),
    "flip_turn_threshold": ("turn_params",
                            lambda d: d.__setitem__("after_set", [91, 15])),
    "undercount_layers": ("map_deck",
                          lambda d: d.__setitem__(
                              "default_deck", {**d["default_deck"],
                                               "n_layers": d["default_deck"]["n_layers"] - 1})),
    "hide_set_style_error": ("quirks",
                             lambda d: d.__setitem__("set_style", "NO_ERROR")),
}


# ----------------------------------------------------------------------
# arm dispatch
# ----------------------------------------------------------------------

def import_arm(arm):
    if arm == "reference":
        sys.path.insert(0, str(REPO / ".refs" / "madina_ref" / "src"))
        import madina.zonal as mz       # noqa: PLC0415
        # upstream quirk: the attribute madina.zonal is the INNER
        # zonal.py module (star-import shadowing); classes are reachable
        # either way and this probe records the identity for evidence.
        return mz
    if arm == "facade":
        sys.path.insert(0, str(REPO / "src"))
        from urban_network_analysis.compat.madina import zonal as mz   # noqa: PLC0415
        return mz
    raise ValueError(f"unknown arm {arm}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["facade", "reference"])
    ap.add_argument("--scenario", required=True, choices=sorted(SCENARIOS))
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=20260930)
    ap.add_argument("--sabotage", choices=sorted(SABOTAGES), default=None)
    args = ap.parse_args()
    digest = {"arm": args.arm, "scenario": args.scenario, "seed": args.seed}
    try:
        mz = import_arm(args.arm)
        digest["state"] = SCENARIOS[args.scenario](mz, args.seed)
        if args.sabotage:
            sab_scenario, mutate = SABOTAGES[args.sabotage]
            if sab_scenario != args.scenario:
                raise ValueError(f"sabotage {args.sabotage} targets "
                                 f"{sab_scenario}, not {args.scenario}")
            mutate(digest["state"])
            digest["sabotage"] = args.sabotage
    except Exception:
        digest["scenario_error"] = traceback.format_exc()[-4000:]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(digest, sort_keys=True, indent=1))
    print(f"digest: {out}")


if __name__ == "__main__":
    main()
