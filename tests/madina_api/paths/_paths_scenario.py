"""MADINA_PATHS parity scenario runner: exercise the all-alternative
paths surface through either the compat facade or the pinned upstream
reference, then dump a BITWISE-strict state digest.

Same discipline as tests/platform_geometry (TOPOLOGY) and
tests/madina_api/zonal (MADINA_ZONAL): both arms run under the SAME
interpreter (the dependency-bridged reference venv) with identical
fixtures, so any digest difference is attributable to the code under
test.  Floats are compared as IEEE-754 bit patterns, geometries as WKB
hashes (GeometryCollections component-by-component), order-sensitive
dict content is recorded as explicit lists.

This runner owns the una.paths surface (MADINA_PATHS):
alternative_paths materialization, the path_generator /
bfs_subgraph_generation / bfs_*_iterative / wandering_messenger engine
primitives, turn_o_scope / turn_penalty_value /
angle_deviation_between_two_lines, turn-penalty distances, detour
route-set semantics (ALL paths within the detour bound — no hidden K
limit, validated by an independent networkx enumeration), and the
pinned validation failures.

Usage:
    python _paths_scenario.py --arm {facade,reference} --scenario NAME
                              --out DIGEST.json --seed 42 [--sabotage NAME]
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import struct
import sys
import traceback
from pathlib import Path

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
import shapely
from shapely.geometry import LineString, Point

REPO = Path(__file__).resolve().parents[3]


# ----------------------------------------------------------------------
# bitwise digest helpers (same discipline as the zonal suite)
# ----------------------------------------------------------------------

def _f(v):
    """float -> IEEE-754 bit-pattern hex (bitwise-strict, NaN-safe)."""
    return struct.pack(">d", float(v)).hex()


def _wkb_hash(geom):
    return hashlib.sha256(shapely.to_wkb(geom, byte_order=1)).hexdigest()[:24]


def _geomcollection_digest(gc):
    """GeometryCollection -> component count + per-component WKB hashes."""
    return {"n_components": len(gc.geoms),
            "component_wkb": [_wkb_hash(g) for g in gc.geoms]}


def _jsonable(v):
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, (int, np.integer)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        return _f(v)
    return v


def _series(s):
    if pd.api.types.is_float_dtype(s):
        return [_f(v) for v in s]
    if pd.api.types.is_bool_dtype(s):
        return [bool(v) for v in s]
    if pd.api.types.is_integer_dtype(s):
        return [_jsonable(v) for v in s]
    if isinstance(s.dtype, pd.CategoricalDtype):
        return {"categories": [str(c) for c in s.cat.categories],
                "codes": [int(c) if c >= 0 else -1 for c in s.cat.codes]}
    return [str(v) for v in s]


def gdf_digest(gdf):
    return {
        "columns": list(gdf.columns),
        "dtypes": [str(t) for t in gdf.dtypes],
        "index": [_jsonable(v) for v in gdf.index],
        "index_name": repr(gdf.index.name),
        "crs": str(gdf.crs),
        "values": {c: _series(gdf[c]) for c in gdf.columns},
    }


def paths_gdf_digest(gdf):
    """alternative_paths output: like gdf_digest but the geometry column
    holds GeometryCollections -> component-level WKB digests."""
    d = gdf_digest(gdf)
    d["values"]["geometry"] = [_geomcollection_digest(g) for g in gdf["geometry"]]
    return d


def _capture_stdout(fn):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        try:
            result = fn()
            error = None
        except Exception as ex:               # noqa: BLE001 - pinned failures
            result = None
            error = f"{type(ex).__name__}: {ex}"
    return result, error, buf.getvalue()


# The paths module of whichever arm imported us (set by import_arm), so
# engine-primitive replays and the independent enumeration walk the SAME
# module the scenario under test used.
_ARM_MARKER = {"paths": None, "tools": None}


def _arm_paths_module():
    if _ARM_MARKER["paths"] is None:
        raise RuntimeError("arm not imported")
    return _ARM_MARKER["paths"]


def _arm_tools():
    if _ARM_MARKER["tools"] is None:
        raise RuntimeError("arm not imported")
    return _ARM_MARKER["tools"]


# ----------------------------------------------------------------------
# fixture: square + diagonal + dead-end stub + approach edge, plus an
# isolated second component with its own destination
#
#   F --------- A ====== o ====== B --- d2 --- C
#               | \                    |       |
#               |   \.... diagonal .....\     d1
#               |                              |
#               D ---------------------------- +
#               |
#               E (dead-end stub)          ISO: iso1 --- iso2 (d999)
#
#   streets: F-A, A-B, B-C, C-D, D-A, A-C (diagonal), D-E, ISO edge.
#   origin o (50,0) mid A-B; destinations d1 (50,100) mid C-D,
#   d2 (100,50) mid B-C; d999 (550,500) on the isolated component.
#   Routes o->d1 and o->d2 each have multiple alternatives within
#   detour_ratio 2.5, including an exact distance tie for d1 (200 + 200).
# ----------------------------------------------------------------------

def _streets_gdf():
    return gpd.GeoDataFrame(
        {
            "id": ["F-A", "A-B", "B-C", "C-D", "D-A", "A-C", "D-E",
                   "ISO"],
            "length": [100.0, 100.0, 100.0, 100.0, 100.0,
                       float(np.hypot(100, 100)), 50.0, 100.0],
        },
        geometry=[
            LineString([(-100, 0), (0, 0)]),
            LineString([(0, 0), (100, 0)]),
            LineString([(100, 0), (100, 100)]),
            LineString([(100, 100), (0, 100)]),
            LineString([(0, 100), (0, 0)]),
            LineString([(0, 0), (100, 100)]),
            LineString([(0, 100), (0, 150)]),
            LineString([(500, 500), (600, 500)]),
        ],
        crs="EPSG:3857",
    )


def _origins_gdf():
    # PINNED: load_layer resets the layer index, so source_id is the
    # positional 0 regardless of any fixture index; origin_id = 0.
    return gpd.GeoDataFrame(
        {"weight": [1.0]},
        geometry=[Point((50, 0))],
        crs="EPSG:3857",
    )


def _destinations_gdf(iso_destination=False):
    ids, weights, geoms = [0, 1], [2.0, 3.0], \
        [Point((50, 100)), Point((100, 50))]
    if iso_destination:
        ids.append(2)
        weights.append(1.0)
        geoms.append(Point((550, 500)))
    # source_ids are positional 0..n-1 after load_layer's pinned index
    # reset (see _origins_gdf)
    return gpd.GeoDataFrame({"weight": weights},
                           geometry=geoms, crs="EPSG:3857")


def _make_zonal(mz, turn_threshold=45, turn_penalty_amount=30,
                iso_destination=False):
    z = mz.Zonal()
    z.load_layer("streets", _streets_gdf())
    z.load_layer("origins", _origins_gdf())
    z.load_layer("destinations",
                 _destinations_gdf(iso_destination=iso_destination))
    z.create_street_network(
        "streets", weight_attribute="length",
        turn_threshold_degree=turn_threshold,
        turn_penalty_amount=turn_penalty_amount)
    z.insert_node("origins", label="origin")
    z.insert_node("destinations", label="destination",
                  weight_attribute="weight")
    z.create_graph()
    return z


def _origin_idx(z):
    return int(z.network.nodes[z.network.nodes["type"] == "origin"].index[0])


# ----------------------------------------------------------------------
# independent enumeration (completion condition: validate the full
# route set and returned order/geometry; no hidden K limit)
# ----------------------------------------------------------------------

def _independent_paths(z, o_idx, search_radius, detour_ratio, turn_penalty):
    """Enumerate ALL simple paths origin->each destination over the
    engine's own scope with networkx, filtered by the pinned acceptance
    bound (weight <= shortest * detour_ratio + 1e-5), and record the
    full route set as scenario state.  The engine-vs-independent
    comparison is pinned bitwise across arms."""
    paths_mod = _arm_paths_module()
    G = z.network.d_graph
    node_gdf = z.network.nodes
    destinations = node_gdf[node_gdf["type"] == "destination"].index

    z.network.add_node_to_graph(G, o_idx)
    try:
        d_idxs, o_scope, _ = paths_mod.turn_o_scope(
            network=z.network, o_idx=o_idx, search_radius=search_radius,
            detour_ratio=detour_ratio, turn_penalty=turn_penalty,
            o_graph=G, return_paths=True)
        scope_nodes = set(o_scope) | {o_idx}
        sub = nx.subgraph(G, scope_nodes)
        full_route_set = set()
        per_dest = {}
        for d_idx in destinations:
            if d_idx not in d_idxs:
                continue
            bound = d_idxs[d_idx] * detour_ratio + 0.00001
            for path in nx.all_simple_paths(sub, o_idx, d_idx):
                weight = 0.0
                for prev, node in zip(path, path[1:]):
                    weight += G.edges[(prev, node)]["weight"]
                if turn_penalty:
                    for a, b, c in zip(path, path[1:], path[2:]):
                        weight += paths_mod.turn_penalty_value(
                            z.network, a, b, c)
                if weight <= bound:
                    hop_ids = [int(G.edges[(a, b)]["id"])
                               for a, b in zip(path, path[1:])]
                    full_route_set.add(tuple(hop_ids))
                    # the engine's path_edges store INTERIOR segments
                    # only (origin-approaching and destination-arriving
                    # hops are excluded by upstream construction) and
                    # DEDUPE repeated parent-street ids along the
                    # accumulated list (upstream bfs_path_edges:
                    # "edges + [edge_id] if edge_id not in edges" — two
                    # segments of one street share the parent id), so
                    # project to the same representation
                    interior = []
                    for eid in hop_ids[1:-1]:
                        if eid not in interior:
                            interior.append(eid)
                    per_dest.setdefault(int(d_idx), []).append(
                        {"hops": hop_ids, "interior": interior,
                         "weight": _f(weight)})
    finally:
        z.network.remove_node_to_graph(G, o_idx)
    return {"d_idxs_order": [int(k) for k in d_idxs],
            "d_idxs": {int(k): _f(v) for k, v in d_idxs.items()},
            "scope_size": len(scope_nodes),
            "full_route_set": sorted(list(r) for r in full_route_set),
            "per_destination":
                {k: {"n_routes": len(v),
                     "distances": sorted(r["weight"] for r in v),
                     "interiors": sorted(r["interior"] for r in v)}
                 for k, v in per_dest.items()}}


# ----------------------------------------------------------------------
# scenarios
# ----------------------------------------------------------------------

def _gdf_counts_by_source(z, ap):
    return {str(int(dest)): len(group)
            for dest, group in ap.groupby("destination")}


def _coherence(z, ap, indep):
    """Cross-check the public materialization against the independent
    enumeration: per-source route counts must agree once network node
    indices are mapped to source ids.  Recorded as scenario state so
    both arms pin it."""
    node_gdf = z.network.nodes
    d_idx_to_source = {
        int(idx): int(node_gdf.at[idx, "source_id"])
        for idx in node_gdf[node_gdf["type"] == "destination"].index}
    indep_counts_by_source = {}
    for d_idx, rec in indep["per_destination"].items():
        src = str(d_idx_to_source[d_idx])
        indep_counts_by_source[src] = \
            indep_counts_by_source.get(src, 0) + rec["n_routes"]
    gdf_counts = _gdf_counts_by_source(z, ap)
    return {
        "gdf_counts": gdf_counts,
        "independent_counts_by_source": indep_counts_by_source,
        "gdf_counts_equal_independent": gdf_counts == indep_counts_by_source,
    }


def scenario_alt_shortest(mz, seed):
    z = _make_zonal(mz)
    ap = _arm_tools().alternative_paths(z, origin_id=0, search_radius=10000, detour_ratio=1)
    indep = _independent_paths(z, _origin_idx(z), search_radius=10000,
                               detour_ratio=1, turn_penalty=False)
    return {
        "gdf": paths_gdf_digest(ap),
        "independent": indep,
        "coherence": _coherence(z, ap, indep),
    }


def scenario_alt_detour(mz, seed):
    z = _make_zonal(mz)
    paths_mod = _arm_paths_module()
    o_idx = _origin_idx(z)

    # engine internals (the exact call sequence of path_generator) so
    # the route set is comparable at the edge-id level:
    z.network.add_node_to_graph(z.network.d_graph, o_idx)
    d_idxs, o_scope, o_scope_paths = paths_mod.turn_o_scope(
        network=z.network, o_idx=o_idx, search_radius=10000,
        detour_ratio=2.5, turn_penalty=False,
        o_graph=z.network.d_graph, return_paths=True)
    scope_nodes, distance_matrix, _ = paths_mod.bfs_subgraph_generation(
        o_idx=o_idx, detour_ratio=2.5, o_graph=z.network.d_graph,
        d_idxs=d_idxs, o_scope=o_scope, o_scope_paths=o_scope_paths)
    d_allowed = {d: d_idxs[d] * 2.5 for d in d_idxs}
    path_edges, distances = paths_mod.bfs_path_edges_many_targets_iterative(
        network=z.network, o_graph=z.network.d_graph, o_idx=o_idx,
        d_idxs=d_allowed, distance_matrix=distance_matrix,
        turn_penalty=False, od_scope=scope_nodes)
    z.network.remove_node_to_graph(z.network.d_graph, o_idx)

    # public materialization through the API under the same detour bound
    ap = _arm_tools().alternative_paths(z, origin_id=0, search_radius=10000, detour_ratio=2.5)
    indep = _independent_paths(z, o_idx, search_radius=10000,
                               detour_ratio=2.5, turn_penalty=False)

    # engine-vs-independent, per destination and representation-coherent:
    # both sides hold INTERIOR edge-id lists and route weights; the
    # comparison pins the FULL route set (no hidden K limit) and the
    # exact per-route distances
    engine_per_dest = {
        int(d): {"n_routes": len(path_edges[d]),
                 "distances": sorted(_f(v) for v in distances[d]),
                 "interiors": sorted([int(e) for e in segs]
                                     for segs in path_edges[d])}
        for d in path_edges}
    indep_per_dest = indep["per_destination"]
    engine_equals_independent = (
        {k: v for k, v in engine_per_dest.items() if v["n_routes"] > 0} ==
        {k: v for k, v in indep_per_dest.items()})
    return {
        "gdf": paths_gdf_digest(ap),
        "engine_d_idxs_order": [int(k) for k in d_idxs],
        "engine_per_destination": engine_per_dest,
        "independent": indep,
        "engine_equals_independent": engine_equals_independent,
        "coherence": _coherence(z, ap, indep),
    }


def scenario_turn_params(mz, seed):
    z = _make_zonal(mz, turn_threshold=45, turn_penalty_amount=30)
    net_before = [float(z.network.turn_threshold_degree),
                  float(z.network.turn_penalty_amount)]

    ap_off = _arm_tools().alternative_paths(z, origin_id=0, search_radius=10000,
                                 detour_ratio=2.5, turn_penalty=False)
    ap_on = _arm_tools().alternative_paths(z, origin_id=0, search_radius=10000,
                                 detour_ratio=2.5, turn_penalty=True)

    z.set_turn_parameters(turn_threshold_degree=91, turn_penalty_amount=15)
    net_after = [float(z.network.turn_threshold_degree),
                 float(z.network.turn_penalty_amount)]
    ap_relaxed = _arm_tools().alternative_paths(z, origin_id=0, search_radius=10000,
                                 detour_ratio=2.5, turn_penalty=True)

    # direct turn primitive probes on the relaxed network: first three
    # street nodes with two neighbors, exact upstream arithmetic
    paths_mod = _arm_paths_module()
    node_gdf = z.network.nodes
    turn_probe = []
    for n in node_gdf[node_gdf["type"] == "street_node"].index:
        nbrs = list(z.network.d_graph.neighbors(n))
        if len(nbrs) >= 2:
            prev, cur, nxt = int(n), int(nbrs[0]), int(nbrs[1])
            geom3 = [node_gdf.at[prev, "geometry"],
                     node_gdf.at[cur, "geometry"],
                     node_gdf.at[nxt, "geometry"]]
            raw = paths_mod.angle_deviation_between_two_lines(geom3,
                                                              raw_angle=True)
            norm = paths_mod.angle_deviation_between_two_lines(geom3)
            pen = paths_mod.turn_penalty_value(z.network, prev, cur, nxt)
            turn_probe.append({"prev_cur_next": [prev, cur, nxt],
                               "raw_angle": _f(raw),
                               "norm_angle": _f(norm),
                               "penalty": float(pen)})

    return {
        "net_params_before": net_before,
        "net_params_after": net_after,
        "gdf_penalty_off": paths_gdf_digest(ap_off),
        "gdf_penalty_on": paths_gdf_digest(ap_on),
        "gdf_penalty_on_relaxed": paths_gdf_digest(ap_relaxed),
        "turn_probes": turn_probe,
    }


def scenario_engine_primitives(mz, seed):
    z = _make_zonal(mz)
    paths_mod = _arm_paths_module()
    G = z.network.d_graph
    o_idx = _origin_idx(z)

    # path_generator end-to-end (adds/removes o_idx itself)
    path_edges, distances, d_idxs = paths_mod.path_generator(
        network=z.network, o_idx=o_idx, search_radius=10000,
        detour_ratio=2.5, turn_penalty=False)

    # turn_o_scope twice: return_paths True and False, o_idx present.
    # o_idx stays in the graph through the whole bfs/enumerator
    # sequence (path_generator's own lifetime for it).
    z.network.add_node_to_graph(G, o_idx)
    d1, scope1, scope_paths1 = paths_mod.turn_o_scope(
        network=z.network, o_idx=o_idx, search_radius=10000,
        detour_ratio=2.5, turn_penalty=False, o_graph=G, return_paths=True)
    d2, scope2, scope_paths2 = paths_mod.turn_o_scope(
        network=z.network, o_idx=o_idx, search_radius=10000,
        detour_ratio=2.5, turn_penalty=False, o_graph=G, return_paths=False)

    # bfs_subgraph_generation + all three enumerators on its outputs
    # (same call sequence as path_generator)
    scope_nodes, distance_matrix, _ = paths_mod.bfs_subgraph_generation(
        o_idx=o_idx, detour_ratio=2.5, o_graph=G, d_idxs=d1,
        o_scope=scope1, o_scope_paths=scope_paths1)
    d_allowed = {d: d1[d] * 2.5 for d in d1}
    edges_a, dist_a = paths_mod.bfs_path_edges_many_targets_iterative(
        network=z.network, o_graph=G, o_idx=o_idx, d_idxs=d_allowed,
        distance_matrix=distance_matrix, turn_penalty=False,
        od_scope=scope_nodes)
    paths_b, dist_b = paths_mod.bfs_paths_many_targets_iterative(
        network=z.network, o_graph=G, o_idx=o_idx, d_idxs=d_allowed,
        distance_matrix=distance_matrix, turn_penalty=False,
        od_scope=scope_nodes)
    edges_w, dist_w = paths_mod.wandering_messenger(
        network=z.network, o_graph=G, o_idx=o_idx, d_idxs=d_allowed,
        distance_matrix=distance_matrix, turn_penalty=False,
        od_scope=scope_nodes)
    z.network.remove_node_to_graph(G, o_idx)

    return {
        "path_generator_d_idxs_order": [int(k) for k in d_idxs],
        "path_generator_distances": {int(k): [_f(v) for v in vs]
                                     for k, vs in distances.items()},
        "path_generator_edges": {int(k): [[int(e) for e in segs]
                                          for segs in path_edges[k]]
                                 for k in path_edges},
        "turn_o_scope_d_idxs": {int(k): _f(v) for k, v in d1.items()},
        "turn_o_scope_scope": {int(k): _f(v) for k, v in scope1.items()},
        "turn_o_scope_paths": {int(k): [int(x) for x in v]
                               for k, v in scope_paths1.items()},
        "turn_o_scope_nopaths_scope": {int(k): _f(v)
                                       for k, v in scope2.items()},
        "scope_paths_empty_when_return_paths_false": scope_paths2 == {},
        "scope_size": sorted(int(n) for n in scope_nodes),
        "distance_matrix": {int(n): {int(d): _f(w) for d, w in row.items()}
                            for n, row in sorted(distance_matrix.items(),
                                                 key=lambda kv: int(kv[0]))},
        "bfs_path_edges": {int(k): [[int(e) for e in segs]
                                    for segs in edges_a[k]]
                           for k in edges_a},
        "bfs_path_edges_distances": {int(k): [_f(v) for v in dist_a[k]]
                                     for k in dist_a},
        "bfs_paths_nodes": {int(k): [[int(x) for x in p] for p in paths_b[k]]
                            for k in paths_b},
        "bfs_paths_distances": {int(k): [_f(v) for v in dist_b[k]]
                                for k in dist_b},
        "wandering_edges": {int(k): [[int(e) for e in segs]
                                     for segs in edges_w[k]]
                            for k in edges_w},
        "wandering_distances": {int(k): [_f(v) for v in dist_w[k]]
                                for k in dist_w},
    }


def scenario_edge_cases(mz, seed):
    # (a) isolated component: a destination exists but is unreachable
    # from the origin component -> empty route set, pinned schema
    z = _make_zonal(mz, iso_destination=True)
    ap_far = _arm_tools().alternative_paths(z, origin_id=0, search_radius=10000,
                                 detour_ratio=2.5)
    digest_far = paths_gdf_digest(ap_far)

    # (b) tiny radius: nothing reachable at all
    ap_none = _arm_tools().alternative_paths(z, origin_id=0, search_radius=1,
                                  detour_ratio=1)

    # (c) int vs float detour_ratio equivalence on valid input
    ap_int = _arm_tools().alternative_paths(z, origin_id=0, search_radius=10000,
                                 detour_ratio=1)
    ap_float = _arm_tools().alternative_paths(z, origin_id=0, search_radius=10000,
                                   detour_ratio=1.0)

    # (d) origin node weight consistent across repeated calls (the
    # dossier's "repeated calls on the same Zonal" combination)
    node_weight_after = float(
        z.network.nodes.loc[z.network.nodes["type"] == "origin", "weight"]
        .iloc[0])

    return {
        "iso_destination_gdf": digest_far,
        "tiny_radius_gdf": paths_gdf_digest(ap_none),
        "detour_int_gdf": paths_gdf_digest(ap_int),
        "detour_float_gdf": paths_gdf_digest(ap_float),
        "detour_int_equals_float":
            json.dumps(paths_gdf_digest(ap_int), sort_keys=True) ==
            json.dumps(paths_gdf_digest(ap_float), sort_keys=True),
        "origin_node_weight_after_calls": _f(node_weight_after),
    }


def scenario_validation(mz, seed):
    results = {}

    def probe(name, fn):
        _, error, _ = _capture_stdout(fn)
        results[name] = error if error is not None else "NO_ERROR"

    z = _make_zonal(mz)

    probe("origin_id_str", lambda: _arm_tools().alternative_paths(z, 
        origin_id="1", search_radius=10000))
    probe("origin_id_unknown", lambda: _arm_tools().alternative_paths(z, 
        origin_id=9999, search_radius=10000))
    probe("origin_id_list", lambda: _arm_tools().alternative_paths(z, 
        origin_id=[0], search_radius=10000))
    probe("search_radius_str", lambda: _arm_tools().alternative_paths(z, 
        origin_id=0, search_radius="10000"))
    probe("search_radius_negative", lambda: _arm_tools().alternative_paths(z, 
        origin_id=0, search_radius=-5))
    probe("detour_below_one", lambda: _arm_tools().alternative_paths(z, 
        origin_id=0, search_radius=10000, detour_ratio=0.9))
    probe("detour_str", lambda: _arm_tools().alternative_paths(z, 
        origin_id=0, search_radius=10000, detour_ratio="1.5"))
    probe("turn_penalty_str", lambda: _arm_tools().alternative_paths(z, 
        origin_id=0, search_radius=10000, turn_penalty="yes"))

    # network but NO graph
    z2 = mz.Zonal()
    z2.load_layer("streets", _streets_gdf())
    z2.load_layer("origins", _origins_gdf())
    z2.load_layer("destinations", _destinations_gdf())
    z2.create_street_network("streets", weight_attribute="length")
    z2.insert_node("origins", "origin")
    z2.insert_node("destinations", "destination")
    probe("no_graph", lambda: _arm_tools().alternative_paths(z2, 
        origin_id=0, search_radius=10000))

    # completely empty zonal
    z3 = mz.Zonal()
    probe("empty_zonal", lambda: _arm_tools().alternative_paths(z3, 
        origin_id=0, search_radius=10000))

    # network + graph but zero origins inserted
    z4 = mz.Zonal()
    z4.load_layer("streets", _streets_gdf())
    z4.load_layer("destinations", _destinations_gdf())
    z4.create_street_network("streets", weight_attribute="length")
    z4.insert_node("destinations", "destination")
    z4.create_graph()
    probe("no_origins", lambda: _arm_tools().alternative_paths(z4, 
        origin_id=0, search_radius=10000))

    return results


SCENARIOS = {
    "alt_shortest": scenario_alt_shortest,
    "alt_detour": scenario_alt_detour,
    "turn_params": scenario_turn_params,
    "engine_primitives": scenario_engine_primitives,
    "edge_cases": scenario_edge_cases,
    "validation": scenario_validation,
}

# Test modules import the scenario list from THIS uniquely-named module,
# never via `from conftest import ...`: bare `conftest` collides across
# non-package test dirs in sys.modules (first suite collected wins), so
# co-collected suites would silently read each other's lists or fail
# collection (zonal review finding; class-wide pattern fix).
SCENARIO_NAMES = list(SCENARIOS)


# ----------------------------------------------------------------------
# comparator-sensitivity mutations: each drops/alters exactly one piece
# of recorded state; an honest comparator MUST select the difference
# (digest-level controls; the stronger end-to-end monkeypatched-mutant
# form lives in TOPOLOGY's suite).
# ----------------------------------------------------------------------

def _sabotage_k_limit(d):
    # the hidden-K-limit control: keep only ONE route for ONE
    # destination — must be selected
    first = list(d["engine_per_destination"])[0]
    rec = d["engine_per_destination"][first]
    d["engine_per_destination"] = {
        first: {"n_routes": 1,
                "distances": rec["distances"][:1],
                "interiors": rec["interiors"][:1]}}


SABOTAGES = {
    "k_limit_paths": ("alt_detour", _sabotage_k_limit),
    "drop_one_path": ("alt_detour",
                      lambda d: d["gdf"]["values"]["distance"].pop()),
    "perturb_distance_bit": ("alt_shortest",
                             lambda d: d["gdf"]["values"]["distance"]
                             .__setitem__(0, "0000000000000000")),
    "drop_geometry_component": ("alt_shortest",
                                lambda d: d["gdf"]["values"]["geometry"][0][
                                    "component_wkb"].pop()),
    "flip_turn_distance": ("turn_params",
                           lambda d: d.__setitem__("gdf_penalty_on",
                                                   d["gdf_penalty_off"])),
    "hide_validation_error": ("validation",
                              lambda d: d.__setitem__("detour_below_one",
                                                      "NO_ERROR")),
}


# ----------------------------------------------------------------------
# arm dispatch
# ----------------------------------------------------------------------

def import_arm(arm):
    if arm == "reference":
        sys.path.insert(0, str(REPO / ".refs" / "madina_ref" / "src"))
        import madina.una.paths as paths_mod     # noqa: PLC0415
        import madina.una.tools as tools_mod     # noqa: PLC0415
        import madina.zonal as mz                # noqa: PLC0415
        _ARM_MARKER["paths"] = paths_mod
        _ARM_MARKER["tools"] = tools_mod
        return mz
    if arm == "facade":
        sys.path.insert(0, str(REPO / "src"))
        from urban_network_analysis.compat.madina import una      # noqa: PLC0415
        import urban_network_analysis.compat.madina.una.tools as una_tools  # noqa: PLC0415
        from urban_network_analysis.compat.madina import zonal as mz  # noqa: PLC0415
        _ARM_MARKER["paths"] = una.paths
        _ARM_MARKER["tools"] = una_tools
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
