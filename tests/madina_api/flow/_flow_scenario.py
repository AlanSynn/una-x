"""MADINA_FLOW parity scenario runner: exercise the betweenness surface
(parallel_betweenness / one_betweenness_2 / betweenness_exposure /
paralell_betweenness_exposure + the tools.betweenness wrapper) through
either the compat facade or the pinned upstream reference, then dump a
BITWISE-strict state digest.

Same discipline as tests/madina_api/paths and tests/madina_api/access
(MADINA_PATHS / MADINA_ACCESS): both arms run under the SAME interpreter
(the dependency-bridged reference venv) with identical fixtures, so any
digest difference is attributable to the code under test.  Floats are
compared as IEEE-754 bit patterns, geometries as WKB hashes,
order-sensitive dict content as explicit lists.

RNG control: ``paralell_betweenness_exposure`` shuffles origins with an
UNSEEDED ``sample(frac=1)`` (pinned upstream behavior — NOT removed).
Every scenario seeds the global numpy RNG immediately before each call
and records the resulting permutation.  (The low-level
``parallel_betweenness`` has that shuffle commented out upstream, so it
is index-ordered.)

Nondeterminism policy (documented in the facade header, not hidden):
  - wall-clock diagnostic columns (destination_discovery_time,
    destination_prep_time, path_generation_time, chunck_time) are
    EXCLUDED from digests, with their presence recorded;
  - content-deterministic diagnostics (memory_stalls count, chunck
    path/segment counts, getsizeof-derived path_segment_memory) ARE
    digested;
  - the time-based "Time spent:" progress prints are recorded as
    presence only, never counts;
  - num_cores=2 is bitwise-stable for the LOW-LEVEL engine (its
    array_split partition is broken upstream — every worker dies, the
    result is all-zero — so trivially deterministic) and is digested
    bitwise; the EXPOSURE engine distributes origins through a shared
    queue, so the per-edge float-addition order carries NO
    bitwise-stability contract at num_cores>1 (load-dependent by
    construction; empirically stable across three retained runs on this
    fixture — evidence logs/nc2_stability_probe/) — exposure
    num_cores=2 runs are recorded structurally (ran / column exists /
    finite / nonzero count), never digest-compared bitwise, and the
    reason is recorded in the digest.

Usage:
    python _flow_scenario.py --arm {facade,reference} --scenario NAME
                             --out DIGEST.json --seed 20260930
                             [--sabotage NAME]
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import math
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

# sabotage selected on the command line, consumed by scenario bodies
# (parameter-level mutants) and by main() (digest-level mutants)
_SABOTAGE = {"name": None}

# wall-clock diagnostic columns: excluded from every digest, presence
# recorded per frame (see module docstring)
TIME_COLUMNS = ("destination_discovery_time", "destination_prep_time",
                "path_generation_time", "chunck_time")

# every origin-stats column betweenness_exposure can write (the stats
# block), used for presence/absence pins across model variants
STATS_COLUMNS = (
    "reach", "gravity", "knn_weight",
    "mean_hazzard", "decayed_mean_hazzad", "expected_hazzard_meters",
    "probable_travel_distance_weighted_hazzard",
    "closest_destination_distance", "furthest_destination_distance",
    "mean_path_length", "probable_travel_distance",
    "eligible_destinations", "path_count", "path_segment_count",
    "path_segment_memory", "scope_node_count", "chunck_count",
    "memory_stalls",
)


# ----------------------------------------------------------------------
# bitwise digest helpers (same discipline as the paths/zonal/access
# suites)
# ----------------------------------------------------------------------

def _f(v):
    """float -> IEEE-754 bit-pattern hex (bitwise-strict, NaN-safe)."""
    return struct.pack(">d", float(v)).hex()


def _wkb_hash(geom):
    return hashlib.sha256(shapely.to_wkb(geom, byte_order=1)).hexdigest()[:24]


def _geomcell_digest(geom):
    """One geometry cell: WKB hash, or component digests for collections."""
    if geom.geom_type == "GeometryCollection":
        return {"collection": [_wkb_hash(g) for g in geom.geoms]}
    if geom.geom_type == "MultiPolygon":
        return {"collection": [_wkb_hash(g) for g in geom.geoms]}
    return _wkb_hash(geom)


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


def _frame_digest(df):
    """Bitwise-strict digest of a DataFrame or GeoDataFrame.  The
    geometry column (GeoDataFrame-registered OR a plain-object 'geometry'
    column, as produced by pd.concat of worker frames) is hashed via WKB;
    everything else by dtype-classed exact values."""
    gcol = getattr(df, "_geometry_column_name", None)
    if gcol is None and "geometry" in df.columns:
        gcol = "geometry"
    d = {
        "columns": list(df.columns),
        "dtypes": [str(t) for t in df.dtypes],
        "index": [_jsonable(v) for v in df.index],
        "index_name": repr(df.index.name),
        "values": {c: _series(df[c]) for c in df.columns if c != gcol},
    }
    if gcol is not None and gcol in df.columns:
        d["values"][gcol] = [_geomcell_digest(g) for g in df[gcol]]
    return d


def gdf_digest(gdf):
    return _frame_digest(gdf)


def _origin_gdf_digest(origin_gdf):
    """Digest of a returned origin frame: wall-clock columns excluded
    (their presence recorded), everything else bitwise."""
    digest = _frame_digest(origin_gdf.drop(columns=[
        c for c in TIME_COLUMNS if c in origin_gdf.columns]))
    digest["excluded_time_columns_present"] = [
        c for c in TIME_COLUMNS if c in origin_gdf.columns]
    digest["stats_columns_present"] = [
        c for c in STATS_COLUMNS if c in origin_gdf.columns]
    return digest


def _betweenness_by_edge(edge_gdf):
    """The betweenness column as {edge_id: IEEE-754 hex} (order is the
    edge index order; ids stringified for JSON)."""
    return {str(int(idx)): _f(v)
            for idx, v in edge_gdf["betweenness"].items()}


def _capture_stdout_stderr(fn):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        try:
            result = fn()
            error = None
        except Exception as ex:               # noqa: BLE001 - pinned failures
            result = None
            error = f"{type(ex).__name__}: {ex}"
    return result, error, out.getvalue(), err.getvalue()


# The betweenness/tools modules of whichever arm imported us (set by
# import_arm), so engine-primitive replays exercise the SAME module the
# scenario under test used.
_ARM_MARKER = {"betweenness": None, "tools": None, "zonal": None}


def _arm_betweenness():
    if _ARM_MARKER["betweenness"] is None:
        raise RuntimeError("arm not imported")
    return _ARM_MARKER["betweenness"]


def _arm_tools():
    if _ARM_MARKER["tools"] is None:
        raise RuntimeError("arm not imported")
    return _ARM_MARKER["tools"]


# ----------------------------------------------------------------------
# fixture: the same square + diagonal + dead-end stub grid as the access
# suite (plus optional isolated component), with per-street 'hazzard'
# values for the exposure scenarios, configurable origin weights, and an
# optional second origin layer.
#
#   F --------- A ======= o1/o2 == B --- d1 --- C --- d? --- G
#               | \                     |                (200,100)
#               |   \.... diagonal .....\
#               |                        |
#               D ---------------------- +
#               |
#               E (stub, d2 at its end)     ISO: iso1 --- iso2 (d3)
#
#   streets: F-A, A-B, B-C, C-D, D-A, A-C (diagonal), D-E, C-G
#            (+ ISO edge when iso=True).
#   origins (source ids 0,1,2): o1 (-50,0) mid F-A, o2 (50,0) mid A-B,
#   o3 (100,50) mid B-C; weights origin_weights (default [1, 2, 1]).
#   destinations: d1 (50,100) mid C-D weight 2, d2 (0,150) = E weight 3
#   (+ d3 (550,500) on the isolated component, weight 1, when iso=True).
#   ORIGINS2 (second_origins=True): o4 (0,50) mid D-A, weight 5, its own
#   layer -> mixed source_layer values across origin rows.
# ----------------------------------------------------------------------

def _streets_gdf(hazzard=False):
    rows = [
        ("F-A", 100.0, LineString([(-100, 0), (0, 0)])),
        ("A-B", 100.0, LineString([(0, 0), (100, 0)])),
        ("B-C", 100.0, LineString([(100, 0), (100, 100)])),
        ("C-D", 100.0, LineString([(100, 100), (0, 100)])),
        ("D-A", 100.0, LineString([(0, 100), (0, 0)])),
        ("A-C", float(np.hypot(100, 100)),
         LineString([(0, 0), (100, 100)])),
        ("D-E", 50.0, LineString([(0, 100), (0, 150)])),
        ("C-G", 100.0, LineString([(100, 100), (200, 100)])),
    ]
    data = {"id": [r[0] for r in rows], "length": [r[1] for r in rows]}
    if hazzard:
        # deterministic per-street exposure values (read by the
        # path_exposure_attribute machinery through parent_street_id)
        data["hazzard"] = [0.5, 1.5, 1.0, 2.0, 0.5, 1.0, 0.25, 0.75]
    return gpd.GeoDataFrame(
        data, geometry=[r[2] for r in rows], crs="EPSG:3857")


def _origins_gdf(origin_weights=None):
    # PINNED: load_layer resets the layer index, so source_id is the
    # positional 0..n-1 regardless of any fixture index.
    weights = [1.0, 2.0, 1.0] if origin_weights is None else origin_weights
    return gpd.GeoDataFrame(
        {"weight": weights},
        geometry=[Point((-50, 0)), Point((50, 0)), Point((100, 50))],
        crs="EPSG:3857")


def _origins2_gdf():
    return gpd.GeoDataFrame(
        {"weight": [5.0]}, geometry=[Point((0, 50))], crs="EPSG:3857")


def _destinations_gdf(zero_weights=False):
    weights = [2.0, 3.0]
    if zero_weights:
        weights = [0.0, 0.0]
    return gpd.GeoDataFrame(
        {"weight": weights},
        geometry=[Point((50, 100)), Point((0, 150))], crs="EPSG:3857")


def _make_zonal(mz, *, origin_weights=None, hazzard=False,
                second_origins=False, zero_dest_weights=False,
                turn_threshold=45, turn_penalty_amount=30):
    z = mz.Zonal()
    z.load_layer("streets", _streets_gdf(hazzard=hazzard))
    z.load_layer("origins", _origins_gdf(origin_weights=origin_weights))
    if second_origins:
        z.load_layer("origins2", _origins2_gdf())
    z.load_layer("destinations",
                 _destinations_gdf(zero_weights=zero_dest_weights))
    z.create_street_network(
        "streets", weight_attribute="length",
        turn_threshold_degree=turn_threshold,
        turn_penalty_amount=turn_penalty_amount)
    z.insert_node("origins", label="origin", weight_attribute="weight")
    if second_origins:
        z.insert_node("origins2", label="origin",
                      weight_attribute="weight")
    z.insert_node("destinations", label="destination",
                  weight_attribute="weight")
    z.create_graph()
    return z


def _origin_idx(z, source_id):
    n = z.network.nodes
    sel = n[(n["type"] == "origin") & (n["source_id"] == source_id)]
    return int(sel.index[0])


def _destination_idxs(z):
    n = z.network.nodes
    return [int(i) for i in n[n["type"] == "destination"].index]


def _seeded_origin_order(z, seed):
    """Replicate paralell_betweenness_exposure's UNSEEDED sample(frac=1)
    permutation under a seeded global RNG, then reseed so the actual call
    consumes the identical permutation.  Recorded in every digest that
    depends on processing order (queue order, tie resolution, first
    processed source_layer)."""
    n = z.network.nodes
    origin_gdf = n[n["type"] == "origin"]
    np.random.seed(seed)
    order = origin_gdf.sample(frac=1).index.astype("int").tolist()
    np.random.seed(seed)
    return [int(i) for i in order]


def _node_state(z):
    return gdf_digest(z.network.nodes)


def _layer_states(z):
    return {name: gdf_digest(z[name].gdf)
            for name in z.layers.layers if name != "streets"}


# ----------------------------------------------------------------------
# scenarios
# ----------------------------------------------------------------------

def _tools_run(z, seed, **kwargs):
    """One seeded tools.betweenness call with its full observable state."""
    tools = _arm_tools()
    order = _seeded_origin_order(z, seed)
    res, err, out, _ = _capture_stdout_stderr(
        lambda: tools.betweenness(z, **kwargs))
    return {
        "edges": gdf_digest(z.network.edges),
        "betweenness_by_edge": _betweenness_by_edge(z.network.edges),
        "layers": _layer_states(z),
        # tools.betweenness ASSIGNS network.knn_weight/knn_plateau on
        # every call — but validation failures raise before that point,
        # so the attributes may be absent entirely (pinned distinction)
        "network_knn_weight": repr(
            getattr(z.network, "knn_weight", "<absent>")),
        "network_knn_plateau": repr(
            getattr(z.network, "knn_plateau", "<absent>")),
        "return": repr(res),
        "error": err,
        "progress_seen": "Time spent:" in out,
        "shuffled_origin_order": order,
    }


def scenario_flow_closest_huff(mz, seed):
    """tools.betweenness (num_cores=1) with the default closest-
    destination model vs the Huff competition model, plus direct
    paralell_betweenness_exposure runs capturing the returned origin
    frame (stats quirk: with closest_destination=True the stats block
    dies on the unbound eligible_destinations_shortest_distance ->
    reach/gravity ARE written, everything after is not)."""
    btd = _arm_betweenness()

    z1 = _make_zonal(mz)
    closest = _tools_run(
        z1, seed, search_radius=250, detour_ratio=1.0, decay=False,
        num_cores=1, save_betweenness_as="bt", save_reach_as="reach",
        save_gravity_as="grav")

    z2 = _make_zonal(mz)
    huff = _tools_run(
        z2, seed, search_radius=250, detour_ratio=1.0, decay=False,
        num_cores=1, closest_destination=False, save_betweenness_as="bt",
        save_reach_as="reach", save_gravity_as="grav")

    # direct engine runs: the origin frame is only observable here
    z3 = _make_zonal(mz)
    _ = _seeded_origin_order(z3, seed)
    res3, err3, out3, _ = _capture_stdout_stderr(
        lambda: btd.paralell_betweenness_exposure(
            z3, search_radius=250, detour_ratio=1.0, decay=False,
            beta=0.003, num_cores=1, closest_destination=True))
    z4 = _make_zonal(mz)
    _ = _seeded_origin_order(z4, seed)
    res4, err4, out4, _ = _capture_stdout_stderr(
        lambda: btd.paralell_betweenness_exposure(
            z4, search_radius=250, detour_ratio=1.0, decay=False,
            beta=0.003, num_cores=1, closest_destination=False))

    # split-street probe.  Pinned upstream fact: load_layer ALWAYS
    # resets the input index (zonal.py 'gdf["id"] = range(gdf.shape[0]);
    # gdf = gdf.set_index("id")'), so a split street (two edge rows
    # sharing parent_street_id) is unreachable through the supported
    # layer surface at this commit — ordinary insertion yields 8 edge
    # rows with 8 unique parents.  The pipeline-produced edge-table
    # state is instead reproduced on the engine's own edge table, the
    # exact shape the preprocessing pipeline hands the una layer:
    # edge 3's parent is merged into edge 2's (edge 3's betweenness is
    # the unique value 4.0, so the discard is visible).  tools.betweenness
    # then maps per-edge betweenness back onto the streets layer via
    # drop_duplicates on parent_street_id: only the FIRST half of the
    # split street keeps its betweenness, the second half's value is
    # silently discarded (street row 3 gets NaN — no surviving parent
    # 3), and the join index lands non-int before the TODO cast at
    # tools.py 'index became a float'.
    z5 = _make_zonal(mz)
    assert int(z5.network.edges["parent_street_id"].nunique()) == 8
    z5.network.edges.loc[3, "parent_street_id"] = 2
    split = _tools_run(
        z5, seed, search_radius=250, detour_ratio=1.0, decay=False,
        num_cores=1, closest_destination=True, save_betweenness_as="bt")
    street_gdf = z5["streets"].gdf
    split_parents = split["edges"]["values"]["parent_street_id"]
    split_row = next(i for i, p in enumerate(split_parents)
                     if split_parents.count(p) >= 2)

    return {
        "closest_true": closest,
        "huff": huff,
        "models_distinguish": closest["betweenness_by_edge"]
        != huff["betweenness_by_edge"],
        "split_probe": {
            "edges": split["edges"],
            "betweenness_by_edge": split["betweenness_by_edge"],
            "parents": split_parents,
            "first_split_row": split_row,
            "first_split_shared_by": split_parents.count(
                split_parents[split_row]),
            "street_bt_values":
                [_jsonable(v) for v in street_gdf["bt"].tolist()],
            "street_bt_index":
                [int(i) for i in street_gdf.index],
            "street_bt_index_dtype": str(street_gdf.index.dtype),
        },
        "direct_closest": {
            "origin_gdf": _origin_gdf_digest(res3["origin_gdf"]),
            "betweenness_by_edge": _betweenness_by_edge(res3["edge_gdf"]),
            "return_keys": sorted(res3.keys()),
            "error": err3, "progress_seen": "Time spent:" in out3},
        "direct_huff": {
            "origin_gdf": _origin_gdf_digest(res4["origin_gdf"]),
            "betweenness_by_edge": _betweenness_by_edge(res4["edge_gdf"]),
            "return_keys": sorted(res4.keys()),
            "error": err4, "progress_seen": "Time spent:" in out4},
    }


def scenario_flow_decay_matrix(mz, seed):
    """Direct paralell_betweenness_exposure (num_cores=1, Huff) decay
    matrix: no decay vs exponent vs power, the destniation_cap cut
    (typo param pinned), and the sabotage hook decay_method_swap."""
    btd = _arm_betweenness()
    exponent_method, power_method = "exponent", "power"
    if _SABOTAGE["name"] == "decay_method_swap":
        # parameter-level mutant (dossier): the digests recorded under
        # each method key are then those of the WRONG method
        exponent_method, power_method = power_method, exponent_method

    def run(decay, decay_method="exponent", cap=None):
        z = _make_zonal(mz)
        _ = _seeded_origin_order(z, seed)
        res, err, out, _ = _capture_stdout_stderr(
            lambda: btd.paralell_betweenness_exposure(
                z, search_radius=250, detour_ratio=1.0, decay=decay,
                decay_method=decay_method, beta=0.003, num_cores=1,
                closest_destination=False, destniation_cap=cap))
        return {
            "origin_gdf": None if res is None else
            _origin_gdf_digest(res["origin_gdf"]),
            "betweenness_by_edge": None if res is None else
            _betweenness_by_edge(res["edge_gdf"]),
            "error": err, "progress_seen": "Time spent:" in out}

    no_decay = run(False)
    exponent = run(True, exponent_method)
    power = run(True, power_method)
    capped = run(False, cap=1)

    # path-detour penalty and detour engagement on the EXPOSURE engine:
    # wandering_messenger paths carry real edge sequences, so per-edge
    # attribution genuinely distinguishes penalties and detour widths
    # (unlike the degenerate low-level engine — see flow_lowlevel_engine)
    def run_penalty(penalty, detour):
        z = _make_zonal(mz)
        _ = _seeded_origin_order(z, seed)
        res, err, out, _ = _capture_stdout_stderr(
            lambda: btd.paralell_betweenness_exposure(
                z, search_radius=250, detour_ratio=detour, decay=False,
                beta=0.003, num_cores=1, closest_destination=False,
                path_detour_penalty=penalty))
        return {
            "betweenness_by_edge": None if res is None else
            _betweenness_by_edge(res["edge_gdf"]),
            "error": err, "progress_seen": "Time spent:" in out}

    pen_equal_15 = run_penalty("equal", 1.5)
    pen_exponent_15 = run_penalty("exponent", 1.5)
    pen_power_15 = run_penalty("power", 1.5)
    detour_10 = run_penalty("equal", 1.0)

    return {
        "no_decay": no_decay,
        "exponent": exponent,
        "power": power,
        "destniation_cap_1": capped,
        "exponent_differs_from_no_decay": (
            exponent["betweenness_by_edge"] != no_decay["betweenness_by_edge"]),
        "power_differs_from_exponent": (
            power["betweenness_by_edge"] != exponent["betweenness_by_edge"]),
        "cap_differs_from_uncapped": (
            capped["betweenness_by_edge"] != no_decay["betweenness_by_edge"]),
        "penalty_equal_1_5": pen_equal_15,
        "penalty_exponent_1_5": pen_exponent_15,
        "penalty_power_1_5": pen_power_15,
        "detour_1_0": detour_10,
        "exposure_penalty_exponent_differs": (
            pen_exponent_15["betweenness_by_edge"]
            != pen_equal_15["betweenness_by_edge"]),
        "exposure_penalty_power_differs": (
            pen_power_15["betweenness_by_edge"]
            != pen_exponent_15["betweenness_by_edge"]),
        "exposure_detour_engages": (
            detour_10["betweenness_by_edge"]
            != pen_equal_15["betweenness_by_edge"]),
    }


def scenario_flow_lowlevel_engine(mz, seed):
    """The Network-level parallel_betweenness engine (path_generator
    path): penalty matrix (at detour_ratio=1.5 — with a single shortest
    path per OD the penalty is provably moot, recorded explicitly), the
    PINNED num_cores>1 defect (np.array_split shards the origins frame
    into ndarray rows, every worker dies on ``origins.index``, the
    as_completed loop swallows it, and sum([]) silently leaves ALL-ZERO
    betweenness), the rertain_expensive_data retention round-trip (with
    its upstream brokenness: only retained_d_idxs is ever populated),
    the honest detour pair (default 1.0 vs 1.5 — the ratio filters the
    path set upstream, so the pair engages), and the sabotage hook
    detour_swap."""
    btd = _arm_betweenness()
    r_small, r_large = 1.0, 1.5
    if _SABOTAGE["name"] == "detour_swap":
        # parameter-level mutant: the pair's digests swap meaning
        r_small, r_large = r_large, r_small

    def run(num_cores=1, **kw):
        z = _make_zonal(mz)
        params = dict(search_radius=250, detour_ratio=r_small if
                      "detour_ratio" not in kw else kw.pop("detour_ratio"),
                      decay=False, beta=0.003, num_cores=num_cores,
                      origin_weights=True, closest_destination=True)
        params.update(kw)
        res, err, out, _ = _capture_stdout_stderr(
            lambda: btd.parallel_betweenness(z.network, **params))
        return {
            "betweenness_by_edge": _betweenness_by_edge(z.network.edges),
            "network_edge_cols": list(z.network.edges.columns),
            "return_keys": None if res is None else sorted(res.keys()),
            "res_edge_is_network_edges":
                res is not None and res["edge_gdf"] is z.network.edges,
            "error": err,
            "swallowed_destination_errors": out.count("faced an error"),
            "worker_error_markers": out.count("'numpy.ndarray' object "
                                              "has no attribute 'index'"),
        }

    base = run()
    exponent_single_path = run(path_detour_penalty="exponent")
    equal_multi = run(detour_ratio=r_large)
    exponent = run(path_detour_penalty="exponent", detour_ratio=r_large)
    power = run(path_detour_penalty="power", detour_ratio=r_large)
    huff = run(closest_destination=False, destination_weights=True,
               perceived_distance=True)
    # the honest detour pair: default (1.0) vs r_large (1.5).  Upstream
    # filters the path set by the ratio (betweenness.py:297
    # 'this_path_weight > shortest_path_distance * detour_ratio'), so
    # 1.5 admits alternates and the per-edge bits change — the earlier
    # 1.0-vs-1.0 pairing was a vacuous comparison (caught in review)
    huff_detour_15 = run(closest_destination=False, destination_weights=True,
                         perceived_distance=True, detour_ratio=r_large)
    nc2 = run(num_cores=2)

    # retention round-trip (typo param rertain_expensive_data pinned):
    # upstream populates ONLY retained_d_idxs (the paths/distances
    # retains are commented out), so the reuse branch is only reachable
    # with d_idxs — passing them back with paths/distances left None
    # regenerates and must reproduce the same bits.
    z_ret = _make_zonal(mz)
    res1, err1, _, _ = _capture_stdout_stderr(
        lambda: btd.parallel_betweenness(
            z_ret.network, search_radius=250, detour_ratio=1.0,
            decay=False, beta=0.003, num_cores=1, origin_weights=True,
            closest_destination=True, rertain_expensive_data=True))
    ret_d = {str(int(o)): {str(int(d)): _f(dist)
                           for d, dist in per_origin.items()}
             for o, per_origin in res1["retained_d_idxs"].items()}
    res2, err2, _, _ = _capture_stdout_stderr(
        lambda: btd.parallel_betweenness(
            z_ret.network, search_radius=250, detour_ratio=1.0,
            decay=False, beta=0.003, num_cores=1, origin_weights=True,
            closest_destination=True,
            retained_d_idxs=res1["retained_d_idxs"]))
    retention = {
        "retained_d_idxs": ret_d,
        "retained_paths_empty": res1["retained_paths"] == {},
        "retained_distances_empty": res1["retained_distances"] == {},
        "first_call_return_keys": sorted(res1.keys()),
        "first_call_error": err1,
        "passthrough_error": err2,
        "passthrough_return_keys": sorted(res2.keys()),
        "passthrough_betweenness": _betweenness_by_edge(z_ret.network.edges),
        # passthrough (retained d_idxs handed back, paths/distances left
        # None -> upstream regenerates) must reproduce the base bits
        "passthrough_reproduces_bits":
            _betweenness_by_edge(z_ret.network.edges)
            == base["betweenness_by_edge"],
    }

    # one_betweenness_2 directly: the tracker starts at INT 0 (vs 0.0 in
    # the exposure engine), and the tiny-radius probe hits the
    # destination_count==0 continue for every origin.
    z_one = _make_zonal(mz)
    origins = z_one.network.nodes[z_one.network.nodes["type"] == "origin"]
    res_one, err_one, _, _ = _capture_stdout_stderr(
        lambda: btd.one_betweenness_2(
            z_one.network, search_radius=250, origins=origins,
            detour_ratio=1.0, decay=False, beta=0.003,
            path_detour_penalty="exponent", origin_weights=True,
            closest_destination=True))
    z_zero = _make_zonal(mz)
    origins_zero = z_zero.network.nodes[
        z_zero.network.nodes["type"] == "origin"]
    res_zero, err_zero, _, _ = _capture_stdout_stderr(
        lambda: btd.one_betweenness_2(
            z_zero.network, search_radius=10, origins=origins_zero,
            detour_ratio=1.0, decay=False, beta=0.003,
            path_detour_penalty="exponent", origin_weights=True,
            closest_destination=True))
    one_direct = {
        "tracker": {str(int(k)): _f(v)
                    for k, v in res_one["batch_betweenness_tracker"].items()},
        "tracker_types_int_start": all(
            v == 0 for v in res_one["batch_betweenness_tracker"].values()
            if v == 0),
        "error": err_one,
        "tiny_radius_tracker_all_zero": all(
            v == 0 for v in
            res_zero["batch_betweenness_tracker"].values()),
        "tiny_radius_error": err_zero,
    }

    return {
        "base_nc1": base,
        "penalty_exponent_single_path": exponent_single_path,
        "penalty_equal_multi_path": equal_multi,
        "penalty_exponent": exponent,
        "penalty_power": power,
        "huff_destination_weights": huff,
        "huff_detour_1_5": huff_detour_15,
        "nc2_silent_zero": nc2,
        "nc2_equals_nc1_bitwise": nc2["betweenness_by_edge"]
        == base["betweenness_by_edge"],
        "nc2_all_zero": all(v == _f(0.0) for v in
                            nc2["betweenness_by_edge"].values()),
        "penalty_moot_at_single_path": (
            exponent_single_path["betweenness_by_edge"]
            == base["betweenness_by_edge"]),
        "penalty_exponent_equals_power_at_multi_path": (
            exponent["betweenness_by_edge"]
            == power["betweenness_by_edge"]),
        # the equal branch computes probabilities in pure python
        # (1/len), which multiplied by the float32 origin weight stays
        # float32 and degrades the total; exponent/power branch
        # probabilities are numpy float64 and stay exact — a pinned
        # deterministic dtype quirk, identical on both arms
        "penalty_equal_bitwise_differs_from_exponent": (
            equal_multi["betweenness_by_edge"]
            != exponent["betweenness_by_edge"]),
        "penalty_equal_degradation_magnitude": abs(
            struct.unpack(">d", bytes.fromhex(
                equal_multi["betweenness_by_edge"]["0"]))[0]
            - struct.unpack(">d", bytes.fromhex(
                exponent["betweenness_by_edge"]["0"]))[0]),
        "lowlevel_detour_engages": (
            huff_detour_15["betweenness_by_edge"]
            != huff["betweenness_by_edge"]),
        "huff_differs": huff["betweenness_by_edge"]
        != base["betweenness_by_edge"],
        "attribution_note":
            "pinned upstream degeneracy: path_generator returns TRIMMED "
            "paths (interior nodes only) and every street node's "
            "nearest_edge_id is 0, so one_betweenness_2's "
            "[nearest(first)] + [] + [nearest(last)] attribution lands "
            "every path's mass on edge 0 twice; per-OD totals are "
            "2*prob_sum*weight.  Detour ratio ENGAGES in this engine "
            "(it filters the path set at betweenness.py:297, so 1.5 "
            "admits alternates and the bits move), and the penalty "
            "split engages exactly at multi-path (equal branch degrades "
            "through float32 only when alternates exist).  What the "
            "low-level engine lacks is REAL route attribution — mass "
            "never leaves edge 0 — which is pinned on the exposure "
            "engine, whose wandering_messenger paths carry real edge "
            "sequences",
        "retention": retention,
        "one_betweenness_2": one_direct,
    }


def scenario_flow_elastic_knn(mz, seed):
    """Elastic trip generation through tools.betweenness: the knn list
    and string forms must agree bitwise, the plateau must engage,
    save_elastic_weight_as joins the knn_weight column, and the pinned
    clobber: every call (elastic or not) overwrites
    network.knn_weight/knn_plateau."""
    list_form = _tools_run(
        _make_zonal(mz), seed, search_radius=250, detour_ratio=1.0,
        decay=False, num_cores=1, elastic_weight=True,
        knn_weight=[0.5, 0.25], knn_plateau=0, save_betweenness_as="bt",
        save_elastic_weight_as="ew")
    str_form = _tools_run(
        _make_zonal(mz), seed, search_radius=250, detour_ratio=1.0,
        decay=False, num_cores=1, elastic_weight=True,
        knn_weight="[0.5, 0.25]", knn_plateau=0, save_betweenness_as="bt",
        save_elastic_weight_as="ew")
    plateau = _tools_run(
        _make_zonal(mz), seed, search_radius=250, detour_ratio=1.0,
        decay=False, num_cores=1, elastic_weight=True,
        knn_weight=[0.5, 0.25], knn_plateau=100, save_betweenness_as="bt",
        save_elastic_weight_as="ew")

    # the clobber pin, on the SAME zonal: an elastic call sets
    # knn_weight/knn_plateau; a following non-elastic call resets
    # knn_weight to None (tools always assigns both attributes)
    z_clob = _make_zonal(mz)
    _tools_run(z_clob, seed, search_radius=250, detour_ratio=1.0,
               decay=False, num_cores=1, elastic_weight=True,
               knn_weight=[0.5, 0.25], knn_plateau=100)
    clobber = _tools_run(z_clob, seed, search_radius=250, detour_ratio=1.0,
                         decay=False, num_cores=1)
    fresh_plain = _tools_run(
        _make_zonal(mz), seed, search_radius=250, detour_ratio=1.0,
        decay=False, num_cores=1)

    tools = _arm_tools()
    z_bad = _make_zonal(mz)
    _, err_bad, _, _ = _capture_stdout_stderr(
        lambda: tools.betweenness(z_bad, search_radius=250,
                                  save_elastic_weight_as="ew"))

    return {
        "knn_list_form": list_form,
        "knn_str_form": str_form,
        "str_form_equals_list_form": (
            str_form["betweenness_by_edge"] == list_form["betweenness_by_edge"]
            and str_form["layers"] == list_form["layers"]),
        "knn_plateau_100": plateau,
        "plateau_engages": plateau["betweenness_by_edge"]
        != list_form["betweenness_by_edge"],
        "clobber_after_elastic": clobber,
        "clobber_resets_knn_weight_to_none": (
            clobber["network_knn_weight"] == "None"),
        "clobber_equals_fresh_plain": (
            clobber["betweenness_by_edge"]
            == fresh_plain["betweenness_by_edge"]),
        "save_elastic_without_elastic_error": err_bad,
    }


def scenario_flow_exposure_diagnostics(mz, seed):
    """Path exposure (hazzard attribute through parent_street_id), the
    keep_diagnostics join (time columns land in the layer — presence
    pinned, values excluded), the typo columns, the direct exposure
    origin frame with all four hazzard stats (sabotage hook
    exposure_column_drop), and the keep_diagnostics-without-save
    TypeError (None + str)."""
    btd = _arm_betweenness()

    z1 = _make_zonal(mz, hazzard=True)
    diag = _tools_run(
        z1, seed, search_radius=250, detour_ratio=1.0, decay=True,
        beta=0.003, decay_method="exponent", num_cores=1,
        closest_destination=False, path_exposure_attribute="hazzard",
        save_path_exposure_as="expo", save_betweenness_as="bt",
        save_reach_as="reach", keep_diagnostics=True)
    # layer digest: exclude the joined wall-clock diagnostic columns
    renamed_time = [f"bt_{c}" for c in TIME_COLUMNS]
    diag_layers = diag["layers"]
    present = [c for c in renamed_time
               if c in diag_layers["origins"]["columns"]]
    for name in ("origins", "destinations"):
        cols = diag_layers[name]["columns"]
        drop = [i for i, c in enumerate(cols) if c in renamed_time]
        diag_layers[name]["columns"] = [
            c for c in cols if c not in renamed_time]
        diag_layers[name]["dtypes"] = [
            t for i, t in enumerate(diag_layers[name]["dtypes"])
            if i not in drop]
        diag_layers[name]["values"] = {
            c: v for c, v in diag_layers[name]["values"].items()
            if c not in renamed_time}
    diag["layers"] = diag_layers
    diag["renamed_time_columns_present"] = present

    # PINNED quirk: the origin-layer joins (expo AND the diagnostics
    # join) live inside the `save_reach/gravity/elastic requested` block
    # — with NO origin save requested, save_path_exposure_as is
    # silently ignored (no expo column, no error)
    z1b = _make_zonal(mz, hazzard=True)
    expo_only = _tools_run(
        z1b, seed, search_radius=250, detour_ratio=1.0, decay=True,
        beta=0.003, num_cores=1, closest_destination=False,
        path_exposure_attribute="hazzard", save_path_exposure_as="expo",
        save_betweenness_as="bt")
    expo_only["expo_joined"] = "expo" in expo_only[
        "layers"]["origins"]["columns"]

    # direct exposure with the attribute: the four hazzard origin stats
    z2 = _make_zonal(mz, hazzard=True)
    _ = _seeded_origin_order(z2, seed)
    res2, err2, out2, _ = _capture_stdout_stderr(
        lambda: btd.paralell_betweenness_exposure(
            z2, search_radius=250, detour_ratio=1.0, decay=True,
            beta=0.003, decay_method="exponent", num_cores=1,
            closest_destination=False, path_exposure_attribute="hazzard"))
    direct = {
        "origin_gdf": _origin_gdf_digest(res2["origin_gdf"]),
        "betweenness_by_edge": _betweenness_by_edge(res2["edge_gdf"]),
        "error": err2, "progress_seen": "Time spent:" in out2}

    # exposure with closest_destination=True: the hazzard stats ARE
    # accumulated and written (they precede the unbound
    # eligible_destinations_shortest_distance line that kills the rest
    # of the stats block)
    z3 = _make_zonal(mz, hazzard=True)
    _ = _seeded_origin_order(z3, seed)
    res3, err3, _, _ = _capture_stdout_stderr(
        lambda: btd.paralell_betweenness_exposure(
            z3, search_radius=250, detour_ratio=1.0, decay=False,
            beta=0.003, num_cores=1, closest_destination=True,
            path_exposure_attribute="hazzard"))
    direct_closest = {
        "origin_gdf": _origin_gdf_digest(res3["origin_gdf"]),
        "error": err3}

    tools = _arm_tools()
    z4 = _make_zonal(mz, hazzard=True)
    _, err_td, _, _ = _capture_stdout_stderr(
        lambda: tools.betweenness(
            z4, search_radius=250, decay=False, num_cores=1,
            closest_destination=False, keep_diagnostics=True,
            save_reach_as="reach"))
    typeerror_probe = {
        "error": err_td,
        "edges_got_betweenness_column":
            "betweenness" in z4.network.edges.columns,
        "origins_layer_untouched":
            "reach" not in z4["origins"].gdf.columns
            and "bt_source_layer" not in z4["origins"].gdf.columns,
    }

    return {
        "tools_diagnostics": diag,
        "expo_only_silently_ignored": expo_only,
        "direct_exposure": direct,
        "direct_exposure_closest": direct_closest,
        "keep_diagnostics_without_save": typeerror_probe,
    }


def scenario_flow_repeats_mutations(mz, seed):
    """Dossier combos: repeated calls on the same Zonal (the save join
    drops+rejoins — idempotent), changed network weights between calls,
    the clear/reinsert round-trip (equals a fresh build), and mixed
    origin source layers (the save join only handles the FIRST
    processed origin's layer — shuffle-dependent, seeded)."""
    # repeated identical calls on one zonal
    z1 = _make_zonal(mz)
    first = _tools_run(z1, seed, search_radius=250, detour_ratio=1.0,
                       decay=False, num_cores=1, closest_destination=False,
                       save_betweenness_as="bt", save_reach_as="reach")
    streets_after_first = gdf_digest(z1["streets"].gdf)
    second = _tools_run(z1, seed, search_radius=250, detour_ratio=1.0,
                        decay=False, num_cores=1, closest_destination=False,
                        save_betweenness_as="bt", save_reach_as="reach")
    streets_after_second = gdf_digest(z1["streets"].gdf)

    # changed weights between calls (network-level node weights are the
    # effective input the engine reads)
    z2 = _make_zonal(mz)
    before = _tools_run(z2, seed, search_radius=250, detour_ratio=1.0,
                        decay=False, num_cores=1, closest_destination=False)
    d_idx = _destination_idxs(z2)[0]
    z2.network.nodes.at[d_idx, "weight"] = 9.9
    after = _tools_run(z2, seed, search_radius=250, detour_ratio=1.0,
                       decay=False, num_cores=1, closest_destination=False)

    # pure clear/reinsert round-trip: equals a fresh build
    z3 = _make_zonal(mz)
    pre = _tools_run(z3, seed, search_radius=250, detour_ratio=1.0,
                     decay=False, num_cores=1, closest_destination=False)
    z3.clear_nodes()
    z3.insert_node("origins", label="origin", weight_attribute="weight")
    z3.insert_node("destinations", label="destination",
                   weight_attribute="weight")
    z3.create_graph()
    post = _tools_run(z3, seed, search_radius=250, detour_ratio=1.0,
                      decay=False, num_cores=1, closest_destination=False)

    # mixed origin source layers: with two origin layers the tools save
    # join resolves origin_layer = origin_gdf.iloc[0]['source_layer'] —
    # whichever origin processed FIRST under the seeded shuffle decides
    # which layer receives reach/gravity; the other layer gets nothing.
    z4 = _make_zonal(mz, second_origins=True)
    order4 = _seeded_origin_order(z4, seed)
    mixed = _tools_run(z4, seed, search_radius=250, detour_ratio=1.0,
                       decay=False, num_cores=1, closest_destination=False,
                       save_reach_as="reach", save_gravity_as="grav")
    layers4 = mixed["layers"]
    got_join = [name for name in ("origins", "origins2")
                if "reach" in layers4[name]["columns"]]
    mixed["layer_receiving_join"] = got_join
    # the first PROCESSED origin's source_layer decides the join target
    mixed["first_processed_source_layer"] = str(
        z4.network.nodes.loc[order4[0], "source_layer"])

    return {
        "repeat_first": first,
        "repeat_second": second,
        "streets_after_first": streets_after_first,
        "streets_after_second": streets_after_second,
        "repeat_idempotent": streets_after_first == streets_after_second,
        "weight_before": before,
        "weight_after": after,
        "weight_change_engages": before["betweenness_by_edge"]
        != after["betweenness_by_edge"],
        "clear_reinsert_pre": pre,
        "clear_reinsert_post": post,
        "clear_reinsert_reproduces": pre["betweenness_by_edge"]
        == post["betweenness_by_edge"],
        "mixed_source_layers": mixed,
    }


def scenario_flow_edge_cases(mz, seed):
    """Edge cases: a search radius that leaves every origin without
    reachable destinations (empty d_idxs continue; the tools save path
    then KeyErrors on 'reach'), a zero-weight origin skipped before any
    work, all-zero destination gravities (the continue that skips
    task_done), zero origins (pd.concat of nothing), and a positive
    turn_penalty engagement."""
    tools = _arm_tools()
    btd = _arm_betweenness()

    # tiny radius: no origin reaches any destination
    z1 = _make_zonal(mz)
    tiny_tools = _tools_run(z1, seed, search_radius=10, detour_ratio=1.0,
                            decay=False, num_cores=1,
                            closest_destination=False,
                            save_reach_as="reach")
    z1b = _make_zonal(mz)
    _ = _seeded_origin_order(z1b, seed)
    res_tiny, err_tiny, out_tiny, _ = _capture_stdout_stderr(
        lambda: btd.paralell_betweenness_exposure(
            z1b, search_radius=10, detour_ratio=1.0, decay=False,
            beta=0.003, num_cores=1, closest_destination=False))
    tiny = {
        "tools_save_reach_error": tiny_tools["error"],
        "tools_edges_all_zero": all(v == _f(0.0) for v in
                                    tiny_tools["betweenness_by_edge"]
                                    .values()),
        "direct_origin_gdf": _origin_gdf_digest(res_tiny["origin_gdf"]),
        "direct_betweenness_all_zero": all(
            v == _f(0.0) for v in _betweenness_by_edge(
                res_tiny["edge_gdf"]).values()),
        "direct_error": err_tiny,
    }

    # zero-weight origin: skipped before path work, row still present
    z2 = _make_zonal(mz, origin_weights=[0.0, 2.0, 1.0])
    zero_w = _tools_run(z2, seed, search_radius=250, detour_ratio=1.0,
                        decay=False, num_cores=1, closest_destination=False,
                        save_reach_as="reach")

    # all destination gravities zero (Huff): the continue that skips
    # task_done — stats never written, betweenness stays zero
    z3 = _make_zonal(mz, zero_dest_weights=True)
    _ = _seeded_origin_order(z3, seed)
    res_zg, err_zg, _, _ = _capture_stdout_stderr(
        lambda: btd.paralell_betweenness_exposure(
            z3, search_radius=250, detour_ratio=1.0, decay=False,
            beta=0.003, num_cores=1, closest_destination=False))
    z3b = _make_zonal(mz, zero_dest_weights=True)
    zero_grav_tools = _tools_run(z3b, seed, search_radius=250,
                                 detour_ratio=1.0, decay=False, num_cores=1,
                                 closest_destination=False,
                                 save_reach_as="reach")
    zero_gravity = {
        "direct_origin_gdf": _origin_gdf_digest(res_zg["origin_gdf"]),
        "direct_error": err_zg,
        "direct_betweenness_all_zero": all(
            v == _f(0.0) for v in _betweenness_by_edge(
                res_zg["edge_gdf"]).values()),
        "tools_save_reach_error": zero_grav_tools["error"],
    }

    # zero origins: the queue drains to the sentinel immediately and the
    # parent pd.concat of an empty list raises
    z4 = mz.Zonal()
    z4.load_layer("streets", _streets_gdf())
    z4.load_layer("destinations", _destinations_gdf())
    z4.create_street_network("streets", weight_attribute="length",
                             turn_threshold_degree=45,
                             turn_penalty_amount=30)
    z4.insert_node("destinations", label="destination",
                   weight_attribute="weight")
    z4.create_graph()
    res_no_o, err_no_o, _, _ = _capture_stdout_stderr(
        lambda: btd.paralell_betweenness_exposure(
            z4, search_radius=250, detour_ratio=1.0, decay=False,
            beta=0.003, num_cores=1, closest_destination=False))
    tools_no_o = _tools_run(z4, seed, search_radius=250, detour_ratio=1.0,
                            decay=False, num_cores=1,
                            closest_destination=False)
    zero_origins = {
        "direct_error": err_no_o,
        "tools_error": tools_no_o["error"],
    }

    # positive turn penalty engagement through tools.betweenness
    z5 = _make_zonal(mz)
    turn = _tools_run(z5, seed, search_radius=250, detour_ratio=1.0,
                      decay=False, num_cores=1, closest_destination=False,
                      turn_penalty=True)
    no_turn = _tools_run(_make_zonal(mz), seed, search_radius=250,
                         detour_ratio=1.0, decay=False, num_cores=1,
                         closest_destination=False)

    return {
        "tiny_radius": tiny,
        "zero_weight_origin": zero_w,
        "zero_destination_gravity": zero_gravity,
        "zero_origins": zero_origins,
        "turn_penalty": turn,
        "turn_engages": turn["betweenness_by_edge"]
        != no_turn["betweenness_by_edge"],
    }


def scenario_flow_nc2_exposure(mz, seed):
    """Exposure engine at num_cores=2: origin processing is distributed
    through a shared mp queue, so per-edge float-addition order across
    the two batch trackers is NOT stable (documented upstream fact —
    the facade header records it).  This scenario therefore digests the
    per-origin LAYER state bitwise (partition-independent: stats are
    written per origin) and records the edge betweenness only
    structurally.  num_cores=1 layer state digested for the equality
    flag."""
    nc1 = _tools_run(_make_zonal(mz), seed, search_radius=250,
                     detour_ratio=1.0, decay=False, num_cores=1,
                     closest_destination=False, save_reach_as="reach")
    nc2 = _tools_run(_make_zonal(mz), seed, search_radius=250,
                     detour_ratio=1.0, decay=False, num_cores=2,
                     closest_destination=False, save_reach_as="reach")

    def edge_structure(run_digest):
        # values unpacked from this run's own recorded hex; never
        # compared across runs (see module docstring)
        vals = [struct.unpack(">d", bytes.fromhex(v))[0]
                for v in run_digest["betweenness_by_edge"].values()]
        return {
            "edges_count": len(vals),
            "all_finite": all(math.isfinite(v) for v in vals),
            "nonzero_edges": sum(1 for v in vals if v != 0.0),
        }

    return {
        "nc1_layers": nc1["layers"],
        "nc2_layers": nc2["layers"],
        "layer_state_equal_across_cores": nc1["layers"] == nc2["layers"],
        "nc1_progress_seen": nc1["progress_seen"],
        "nc2_progress_seen": nc2["progress_seen"],
        "nc1_edges_structural": edge_structure(nc1),
        "nc2_edges_structural": edge_structure(nc2),
        "nondeterminism_note":
            "exposure engine num_cores>1: origins are distributed through "
            "a shared mp queue, so the partition — and with it the "
            "per-edge addition order — is load-dependent BY CONSTRUCTION "
            "and carries no bitwise-stability contract upstream; "
            "empirically stable across three retained runs on this "
            "fixture (evidence logs/nc2_stability_probe/), so edges are "
            "digested structurally only as the conservative choice; "
            "layers are per-origin and partition-independent (bitwise)",
    }


def scenario_flow_validation(mz, seed):
    """The full tools.betweenness argument-validation matrix: every
    probe records the exact error (no silent acceptance), including the
    duplicated save_gravity_as check (same message) and the
    keep_diagnostics-without-save TypeError that fires only AFTER a full
    engine run."""
    tools = _arm_tools()
    z = _make_zonal(mz)

    probes = {
        "search_radius_str": dict(search_radius="250"),
        "search_radius_negative": dict(search_radius=-1),
        "detour_ratio_str": dict(detour_ratio="1"),
        "detour_ratio_below_1": dict(detour_ratio=0.5),
        "decay_str": dict(decay="False"),
        "decay_method_bad_str": dict(decay=True, decay_method="linear"),
        "decay_method_not_str": dict(decay=True, decay_method=3),
        "beta_str_with_decay": dict(decay=True, beta="0.003"),
        "num_cores_str": dict(num_cores="1"),
        "num_cores_zero": dict(num_cores=0),
        "closest_destination_str": dict(closest_destination="True"),
        "elastic_weight_str": dict(elastic_weight="True"),
        "elastic_without_knn": dict(elastic_weight=True),
        "knn_weight_bad_type": dict(elastic_weight=True, knn_weight=5),
        "knn_plateau_str": dict(elastic_weight=True, knn_weight=[0.5, 0.25],
                                knn_plateau="100"),
        "knn_plateau_negative": dict(elastic_weight=True,
                                     knn_weight=[0.5, 0.25],
                                     knn_plateau=-1),
        "turn_penalty_str": dict(turn_penalty="True"),
        "save_betweenness_as_int": dict(save_betweenness_as=5),
        "save_reach_as_int": dict(save_reach_as=5),
        "save_gravity_as_int": dict(save_gravity_as=5),
        "save_elastic_as_int": dict(save_elastic_weight_as=5),
        "save_elastic_without_elastic": dict(save_elastic_weight_as="ew"),
        "keep_diagnostics_str": dict(keep_diagnostics="True"),
        "path_exposure_attr_int": dict(path_exposure_attribute=5),
        "path_exposure_attr_missing": dict(path_exposure_attribute="nope"),
        "save_path_exposure_as_int": dict(save_path_exposure_as=5),
        "save_path_exposure_without_attr": dict(save_path_exposure_as="px"),
    }
    errors = {}
    for name, kwargs in probes.items():
        # validation failures must leave the zonal untouched: give every
        # probe the same pristine zonal
        probe_z = _make_zonal(mz)
        kwargs = dict(kwargs, search_radius=kwargs.get("search_radius", 250),
                      detour_ratio=kwargs.get("detour_ratio", 1.0),
                      decay=kwargs.get("decay", False),
                      num_cores=kwargs.get("num_cores", 1))
        _, err, _, _ = _capture_stdout_stderr(
            lambda kwargs=kwargs, probe_z=probe_z:
                tools.betweenness(probe_z, **kwargs))
        errors[name] = err if err is not None else "NO_ERROR"

    # the keep_diagnostics TypeError fires only after a full engine run
    # (it happens in the save block, post paralell_betweenness_exposure)
    z_td = _make_zonal(mz)
    _, err_td, _, _ = _capture_stdout_stderr(
        lambda: tools.betweenness(z_td, search_radius=250, decay=False,
                                  num_cores=1, closest_destination=False,
                                  keep_diagnostics=True,
                                  save_reach_as="reach"))
    after_typeerror = {
        "error": err_td,
        "edges_got_betweenness_column":
            "betweenness" in z_td.network.edges.columns,
        "origins_layer_untouched":
            "reach" not in z_td["origins"].gdf.columns,
    }

    return {
        "errors": errors,
        "zonal_untouched_after_probes": _node_state(z) == _node_state(
            _make_zonal(mz)),
        "keep_diagnostics_typeerror_after_run": after_typeerror,
    }


SCENARIOS = {
    "flow_closest_huff": scenario_flow_closest_huff,
    "flow_decay_matrix": scenario_flow_decay_matrix,
    "flow_lowlevel_engine": scenario_flow_lowlevel_engine,
    "flow_elastic_knn": scenario_flow_elastic_knn,
    "flow_exposure_diagnostics": scenario_flow_exposure_diagnostics,
    "flow_repeats_mutations": scenario_flow_repeats_mutations,
    "flow_edge_cases": scenario_flow_edge_cases,
    "flow_nc2_exposure": scenario_flow_nc2_exposure,
    "flow_validation": scenario_flow_validation,
}

# Test modules import the scenario list from THIS uniquely-named module,
# never via `from conftest import ...`: bare `conftest` collides across
# non-package test dirs in sys.modules (first suite collected wins), so
# co-collected suites would silently read each other's lists or fail
SCENARIO_NAMES = list(SCENARIOS)


def import_arm(arm):
    if arm == "reference":
        sys.path.insert(0, str(REPO / ".refs" / "madina_ref" / "src"))
        import madina.una.betweenness as btd        # noqa: PLC0415
        import madina.una.tools as tools_mod        # noqa: PLC0415
        import madina.zonal as mz                   # noqa: PLC0415
        _ARM_MARKER["betweenness"] = btd
        _ARM_MARKER["tools"] = tools_mod
        _ARM_MARKER["zonal"] = mz
        return mz
    if arm == "facade":
        sys.path.insert(0, str(REPO / "src"))
        from urban_network_analysis.compat.madina import una    # noqa: PLC0415
        import urban_network_analysis.compat.madina.una.tools as una_tools  # noqa: PLC0415
        from urban_network_analysis.compat.madina import zonal as mz  # noqa: PLC0415
        _ARM_MARKER["betweenness"] = una.betweenness
        _ARM_MARKER["tools"] = una_tools
        _ARM_MARKER["zonal"] = mz
        return mz
    raise ValueError(f"unknown arm {arm}")


# ----------------------------------------------------------------------
# digest-level sabotage mutators (parameter-level ones are consumed
# inside the scenario bodies; these reshape the already-recorded state
# and must be selected by the comparator)
# ----------------------------------------------------------------------

def _sabotage_noop(state):
    return state


def _drop_gdf_row(digest, pos=0):
    d = {k: (v if not isinstance(v, list) else list(v))
         for k, v in digest.items()}
    d["index"] = d["index"][:pos] + d["index"][pos + 1:]
    d["values"] = {c: v[:pos] + v[pos + 1:]
                   for c, v in d["values"].items()}
    return d


def _drop_gdf_column(digest, col):
    d = {k: (v if not isinstance(v, list) else list(v))
         for k, v in digest.items()}
    pos = d["columns"].index(col)
    d["columns"] = [c for c in d["columns"] if c != col]
    d["dtypes"] = [t for i, t in enumerate(d["dtypes"]) if i != pos]
    d["values"] = {c: v for c, v in d["values"].items() if c != col}
    return d


def _sabotage_split_edge_drop(state):
    """Dossier mutant: drop one half of a SPLIT street (the split_probe
    edge-table row found first whose parent_street_id is shared — row 2,
    the first of the two rows carrying parent 2) from the recorded edge
    digests.  A comparator that only checks the streets layer or
    aggregate sums would miss this — the row must be selected."""
    sub = state["split_probe"]
    parents = sub["edges"]["values"]["parent_street_id"]
    row = sub["first_split_row"]
    parent = parents[row]
    shared = parents.count(parent)
    if shared < 2:
        raise RuntimeError(
            f"sabotage split_edge_drop: edge row {row} is not a split "
            f"edge (parent_street_id {parent} appears {shared}x)")
    sub["edges"] = _drop_gdf_row(sub["edges"], row)
    sub["parents"] = parents[:row] + parents[row + 1:]
    sub["betweenness_by_edge"] = dict(
        (k, v) for i, (k, v) in enumerate(sub["betweenness_by_edge"].items())
        if i != row)
    sub["split_edge_dropped"] = {
        "row": row, "parent_street_id": parent, "shared_by": shared}
    return state


def _sabotage_exposure_column_drop(state):
    """Dossier mutant: omit an exposure column (the typo-named
    decayed_mean_hazzad) from the recorded origin-frame digest."""
    sub = state["direct_exposure"]["origin_gdf"]
    if "decayed_mean_hazzad" not in sub["columns"]:
        raise RuntimeError(
            "sabotage exposure_column_drop: decayed_mean_hazzad missing "
            "from the recorded origin frame")
    sub = _drop_gdf_column(sub, "decayed_mean_hazzad")
    sub["stats_columns_present"] = [
        c for c in sub["stats_columns_present"]
        if c != "decayed_mean_hazzad"]
    state["direct_exposure"]["origin_gdf"] = sub
    return state


SABOTAGES = {
    # parameter-level (dossier): decay_method_swap lives in
    # flow_decay_matrix, detour_swap in flow_lowlevel_engine
    "decay_method_swap": ("flow_decay_matrix", _sabotage_noop),
    "detour_swap": ("flow_lowlevel_engine", _sabotage_noop),
    # digest-level (dossier): drop a split edge / omit an exposure column
    "split_edge_drop": ("flow_closest_huff", _sabotage_split_edge_drop),
    "exposure_column_drop": ("flow_exposure_diagnostics",
                             _sabotage_exposure_column_drop),
}


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
        if args.sabotage:
            _SABOTAGE["name"] = args.sabotage
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
