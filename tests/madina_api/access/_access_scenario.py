"""MADINA_ACCESS parity scenario runner: exercise the accessibility
surface (validate_zonal_ready / accessibility / service_area + the
access-engine trio get_origin_properties / one_access / parallel_access)
through either the compat facade or the pinned upstream reference, then
dump a BITWISE-strict state digest.

Same discipline as tests/madina_api/paths (MADINA_PATHS): both arms run
under the SAME interpreter (the dependency-bridged reference venv) with
identical fixtures, so any digest difference is attributable to the code
under test.  Floats are compared as IEEE-754 bit patterns, geometries as
WKB hashes, order-sensitive dict content as explicit lists.

RNG control: ``parallel_access`` processes origins in an UNSEEDED
``sample(frac=1)`` order (pinned upstream behavior — NOT removed).  Every
scenario seeds the global numpy RNG immediately before each call and
records the resulting permutation, so tie resolution (e.g. closest
facility) is deterministic and the shuffle is exercised, not skipped.

Presentation randomness: ``one_access``/``parallel_access`` print
time-based progress ("Time spent: Ns ...") — digests record only the
DETERMINISTIC progress-line count and structured state, never raw
timings or traceback text (file paths differ between arms).

Usage:
    python _access_scenario.py --arm {facade,reference} --scenario NAME
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


# ----------------------------------------------------------------------
# bitwise digest helpers (same discipline as the paths/zonal suites)
# ----------------------------------------------------------------------

def _f(v):
    """float -> IEEE-754 bit-pattern hex (bitwise-strict, NaN-safe)."""
    return struct.pack(">d", float(v)).hex()


def _wkb_hash(geom):
    return hashlib.sha256(shapely.to_wkb(geom, byte_order=1)).hexdigest()[:24]


def _geomcell_digest(geom):
    """One geometry cell: WKB hash, or component digests for collections."""
    if geom.geom_type == "GeometryCollection":
        return {"collection": [ _wkb_hash(g) for g in geom.geoms ]}
    if geom.geom_type == "MultiPolygon":
        return {"collection": [ _wkb_hash(g) for g in geom.geoms ]}
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


def gdf_digest(gdf):
    """Bitwise-strict GeoDataFrame digest; the geometry column is hashed
    via WKB (collections component-by-component), everything else by
    dtype-classed exact values."""
    d = {
        "columns": list(gdf.columns),
        "dtypes": [str(t) for t in gdf.dtypes],
        "index": [_jsonable(v) for v in gdf.index],
        "index_name": repr(gdf.index.name),
        "crs": str(gdf.crs),
        "values": {c: _series(gdf[c]) for c in gdf.columns
                   if c != gdf._geometry_column_name},
    }
    gcol = gdf._geometry_column_name
    if gcol is not None and gcol in gdf.columns:
        d["values"][gcol] = [_geomcell_digest(g) for g in gdf[gcol]]
    return d


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
# fixture: square + diagonal + dead-end stub + one stub whose far corner
# is the lexicographic maximum point of the fixture (keeps the pinned
# node_builder truncation defect from biting the CLEAN fixture), plus an
# optional isolated second component with its own destination.
#
#   F --------- A ======= o1/o2 == B --- d1 --- C --- d? --- G
#               | \                     |                (200,100)
#               |   \.... diagonal .....\
#               |                        |
#               D ---------------------- +
#               |
#               E (stub, d2 at its end)     ISO: iso1 --- iso2 (d3)
#
#   streets: F-A, A-B, B-C, C-D, D-A, A-C (diagonal), D-E, C-G,
#            (+ ISO edge when iso=True).
#   origins (source ids 0,1,2): o1 (-50,0) mid F-A, o2 (50,0) mid A-B,
#   o3 (100,50) mid B-C; weights [1, 2, 1].
#   destinations: d1 (50,100) mid C-D weight 2, d2 (0,150) = E weight 3,
#   (+ d3 (550,500) on the isolated component, weight 1, when iso=True;
#   optionally its layer weight is NaN to exercise the fillna(0) path).
#
#   CLEAN fixture: all street-edge endpoint labels resolve correctly
#   (the lexicographic maximum (600,500) occurs exactly once).
#   BROKEN fixture (clean=False, no C-G edge): C=(100,100) is the
#   lexicographic maximum and occurs three times; the pinned upstream
#   vectorized_node_edge_builder truncation leaves B-C's `end` labeled 0
#   (F) and drops the A-C diagonal as a false "redundant_edge" —
#   scenario node_builder_truncation pins this bitwise on both arms.
# ----------------------------------------------------------------------

def _streets_gdf(clean=True, iso=False):
    rows = [
        ("F-A", 100.0, LineString([(-100, 0), (0, 0)])),
        ("A-B", 100.0, LineString([(0, 0), (100, 0)])),
        ("B-C", 100.0, LineString([(100, 0), (100, 100)])),
        ("C-D", 100.0, LineString([(100, 100), (0, 100)])),
        ("D-A", 100.0, LineString([(0, 100), (0, 0)])),
        ("A-C", float(np.hypot(100, 100)),
         LineString([(0, 0), (100, 100)])),
        ("D-E", 50.0, LineString([(0, 100), (0, 150)])),
    ]
    if clean:
        rows.append(("C-G", 100.0, LineString([(100, 100), (200, 100)])))
    if iso:
        rows.append(("ISO", 100.0, LineString([(500, 500), (600, 500)])))
    return gpd.GeoDataFrame(
        {"id": [r[0] for r in rows], "length": [r[1] for r in rows]},
        geometry=[r[2] for r in rows],
        crs="EPSG:3857")


def _origins_gdf():
    # PINNED: load_layer resets the layer index, so source_id is the
    # positional 0..n-1 regardless of any fixture index.
    return gpd.GeoDataFrame(
        {"weight": [1.0, 2.0, 1.0]},
        geometry=[Point((-50, 0)), Point((50, 0)), Point((100, 50))],
        crs="EPSG:3857")


def _destinations_gdf(iso=False, nan_weight=False):
    weights = [2.0, 3.0]
    if nan_weight:
        weights = [2.0, float("nan")]
    ids, geoms = [0, 1], [Point((50, 100)), Point((0, 150))]
    if iso:
        ids.append(2)
        weights.append(float("nan") if nan_weight else 1.0)
        geoms.append(Point((550, 500)))
    return gpd.GeoDataFrame(
        {"weight": weights}, geometry=geoms, crs="EPSG:3857")


def _make_zonal(mz, *, clean=True, iso=False, nan_weight=False,
                turn_threshold=45, turn_penalty_amount=30):
    z = mz.Zonal()
    z.load_layer("streets", _streets_gdf(clean=clean, iso=iso))
    z.load_layer("origins", _origins_gdf())
    z.load_layer("destinations",
                 _destinations_gdf(iso=iso, nan_weight=nan_weight))
    z.create_street_network(
        "streets", weight_attribute="length",
        turn_threshold_degree=turn_threshold,
        turn_penalty_amount=turn_penalty_amount)
    z.insert_node("origins", label="origin")
    z.insert_node("destinations", label="destination",
                  weight_attribute="weight")
    z.create_graph()
    return z


def _origin_idx(z, source_id):
    n = z.network.nodes
    sel = n[(n["type"] == "origin") & (n["source_id"] == source_id)]
    return int(sel.index[0])


def _seeded_origin_order(z, seed):
    """Replicate parallel_access's UNSEEDED sample(frac=1) permutation
    under a seeded global RNG, then reseed so the actual call consumes
    the identical permutation.  Recorded in every digest that depends on
    processing order (tie resolution)."""
    n = z.network.nodes
    origin_gdf = n[n["type"] == "origin"]
    np.random.seed(seed)
    order = origin_gdf.sample(frac=1).index.astype("int").tolist()
    np.random.seed(seed)
    return [int(i) for i in order]


def _progress_line_count(text):
    # deterministic content of the time-based progress prints: their
    # COUNT equals the number of processed-and-reported origins
    return sum(1 for line in text.splitlines() if "origins (" in line)


def _node_state(z):
    return gdf_digest(z.network.nodes)


def _layer_states(z):
    return {name: gdf_digest(z[name].gdf) for name in ("origins",
                                                       "destinations")}


# ----------------------------------------------------------------------
# scenarios
# ----------------------------------------------------------------------

def scenario_reach_gravity(mz, seed):
    """reach (alpha=1, weightless) and gravity (alpha=2, beta,
    destination_weight) on fresh zonals; the layer save-join surface;
    NaN destination weights hitting the fillna(0) path; the unseeded
    shuffle recorded per call."""
    tools = _arm_tools()
    alpha_1, alpha_2 = 1, 2
    if _SABOTAGE["name"] == "alpha_swap":
        # parameter-level mutant (dossier): the recorded digests are
        # then those of the WRONG alphas; comparator must select
        alpha_1, alpha_2 = 2, 1

    z1 = _make_zonal(mz)
    order1 = _seeded_origin_order(z1, seed)
    res1, err1, out1, _ = _capture_stdout_stderr(
        lambda: tools.accessibility(z1, search_radius=250, alpha=alpha_1))

    z2 = _make_zonal(mz)
    order2 = _seeded_origin_order(z2, seed)
    res2, err2, out2, _ = _capture_stdout_stderr(
        lambda: tools.accessibility(z2, search_radius=250, alpha=alpha_2,
                                    beta=0.0, destination_weight="weight",
                                    save_reach_as="reach",
                                    save_gravity_as="grav"))

    z3 = _make_zonal(mz, nan_weight=True)
    order3 = _seeded_origin_order(z3, seed)
    res3, err3, out3, _ = _capture_stdout_stderr(
        lambda: tools.accessibility(z3, search_radius=250, alpha=1,
                                    destination_weight="weight"))

    # review M4: positive turn_penalty=True exercise for accessibility
    # itself (the service_areas scenario covers only the scope path)
    z4 = _make_zonal(mz)
    order4 = _seeded_origin_order(z4, seed)
    res4, err4, out4, _ = _capture_stdout_stderr(
        lambda: tools.accessibility(z4, search_radius=250, alpha=1,
                                    turn_penalty=True))
    tp_node = _node_state(z4)

    return {
        "reach": {"node": _node_state(z1), "error": err1,
                  "progress_lines": _progress_line_count(out1),
                  "shuffled_origin_order": order1},
        "gravity": {"node": _node_state(z2), "layers": _layer_states(z2),
                    "error": err2, "progress_lines": _progress_line_count(out2),
                    "shuffled_origin_order": order2},
        "nan_destination_weight": {"node": _node_state(z3), "error": err3,
                                   "shuffled_origin_order": order3},
        "reach_turn_penalty": {"node": tp_node, "error": err4,
                               "progress_lines": _progress_line_count(out4),
                               "shuffled_origin_order": order4,
                               "turn_engages": tp_node != _node_state(z1)},
    }


def scenario_knn(mz, seed):
    """KNN access: list and string knn_weights forms must agree bitwise
    (the str parse path), and a sub-radius knn_plateau must engage the
    exponential decay (digest differs from the default plateau)."""
    tools = _arm_tools()

    z1 = _make_zonal(mz)
    order1 = _seeded_origin_order(z1, seed)
    tools.accessibility(z1, search_radius=250, beta=0.002,
                        save_knn_access_as="knn",
                        knn_weights=[0.5, 0.25])

    z2 = _make_zonal(mz)
    order2 = _seeded_origin_order(z2, seed)
    tools.accessibility(z2, search_radius=250, beta=0.002,
                        save_knn_access_as="knn",
                        knn_weights="[0.5, 0.25]")

    z3 = _make_zonal(mz)
    order3 = _seeded_origin_order(z3, seed)
    tools.accessibility(z3, search_radius=250, beta=0.002,
                        save_knn_access_as="knn",
                        knn_weights=[0.5, 0.25], knn_plateau=100)

    list_form = _node_state(z1)
    str_form = _node_state(z2)
    return {
        "knn_list_form": {"node": list_form,
                          "layers": _layer_states(z1),
                          "shuffled_origin_order": order1},
        "knn_str_form": {"node": str_form,
                         "shuffled_origin_order": order2},
        "str_form_equals_list_form": list_form == str_form,
        "knn_plateau_100": {"node": _node_state(z3),
                            "shuffled_origin_order": order3},
        "plateau_engages_decay": _node_state(z3) != list_form,
    }


def scenario_closest_facility(mz, seed):
    """closest_facility on the iso fixture: d1's unique closest origin
    is o3 (100 m) regardless of processing order; d2 is a three-way
    200 m tie resolved by the FIRST processed origin (the seeded
    shuffle); the isolated d3 is unreachable -> cf/cfd stay NaN and the
    destination-layer join does NOT fill them (upstream comment: NA
    must survive).  Origin reach/gravity are overwritten by the
    joint-destination recomputation (origins without joint destinations
    get 0)."""
    tools = _arm_tools()

    z = _make_zonal(mz, iso=True)
    order = _seeded_origin_order(z, seed)
    if _SABOTAGE["name"] == "skip_shuffle":
        # behavioral mutant (process-local monkeypatch, no file edits):
        # sample(frac=1) returns the frame unshuffled -> origins are
        # processed in index order -> the d2 tie winner flips unless the
        # seeded permutation happens to equal index order
        def _identity_sample(self, *a, **k):
            return self
        gpd.GeoDataFrame.sample = _identity_sample
        pd.DataFrame.sample = _identity_sample
        order = sorted(order)
    res, err, out, _ = _capture_stdout_stderr(
        lambda: tools.accessibility(z, search_radius=250,
                                    closest_facility=True,
                                    save_closest_facility_as="cf",
                                    save_closest_facility_distance_as="cfd"))

    n = z.network.nodes
    o_ids = sorted(n[n["type"] == "origin"]["source_id"].astype(int).tolist())
    d_rows = n[n["type"] == "destination"]
    cf_by_source = {int(r.source_id):
                    (None if pd.isna(r.closest_facility)
                     else int(r.closest_facility),
                     _f(r.closest_facility_distance))
                    for r in d_rows.itertuples()}
    return {
        "node": _node_state(z),
        "layers": _layer_states(z),
        "error": err,
        "progress_lines": _progress_line_count(out),
        "shuffled_origin_order": order,
        "origin_source_ids": o_ids,
        "closest_facility_by_destination_source_id": cf_by_source,
    }


def scenario_layer_mutations(mz, seed):
    """Repeated accessibility calls on ONE zonal: the named save-join
    drops and rejoins same-name columns (no duplication), different
    names coexist, and the layer index is int after the joins."""
    tools = _arm_tools()

    z = _make_zonal(mz)
    order1 = _seeded_origin_order(z, seed)
    tools.accessibility(z, search_radius=250, alpha=1,
                        save_reach_as="reach")
    after1 = _layer_states(z)

    order2 = _seeded_origin_order(z, seed)
    tools.accessibility(z, search_radius=250, alpha=2, beta=0.0,
                        destination_weight="weight",
                        save_reach_as="reach", save_gravity_as="grav")
    after2 = _layer_states(z)

    order3 = _seeded_origin_order(z, seed)
    tools.accessibility(z, search_radius=250, alpha=1,
                        save_reach_as="reach2")
    after3 = _layer_states(z)

    return {
        "after_first_save": after1,
        "after_same_name_resave": after2,
        "after_second_name_save": after3,
        "shuffled_origin_orders": [order1, order2, order3],
        "node_columns_final": list(z.network.nodes.columns),
        "origin_layer_index_int": str(after3["origins"]["index"][0]),
    }


def scenario_service_areas(mz, seed):
    """service_area: single int id, all origins (origin_ids=None), a
    list, the turn_penalty variant, the below-radius origin skip, the
    silent int-miss, and the bad-list ValueError.  Returns a 3-tuple
    (destinations, network_edges, scope_gdf) — all three digested."""
    tools = _arm_tools()

    def run(z, *a, **k):
        res, err, _, _ = _capture_stdout_stderr(
            lambda: tools.service_area(z, *a, **k))
        if err is not None:
            return {"error": err}
        destinations, network_edges, scope_gdf = res
        return {"destinations": gdf_digest(destinations),
                "network_edges": gdf_digest(network_edges),
                "scope_gdf": gdf_digest(scope_gdf)}

    z1 = _make_zonal(mz)
    single = run(z1, 250, 0)

    z2 = _make_zonal(mz)
    all_origins = run(z2, 250, None)

    z3 = _make_zonal(mz)
    list_ids = run(z3, 250, [0, 1, 2])

    z4 = _make_zonal(mz)
    turn_variant = run(z4, 250, [0, 1, 2], turn_penalty=True)

    z5 = _make_zonal(mz)
    below_radius = run(z5, 150, 0)  # o1 reaches nothing within 150

    z6 = _make_zonal(mz)
    int_miss = run(z6, 250, 99)  # silent empty result (no error)

    z7 = _make_zonal(mz)
    bad_list = run(z7, 250, [0, 999])

    # MADINA_PATHS review N3 strengthening: the list-origin
    # alternative_paths result, digest-strict (queued from the paths
    # suite to the ACCESS validation surface)
    z8 = _make_zonal(mz)
    alt_list, alt_list_err, _, _ = _capture_stdout_stderr(
        lambda: tools.alternative_paths(z8, origin_id=[0],
                                        search_radius=10000,
                                        detour_ratio=1))

    return {
        "single_int_id": single,
        "all_origins": all_origins,
        "list_ids": list_ids,
        "turn_penalty_variant": turn_variant,
        "below_radius_skip": below_radius,
        "int_miss_silent": int_miss,
        "bad_list_error": bad_list,
        "alt_paths_list_origin": {
            "gdf": None if alt_list is None else gdf_digest(alt_list),
            "error": alt_list_err,
        },
        "turn_engages": turn_variant != list_ids,
    }


def scenario_validation(mz, seed):
    """Argument-validation matrix: exact error types and messages pinned
    for accessibility/service_area/validate_zonal_ready, including the
    asymmetric pinned quirks (accessibility reads origin/destination
    iloc[0] BEFORE validate_zonal_ready; service_area validates first;
    an empty Zonal fails earlier still on network=None)."""
    tools = _arm_tools()
    results = {}

    def probe(name, fn):
        _, error, _, _ = _capture_stdout_stderr(fn)
        if error is not None:
            # annotation-alias delta (paths-reflection rule): the class
            # qualname in validate_zonal_ready's TypeError is each arm's
            # own zonal package; normalize the exact repr token so the
            # arms compare equal modulo the documented alias
            error = error.replace(str(mz.Zonal), "<Zonal>")
        results[name] = error if error is not None else "NO_ERROR"

    def fresh():
        return _make_zonal(mz)

    # accessibility argument matrix
    z = fresh()
    probe("search_radius_str", lambda: tools.accessibility(
        z, search_radius="250"))
    probe("search_radius_negative", lambda: tools.accessibility(
        z, search_radius=-5))
    probe("destination_weight_int", lambda: tools.accessibility(
        z, search_radius=250, destination_weight=5))
    probe("destination_weight_missing_column", lambda: tools.accessibility(
        z, search_radius=250, destination_weight="nope"))
    probe("alpha_str", lambda: tools.accessibility(
        z, search_radius=250, alpha="1"))
    probe("beta_str", lambda: tools.accessibility(
        z, search_radius=250, beta="0.5"))
    probe("save_reach_as_int", lambda: tools.accessibility(
        z, search_radius=250, save_reach_as=5))
    probe("save_gravity_as_int", lambda: tools.accessibility(
        z, search_radius=250, beta=0.0, save_gravity_as=5))
    probe("save_gravity_as_without_beta", lambda: tools.accessibility(
        z, search_radius=250, save_gravity_as="g"))
    probe("save_knn_without_knn_weights", lambda: tools.accessibility(
        z, search_radius=250, beta=0.0, save_knn_access_as="k"))
    probe("knn_weights_bad_type", lambda: tools.accessibility(
        z, search_radius=250, beta=0.0, save_knn_access_as="k",
        knn_weights=5))
    probe("knn_plateau_str", lambda: tools.accessibility(
        z, search_radius=250, beta=0.0, save_knn_access_as="k",
        knn_weights=[0.5, 0.25], knn_plateau="100"))
    probe("knn_plateau_negative", lambda: tools.accessibility(
        z, search_radius=250, beta=0.0, save_knn_access_as="k",
        knn_weights=[0.5, 0.25], knn_plateau=-1))
    probe("knn_plateau_above_radius", lambda: tools.accessibility(
        z, search_radius=250, beta=0.0, save_knn_access_as="k",
        knn_weights=[0.5, 0.25], knn_plateau=999))
    probe("knn_plateau_without_beta", lambda: tools.accessibility(
        z, search_radius=250, save_knn_access_as="k",
        knn_weights=[0.5, 0.25], knn_plateau=100))
    probe("closest_facility_str", lambda: tools.accessibility(
        z, search_radius=250, closest_facility="yes"))
    probe("closest_facility_true_without_saves", lambda: tools.accessibility(
        z, search_radius=250, closest_facility=True))
    probe("closest_facility_true_missing_distance_save",
          lambda: tools.accessibility(
              z, search_radius=250, closest_facility=True,
              save_closest_facility_as="cf"))
    probe("closest_facility_false_with_save", lambda: tools.accessibility(
        z, search_radius=250, save_closest_facility_as="cf"))
    probe("closest_facility_false_with_distance_save",
          lambda: tools.accessibility(
              z, search_radius=250,
              save_closest_facility_distance_as="cfd"))
    probe("turn_penalty_str", lambda: tools.accessibility(
        z, search_radius=250, turn_penalty="True"))

    # structural quirk probes
    probe("empty_zonal", lambda: tools.accessibility(
        mz.Zonal(), search_radius=250))

    z_late = mz.Zonal()
    z_late.load_layer("streets", _streets_gdf())
    z_late.load_layer("origins", _origins_gdf())
    z_late.load_layer("destinations", _destinations_gdf())
    probe("layers_without_network", lambda: tools.accessibility(
        z_late, search_radius=250))

    z_no_o = fresh()
    z_no_o.network.nodes = z_no_o.network.nodes[
        z_no_o.network.nodes["type"] != "origin"]
    probe("no_origin_rows", lambda: tools.accessibility(
        z_no_o, search_radius=250))

    z_no_d = fresh()
    z_no_d.network.nodes = z_no_d.network.nodes[
        z_no_d.network.nodes["type"] != "destination"]
    probe("no_destination_rows", lambda: tools.accessibility(
        z_no_d, search_radius=250))

    # service_area validation (validates BEFORE the iloc reads — the
    # asymmetry with accessibility is the pin)
    z2 = fresh()
    probe("sa_search_radius_str", lambda: tools.service_area(
        z2, search_radius="250"))
    probe("sa_search_radius_negative", lambda: tools.service_area(
        z2, search_radius=-5))
    probe("sa_origin_ids_str", lambda: tools.service_area(
        z2, origin_ids="0", search_radius=250))
    probe("sa_turn_penalty_int", lambda: tools.service_area(
        z2, origin_ids=0, search_radius=250, turn_penalty=1))

    # validate_zonal_ready direct
    probe("vzr_layers_empty", lambda: tools.validate_zonal_ready(
        mz.Zonal()))
    probe("vzr_not_a_zonal", lambda: tools.validate_zonal_ready(
        "not a zonal"))
    z3 = mz.Zonal()
    z3.load_layer("streets", _streets_gdf())
    z3.create_street_network("streets", weight_attribute="length")
    z3.create_graph()
    probe("vzr_no_origins_or_destinations",
          lambda: tools.validate_zonal_ready(z3))

    return results


class _QueueStub:
    """Single-threaded stand-in for the mp.Manager queue one_access
    consumes (get/task_done only)."""

    def __init__(self, items):
        self._items = list(items)

    def get(self):
        return self._items.pop(0)

    def task_done(self):
        pass


class _ExplodingQueueStub(_QueueStub):
    def __init__(self, items, raise_after):
        super().__init__(items)
        self._gets = 0
        self._raise_after = raise_after

    def get(self):
        if self._gets >= self._raise_after:
            raise RuntimeError("synthetic queue failure "
                               "(pinned exception-swallowing probe)")
        self._gets += 1
        return super().get()


def scenario_engine_primitives(mz, seed):
    """Direct trio exercise: get_origin_properties (with and without the
    knn fields set on the network), one_access via a queue stub
    (reporting on; deterministic progress-line count), the pinned
    exception-swallowing path, parallel_access num_cores=1, and the
    pinned parallel-path consolidation failures (num_cores=2 with
    beta=None -> KeyError 'gravity'; num_cores=2 + closest_facility ->
    KeyError 'reach')."""
    btd = _arm_betweenness()
    tools = _arm_tools()
    results = {}

    # --- get_origin_properties, no knn fields ---
    z1 = _make_zonal(mz)
    o1 = _origin_idx(z1, 0)
    z1.network.add_node_to_graph(z1.network.d_graph, o1)
    d_idxs, _, _ = tools.turn_o_scope(
        network=z1.network, o_idx=o1, search_radius=250, detour_ratio=1,
        turn_penalty=False, o_graph=z1.network.d_graph,
        return_paths=False)
    z1.network.remove_node_to_graph(z1.network.d_graph, o1)
    d_idxs = dict(sorted(d_idxs.items(), key=lambda item: item[1]))
    # PINNED: get_origin_properties reads self.network.knn_weight
    # unconditionally (`is not None` on a MISSING attribute); a fresh
    # Network lacks the attribute (one_access sets it right before
    # calling), so a direct call on a pristine network raises.
    _, pristine_err, _, _ = _capture_stdout_stderr(
        lambda: btd.get_origin_properties(
            z1, search_radius=250, beta=0.001, turn_penalty=False,
            o_idx=o1, d_idxs=d_idxs, o_graph=z1.network.d_graph))
    z1.network.knn_weight = None
    z1.network.knn_plateau = None
    btd.get_origin_properties(
        z1, search_radius=250, beta=0.001, turn_penalty=False,
        o_idx=o1, d_idxs=d_idxs, o_graph=z1.network.d_graph)
    results["gop_reach_gravity"] = {
        "pristine_network_error": pristine_err,
        "d_idxs": {int(k): _f(v) for k, v in d_idxs.items()},
        "origin_row": gdf_digest(
            z1.network.nodes[z1.network.nodes.index == o1]),
    }

    # --- get_origin_properties with knn fields on the network ---
    z2 = _make_zonal(mz)
    o2 = _origin_idx(z2, 1)
    z2.network.knn_weight = [0.5, 0.25]
    z2.network.knn_plateau = 250
    z2.network.add_node_to_graph(z2.network.d_graph, o2)
    d_idxs2, _, _ = tools.turn_o_scope(
        network=z2.network, o_idx=o2, search_radius=250, detour_ratio=1,
        turn_penalty=False, o_graph=z2.network.d_graph,
        return_paths=False)
    z2.network.remove_node_to_graph(z2.network.d_graph, o2)
    d_idxs2 = dict(sorted(d_idxs2.items(), key=lambda item: item[1]))
    btd.get_origin_properties(
        z2, search_radius=250, beta=0.002, turn_penalty=False,
        o_idx=o2, d_idxs=d_idxs2, o_graph=z2.network.d_graph)
    results["gop_knn"] = {
        "origin_row": gdf_digest(
            z2.network.nodes[z2.network.nodes.index == o2]),
    }

    # --- one_access via queue stub, reporting on ---
    z3 = _make_zonal(mz)
    o_idxs = sorted(z3.network.nodes[
        z3.network.nodes["type"] == "origin"].index.tolist())
    queue = _QueueStub(o_idxs + ["done"])
    ret, err, out, errstream = _capture_stdout_stderr(
        lambda: btd.one_access(
            z3, origin_queue=queue, search_radius=250, alpha=1,
            closest_facility=False, turn_penalty=False, reporting=True))
    o_gdf, d_gdf = ret
    results["one_access_stub_queue"] = {
        "error": err,
        "queue_order": [int(i) for i in o_idxs],
        "processed_origins": [int(i) for i in o_gdf.index],
        "progress_lines": _progress_line_count(out),
        "origin_rows_reach": [_f(v) for v in o_gdf["reach"]],
    }

    # --- pinned exception swallowing: mid-queue failure is printed and
    # swallowed; one_access returns the PARTIAL processed rows ---
    z4 = _make_zonal(mz)
    queue4 = _ExplodingQueueStub(o_idxs + ["done"], raise_after=2)
    ret4, err4, out4, err4s = _capture_stdout_stderr(
        lambda: btd.one_access(
            z4, origin_queue=queue4, search_radius=250, alpha=1,
            closest_facility=False, turn_penalty=False, reporting=True))
    o_gdf4, _ = ret4
    results["one_access_exception_swallowed"] = {
        "returned_error": err4,
        "processed_before_failure": [int(i) for i in o_gdf4.index],
        "notice_in_stdout": "Issues in one access" in out4,
        "traceback_in_stderr": "Traceback" in err4s,
    }

    # --- parallel_access num_cores=1 (inline reporting path) ---
    z5 = _make_zonal(mz)
    order5 = _seeded_origin_order(z5, seed)
    res5, err5, out5, _ = _capture_stdout_stderr(
        lambda: btd.parallel_access(
            z5, search_radius=250, alpha=1, closest_facility=False,
            turn_penalty=False, num_cores=1))
    results["parallel_access_num_cores_1"] = {
        "error": err5,
        "node": _node_state(z5),
        "progress_lines": _progress_line_count(out5),
        "shuffled_origin_order": order5,
    }

    # --- pinned parallel consolidation failures ---
    z6 = _make_zonal(mz)
    _seeded_origin_order(z6, seed)
    _, err6, _, _ = _capture_stdout_stderr(
        lambda: btd.parallel_access(
            z6, search_radius=250, alpha=1, closest_facility=False,
            turn_penalty=False, num_cores=2))
    results["parallel_access_num_cores_2_beta_none"] = {"error": err6}

    z7 = _make_zonal(mz)
    order7 = _seeded_origin_order(z7, seed)
    res7, err7, out7, _ = _capture_stdout_stderr(
        lambda: btd.parallel_access(
            z7, search_radius=250, alpha=2, beta=0.0,
            destination_weight="weight", closest_facility=False,
            turn_penalty=False, num_cores=2))
    results["parallel_access_num_cores_2_beta_set"] = {
        "error": err7,
        "node": None if err7 else _node_state(z7),
        "shuffled_origin_order": order7,
    }

    z8 = _make_zonal(mz)
    _seeded_origin_order(z8, seed)
    _, err8, _, _ = _capture_stdout_stderr(
        lambda: btd.parallel_access(
            z8, search_radius=250, alpha=1, closest_facility=True,
            turn_penalty=False, num_cores=2))
    results["parallel_access_num_cores_2_closest_facility"] = {
        "error": err8}

    return results


def scenario_node_builder_truncation(mz, seed):
    """Bounded reproduction of the pinned upstream node-builder defect:
    vectorized_node_edge_builder's final occurrence range is truncated
    to the lexicographic maximum point's FIRST occurrence, so that
    point's remaining edge endpoints keep their zeros-initialized labels
    (the dead ``np.append(node_first_occurance, point_count - 1)`` is
    the vestigial intended fix).  On the broken fixture C=(100,100) is
    the lexicographic maximum with three occurrences: B-C's ``end`` is
    labeled 0 (F), and the A-C diagonal is mislabeled {1,0} — the same
    endpoint set as F-A — and is discarded as a false "redundant_edge".
    Consequence: origin o3 (mid B-C) attaches a phantom 50 m edge to F
    and its accessibility scope is EMPTY.  Both arms must reproduce all
    of this bitwise (the facade is a faithful copy, bugs included)."""
    tools = _arm_tools()

    broken = _make_zonal(mz, clean=False)
    e_broken = broken.network.edges
    broken_edges = {int(i): {"start": int(r["start"]), "end": int(r["end"]),
                             "length": _f(r["length"]),
                             "parent_street_id": int(r["parent_street_id"]),
                             "wkb": _wkb_hash(r["geometry"])}
                    for i, r in e_broken.iterrows()}
    broken_nodes = gdf_digest(broken.network.nodes)

    # accessibility consequence: o3's scope must be empty
    o3 = _origin_idx(broken, 2)
    broken.network.add_node_to_graph(broken.network.d_graph, o3)
    d_idxs_b, _, _ = tools.turn_o_scope(
        network=broken.network, o_idx=o3, search_radius=250,
        detour_ratio=1, turn_penalty=False,
        o_graph=broken.network.d_graph, return_paths=False)
    broken.network.remove_node_to_graph(broken.network.d_graph, o3)
    o1_idx = _origin_idx(broken, 0)
    broken.network.add_node_to_graph(broken.network.d_graph, o1_idx)
    d_idxs_o1, _, _ = tools.turn_o_scope(
        network=broken.network, o_idx=o1_idx, search_radius=250,
        detour_ratio=1, turn_penalty=False,
        o_graph=broken.network.d_graph, return_paths=False)
    broken.network.remove_node_to_graph(broken.network.d_graph, o1_idx)

    order = _seeded_origin_order(broken, seed)
    res, err, out, _ = _capture_stdout_stderr(
        lambda: tools.accessibility(broken, search_radius=250, alpha=1))

    clean = _make_zonal(mz)
    e_clean = clean.network.edges
    clean_edges = {int(i): {"start": int(r["start"]), "end": int(r["end"]),
                            "length": _f(r["length"])}
                   for i, r in e_clean.iterrows()}

    return {
        "broken_edge_table": broken_edges,
        "broken_node_table": broken_nodes,
        "o3_scope_empty": len(d_idxs_b) == 0,
        "o1_scope": {int(k): _f(v) for k, v in d_idxs_o1.items()},
        "accessibility_on_broken": {
            "node": _node_state(broken), "error": err,
            "progress_lines": _progress_line_count(out),
            "shuffled_origin_order": order,
        },
        "clean_edge_table": clean_edges,
        "diagonal_dropped_in_broken": (
            sum(1 for v in broken_edges.values()
                if math.isclose(float.fromhex(v["length"]),
                                math.hypot(100, 100))) == 0),
    }


SCENARIOS = {
    "reach_gravity": scenario_reach_gravity,
    "knn": scenario_knn,
    "closest_facility": scenario_closest_facility,
    "layer_mutations": scenario_layer_mutations,
    "service_areas": scenario_service_areas,
    "validation": scenario_validation,
    "engine_primitives": scenario_engine_primitives,
    "node_builder_truncation": scenario_node_builder_truncation,
}

# Test modules import the scenario list from THIS uniquely-named module,
# never via `from conftest import ...`: bare `conftest` collides across
# non-package test dirs in sys.modules (first suite collected wins), so
# co-collected suites would silently read each other's lists or fail
# collection (zonal review finding; class-wide pattern fix).
SCENARIO_NAMES = list(SCENARIOS)


# ----------------------------------------------------------------------
# comparator-sensitivity mutations: each drops/alters exactly one piece
# of recorded state; an honest comparator MUST select the difference.
# alpha_swap and skip_shuffle are REAL behavioral mutants (applied
# inside the scenario body); the rest are digest-level controls.
# ----------------------------------------------------------------------

def _sabotage_noop(d):
    # alpha_swap (parameter-level) and skip_shuffle (behavioral
    # monkeypatch) are applied inside the scenario body; nothing left
    # for the digest-level pass to do.
    pass


def _sabotage_reverse_closest_facility(d):
    # dossier mutant (direction reversal): swap the closest-facility
    # assignment between the two reachable destinations (the last node
    # row is the unreachable iso destination, still NaN).  Guaranteed
    # selected: the winners' distances differ (100 vs 200), so both the
    # cf and the cfd lists change.
    node = d["node"]["values"]
    for col in ("closest_facility", "closest_facility_distance"):
        vals = node[col]
        vals[-3], vals[-2] = vals[-2], vals[-3]


def _sabotage_fillna_closest_distance(d):
    # the NA-not-filled pin: the iso destination's NaN must survive into
    # the destination layer; replacing it with a filled 0 must be
    # selected
    d["layers"]["destinations"]["values"]["cfd"][-1] = _f(0.0)


def _sabotage_perturb_gravity_bit(d):
    # bit-level comparator sensitivity on the gravity column (node gdf):
    # zero the bit pattern of the first non-NaN entry
    grav = d["gravity"]["node"]["values"]["gravity"]
    grav[grav.index([v for v in grav
                     if v != _f(float("nan"))][0])] = "0000000000000000"


SABOTAGES = {
    "alpha_swap": ("reach_gravity", _sabotage_noop),
    "skip_shuffle": ("closest_facility", _sabotage_noop),
    "reverse_closest_facility": ("closest_facility",
                                 _sabotage_reverse_closest_facility),
    "fillna_closest_distance": ("closest_facility",
                                _sabotage_fillna_closest_distance),
    "perturb_gravity_bit": ("reach_gravity", _sabotage_perturb_gravity_bit),
    "hide_validation_error": ("validation",
                              lambda d: d.__setitem__(
                                  "search_radius_negative", "NO_ERROR")),
}


# ----------------------------------------------------------------------
# arm dispatch
# ----------------------------------------------------------------------

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
