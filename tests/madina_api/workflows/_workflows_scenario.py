"""MADINA_WORKFLOWS parity scenario runner: exercise the pairing-workflow
surface (Logger / betweenness_flow_simulation / KNN_accessibility)
through either the compat facade or the pinned upstream reference, then
dump a BITWISE-strict state digest.

Same discipline as tests/madina_api/{zonal,paths,access,flow}
(MADINA_ZONAL / MADINA_PATHS / MADINA_ACCESS / MADINA_FLOW): both arms
run under the SAME interpreter (the dependency-bridged reference venv)
with identical file fixtures, so any digest difference is attributable
to the code under test.  Floats are compared as IEEE-754 bit patterns,
geometries as WKB hashes, order-sensitive content as explicit lists.

The workflows build their own Zonal from FILES in a data folder and
write all observable state as FILES in an output folder — the digests
cover exactly those files (plus in-process ValueErrors):

Nondeterminism policy (documented in the facade header, not hidden):
  - ``Logger.flow_map_template_1`` HTML bytes differ run-to-run INSIDE
    pydeck (same inputs -> different files, evidence probes C1/C2), so
    ``*.html`` files are pinned as presence + nonzero size only;
  - wall-clock columns (``time``, ``seconds_elapsed``,
    ``cumulative_seconds`` in time_log.csv; any ``*_chunck_time``
    diagnostic column) are EXCLUDED from digests, with presence and
    column names recorded — time_log.csv is additionally digested via
    its clock-free ``flow_name``/``event`` columns in row order;
  - every other produced file is digested structurally (columns,
    dtypes, IEEE-754 value hex, WKB geometry hashes) AND by raw md5.
  - num_cores>1 exposure accumulation carries no bitwise-stability
    contract upstream (queue partition; see the flow suite header), so
    the num_cores=8 workflow scenario digests the streets record
    STRUCTURALLY (exists/columns/rows/dtypes) and the origin record
    BITWISE (its content is the deterministic stats propagation),
    never the edge bits.

Sabotages (mutation protocol): source-level mutants are applied to the
arm's OWN workflows.py source (reference arm patches the upstream file,
facade arm patches the facade file — same literals, count asserted ==
1) and the patched clone is exercised; integrity requires
mutant-facade digests == mutant-reference digests, selection requires
mutant != clean.  Digest-level mutants reshape the recorded state and
must be selected by the comparator.

Usage:
    python _workflows_scenario.py --arm {facade,reference}
                                  --scenario NAME --out DIGEST.json
                                  --seed 20260930 [--sabotage NAME]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import struct
import sys
import traceback
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import LineString, Point

REPO = Path(__file__).resolve().parents[3]

_SABOTAGE = {"name": None}

# wall-clock diagnostic columns: excluded from every digest, presence
# recorded (see module docstring)
TIME_COLUMN_SUFFIXES = ("_chunck_time",)
TIME_COLUMN_NAMES = ("destination_discovery_time", "destination_prep_time",
                     "path_generation_time", "chunck_time")
TIME_COLUMNS = ("time", "seconds_elapsed", "cumulative_seconds",
                "destination_discovery_time", "destination_prep_time",
                "path_generation_time", "chunck_time")

_ARM_MARKER = {}

_SCENARIO_TMP = None  # per-arm scratch root, set in main()


# ----------------------------------------------------------------------
# arm loading (same convention as the flow suite)
# ----------------------------------------------------------------------

def import_arm(arm):
    if arm == "reference":
        sys.path.insert(0, str(REPO / ".refs" / "madina_ref" / "src"))
        import madina.una.workflows as wf_mod        # noqa: PLC0415
        import madina.una.tools as tools_mod         # noqa: PLC0415
        import madina.zonal as mz                    # noqa: PLC0415
        _ARM_MARKER["workflows"] = wf_mod
        _ARM_MARKER["tools"] = tools_mod
        _ARM_MARKER["zonal"] = mz
        return wf_mod
    if arm == "facade":
        sys.path.insert(0, str(REPO / "src"))
        import urban_network_analysis.compat.madina.una.workflows as wf_mod  # noqa: PLC0415
        import urban_network_analysis.compat.madina.una.tools as tools_mod   # noqa: PLC0415
        from urban_network_analysis.compat.madina import zonal as mz         # noqa: PLC0415
        _ARM_MARKER["workflows"] = wf_mod
        _ARM_MARKER["tools"] = tools_mod
        _ARM_MARKER["zonal"] = mz
        return wf_mod
    raise ValueError(f"unknown arm {arm}")


def load_patched_workflows(arm, replacements: list[tuple[str, str]]):
    """Import the arm's workflows module, then exec a source-patched
    clone of ITS OWN file (reference -> upstream source, facade ->
    facade source).  Every replacement must occur exactly once, in both
    arms, so a mutant that stops engaging fails loudly instead of
    silently pinning the clean behavior."""
    real = import_arm(arm)
    import types
    src = Path(real.__file__).read_text()
    for old, new in replacements:
        count = src.count(old)
        assert count == 1, (
            f"sabotage {_SABOTAGE['name']}: replacement target {old!r} "
            f"occurs {count}x in {real.__file__} (expected exactly 1)")
        src = src.replace(old, new)
    clone = types.ModuleType(f"{real.__name__}_mutant")
    clone.__file__ = real.__file__
    clone.__package__ = real.__package__
    exec(compile(src, real.__file__, "exec"), clone.__dict__)  # noqa: S102
    return clone


# ----------------------------------------------------------------------
# digest helpers (bitwise strict)
# ----------------------------------------------------------------------

def _float_hex(v):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "nan"
    if isinstance(v, (bool, np.bool_)):
        return str(bool(v))
    if isinstance(v, (int, np.integer)):
        return f"int:{int(v)}"
    if isinstance(v, (float, np.floating)):
        return struct.pack(">d", float(v)).hex()
    return str(v)


def _jsonable(v):
    if isinstance(v, (list, tuple, np.ndarray)):
        return [_jsonable(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    return _float_hex(v)


def _geom_digest(geom):
    # workflow CSVs carry geometry as WKT TEXT (pd.read_csv), output
    # geoJSONs carry real shapely geometries — digest both by content
    # (a WKT string is deterministic text, same contract as WKB bytes)
    if geom is None:
        return None
    if isinstance(geom, str):
        return hashlib.sha256(geom.encode("utf-8")).hexdigest()
    return hashlib.sha256(geom.wkb).hexdigest()


def _is_time_col(col):
    # workflow outputs prefix diagnostic columns with the Flow_Name
    # (e.g. 'elastic_huff_destination_discovery_time'), so suffix
    # matching is required, not exact names
    return (col in TIME_COLUMNS
            or col.endswith(TIME_COLUMN_SUFFIXES)
            or col.endswith(TIME_COLUMN_NAMES))


def _digest_frame(df_or_gdf, *, label):
    """Bitwise digest of a frame read from a workflow output file.
    Time columns: presence only.  Geometry: WKB sha256."""
    gdf = df_or_gdf
    out = {
        "label": label,
        "columns": [str(c) for c in gdf.columns],
        "dtypes": [str(t) for t in gdf.dtypes],
        "shape": [int(gdf.shape[0]), int(gdf.shape[1])],
        "time_columns_present": [c for c in out_cols(gdf) if _is_time_col(c)],
    }
    values = {}
    for col in gdf.columns:
        c = str(col)
        if c == "geometry":
            values[c] = [_geom_digest(g) for g in gdf["geometry"]]
        elif _is_time_col(col):
            continue
        else:
            values[c] = [_jsonable(v) for v in gdf[col].tolist()]
    out["values"] = values
    return out


def out_cols(gdf):
    return [str(c) for c in gdf.columns]


def _md5(path: Path):
    return hashlib.md5(path.read_bytes()).hexdigest()


def _digest_file(path: Path, rel: str):
    """Digest one produced file.  HTML: presence + size only (pydeck
    bytes are nondeterministic run-to-run).  Everything else: md5 +
    structured content where the format is one we parse."""
    if path.suffix == ".html":
        entry = {"bytes": path.stat().st_size,
                 "policy": "presence_only_pydeck_nondeterministic"}
        return entry
    if path.name == "time_log.csv":
        # wall-clock columns make the raw bytes nondeterministic
        # run-to-run; digest the clock-free columns only (policy, not
        # an exemption from parity — events are still compared in
        # order)
        log = pd.read_csv(path)
        entry = {
            "policy": "structured_only_wall_clock_columns",
            "row_count": int(log.shape[0]),
            "columns": [str(c) for c in log.columns],
            "events": [
                [None if pd.isna(f) else str(f),
                 None if pd.isna(e) else str(e)]
                for f, e in zip(log.get("flow_name"), log.get("event"))],
        }
        return entry
    entry = {"bytes": path.stat().st_size}
    name = path.name
    try:
        if name.endswith(".csv"):
            frame = _digest_frame(pd.read_csv(path), label=rel)
        elif name.endswith(".geojson") or name.endswith(".geoJSON"):
            frame = _digest_frame(gpd.read_file(path), label=rel)
        else:
            frame = None
        if frame is not None:
            entry["frame"] = frame
            if frame["time_columns_present"]:
                # volatile bytes embed the wall-clock diagnostics; the
                # structured digest above is the parity contract
                entry["policy"] = ("md5_excluded_wall_clock_columns_in_frame")
                del entry["bytes"]
            else:
                entry["md5"] = _md5(path)
        else:
            entry["md5"] = _md5(path)
    except Exception as exc:  # noqa: BLE001
        entry["parse_error"] = f"{type(exc).__name__}: {exc}"
    return entry


def _digest_tree(out_folder: Path, structural_files=()):
    """structural_files: output-tree-relative names whose content carries
    NO bitwise contract (e.g. the num_cores>1 edge record — queue
    partition, same policy as the flow suite's nc=2 edges); recorded as
    presence + size + policy only, with the structural read living in
    the scenario's own extra block."""
    files = {}
    for p in sorted(out_folder.rglob("*")):
        if p.is_file():
            rel = str(p.relative_to(out_folder))
            if rel in structural_files:
                files[rel] = {
                    "bytes": p.stat().st_size,
                    "policy": "structural_only_nc_gt1_edge_record",
                }
            else:
                files[rel] = _digest_file(p, rel)
    return {"files": files}


# ----------------------------------------------------------------------
# file fixtures (deterministic; written fresh per arm under the
# arm-owned scratch root so the two arms never share writable state)
# ----------------------------------------------------------------------

def _streets_rows(hazzard=False):
    rows = [
        ("F-A", 100.0, LineString([(-100, 0), (0, 0)])),
        ("A-B", 100.0, LineString([(0, 0), (100, 0)])),
        ("B-C", 100.0, LineString([(100, 0), (100, 100)])),
        ("C-D", 100.0, LineString([(100, 100), (0, 100)])),
        ("D-A", 100.0, LineString([(0, 100), (0, 0)])),
        ("A-C", float(np.hypot(100, 100)), LineString([(0, 0), (100, 100)])),
        ("D-E", 50.0, LineString([(0, 100), (0, 150)])),
        ("C-G", 100.0, LineString([(100, 100), (200, 100)])),
    ]
    data = {"id": [r[0] for r in rows], "length": [r[1] for r in rows]}
    if hazzard:
        data["hazzard"] = [0.5, 1.5, 1.0, 2.0, 0.5, 1.0, 0.25, 0.75]
    return data, [r[2] for r in rows]


def _write_geojson(folder: Path, name: str, data, geoms):
    gpd.GeoDataFrame(data, geometry=geoms, crs="EPSG:3857").to_file(
        folder / name, driver="GeoJSON", engine="pyogrio")


def _write_base_layers(folder: Path, *, hazzard=False, second_streets=False):
    folder.mkdir(parents=True, exist_ok=True)
    data, geoms = _streets_rows(hazzard=hazzard)
    _write_geojson(folder, "streets.geojson", data, geoms)
    if second_streets:
        # same geometry, different id labels: exercises the KNN
        # per-row Network_File change (reload path) without geometry
        # ambiguity
        data2 = {"id": [f"S{i}" for i in range(len(geoms))],
                 "length": [100.0] * len(geoms)}
        _write_geojson(folder, "streets2.geojson", data2, geoms)
    _write_geojson(
        folder, "origins.geojson", {"weight": [1.0, 2.0, 1.0]},
        [Point((-50, 0)), Point((50, 0)), Point((100, 50))])
    _write_geojson(
        folder, "destinations.geojson", {"weight": [2.0, 3.0]},
        [Point((50, 100)), Point((0, 150))])


def _flow_row(**over):
    row = {
        "Flow_Name": "huff_flow",
        "Origin_Name": "origins", "Origin_File": "origins.geojson",
        "Origin_Weight": "Count",
        "Destination_Name": "destinations",
        "Destination_File": "destinations.geojson",
        "Destination_Weight": "Count",
        "Network_File": "streets.geojson", "Network_Cost": "Geometric",
        "Turn_Penalty": 0, "Turn_Threshold": 45, "Turns": False,
        "Radius": 250, "Detour": 1.0, "Decay": False,
        "Decay_Mode": "exponent", "Beta": 0.003, "Elastic_Weights": False,
        "KNN_Weight": "[0.5,0.5]", "Plateau": 0,
        "Closest_destination": True,
    }
    row.update(over)
    return row


def _write_pairings(folder: Path, rows, name="pairings.csv"):
    pd.DataFrame(rows).to_csv(folder / name, index=False)


def _fresh(tag):
    """Fresh per-arm fixture/output pair under the arm scratch root."""
    root = _SCENARIO_TMP / tag
    shutil.rmtree(root, ignore_errors=True)
    return root / "data", root / "out"


def _norm_err(exc):
    # error text embeds absolute fixture paths; normalize the arm-owned
    # scratch root so the two arms' verbatim texts are comparable (the
    # arm-specific prefix is scratch layout, not arm behavior)
    text = f"{type(exc).__name__}: {exc}"
    if _SCENARIO_TMP is not None:
        text = text.replace(str(_SCENARIO_TMP), "<scratch>")
    return text


def _run_flow(wf, data, out, *, pairings_file="pairings.csv", **kw):
    try:
        wf.betweenness_flow_simulation(
            data_folder=str(data), output_folder=str(out),
            pairings_file=pairings_file, **kw)
        return None
    except Exception as exc:  # noqa: BLE001
        return _norm_err(exc)


def _run_knn(wf, data, out, *, pairings_file="pairing.csv", **kw):
    try:
        wf.KNN_accessibility(
            city_name="suite_city", data_folder=str(data),
            output_folder=str(out), pairings_file=pairings_file, **kw)
        return None
    except Exception as exc:  # noqa: BLE001
        return _norm_err(exc)


def _state(data, out, *, extra=None, errors=None, structural_files=()):
    state = {
        "pinned_fact": (
            "workflows build a Zonal from files and publish FILES; "
            "digests cover the output tree + captured errors"),
        "env_marker": {
            "workflows_file": Path(_ARM_MARKER["workflows"].__file__).name,
        },
    }
    if out.is_dir():
        state.update(_digest_tree(out, structural_files=structural_files))
    else:
        state["files"] = {}
    if errors:
        state["errors"] = errors
    if extra:
        state.update(extra)
    return state


# ----------------------------------------------------------------------
# scenarios
# ----------------------------------------------------------------------

def scenario_wf_flow_defaults():
    """Single closest-destination pairing, num_cores=1: the CSV-default
    output contract (save_flow_csv/save_origin_csv default False ->
    files ABSENT; geoJSONs present), the deterministic record even with
    the pinned per-origin stats UnboundLocalError, the uniform
    'Count' origin weights, and the clock-free log events."""
    wf = _WF()
    data, out = _fresh("flow_defaults")
    _write_base_layers(data)
    _write_pairings(data, [_flow_row()])
    err = _run_flow(wf, data, out, num_cores=1)
    pairing_dir = ("huff_flow_O(origins)_D(destinations)")
    extra = {
        "record_csv_absent_by_default":
            not (out / pairing_dir / "betweenness_record_so_far.csv").exists(),
        "origin_csv_absent_by_default": not any(
            p.name == "origin_record.csv" and p.parent == out
            for p in out.rglob("origin_record.csv")),
        "origin_geojson_present":
            (out / pairing_dir / "origin_record_(origins).geoJSON").is_file(),
        "flow_maps_present": sorted(
            p.name for p in (out / pairing_dir).glob("flow_map_*.html")),
        "workflow_error": err,
    }
    return _state(data, out, extra=extra)


def scenario_wf_flow_huff_elastic():
    """Huff competition (closest_destination=False) + elastic weights:
    the healthy stats path (all stats columns present), the pinned
    elastic decay suppression (decay=False because Elastic_Weights,
    even though Decay=True in the row), save_elastic_weight_as wiring,
    and the KNN_Weight/Plateau forwarding."""
    wf = _WF()
    data, out = _fresh("flow_huff_elastic")
    _write_base_layers(data)
    _write_pairings(data, [_flow_row(
        Flow_Name="elastic_huff", Closest_destination=False,
        Elastic_Weights=True, Decay=True)])
    err = _run_flow(wf, data, out, num_cores=1)
    pairing_dir = "elastic_huff_O(origins)_D(destinations)"
    origin = out / pairing_dir / "origin_record_(origins).geoJSON"
    extra = {"workflow_error": err}
    if origin.is_file():
        og = gpd.read_file(origin)
        cols = out_cols(og)
        extra["origin_columns"] = cols
        extra["elastic_weight_column_present"] = any(
            c.startswith("elastic_weight_") for c in cols)
        extra["stats_columns_present"] = [
            c for c in ("elastic_huff_closest_destination_distance",
                        "elastic_huff_mean_path_length",
                        "elastic_huff_eligible_destinations",
                        "elastic_huff_path_count") if c in cols]
        extra["reach_gravity_present"] = sorted(
            c for c in cols
            if c.startswith(("reach_", "gravity_")))
    return _state(data, out, extra=extra)


def scenario_wf_flow_two_pairings():
    """Two pairings, SAME Network_Cost: the flush branch (nodes restored
    from clean_network_nodes, no rebuild), layer reuse (origin/
    destination files loaded ONCE — clock-free log events), and the
    streets record carrying BOTH flow columns after the second run."""
    wf = _WF()
    data, out = _fresh("flow_two_pairings")
    _write_base_layers(data)
    _write_pairings(data, [
        _flow_row(Flow_Name="flow_a", Radius=250),
        _flow_row(Flow_Name="flow_b", Radius=200),
    ])
    err = _run_flow(wf, data, out, num_cores=1)
    rec = out / "betweenness_record.csv"
    extra = {"workflow_error": err}
    if rec.is_file():
        df = pd.read_csv(rec)
        extra["record_columns"] = [str(c) for c in df.columns]
        extra["both_flow_columns"] = [
            c for c in ("flow_a", "flow_b") if c in df.columns]
    return _state(data, out, extra=extra)


def scenario_wf_flow_cost_change():
    """Second pairing with a CHANGED Network_Cost ('length' attribute):
    the rebuild branch runs (create_street_network with
    weight_attribute='length' vs Geometric) — per-row input changes
    must change the profile, not be silently ignored."""
    wf = _WF()
    data, out = _fresh("flow_cost_change")
    _write_base_layers(data)
    _write_pairings(data, [
        _flow_row(Flow_Name="geo_leg", Network_Cost="Geometric"),
        _flow_row(Flow_Name="len_leg", Network_Cost="length"),
    ])
    err = _run_flow(wf, data, out, num_cores=1)
    rec = out / "betweenness_record.csv"
    extra = {"workflow_error": err}
    if rec.is_file():
        df = pd.read_csv(rec)
        extra["record_columns"] = [str(c) for c in df.columns]
        extra["both_flow_columns"] = [
            c for c in ("geo_leg", "len_leg") if c in df.columns]
        extra["record_row_count"] = int(df.shape[0])
    return _state(data, out, extra=extra)


def scenario_wf_flow_exposure_row():
    """A row carrying Exposure_Attribute='hazzard': the path-exposure
    machinery at workflow level (edge exposure column on the record,
    per-origin hazzard stats), with closest_destination=False so the
    stats block completes (the closest-destination stats crash is
    pinned separately in wf_flow_defaults)."""
    wf = _WF()
    data, out = _fresh("flow_exposure_row")
    _write_base_layers(data, hazzard=True)
    _write_pairings(data, [_flow_row(
        Flow_Name="expo", Closest_destination=False,
        Exposure_Attribute="hazzard")])
    err = _run_flow(wf, data, out, num_cores=1)
    pairing_dir = "expo_O(origins)_D(destinations)"
    rec = out / pairing_dir / "betweenness_record_so_far.geoJSON"
    origin = out / pairing_dir / "origin_record_(origins).geoJSON"
    extra = {"workflow_error": err}
    if rec.is_file():
        rg = gpd.read_file(rec)
        extra["record_columns"] = out_cols(rg)
        extra["exposure_column_on_record"] = [
            c for c in out_cols(rg) if "exposure" in c.lower()]
    if origin.is_file():
        og = gpd.read_file(origin)
        extra["origin_hazzard_columns"] = [
            c for c in out_cols(og)
            if "hazz" in c.lower() or "exposure" in c.lower()]
    return _state(data, out, extra=extra)


def scenario_wf_flow_nc3_structural():
    """Default num_cores=8 with 3 origins -> num_cores=3 engages the
    exposure engine's multi-core path (no bitwise-stability contract; see
    module docstring): workflow completes, origin record digested BITWISE
    (deterministic stats propagation incl. the missing stats-block
    columns), BOTH edge-record files digested STRUCTURALLY only.  The
    edge bits are additionally observed stable on this fixture across
    the retained runs (review probe: edge md5
    41cbac402fe59a5439cc8253f127b056 across 4 runs) — an observation,
    not an engine contract, so the parity contract stays structural."""
    wf = _WF()
    data, out = _fresh("flow_nc3_structural")
    _write_base_layers(data)
    _write_pairings(data, [_flow_row()])
    err = _run_flow(wf, data, out)  # default num_cores=8
    extra = {"workflow_error": err, "num_cores_passed": None}
    rec = out / "betweenness_record.csv"
    if rec.is_file():
        df = pd.read_csv(rec)
        extra["record_structural"] = {
            "columns": [str(c) for c in df.columns],
            "row_count": int(df.shape[0]),
            "dtypes": [str(t) for t in df.dtypes],
        }
    gj = out / "betweenness_record.geoJSON"
    if gj.is_file():
        gdf = gpd.read_file(gj)
        flow_cols = [c for c in gdf.columns
                     if c not in ("id", "geometry")
                     and not _is_time_col(c)]
        vals = gdf[flow_cols[0]].tolist() if flow_cols else []
        extra["record_geojson_structural"] = {
            "columns": [str(c) for c in gdf.columns],
            "row_count": int(gdf.shape[0]),
            "flow_column": flow_cols[0] if flow_cols else None,
            "nonzero_entries": int(sum(1 for v in vals if v)),
            "all_finite": all(
                bool(math.isfinite(float(v))) for v in vals if v is not None),
        }
    return _state(data, out, extra=extra,
                  structural_files=("betweenness_record.geoJSON",
                                    "betweenness_record.csv"))


def scenario_wf_flow_errors():
    """Validation matrix at workflow level: the verbatim no-args
    ValueError, the missing pairings-file FileNotFoundError, the EMPTY
    pairings table (headers only -> empty output tree / engine error,
    whichever the profile produces), and a custom pairings_file name."""
    wf = _WF()
    errors = {}
    try:
        wf.betweenness_flow_simulation()
    except Exception as exc:  # noqa: BLE001
        errors["noargs"] = _norm_err(exc)

    data, out = _fresh("flow_errors")
    _write_base_layers(data)
    missing = _run_flow(wf, data / "nope", out / "nope_out")
    errors["missing_data_folder"] = missing

    # empty pairings: headers only, zero rows
    data2, out2 = _fresh("flow_errors_empty")
    _write_base_layers(data2)
    _write_pairings(data2, [], name="pairings.csv")
    errors["empty_pairings"] = _run_flow(wf, data2, out2)
    empty_state = {
        "empty_output_files": sorted(
            str(p.relative_to(out2)) for p in out2.rglob("*")
            if p.is_file()) if out2.is_dir() else [],
    }

    # custom pairings_file name
    data3, out3 = _fresh("flow_errors_custom")
    _write_base_layers(data3)
    _write_pairings(data3, [_flow_row(Flow_Name="custom_flow")],
                    name="my_pairs.csv")
    err3 = _run_flow(wf, data3, out3, pairings_file="my_pairs.csv",
                     num_cores=1)
    errors["custom_pairings_file"] = err3
    custom_state = {
        "custom_pairing_dir_present": (out3 / (
            "custom_flow_O(origins)_D(destinations)")).is_dir(),
    }

    state = _state(data, out, errors=errors)
    state["empty_pairings"] = empty_state
    state["custom_pairings"] = custom_state
    return state


def scenario_wf_knn_defaults():
    """KNN workflow defaults: singular 'pairing.csv' resolved via the
    pairings_file parameter, city_name REQUIRED even with explicit
    folders, two same-cost pairings (clear_nodes branch), the
    total_knn_access / normalized_knn_access min-max columns, and the
    verbatim bad-KNN_Weight ValueError for an empty CSV cell."""
    wf = _WF()
    errors = {}

    # city_name-only guard: explicit folders WITHOUT city_name
    data0, out0 = _fresh("knn_guard")
    _write_base_layers(data0)
    _write_pairings(data0, [_knn_row()], name="pairing.csv")
    try:
        wf.KNN_accessibility(data_folder=str(data0),
                             output_folder=str(out0), num_cores=1)
    except Exception as exc:  # noqa: BLE001
        errors["city_name_required"] = _norm_err(exc)

    # empty KNN_Weight cell -> NaN -> verbatim accessibility ValueError
    # (rewrite pairing.csv with an empty cell first)
    _write_pairings(data0, [_knn_row(KNN_Weight="")], name="pairing.csv")
    try:
        wf.KNN_accessibility(city_name="suite_city",
                             data_folder=str(data0),
                             output_folder=str(out0 / "nan_kw"),
                             num_cores=1)
    except Exception as exc:  # noqa: BLE001
        errors["nan_knn_weight"] = _norm_err(exc)

    # the real run
    data, out = _fresh("knn_defaults")
    _write_base_layers(data)
    _write_pairings(data, [
        _knn_row(Flow_Name="knn_a", Radius=250),
        _knn_row(Flow_Name="knn_b", Radius=200),
    ], name="pairing.csv")
    err = _run_knn(wf, data, out, num_cores=1)
    rec = out / "origin_record.csv"
    extra = {"workflow_error": err, "guard_probes": errors}
    if rec.is_file():
        df = pd.read_csv(rec)
        extra["origin_columns"] = [str(c) for c in df.columns]
        extra["totals_present"] = sorted(
            c for c in ("total_knn_access", "normalized_knn_access")
            if c in df.columns)

    # PINNED QUIRK: with Destination_Weight="Count" the workflow's
    # insert_node gets weight_attribute=None -> every destination NODE
    # weight is 1.0, so the hardcoded alpha is bit-inert through this
    # profile (1**a == 1**1 bitwise for every a) — the alpha term only
    # engages when the pairing forwards a real weight attribute (pinned
    # selecting in wf_knn_weights).  Sensitivity pin over the
    # DETERMINISTIC files only (html bytes are nondeterministic every
    # run): an alpha=999 patched clone must be bitwise-equal here.
    alpha_out = out.parent / "out_alpha999"
    clone = load_patched_workflows(
        _ARM(), [("            alpha=1,\n", "            alpha=999,\n")])
    extra["alpha_sensitivity_error"] = _run_knn(clone, data, alpha_out,
                                                num_cores=1)
    if rec.is_file() and (alpha_out / "origin_record.csv").is_file():
        extra["alpha_999_deterministic_outputs_equal"] = all(
            _md5(p) == _md5(alpha_out / p.relative_to(out))
            for p in sorted(out.rglob("*"))
            if p.is_file() and not p.suffix == ".html"
            and p.name != "time_log.csv")
    return _state(data, out, extra=extra)


def _knn_row(**over):
    row = {
        "Flow_Name": "knn_a",
        "Origin_Name": "origins", "Origin_File": "origins.geojson",
        "Origin_Weight": "Count",
        "Destination_Name": "destinations",
        "Destination_File": "destinations.geojson",
        "Destination_Weight": "Count",
        "Network_File": "streets.geojson", "Network_Cost": "Geometric",
        "Turn_Penalty": 0, "Turn_Threshold": 45, "Turns": False,
        "Radius": 250, "Beta": 0.003,
        "KNN_Weight": "[0.5,0.5]", "Plateau": 0,
    }
    row.update(over)
    return row


def scenario_wf_knn_weights():
    """Destination_Weight='weight' (a real layer attribute, not
    'Count'): the weight attribute is forwarded through insert_node, so
    the hardcoded gravity alpha ENGAGES (weights [2.0, 3.0] -> 2^a /
    3^a terms) and the Origin_Weight='Count' control keeps origin node
    weights uniform.  Target profile of the dossier knn_alpha_2
    mutant."""
    wf = _WF()
    data, out = _fresh("knn_weights")
    _write_base_layers(data)
    _write_pairings(data, [
        _knn_row(Flow_Name="wgt_a", Radius=250,
                 Destination_Weight="weight"),
        _knn_row(Flow_Name="wgt_b", Radius=200,
                 Destination_Weight="weight"),
    ], name="pairing.csv")
    err = _run_knn(wf, data, out, num_cores=1)
    rec = out / "origin_record.csv"
    extra = {"workflow_error": err}
    if rec.is_file():
        df = pd.read_csv(rec)
        extra["origin_columns"] = [str(c) for c in df.columns]
    return _state(data, out, extra=extra)


def scenario_wf_knn_changes():
    """Per-row input changes on the KNN profile: row 2 changes
    Network_File / Network_Cost / Turns.  PINNED CRASH — the
    Network_File change branch calls load_layer on the ALREADY-LOADED
    'streets' label and the workflow dies with
    ``KeyError: 'Layer with label streets is already in Zonal object'``
    BEFORE any row-2 work: the KNN profile's per-row network-file
    change path is broken upstream.  Row 1's per-pairing
    origin_record.csv exists; every final output (totals, geoJSON,
    time_log) is unreachable.  Digest = the verbatim error + the
    partial tree."""
    wf = _WF()
    data, out = _fresh("knn_changes")
    _write_base_layers(data, second_streets=True)
    _write_pairings(data, [
        _knn_row(Flow_Name="knn_geo", Radius=250),
        _knn_row(Flow_Name="knn_len", Radius=220,
                 Network_File="streets2.geojson", Network_Cost="length",
                 Turns=True, Turn_Penalty=30, Turn_Threshold=60),
    ], name="pairing.csv")
    err = _run_knn(wf, data, out, num_cores=1)
    rec = out / "origin_record.csv"
    extra = {"workflow_error": err}
    if rec.is_file():
        df = pd.read_csv(rec)
        extra["origin_columns"] = [str(c) for c in df.columns]
    return _state(data, out, extra=extra)


def scenario_wf_knn_zero_reach():
    """No reachable destination (Radius=10 below every network path):
    PINNED CRASH — the workflow dies with ``KeyError: 'reach'`` BEFORE
    any output file is written (the empty-output dossier combination
    manifests as this exception; the origin-record normalization code
    is never reached).  Digest = the verbatim error + the EMPTY output
    tree, bitwise across arms."""
    wf = _WF()
    data, out = _fresh("knn_zero_reach")
    _write_base_layers(data)
    _write_pairings(data, [_knn_row(Flow_Name="knn_zero", Radius=10)],
                    name="pairing.csv")
    err = _run_knn(wf, data, out, num_cores=1)
    rec = out / "origin_record.csv"
    extra = {"workflow_error": err}
    if rec.is_file():
        df = pd.read_csv(rec)
        cols = [c for c in ("knn_zero_reach", "knn_zero_gravity",
                            "knn_zero_knn_access", "total_knn_access",
                            "normalized_knn_access") if c in df.columns]
        extra["metric_columns_present"] = cols
        extra["metric_values"] = {
            c: [_jsonable(v) for v in df[c].tolist()] for c in cols}
    return _state(data, out, extra=extra)


def _ARM():
    arm = _SABOTAGE.get("arm")
    if arm is None:
        raise RuntimeError("arm not set")
    return arm


def _WF():
    """The arm's workflows module (the patched clone when a source
    sabotage is active — main() owns loading; scenarios never
    re-import, which would silently drop the patch)."""
    wf = _ARM_MARKER.get("workflows")
    if wf is None:
        raise RuntimeError("arm module not loaded")
    return wf


# ----------------------------------------------------------------------
# sabotage registry
# ----------------------------------------------------------------------

SABOTAGE_REPLACEMENTS = {
    # Dossier mutant: swap alpha=1 with alpha=2 (the KNN workflow's
    # hardcoded gravity alpha, upstream workflows.py:527; the adjacent
    # destination_weight=None is :526).  Inert on
    # wf_knn_defaults and SELECTED on wf_knn_weights — the profile
    # split comes from insert_node's weight_attribute (upstream
    # :381/:510): Destination_Weight='Count' -> weight_attribute=None
    # -> unit destination NODE weights (1**a == 1**1, alpha bit-inert),
    # while a real weight attribute gives the gravity term non-unit
    # weights and alpha engages.  accessibility's destination_weight
    # parameter is hardcoded None upstream in BOTH profiles (gravity
    # destination weights, a different knob than the node weights).
    "knn_alpha_2": [("            alpha=1,\n",
                     "            alpha=2,\n")],
    # CSV naming contract: the reach column must arrive as
    # "<Flow_Name>_reach" on the origin record; dropping the suffix
    # must be selected by the comparator.
    "knn_reach_suffix_drop": [(
        "save_reach_as=pairing['Flow_Name']+\"_reach\", ",
        "save_reach_as=pairing['Flow_Name'], ")],
    # Dossier mutant: reverse the closest-facility direction flag at
    # the KNN workflow call site.
    "knn_closest_facility_true": [(
        "            closest_facility = False,\n",
        "            closest_facility = True,\n")],
    # Workflow-specific mutant: drop the elastic decay suppression so
    # elastic rows are decayed again (double decay).
    "flow_elastic_decay_always": [(
        "decay=False if pairing['Elastic_Weights'] else pairing['Decay'],",
        "decay=pairing['Decay'],")],
}

DIGEST_SABOTAGES = {
    # Dossier mutant: omit the exposure column from the recorded
    # streets digest — a comparator that only checks presence/rows
    # would pass a broken surface.
    "omit_exposure_column": "wf_flow_exposure_row",
}


def _apply_digest_sabotage(state):
    name = _SABOTAGE["name"]
    if name == "omit_exposure_column":
        rec = state.get("record_columns")
        if rec:
            state["record_columns"] = [
                c for c in rec if "exposure" not in c.lower()]
        if "exposure_column_on_record" in state:
            state["exposure_column_on_record"] = []
        if "origin_hazzard_columns" in state:
            state["origin_hazzard_columns"] = []
        return state
    raise ValueError(f"unknown digest sabotage {name}")


# ----------------------------------------------------------------------
# main
# ----------------------------------------------------------------------

SCENARIOS = {
    "wf_flow_defaults": scenario_wf_flow_defaults,
    "wf_flow_huff_elastic": scenario_wf_flow_huff_elastic,
    "wf_flow_two_pairings": scenario_wf_flow_two_pairings,
    "wf_flow_cost_change": scenario_wf_flow_cost_change,
    "wf_flow_exposure_row": scenario_wf_flow_exposure_row,
    "wf_flow_nc3_structural": scenario_wf_flow_nc3_structural,
    "wf_flow_errors": scenario_wf_flow_errors,
    "wf_knn_defaults": scenario_wf_knn_defaults,
    "wf_knn_weights": scenario_wf_knn_weights,
    "wf_knn_changes": scenario_wf_knn_changes,
    "wf_knn_zero_reach": scenario_wf_knn_zero_reach,
}

SCENARIO_NAMES = list(SCENARIOS)

# which scenario each sabotage engages (single source of truth for the
# test-side selection/integrity matrix)
SABOTAGE_SCENARIOS = {
    "knn_alpha_2": "wf_knn_weights",
    "knn_reach_suffix_drop": "wf_knn_defaults",
    "knn_closest_facility_true": "wf_knn_defaults",
    "flow_elastic_decay_always": "wf_flow_huff_elastic",
    "omit_exposure_column": "wf_flow_exposure_row",  # digest-level
}


def main():
    global _SCENARIO_TMP
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["facade", "reference"], required=True)
    ap.add_argument("--scenario", required=True, choices=SCENARIO_NAMES)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=20260930)
    ap.add_argument("--sabotage", default=None)
    ap.add_argument("--scratch", default=None,
                    help="per-arm scratch root for fixtures/outputs")
    args = ap.parse_args()

    np.random.seed(args.seed)
    _SABOTAGE["name"] = args.sabotage
    _SABOTAGE["arm"] = args.arm

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        if args.scratch:
            _SCENARIO_TMP = Path(args.scratch).resolve()
            _SCENARIO_TMP.mkdir(parents=True, exist_ok=True)
        else:
            _SCENARIO_TMP = Path(out_path.resolve().parent / "scratch")

        # arm loaded FIRST for every run (clean or mutant), so the
        # digests always record a fully initialized arm module
        import_arm(args.arm)
        if args.sabotage in SABOTAGE_REPLACEMENTS:
            _ARM_MARKER["workflows"] = load_patched_workflows(
                args.arm, SABOTAGE_REPLACEMENTS[args.sabotage])
        state = SCENARIOS[args.scenario]()
        if args.sabotage in DIGEST_SABOTAGES:
            state = _apply_digest_sabotage(state)
        state["sabotage"] = args.sabotage
        state["arm"] = args.arm
        state["scenario"] = args.scenario
    except Exception:
        state = {
            "arm": args.arm,
            "scenario": args.scenario,
            "sabotage": args.sabotage,
            "scenario_error": traceback.format_exc(),
        }
    out_path.write_text(json.dumps(state, indent=1, default=str))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
