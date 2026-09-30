"""Expand API_PARITY.json seed into evidence/api/api_matrix.json.

AST-driven census of the pinned Madina tree and the UNA baseline tree. Every
public module-level callable/class and public class method/attribute becomes one
record with its exact signature and defaults. The 18 seed families (M01..M18)
carry semantic annotations transcribed from the pinned source reading recorded
in INVENTORY's evidence notes. Run from repository root:

    python campaigns/una_platform/evidence/api/generate_api_matrix.py
"""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
MADINA = REPO / ".refs" / "madina" / "src" / "madina"
UNA = REPO / "src" / "urban_network_analysis"

SKIP_PARTS = {".git", "__pycache__", ".venv", "venv"}


def _sig(node: ast.FunctionDef | ast.AsyncFunctionDef) -> dict:
    a = node.args
    defaults = [None] * (len(a.args) - len(a.defaults)) + [ast.unparse(d) for d in a.defaults]
    params = []
    for arg, default in zip(a.args, defaults):
        params.append({
            "name": arg.arg,
            "annotation": ast.unparse(arg.annotation) if arg.annotation else None,
            "default": default,
        })
    for i, arg in enumerate(a.kwonlyargs):
        d = a.kw_defaults[i]
        params.append({
            "name": arg.arg, "kind": "keyword_only",
            "annotation": ast.unparse(arg.annotation) if arg.annotation else None,
            "default": ast.unparse(d) if d else None,
        })
    if a.vararg:
        params.append({"name": a.vararg.arg, "kind": "var_positional"})
    if a.kwarg:
        params.append({"name": a.kwarg.arg, "kind": "var_keyword"})
    return {
        "signature": "(" + ", ".join(
            p["name"] + ("" if p.get("default") is None else "=" + p["default"]) for p in params
        ) + ")",
        "parameters": params,
        "returns": ast.unparse(node.returns) if node.returns else None,
        "decorators": [ast.unparse(d) for d in node.decorator_list],
    }


def census(root: Path, prefix: str) -> list[dict]:
    records = []
    for path in sorted(root.rglob("*.py")):
        if any(p in SKIP_PARTS for p in path.parts):
            continue
        raw = path.read_bytes()
        tree = ast.parse(raw, filename=str(path))
        rel = path.relative_to(root).as_posix()
        module = prefix + "." + rel[:-3].replace("/", ".")

        def walk(nodes, owner: str | None, in_class: bool):
            for node in nodes:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    dunder = node.name.startswith("__") and node.name.endswith("__")
                    if node.name.startswith("_") and not dunder:
                        continue
                    rec = {
                        "symbol": (owner + "." if owner else module + ".") + node.name,
                        "kind": "method" if in_class else "function",
                        "source": rel,
                        "line": node.lineno,
                        **_sig(node),
                    }
                    if dunder:
                        rec["kind"] = "dunder_method"
                    body_src = ast.unparse(node)
                    if "NotImplementedError" in body_src:
                        rec["unsupported_stub"] = "raises NotImplementedError"
                    if "raise NotImplementedError" in body_src:
                        rec["unsupported_stub"] = "raises NotImplementedError"
                    records.append(rec)
                elif isinstance(node, ast.ClassDef) and not node.name.startswith("_"):
                    cls = module + "." + node.name
                    attrs = []
                    for sub in node.body:
                        if isinstance(sub, ast.Assign):
                            for t in sub.targets:
                                if isinstance(t, ast.Name) and not t.id.startswith("_"):
                                    attrs.append({"name": t.id, "value": ast.unparse(sub.value)})
                        elif isinstance(sub, ast.AnnAssign) and isinstance(sub.target, ast.Name):
                            if not sub.target.id.startswith("_"):
                                attrs.append({"name": sub.target.id,
                                              "value": ast.unparse(sub.value) if sub.value else None})
                    records.append({"symbol": cls, "kind": "class", "source": rel,
                                    "line": node.lineno,
                                    "bases": [ast.unparse(b) for b in node.bases],
                                    "class_attributes": attrs})
                    walk(node.body, cls, True)
        walk(tree.body, None, False)
    return records


# Semantic annotations transcribed from the pinned-source reading (see
# evidence/api/semantic_notes.md for the full derivation per row).
SEED_ANNOTATIONS = {
    "M01": {
        "behavior": ["Zonal() starts network=None, geo_center=(None,None), empty Layers",
                     "__getitem__ delegates to Layers (str label or int position)",
                     "class constants DEFAULT_PROJECTED_CRS='EPSG:3857', DEFAULT_GEOGRAPHIC_CRS='EPSG:4326', DEFAULT_COLORS reexport",
                     "VERSION='0.0.15', RELEASE_DATE='2023-02-16' module constants"],
        "mutation": ["layers", "network", "geo_center mutated by load_layer/create_street_network/insert_node"],
    },
    "M02": {
        "behavior": ["TypeError when name not str; TypeError when source not Path|str|GeoDataFrame",
                     "GeoDataFrame sources are deep-copied; files read with engine='pyogrio'",
                     "gdf gets 'id' column = range(rows), set as index; original crs captured before prepare_geometry",
                     "prepare_geometry: Polygon->boundary, all-MultiLineString->first geom, Z dropped via transform(_to_2d) where _to_2d=tuple(filter(None,[x,y])) -- drops zero coordinates (quirk)",
                     "Layer default_color = [round(random.random()*255)]*3 (unseeded RNG)",
                     "geo_center set from first loaded layer's EPSG:4326 unary_union centroid",
                     "insertion respects pos/first/before/after via Layers.add"],
    },
    "M03": {
        "behavior": ["actual default redundant_edge_treatment='discard' (docstring text says 'split'; source default wins per CONTRACTS)",
                     "validations: source_layer str+exists, weight_attribute str+in columns, tolerance numeric >=0, treatment in keep|discard|split",
                     "tolerance==0 vectorized path; tolerance>0 tolerance_network_nodes_edges (sorts by weight desc, adds 'weight' column to the source layer gdf, snapped/snapping_distance columns on edges)",
                     "weight rule: geometric length if weight_attribute None else max(attr,0.01), but attr==0 falls back to geometric length",
                     "discard: shortest duplicate kept; self-loop edges always dropped (start==end continue)",
                     "split: shortest duplicate kept intact, others split at normalized midpoint, weight/2 halves, new node ids from max+1, 'streets' hardcoded as split-node source_layer",
                     "edge 'type' column dropped after coloring; Network(None,None turn params) then set_turn_parameters called"],
    },
    "M04": {
        "behavior": ["threshold numeric 0..180 else TypeError/ValueError; penalty numeric >=0",
                     "writes network.turn_threshold_degree / network.turn_penalty_amount",
                     "crashes with AttributeError if called before create_street_network (network None)"],
    },
    "M05": {
        "behavior": ["label must be 'origin'|'destination' else TypeError/ValueError",
                     "efficient_node_insertion: sindex.nearest edge match, centroid projected onto edge, weight_to_start=weight*(dist/len), weight_to_end=weight-weight_to_start",
                     "node weight column dtype float32 (fractional weights truncated toward float32)",
                     "missing node columns initialized to 0 before concat",
                     "network.nodes recolored categorically after every insertion (random colors for unseen types)"],
    },
    "M06": {
        "behavior": ["Zonal defaults (True,True,False) forwarded to Network.create_graph which has different signature defaults (False,True,False)",
                     "light_graph: nx.Graph of street edges weight=max(w,0), edge attr id, node attr type",
                     "d_graph: light_graph.copy() + destination insertion via update_light_graph",
                     "od_graph: light_graph.copy() + origin+destination insertion",
                     "street_node_ids set refreshed on each call"],
    },
    "M07": {
        "behavior": ["network.nodes filtered to type=='street_node' (origins and destinations removed)",
                     "network/edges/graphs retained; d_graph/od_graph NOT rebuilt -> stale graphs keep old od nodes (lifecycle quirk)"],
    },
    "M08": {
        "behavior": ["layer_list None -> all visible layers colored with their default_color",
                     "'layer' key in dicts -> gdf copied from that layer and colored",
                     "create_deckGL_map: per layer deep copy, reset_index, to EPSG:4326, radius/width/opacity/color overrides, GeoJsonLayer; 'text' adds TextLayer",
                     "returns pdk.Deck; save_as writes HTML with css_background_color='cornflowerblue'",
                     "requires pydeck; geo_center (None,None) yields invalid ViewState lat/lon"],
    },
    "M09": {
        "behavior": ["prints table (Layer name | Visible | projection | rows | File path)",
                     "'No zonal_layers yet...' when empty",
                     "geo_center None check compares tuple to None (always False; prints center line with values) (quirk)",
                     "network None prints setup instructions"],
    },
    "M10": {
        "behavior": ["Layers: label list + dict; getitem str/int; setitem replaces; str form position:N reverse order; iteration yields LABELS (strings) not Layer objects",
                     "add/insert_at with duplicate-label KeyError, int-typed pos checks (bool is int)",
                     "pos_at_label raises ValueError for missing label (list.index)",
                     "Layer: gdf,label,show,crs,file_path,default_color,other_fields=kwargs",
                     "Layer.set_style/color_layer reference self.default_colors which Layer never defines -> AttributeError on call (broken path; registered correction required, not silent reinvention)"],
    },
    "M11": {
        "behavior": ["Network(nodes,edges,turn_threshold_degree,turn_penalty_amount,weight_attribute,edge_source_layer)",
                     "empty nodes/edges: 'pass # throw Error here' -- no actual validation",
                     "set_node_value -> nodes.at; create_graph (False,True,False defaults); visualize_graph raises NotImplementedError (documented unsupported)",
                     "add_node_to_graph: broad try/except print-and-continue; epsilon perturbation 1e-7*(weight_sec+1); chain rebuild sorted by weight_to_end",
                     "remove_node_to_graph: degree-2 rejoin; self-loop edge restore; silently returns for other degree!=2",
                     "update_light_graph: mutates add_nodes during iteration (RuntimeError risk), flat 1e-7 epsilon (differs from add_node_to_graph), prints on anomalies",
                     "_get_nodes_at_geometric_distance/_get_nodes_at_network_distance/_get_nodes_at_bf_distance/scan_for_intersections/fuse_degree_2_nodes raise NotImplementedError (documented unsupported)",
                     "network_to_layer returns two Layer objects (breaks: Layer requires 6 args) (broken path)"],
    },
    "M12": {
        "behavior": ["origin_layer/destination_layer read via .iloc[0] BEFORE validate_zonal_ready -> IndexError instead of ValueError on empty zonal (quirk)",
                     "search_radius numeric >=0; destination_weight str in destination layer columns",
                     "save_gravity_as requires beta; knn path requires knn_weights; knn_weights str '[0.5, 0.25]' parsed by [1:-1].split(','); knn_plateau None->search_radius, 0<=plateau<=radius, plateau requires beta",
                     "closest_facility=True requires BOTH save_closest_facility_as and save_closest_facility_distance_as strings",
                     "num_cores default 1; results written into origin layer gdf via source_id join; reach/gravity fillna(0); closest_facility columns joined to DESTINATION layer without fillna (unreachable -> NaN preserved)",
                     "returns None",
                     "closest_facility assigns each DESTINATION its closest ORIGIN (direction fixed by dossier D01); parallel tie-break: first-come wins; single-core tie-break: dict iteration over distance-sorted d_idxs",
                     "reach = sum of assigned-destination weights; gravity = sum(w^alpha / e^(beta*dist)) over assigned destinations when closest_facility"],
    },
    "M13": {
        "behavior": ["returns (destinations, network_edges, scope_gdf)",
                     "origin_ids None->all origins; int->source_id match; list validated against origin layer index then mapped via source_id",
                     "per origin: add origin to d_graph, turn_o_scope detour 1.0, remove origin",
                     "empty d_idxs -> origin skipped entirely (no scope rows)",
                     "destination_scope = reachable destination geometries unary_union convex_hull; network_scope = o_scope node geometries hull; scope = union",
                     "network_edges = edge_gdf.clip(scope) per origin, unary_union, exploded; Issue #7: scope can be GeometryCollection -> clip failure in shapely 2.x",
                     "scope_gdf rows alternate 'service area border'/'origin' with origin_id",
                     "crs taken from destination layer"],
    },
    "M14": {
        "behavior": ["ALL admissible alternatives within detour_ratio of shortest path (NOT K-alternatives)",
                     "path_generator: add origin to d_graph, turn_o_scope(return_paths=True), bfs_subgraph_generation, bfs_path_edges_many_targets_iterative, remove origin",
                     "0.00001 tolerance on path-vs-shortest comparisons",
                     "paths recorded as node sequences; output geometry = GeometryCollection([origin whole edge] + interior edge geometries + [destination whole edge])",
                     "rows: destination (source_id), distance (path weight incl turn costs), geometry; sorted by distance, reset_index; tie order from enumeration (set-iteration dependent)",
                     "origin_id typed int|list in error text but only single int supported (iloc[0])",
                     "interior edge ids exclude first segment and destination segment during enumeration"],
    },
    "M15": {
        "behavior": ["all-path assignment via paralell_betweenness_exposure with path_detour_penalty hardcoded 'equal', destniation_cap=None",
                     "edge_gdf['betweenness']=0.0 then per-origin accumulation",
                     "closest_destination=True default: single min-distance destination, destination_probability=1",
                     "Huff mode: gravities=weight/e^(beta*dist), probabilities normalized; all-zero gravities -> origin skipped entirely, and (observed in source) origin not removed from d_graph (graph-poisoning candidate; must be captured by executed reference, not assumed)",
                     "origins.sample(frac=1) unseeded shuffle before queueing (order affects accumulation order; fixtures must seed RNG)",
                     "d_path_weights < 0.01 raised to 0.01",
                     "decay exponent uses min shortest-distance over the origin's d_idxs (1/e^(beta*min)), power uses per-path 1/w^2",
                     "diagnostics columns written per origin AFTER the stats block; with closest_destination=True the stats block raises NameError on eligible_destinations_shortest_distance -> caught -> later columns and graph removal skipped (candidate legacy behavior; verify by execution)",
                     "save_betweenness_as joins edge betweenness to source layer by parent_street_id drop_duplicates; index cast to int",
                     "keep_diagnostics saves every non-saved origin_gdf column as save_betweenness_as+'_'+column",
                     "num_cores>1: ProcessPoolExecutor of betweenness_exposure workers fed by Manager queue; as_completed merge order nondeterministic",
                     "path_exposure: mean exposure over interior path edges weighted by segment weight; expected_hazzard_meters accumulated per origin"],
    },
    "M16": {
        "behavior": ["ValueError when city_name/data_folder/output_folder all None",
                     "defaults data_folder='Cities/<city>/Data'; output 'Cities/<city>/Simulations/<YYYY-MM-DD HH-MM>'",
                     "pairings.csv columns: Flow_Name, Network_File, Origin_Name, Origin_File, Origin_Weight, Destination_Name, Destination_File, Destination_Weight, Network_Cost, Radius, Detour, Decay, Decay_Mode, Beta, Closest_destination, Elastic_Weights, KNN_Weight, Plateau, Turns, Turn_Penalty, Turn_Threshold",
                     "network rebuilt when Network_Cost changes; else nodes reset from clean copy",
                     "'Count' weight -> None (unit weights); snap tolerance 0.00001",
                     "num_cores=min(origin rows, num_cores)",
                     "decay disabled when Elastic_Weights true",
                     "keep_diagnostics=True always; outputs per pairing folder + time_log.csv + betweenness_record.geoJSON/csv",
                     "Logger prints cumulative/seconds-elapsed table; timestamps nondeterministic (test format/order under controlled clock)"],
    },
    "M17": {
        "behavior": ["ValueError when city_name None; default pairings_file='pairing.csv' (singular; distinct from M16)",
                     "streets layer reloaded when Network_File changes; network rebuilt when cost/file changes; else clear_nodes()",
                     "turn parameters set only when pairing['Turns'] truthy",
                     "accessibility with destination_weight=None, alpha=1, saves <Flow_Name>_reach/_gravity/_knn_access",
                     "final: total_knn_access = row sum of all <Flow_Name>_knn_access; normalized_knn_access min-max; origin_record.geoJSON + csv; assumes single origin layer across pairings (last pairing's Origin_Name)"],
    },
    "M18": {
        "behavior": ["madina/__init__: from .zonal import *; from .una import *",
                     "zonal/__init__: zonal,layer,network,network_utils star imports",
                     "una/__init__: betweenness, paths star imports (tools and workflows NOT reexported; imported as madina.una.tools / madina.una.workflows)",
                     "network_utils publics: node_edge_builder, tolerance_network_nodes_edges, _tag_edges is private, efficient_node_insertion, GeoPandaExtractor, vectorized_node_edge_builder",
                     "utils publics (not star-reexported from madina): prepare_geometry, color_gdf, create_deckGL_map, color_layer(module-level fn taking self), DEFAULT_COLORS",
                     "tools publics: validate_zonal_ready, accessibility, service_area, alternative_paths, betweenness",
                     "paths publics: path_generator, wandering_messenger, bfs_* family, turn_o_scope, turn_penalty_value, angle_deviation_between_two_lines",
                     "betweenness publics: parallel_betweenness, one_betweenness_2, betweenness_exposure, paralell_betweenness_exposure, parallel_access, one_access, get_origin_properties, clockwiseangle_and_distance",
                     "Layer/Layers/Network/Zonal/DEFAULT_COLORS reachable via madina.zonal and madina top-level"],
    },
}


def main() -> None:
    seed = json.loads((REPO / "campaigns" / "una_platform" / "API_PARITY.json").read_text())
    madina_records = census(MADINA, "madina")
    una_records = census(UNA, "urban_network_analysis")

    # group madina records under seed families by symbol prefix mapping
    family_of = []
    for rec in madina_records:
        sym = rec["symbol"]
        if sym.startswith("madina.zonal.Zonal"):
            fam = "M01"
            if sym.startswith("madina.zonal.Zonal.load_layer"):
                fam = "M02"
            elif sym.startswith("madina.zonal.Zonal.create_street_network"):
                fam = "M03"
            elif sym.startswith("madina.zonal.Zonal.set_turn_parameters"):
                fam = "M04"
            elif sym.startswith("madina.zonal.Zonal.insert_node"):
                fam = "M05"
            elif sym.startswith("madina.zonal.Zonal.create_graph"):
                fam = "M06"
            elif sym.startswith("madina.zonal.Zonal.clear_nodes"):
                fam = "M07"
            elif sym.startswith("madina.zonal.Zonal.create_map"):
                fam = "M08"
            elif sym.startswith("madina.zonal.Zonal.describe"):
                fam = "M09"
        elif sym.startswith("madina.zonal.Layer") or sym.startswith("madina.zonal.Layers"):
            fam = "M10"
        elif sym.startswith("madina.zonal.Network"):
            fam = "M11"
        elif sym.startswith("madina.una.tools.accessibility"):
            fam = "M12"
        elif sym.startswith("madina.una.tools.service_area"):
            fam = "M13"
        elif sym.startswith("madina.una.tools.alternative_paths"):
            fam = "M14"
        elif sym.startswith("madina.una.tools.betweenness"):
            fam = "M15"
        elif sym.startswith("madina.una.workflows.betweenness_flow_simulation"):
            fam = "M16"
        elif sym.startswith("madina.una.workflows.KNN_accessibility"):
            fam = "M17"
        elif sym.startswith("madina.una.tools.validate_zonal_ready"):
            fam = "M18"
        elif sym.startswith("madina.una."):
            fam = "M18"
        elif sym.startswith("madina.zonal."):
            fam = "M18"
        else:
            fam = "M18"
        family_of.append(fam)

    # attach family + annotations
    fam_symbols: dict[str, list[str]] = {}
    for rec, fam in zip(madina_records, family_of):
        rec["seed_family"] = fam
        ann = SEED_ANNOTATIONS.get(fam)
        if ann:
            for key in ("behavior", "mutation"):
                if key in ann:
                    rec.setdefault("semantic_notes", {})
                    # family-level notes go on the family anchor record only
        fam_symbols.setdefault(fam, []).append(rec["symbol"])

    matrix = {
        "schema_version": 1,
        "generated_from": {
            "madina": {"commit": "8b5c3bd3b1c0048ae8b04054daed92bfb201b9d6",
                        "path": ".refs/madina (read-only clone)",
                        "public_records": len(madina_records)},
            "una_baseline": {"commit": "16bf404d2cc5e776a78dc073c80405f6d186da29",
                              "public_records": len(una_records)},
        },
        "seed": {"file": "campaigns/una_platform/API_PARITY.json", "rows": len(seed["rows"]),
                  "note": "seed file untouched; this matrix is the expansion"},
        "required_checks": seed["required_checks"],
        "families": [],
        "madina_symbols": madina_records,
        "una_baseline_symbols": una_records,
    }

    for row in seed["rows"]:
        fam = row["id"]
        entry = {
            "id": fam,
            "symbol": row["symbol"],
            "signature_seed": row["signature_seed"],
            "required_behavior": row["required_behavior"],
            "status": row["status"],
            "expanded_symbols": fam_symbols.get(fam, []),
            "semantic_notes": SEED_ANNOTATIONS.get(fam, {}),
            "tests": [],
        }
        matrix["families"].append(entry)

    out = REPO / "campaigns" / "una_platform" / "evidence" / "api" / "api_matrix.json"
    out.write_text(json.dumps(matrix, indent=2))
    digest = hashlib.sha256(out.read_bytes()).hexdigest()
    print(json.dumps({
        "wrote": str(out.relative_to(REPO)),
        "sha256": digest,
        "madina_public_records": len(madina_records),
        "una_public_records": len(una_records),
        "unsupported_stubs": [r["symbol"] for r in madina_records if r.get("unsupported_stub")],
    }, indent=2))


if __name__ == "__main__":
    main()
