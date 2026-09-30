"""Incident payload: build the tiny pinned-Madina project and enter the
reported stage (dossier 03 reproduction matrix).

Actions (params["action"]):
  build_only     load_layer x3 + create_street_network
  service_area   + insert_node(origin/destination) + create_graph +
                 una.tools.service_area          (issue #7)
  btn_exposure   + create_graph + betweenness_exposure(closest_destination=
                 True/False via params)          (BTN stats NameError + graph
                                                  poisoning)
  btn_parallel   + create_graph + parallel_betweenness(num_cores=2)
                                                  (issue #12b array_split)

Every stage transition is recorded in `state` (heartbeat phase + units)
and the returned dict carries the per-stage evidence the envelope needs.
"""
import io
import json
import sys
import traceback
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))


def payload(params, state):
    from madina_bridge import build_project, ensure_fixture, \
        ensure_fixture_redundant

    madina_src = params["madina_src"]
    interventions = params.get("interventions") or []
    tolerance = params.get("node_snapping_tolerance", 0.0)
    streets_file = None
    dataset = ensure_fixture()
    if params.get("fixture") == "redundant":
        redundant = ensure_fixture_redundant()
        streets_file = redundant["file"]
        # provenance: the envelope must carry the ACTUAL streets file used,
        # not only the base fixture
        dataset["fixture_variant"] = "redundant"
        dataset["streets_redundant_sha256"] = redundant["sha256"]
    state["phase"] = "build_project"
    z, info = build_project(madina_src, interventions, tolerance,
                            streets_file=streets_file,
                            redundant_edge_treatment=params.get(
                                "redundant_edge_treatment"))
    out = {"dataset": dataset, "build": info}

    action = params.get("action", "build_only")
    state["units_done"] = 1
    if action == "build_only":
        return out

    state["phase"] = "insert_nodes"
    z.insert_node(layer_name="origins", label="origin", weight_attribute=None)
    z.insert_node(layer_name="destinations", label="destination",
                  weight_attribute=None)
    state["units_done"] = 2
    state["phase"] = "create_graph"
    z.create_graph()
    state["units_done"] = 3

    if action == "service_area":
        from madina.una.tools import service_area
        radius = params.get("search_radius", 1000)
        state["phase"] = "service_area"
        d_graph_before = z.network.d_graph.number_of_nodes() \
            if hasattr(z.network, "d_graph") and z.network.d_graph else None
        try:
            result = service_area(z, search_radius=radius)
            out["service_area"] = {
                "entered": True,
                "scopes": int(len(result[2])) if result is not None else None,
                "d_graph_nodes_before": d_graph_before,
            }
        except Exception as exc:
            out["service_area"] = {
                "entered": True,
                "raised": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
        return out

    if action == "btn_exposure":
        # Replicates paralell_betweenness_exposure's worker feeding
        # (manager queue + betweenness_exposure) but IN-PROCESS, so the
        # per-origin stats handler's d_graph mutations are directly
        # observable (the real driver ships pickled copies to workers,
        # hiding worker-side poisoning).
        import multiprocessing as mp
        from madina.una.betweenness import betweenness_exposure
        closest = bool(params.get("closest_destination", True))
        d_graph = z.network.d_graph
        nodes_before = d_graph.number_of_nodes()
        state["phase"] = "btn_exposure"
        buf = io.StringIO()
        errbuf = io.StringIO()
        raised = None
        tb = None
        with mp.Manager() as manager:
            origin_queue = manager.Queue()
            origin_ids = [int(i) for i in
                          z.network.nodes[z.network.nodes["type"] == "origin"]
                          .index]
            for o in origin_ids:
                origin_queue.put(o)
            origin_queue.put("done")
            try:
                with redirect_stdout(buf), redirect_stderr(errbuf):
                    betweenness_exposure(
                        z, core_index=0, origin_queue=origin_queue,
                        search_radius=params.get("search_radius", 1000),
                        closest_destination=closest)
            except Exception as exc:
                raised = f"{type(exc).__name__}: {exc}"
                tb = traceback.format_exc()
        err_lines = [ln.strip()[:400] for ln in buf.getvalue().splitlines()
                     if "error" in ln.lower() or "UnboundLocalError" in ln]
        nodes_after = d_graph.number_of_nodes()
        inserted = z.network.nodes[z.network.nodes["type"].isin(
            ["origin", "destination"])].index.tolist()
        leaked = [int(n) for n in inserted if n in d_graph.nodes]
        origin_ids = [int(i) for i in
                      z.network.nodes[z.network.nodes["type"] == "origin"]
                      .index]
        dest_ids = [int(i) for i in
                    z.network.nodes[z.network.nodes["type"] == "destination"]
                    .index]
        out["btn"] = {
            "origin_nodes_still_in_d_graph": [n for n in origin_ids
                                              if n in d_graph.nodes],
            "destination_nodes_still_in_d_graph": [n for n in dest_ids
                                                   if n in d_graph.nodes],
            "raised": raised,
            "traceback": tb,
            "stdout_tail": buf.getvalue()[-4000:],
            "stderr_tail": errbuf.getvalue()[-4000:],
            "per_origin_error_lines": err_lines,
            "d_graph_nodes_before": int(nodes_before),
            "d_graph_nodes_after": int(nodes_after),
            "od_node_ids": [int(n) for n in inserted],
            "od_nodes_still_in_d_graph": leaked,
            "stats_columns": {
                col: (None if col not in z.network.nodes else
                      (None if z.network.nodes[col].isna().all()
                       else "populated"))
                for col in ("reach", "gravity", "closest_destination_distance",
                            "furthest_destination_distance")},
        }
        return out

    if action == "btn_parallel":
        from madina.una.betweenness import parallel_betweenness
        state["phase"] = "btn_parallel"
        try:
            with redirect_stdout(buf := io.StringIO()):
                parallel_betweenness(
                    z.network, search_radius=params.get("search_radius", 1000),
                    num_cores=int(params.get("num_cores", 2)),
                    closest_destination=bool(params.get(
                        "closest_destination", False)))
            out["btn_parallel"] = {"completed": True,
                                   "stdout_tail": buf.getvalue()[-2000:]}
        except Exception as exc:
            out["btn_parallel"] = {
                "completed": False,
                "raised": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
                "stdout_tail": buf.getvalue()[-2000:]}
        return out

    raise ValueError(f"unknown action {action!r}")
