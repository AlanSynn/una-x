"""Incident payload: all-path enumeration scaling on square lattices
(dossier 03 failure class "finite but combinatorial all-path work").

The pinned source itself warns of "large detour all-path memory and
performance blowup".  Mechanism fixture: (N+1)x(N+1) unit lattice, one
corner-to-corner OD pair, the pinned bfs_paths_many_targets_iterative
enumerating EVERY path within detour_ratio of the shortest distance.
For each (N, ratio) we record paths generated, per-path length, and the
output-size lower bound (sum of stored node ids) — the growth factor is
the mechanism evidence.  Inputs are kept small (N <= 4) so every attempt
is bounded by construction; each input executes once.

This is a mechanism classification, NOT a named-city root cause and NOT
a timeout-based claim.
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))


def _lattice(n: int):
    """networkx unit lattice (n+1)x(n+1) + shortest-distance matrices."""
    import networkx as nx
    g = nx.Graph()
    for r in range(n + 1):
        for c in range(n + 1):
            g.add_node((r, c))
            if c < n:
                g.add_edge((r, c), (r, c + 1), weight=1.0)
            if r < n:
                g.add_edge((r, c), (r + 1, c), weight=1.0)
    return g


class _StubNetwork:
    """Only member bfs_paths_many_targets_iterative reads."""
    street_node_ids = set()  # filled per attempt


def payload(params, state):
    from madina_bridge import bridge_import
    applied, shas = bridge_import(params["madina_src"], [])
    from madina.una.paths import bfs_paths_many_targets_iterative

    results = []
    o_idx, d_node = (0, 0), (params["n"], params["n"])
    g = _lattice(params["n"])
    import networkx as nx
    dist = dict(nx.single_source_dijkstra_path_length(g, o_idx,
                                                      weight="weight"))
    o_graph = g
    d_idxs = {d_node: dist[d_node]}
    # distance_matrix[target][node] = shortest node->target length
    distance_matrix = {}
    for t in d_idxs:
        dmap = dict(nx.single_source_dijkstra_path_length(g, t,
                                                          weight="weight"))
        distance_matrix[t] = dmap
    stub = _StubNetwork()
    stub.street_node_ids = {v for v in g.nodes if v not in (o_idx, d_node)}

    ratio = float(params["detour_ratio"])
    state["phase"] = f"enumerate N={params['n']} ratio={ratio}"
    paths, distances = bfs_paths_many_targets_iterative(
        stub, o_graph, o_idx, d_idxs, distance_matrix=distance_matrix,
        turn_penalty=False, od_scope=None)
    per_d = paths[d_node]
    total_node_ids = sum(len(p) for p in per_d)
    results.append({
        "n": params["n"], "detour_ratio": ratio,
        "paths_generated": len(per_d),
        "sum_path_lengths": total_node_ids,
        "output_lower_bound_bytes_estimate": total_node_ids * 8,
        "shortest_distance": dist[d_node],
        "max_path_length": max((len(p) for p in per_d), default=0),
    })
    state["units_done"] = 1
    return {"applied_interventions": applied, "attempts": results}
