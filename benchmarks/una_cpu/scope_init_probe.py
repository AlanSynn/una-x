"""T08 admission probe: allocation/init vs search share of the scope kernel.

Dossier 06 admits the optional scratch-state experiment only if the post-CSR
profile shows MATERIAL end-to-end service in per-origin allocation and
initialization of Dijkstra search state. This probe measures, on the REAL W3
topology built through the public API, the same unmodified upstream kernel
`compact_vector_node_view_scope` at three search radii:

  r ~ 0    : no neighbour relaxations occur -> allocation + O(V) init only
  r mid    : partial search
  r full   : the workload's actual search radius

share_init = t(r~0) / t(r_full) bounds the allocation/init fraction of the
kernel; scaled by the centrality stage share of the end-to-end job, it bounds
the Amdahl headroom the dossier gates on. The kernel itself is upstream code,
identical in both trees; this probe runs on the candidate tree only (values
are about the kernel, not about the CSR change).
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-root", required=True)
    ap.add_argument("--workload-dir", type=Path, required=True)
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    sys.path.insert(0, str(HERE))
    from workload_gen import settings_for

    import numpy as np
    from urban_network_analysis import Settings
    from urban_network_analysis.Topology import Topology
    from urban_network_analysis.Engines.AccessibilityWElevation import (
        AccessibilityWElevation,
        compact_vector_node_view_scope,
    )

    fixture = args.workload_dir / "W3_medium_proxy"
    out_root = args.out / "outputs"
    out_root.mkdir(parents=True, exist_ok=True)
    s = Settings()
    for k, v in settings_for("W3_medium_proxy", fixture, out_root,
                             "accessibility").items():
        if k == "knn_weights":
            v = np.asarray(v, dtype=np.float64)
        setattr(s, k, v)
    topo = Topology(verbosity=0)
    topo.AddNetwork(s)
    topo.AddOrigins(s)
    topo.AddDestinations(s)
    acc = AccessibilityWElevation(topo, s)
    ge = acc.graph_engine
    full_radius = float(s.search_radius)

    # Representative origin: median terminal count (origins may snap to more
    # than one network terminal; the kernel runs multi-source from all of
    # them). o_terminal_idxs[i] / o_terminal_weights[i] are parallel arrays.
    n_probe = min(24, len(ge.o_terminal_idxs))
    term_counts = [len(ge.o_terminal_idxs[i]) for i in range(n_probe)]
    med_count = statistics.median(term_counts)
    pick = next(i for i in range(n_probe)
                if len(ge.o_terminal_idxs[i]) == med_count)
    term = np.asarray(ge.o_terminal_idxs[pick])
    tw = np.asarray(ge.o_terminal_weights[pick])

    def time_radius(radius):
        walls = []
        for _ in range(args.reps):
            t0 = time.perf_counter_ns()
            compact_vector_node_view_scope(
                term, tw, ge.adjacency_pointer, ge.adjacency_vector,
                ge.adjacency_vector_weights,
                ge.adjacynct_vector_network_node, radius, ge.d_count)
            walls.append(time.perf_counter_ns() - t0)
        return statistics.median(walls)

    result = {
        "kernel_module": sys.modules[
            "urban_network_analysis.Engines.AccessibilityWElevation"].__file__,
        "probe_origins": n_probe, "reps": args.reps,
        "graph_nodes": int(ge.adjacency_pointer.shape[0]) - 1,
        "graph_edges_directed": int(ge.adjacency_vector.shape[0]),
        "origin_terminal_counts_first24": term_counts,
        "picked_origin_index": pick,
        "radii_tested": {},
    }
    for label, radius in (("r_zero", 0.0),
                          ("r_small", full_radius * 0.01),
                          ("r_full", full_radius)):
        med = time_radius(radius)
        result["radii_tested"][label] = {"radius": radius, "median_ns": med}

    r0 = result["radii_tested"]["r_zero"]["median_ns"]
    rf = result["radii_tested"]["r_full"]["median_ns"]
    result["init_share_of_kernel"] = r0 / rf if rf else None

    args.out.mkdir(parents=True, exist_ok=True)
    with open(args.out / "scope_init_probe.json", "w") as f:
        json.dump(result, f, indent=1, default=str)
    print(json.dumps({k: result[k] for k in
                      ("radii_tested", "init_share_of_kernel")}, indent=1))


if __name__ == "__main__":
    main()
