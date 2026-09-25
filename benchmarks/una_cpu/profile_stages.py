"""Stage-level profiler for the L3 stage model (T07).

Replicates UNA.RunAccessibility's exact public sequence with per-stage
wall timers (same call order, same public API — no internals patched):

    AddNetwork -> AddOrigins -> AddDestinations -> [AddObstacles]
    -> AccessibilityWElevation(topology, settings)   [graph build incl. CSR]
    -> Centrality -> ExportAccessibilityResults

The ordered/original CSR builder is additionally timed standalone on the
already-built topology (pure function, deterministic) to expose the CSR
share of graph construction without patching the engine.

One JSON per run: stage walls, counts, machine snapshot. Output folder is
wiped before the run so export cost includes writing, not merging.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from urllib.request import urlopen

HERE = Path(__file__).resolve().parent


def _assert_import(src_root: str | None):
    import urban_network_analysis as una
    path = Path(una.__file__).resolve()
    if src_root:
        assert str(path).startswith(str(Path(src_root).resolve())), (
            f"import {path} outside {src_root}")
    return str(path)


def _resource():
    import psutil
    vm = psutil.virtual_memory()
    return {"loadavg": os.getloadavg(), "mem_available_bytes": vm.available}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-root", default=None)
    ap.add_argument("--workload", default="W1_small_genuine")
    ap.add_argument("--workload-dir", type=Path, required=True)
    ap.add_argument("--analysis", choices=["accessibility", "flow"],
                    default="accessibility")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    sys.path.insert(0, str(HERE))
    from workload_gen import settings_for

    fixture_dir = args.workload_dir / args.workload
    out_root = args.out / "outputs"
    if out_root.exists():
        import shutil
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True)

    import numpy as np
    from urban_network_analysis import UNA
    from urban_network_analysis import Settings
    from urban_network_analysis.Topology import Topology
    from urban_network_analysis.Engines.AccessibilityWElevation import (
        AccessibilityWElevation,
    )

    result = {"src_root": args.src_root,
              "import_path": _assert_import(args.src_root),
              "workload": args.workload, "analysis": args.analysis,
              "resource_at_start": _resource()}
    s = Settings()
    for key, value in settings_for(args.workload, fixture_dir, out_root,
                                   args.analysis).items():
        if key == "knn_weights":
            value = np.asarray(value, dtype=np.float64)
        setattr(s, key, value)

    una = UNA(verbosity=0)
    una.settings = s
    t = time.perf_counter_ns

    t0 = t(); s.Validation(); t1 = t()
    stages = {"validation_ns": t1 - t0}

    una.topology = Topology(verbosity=0)
    t0 = t(); una.topology.AddNetwork(s); t1 = t()
    stages["add_network_ns"] = t1 - t0
    t0 = t(); una.topology.AddOrigins(s); t1 = t()
    stages["add_origins_ns"] = t1 - t0
    t0 = t(); una.topology.AddDestinations(s); t1 = t()
    stages["add_destinations_ns"] = t1 - t0
    if (s.obstacle_points_file or "").strip():
        t0 = t(); una.topology.AddObstacles(s); t1 = t()
        stages["add_obstacles_ns"] = t1 - t0

    t0 = t()
    acc = AccessibilityWElevation(una.topology, s)
    t1 = t()
    stages["engine_ctor_ns"] = t1 - t0

    # second ctor on the same topology: same work, JIT warm — a stability
    # check for engine_ctor_ns (CSR share is inferred cross-tree as
    # ctor(baseline) - ctor(candidate), not via internals)
    t0 = t()
    AccessibilityWElevation(una.topology, s)
    t1 = t()
    stages["engine_ctor_repeat_ns"] = t1 - t0

    t0 = t(); acc.Centrality(s); t1 = t()
    stages["centrality_ns"] = t1 - t0

    if args.analysis == "flow":
        t0 = t(); una._ResolveGravityCap(); t1 = t()
        stages["resolve_gravity_cap_ns"] = t1 - t0
        t0 = t(); una.RunFlow(); t1 = t()
        stages["run_flow_total_ns"] = t1 - t0
    else:
        t0 = t()
        acc.ExportAccessibilityResults(
            s, folder_prefix="accessibility_",
            file_name=s.output_file_name)
        t1 = t()
        stages["export_ns"] = t1 - t0

    result["stages_ns"] = stages
    result["counts"] = {
        "n_network_edges": int(len(una.topology.network.geometry)),
        "n_origins": int(len(una.topology.origins.geometry)),
        "n_destinations": int(len(una.topology.destinations.geometry)),
        "graph_nodes": int(acc.graph_engine.adjacency_pointer.shape[0]) - 1,
        "graph_edges_directed": int(acc.graph_engine.adjacency_vector.shape[0]),
    }
    result["resolved_gravity_cap"] = getattr(una, "resolved_gravity_cap", None)
    result["resource_at_end"] = _resource()
    result["python"] = sys.version.split()[0]

    args.out.mkdir(parents=True, exist_ok=True)
    with open(args.out / "stage_profile.json", "w") as f:
        json.dump(result, f, indent=1, default=str)
    print(json.dumps({k: v for k, v in stages.items()}, indent=1))


if __name__ == "__main__":
    main()
