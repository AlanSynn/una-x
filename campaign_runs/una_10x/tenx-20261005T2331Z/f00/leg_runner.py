#!/usr/bin/env python3
"""F00 leg runner: one real public RunFlow leg on the Boston InnerCore data.

Modes
  clean : NO instrumentation. One perf_counter bracket around the single
          public call project.RunFlow(). This wall is the only claimed figure.
  diag  : the SAME public bracket, plus runtime-only monkeypatch timers
          around internal stages (topology adds, build, backward-gradient
          stage, origin loop, export) and around the module-level scipy
          dijkstra entry and the np.isfinite / np.where call sites used by
          the gradient packer. No src/ bytes are touched; patches exist
          only inside this process.

Every leg: fresh NUMBA_CACHE_DIR + fresh output dir (both under /tmp), the
frozen cache-off execution policy, threads pinned to --threads, and a JSON
leg record plus console custody file. Nothing under src/, tests/ or
campaign_data/ is written.
"""
from __future__ import annotations

import os
import sys

# Before any heavy import: no bytecode writes anywhere.
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

import argparse
import functools
import hashlib
import json
import multiprocessing as mp
import shutil
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path("/Users/alansynn/orca/workspaces/una-x/main")


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--leg", required=True)
    ap.add_argument("--mode", choices=["clean", "diag"], required=True)
    ap.add_argument("--origins-fixture", required=True)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--outdir", required=True, help="leg record + custody dir under f00/")
    ap.add_argument("--scratch-root", required=True, help="fresh /tmp root for outputs + numba cache")
    args = ap.parse_args()

    scratch = Path(args.scratch_root)
    if scratch.exists():
        shutil.rmtree(scratch)  # our own per-leg scratch; safe by construction
    out_dir = scratch / "out"
    nbcache = scratch / "nbcache"
    out_dir.mkdir(parents=True)
    nbcache.mkdir(parents=True)
    os.environ["NUMBA_CACHE_DIR"] = str(nbcache)

    rec_dir = Path(args.outdir)
    rec_dir.mkdir(parents=True, exist_ok=True)

    import numpy as np
    import scipy

    from urban_network_analysis import UNA
    from urban_network_analysis.Engines import AggregateFlow as AF
    from urban_network_analysis.Engines.Base import Base
    from urban_network_analysis.Settings import Settings
    from urban_network_analysis.Topology import Topology

    records = {
        "scipy_calls": [],
        "isfinite_dense": {"calls": 0, "s": 0.0, "elems": 0},
        "isfinite_small": {"calls": 0, "s": 0.0, "elems": 0},
        "where_calls": {"calls": 0, "s": 0.0, "elems": 0},
    }
    rec_lock = threading.Lock()
    stage_walls = {}
    stage_order = []

    def time_stage(name, fn):
        @functools.wraps(fn)
        def wrapper(*a, **kw):
            t0 = time.perf_counter()
            try:
                return fn(*a, **kw)
            finally:
                dt = time.perf_counter() - t0
                with rec_lock:
                    stage_walls[name] = stage_walls.get(name, 0.0) + dt
                    stage_order.append(name)
        return wrapper

    if args.mode == "diag":
        # -- scipy dijkstra entry (traversal + dense return) ------------
        _real_dij = AF._scipy_dijkstra

        @functools.wraps(_real_dij)
        def dij(*a, **kw):
            t0 = time.perf_counter()
            r = _real_dij(*a, **kw)
            dt = time.perf_counter() - t0
            dist = r[0] if isinstance(r, tuple) else r
            if dist.ndim == 1:
                k, n = 1, int(dist.shape[0])
            else:
                k, n = dist.shape
            with rec_lock:
                records["scipy_calls"].append({
                    "s": dt, "rows": int(k), "cols": int(n),
                    "dense_mb": k * n * 12.0 / 1e6,
                    "return_pred": bool(kw.get("return_predecessors", False)),
                    "limit": (None if kw.get("limit") is None
                              else float(kw["limit"])),
                })
            return r

        AF._scipy_dijkstra = dij

        # -- np.isfinite / np.where at the packer's call sites ----------
        class _NPProxy:
            def __init__(self, real):
                self._real = real

            def __getattr__(self, name):
                return getattr(self._real, name)

            def isfinite(self, x, /, *a, **kw):
                dense = getattr(x, "size", 1) > 1000
                t0 = time.perf_counter()
                r = self._real.isfinite(x, *a, **kw)
                dt = time.perf_counter() - t0
                with rec_lock:
                    bucket = (records["isfinite_dense"] if dense
                              else records["isfinite_small"])
                    bucket["calls"] += 1
                    bucket["s"] += dt
                    bucket["elems"] += int(getattr(x, "size", 1))
                return r

            def where(self, cond, /, *a, **kw):
                t0 = time.perf_counter()
                r = self._real.where(cond, *a, **kw)
                dt = time.perf_counter() - t0
                with rec_lock:
                    b = records["where_calls"]
                    b["calls"] += 1
                    b["s"] += dt
                    b["elems"] += int(getattr(cond, "size", 1))
                return r

        # -- stage timers ------------------------------------------------
        stage_map = [
            ("topology_add_network", Topology, "AddNetwork"),
            ("topology_add_origins", Topology, "AddOrigins"),
            ("topology_add_destinations", Topology, "AddDestinations"),
            ("topology_add_observers", Topology, "AddObservers"),
            ("topology_add_obstacles", Topology, "AddObstacles"),
            ("topology_build_clusters", Topology, "BuildClusters"),
            ("flow_build_digraph", AF.AggregateFlow, "_build_digraph"),
            ("flow_build_csr", AF.AggregateFlow, "_build_csr"),
            ("flow_origin_loop", AF.AggregateFlow, "_process_origins_aggregate"),
            ("flow_observer_accum", AF.AggregateFlow, "_accumulate_observer_flows"),
            ("export_flow_result", Base, "ExportFlowResult"),
        ]
        for name, cls, meth in stage_map:
            setattr(cls, meth, time_stage(name, getattr(cls, meth)))
        Settings.Validation = time_stage("settings_validation", Settings.Validation)

        # The np proxy must be visible to the gradient packer but NOT to
        # numba's compiler (the jitted kernels resolve the module global
        # `np` at JIT time).  The gradient stage is pure Python and runs
        # strictly before the origin loop, so swap the proxy in around
        # exactly that one stage.  The restore target is the REAL numpy
        # module object (never the current AF.np, which may be stale).
        _real_np = np
        _proxy_np = _NPProxy(np)

        _grad_timed = AF.AggregateFlow._precompute_dest_gradients

        @functools.wraps(_grad_timed)
        def grad_with_proxy(*a, **kw):
            t0 = time.perf_counter()
            AF.np = _proxy_np
            try:
                return _grad_timed(*a, **kw)
            finally:
                AF.np = _real_np
                with rec_lock:
                    stage_walls["flow_dest_gradients"] = (
                        stage_walls.get("flow_dest_gradients", 0.0)
                        + time.perf_counter() - t0)
                    stage_order.append("flow_dest_gradients")

        AF.AggregateFlow._precompute_dest_gradients = grad_with_proxy

    # ---- assemble the project (outside the public timer) ----------------
    real_data = str(REPO / "campaign_data" / "inputs" / "upstream_Boston" / "Boston")
    network_file = "20260703_PercLenNetwork_InnerCore.geojson"
    dest_file = "MA_transit_stations.geojson"

    # The fixture lives in f00/ (campaign_data stays untouched).  Settings
    # joins data_folder + origins_file for every layer, so each leg stages
    # a scratch data dir holding SYMLINKS to the real network and
    # destination bytes plus the fixture as origins.  Symlinks are reads
    # only; the referenced bytes are pinned by the sha256s below.
    leg_data = scratch / "data"
    leg_data.mkdir(parents=True, exist_ok=True)
    os.symlink(os.path.join(real_data, network_file), leg_data / network_file)
    os.symlink(os.path.join(real_data, dest_file), leg_data / dest_file)
    origins_name = os.path.basename(args.origins_fixture)
    os.symlink(os.path.abspath(args.origins_fixture), leg_data / origins_name)

    project = UNA(verbosity=1)
    project.topology.num_threads = int(args.threads)  # matched schedule
    s = project.settings
    s.data_folder = str(leg_data)
    s.network_file = network_file
    s.origins_file = origins_name
    s.destinations_file = dest_file
    s.output_folder = str(out_dir)
    s.flow_engine = "aggregate_flow"
    s.search_radius = 1500
    s.turns = False

    cfg = {
        "leg": args.leg,
        "mode": args.mode,
        "threads_pinned": int(args.threads),
        "numba_cache_dir": str(nbcache),
        "output_dir": str(out_dir),
        "data_folder": str(leg_data),
        "real_data_folder": real_data,
        "network_file": s.network_file,
        "network_sha256": sha256_file(os.path.join(real_data, s.network_file)),
        "origins_fixture": args.origins_fixture,
        "origins_fixture_sha256": sha256_file(args.origins_fixture),
        "destinations_file": "MA_transit_stations.geojson",
        "destinations_sha256": sha256_file(
            os.path.join(real_data, s.destinations_file)),
        "search_radius": s.search_radius,
        "flow_settings": {
            k: getattr(s, k)
            for k in sorted(dir(s))
            if k.startswith("flow_") and not callable(getattr(s, k))
        },
        "cache_policy": str(s.execution.cache.mode),
        "semantic_profile": str(s.execution.semantic_profile),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "numba": __import__("numba").__version__,
        "mp_cpu_count": mp.cpu_count(),
        "host_loadavg_1m_at_start": os.getloadavg()[0],
        "started_utc": datetime.now(timezone.utc).isoformat(),
    }
    cfg["data_folder_symlinks"] = {
        network_file: "real file (symlink)",
        dest_file: "real file (symlink)",
        origins_name: args.origins_fixture,
    }

    # ---- ONE public call, ONE bracket ------------------------------------
    t0 = time.perf_counter()
    project.RunFlow()
    wall_s = time.perf_counter() - t0

    cfg["public_wall_s"] = wall_s
    cfg["host_loadavg_1m_at_end"] = os.getloadavg()[0]
    cfg["finished_utc"] = datetime.now(timezone.utc).isoformat()

    # ---- engine facts (after the call; not timed) ------------------------
    flow = project.flow
    cfg["engine_facts"] = {
        "engine": type(flow).__name__,
        "n_network_nodes": int(flow._n_network_nodes),
        "n_destinations": int(flow._n_destinations),
        "n_origins_virtual": int(len(project.topology.origins.geometry)),
        "csr_nodes_total": int(flow._csr_indptr.shape[0] - 1),
        "csr_arcs": int(len(flow._csr_indices)),
        "n_edges_out": int(flow.edge_flow.shape[0]),
        "edge_flow_nonzero": int(np.count_nonzero(flow.edge_flow)),
        "f2_stats": dict(getattr(flow, "_f2_stats", {})),
        "assigned_stats": dict(getattr(flow, "_assigned_stats", {})),
        "num_threads_effective": int(flow.num_threads),
    }
    if args.mode == "diag":
        fwd = [c for c in records["scipy_calls"] if c["rows"] == 1]
        grad = [c for c in records["scipy_calls"] if c["rows"] > 1]
        cfg["stage_walls_s"] = stage_walls
        cfg["stage_order_first8"] = stage_order[:8]
        cfg["stage_sum_s"] = float(sum(stage_walls.values()))
        cfg["scipy_summary"] = {
            "gradient_calls": len(grad),
            "gradient_dense_mb_total": sum(c["dense_mb"] for c in grad),
            "gradient_s_total": sum(c["s"] for c in grad),
            "forward_calls": len(fwd),
            "forward_dense_mb_total": sum(c["dense_mb"] for c in fwd),
            "forward_s_total": sum(c["s"] for c in fwd),
            "limit_value": (grad[0]["limit"] if grad else
                            (fwd[0]["limit"] if fwd else None)),
        }
        cfg["np_pack"] = {k: v for k, v in records.items()
                          if k.startswith(("isfinite", "where"))}
        cfg["scipy_calls_first10"] = records["scipy_calls"][:10]

    # output custody: sha256 of every exported file (ExportFlowResult
    # writes into a timestamped subfolder — walk recursively)
    outs = {}
    for p in sorted(out_dir.rglob("*")):
        if p.is_file():
            rel = str(p.relative_to(out_dir))
            outs[rel] = {"bytes": p.stat().st_size,
                         "sha256": sha256_file(str(p))}
    cfg["output_files"] = outs

    rec_path = rec_dir / f"leg_{args.leg}_{args.mode}.json"
    with open(rec_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, default=str)
    print(json.dumps({k: cfg[k] for k in (
        "leg", "mode", "public_wall_s")}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
