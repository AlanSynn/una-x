"""Single UNA analysis job executor with explicit timer boundaries.

Runs as a subprocess launched by the campaign runner. The process imports
the package normally (source arm via PYTHONPATH, wheel arm from
site-packages), asserts where the import resolved, executes ONE fresh public
job under frozen Settings, and reports wall times, peak process RSS, and
post-boundary output hashes as a JSON object on stdout's last line.

Timer boundary ("whole fresh job including required outputs"):
    t0 = immediately before UNA() construction
    t1 = after RunAccessibility()/RunFlow() returns (exports are
         synchronous inside the public call, so required outputs are
         materialized and flushed at t1)
Output hashing happens AFTER t1 and is timed separately.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time


def _sample_rss_peak(stop_event, out_dict, interval=0.02):
    import psutil
    proc = psutil.Process()
    peak = 0
    while not stop_event.is_set():
        try:
            rss = proc.memory_info().rss
            try:
                for c in proc.children(recursive=False):
                    try:
                        rss += c.memory_info().rss
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
            except psutil.NoSuchProcess:
                pass
            peak = max(peak, rss)
        except psutil.Error:
            pass
        stop_event.wait(interval)
    out_dict["peak_rss_bytes"] = peak


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True, help="job spec JSON path")
    ap.add_argument("--result", required=True, help="result JSON out path")
    args = ap.parse_args()

    with open(args.spec) as f:
        spec = json.load(f)

    result = {"job_id": spec.get("job_id"), "pid": os.getpid()}
    mem = {}
    stop = threading.Event()
    sampler = threading.Thread(target=_sample_rss_peak, args=(stop, mem), daemon=True)
    sampler.start()

    import urban_network_analysis as una_pkg
    import numba
    from numba.np.ufunc import parallel as nb_parallel

    import_path = una_pkg.__file__
    result["import_path"] = import_path
    result["package_version"] = getattr(una_pkg, "__version__", None)
    mode = spec.get("assert_mode", "source")
    if mode == "wheel":
        import site
        ok = any(import_path.startswith(os.path.join(sp, ""))
                 for sp in site.getsitepackages() + [site.getusersitepackages()])
        if not ok:
            result["fatal"] = f"wheel mode: import resolved outside site-packages: {import_path}"
            _emit(result, args.result, stop)
            sys.exit(3)
    else:
        src_root = os.path.realpath(spec["source_root"])
        if not os.path.realpath(import_path).startswith(src_root):
            result["fatal"] = f"source mode: import resolved outside {src_root}: {import_path}"
            _emit(result, args.result, stop)
            sys.exit(3)

    result["numba_requested_threads"] = os.environ.get("NUMBA_NUM_THREADS")
    result["numba_effective_threads"] = int(nb_parallel.get_num_threads())
    result["numba_threading_layer_configured"] = numba.threading_layer() \
        if numba.config.DISABLE_JIT else "jit-disabled"

    import numpy as np
    from urban_network_analysis import UNA
    from urban_network_analysis import Settings

    s = Settings()
    for key, value in spec["settings"].items():
        if key == "knn_weights":
            value = np.asarray(value, dtype=np.float64)
        setattr(s, key, value)

    analysis = spec.get("analysis", "accessibility")
    t0 = time.perf_counter_ns()
    una = UNA(verbosity=0)
    una.settings = s
    if analysis == "flow":
        una.RunFlow()
    elif analysis == "accessibility":
        una.RunAccessibility()
    else:
        raise ValueError(f"unknown analysis {analysis}")
    t1 = time.perf_counter_ns()

    stop.set()
    sampler.join(timeout=1.0)

    result["analysis"] = analysis
    result["wall_ns"] = t1 - t0
    result["peak_rss_bytes"] = mem.get("peak_rss_bytes")
    result["output_root"] = spec["output_root"]

    # post-boundary verification hashing (timed separately, outside wall)
    th0 = time.perf_counter_ns()
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
    from workload_gen import hash_dir
    result["output_files"] = hash_dir(spec["output_root"])
    result["hash_ns"] = time.perf_counter_ns() - th0

    # small artifacts that prove the run actually executed the engine path
    acc = getattr(una, "accessibility", None)
    if acc is not None:
        reach = getattr(acc, "reach", None)
        result["reach_len"] = None if reach is None else int(len(reach))
    flow = getattr(una, "flow", None)
    if flow is not None:
        ef = getattr(flow, "edge_flow", None)
        result["edge_flow_len"] = None if ef is None else int(len(ef))
    result["resolved_gravity_cap"] = getattr(una, "resolved_gravity_cap", None)

    _emit(result, args.result, stop)


def _emit(result, path, stop):
    stop.set()
    import platform
    result["loadavg_at_finish"] = os.getloadavg()
    result["python"] = platform.python_version()
    with open(path, "w") as f:
        json.dump(result, f, indent=1, default=str)


if __name__ == "__main__":
    main()
