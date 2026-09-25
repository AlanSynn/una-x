"""Persistent batch worker for independent-job concurrency experiments.

One worker owns one process: it imports the package once, performs a
small warm-up job (recorded, excluded from the timed batch window), then
consumes job specs from a bounded queue until a None sentinel arrives.
Results flow back over a result queue. Spawn-style creation is enforced by
the runner (macOS default) and the module guard below; workers never fork an
initialized JIT runtime.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def worker_loop(worker_id, spec_queue, result_queue, warmup_spec):
    import threading

    sys.path.insert(0, str(HERE))
    from job import _sample_rss_peak

    t_spawn = time.perf_counter_ns()
    import urban_network_analysis as una_pkg  # noqa: F401 (import timing)
    import numba
    from numba.np.ufunc import parallel as nb_parallel
    t_import = time.perf_counter_ns()

    mem = {}
    stop = threading.Event()
    sampler = threading.Thread(target=_sample_rss_peak, args=(stop, mem), daemon=True)
    sampler.start()

    # Warm-up: compile jitted kernels + GIS writer once per worker.
    t_warm0 = time.perf_counter_ns()
    with open(warmup_spec) as f:
        wspec = json.load(f)
    _run_job_body(wspec)
    t_warm1 = time.perf_counter_ns()
    del wspec
    stop.set()
    sampler.join(timeout=1.0)

    result_queue.put({
        "type": "worker_ready", "worker_id": worker_id,
        "import_ns": t_import - t_spawn,
        "warmup_ns": t_warm1 - t_warm0,
        "warmup_peak_rss_bytes": mem.get("peak_rss_bytes"),
        "numba_effective_threads": int(nb_parallel.get_num_threads()),
        "numba_requested_threads": os.environ.get("NUMBA_NUM_THREADS"),
        "threading_layer": numba.threading_layer(),
        "import_path": una_pkg.__file__,
        "pid": os.getpid(),
    })

    # Re-arm the sampler for the measurement phase.
    mem2 = {}
    stop2 = threading.Event()
    sampler2 = threading.Thread(target=_sample_rss_peak, args=(stop2, mem2), daemon=True)
    sampler2.start()

    while True:
        spec = spec_queue.get()
        if spec is None:
            break
        try:
            wall_ns, hash_ns, files, reach_len = _run_job_body(spec)
            result_queue.put({"type": "job_done", "worker_id": worker_id,
                              "job_id": spec.get("job_id"),
                              "wall_ns": wall_ns, "hash_ns": hash_ns,
                              "output_root": spec["output_root"],
                              "n_output_files": len(files),
                              "output_hash": files,
                              "reach_len": reach_len})
        except Exception as exc:  # noqa: BLE001 - report and keep the pool alive
            result_queue.put({"type": "job_failed", "worker_id": worker_id,
                              "job_id": spec.get("job_id"),
                              "error": repr(exc)})
    stop2.set()
    sampler2.join(timeout=1.0)
    result_queue.put({"type": "worker_exit", "worker_id": worker_id,
                      "phase_peak_rss_bytes": mem2.get("peak_rss_bytes")})


def _run_job_body(spec):
    """Execute one accessibility job in this (already initialized) process.

    Returns (job_wall_ns, hash_wall_ns, output_files, reach_len). The job
    timer covers UNA construction through export completion; output hashing
    is timed and reported separately, outside the job wall.
    """
    import numpy as np
    from urban_network_analysis import UNA
    from urban_network_analysis import Settings

    s = Settings()
    for key, value in spec["settings"].items():
        if key == "knn_weights":
            value = np.asarray(value, dtype=np.float64)
        setattr(s, key, value)

    t0 = time.perf_counter_ns()
    una = UNA(verbosity=0)
    una.settings = s
    una.RunAccessibility()
    t1 = time.perf_counter_ns()

    th0 = time.perf_counter_ns()
    sys.path.insert(0, str(HERE))
    from workload_gen import hash_dir
    files = hash_dir(spec["output_root"])
    th1 = time.perf_counter_ns()
    return (t1 - t0, th1 - th0, files, int(len(una.accessibility.reach)))


if __name__ == "__main__":
    # Spawn guard: this module is only ever executed as the mp spawn target.
    pass
