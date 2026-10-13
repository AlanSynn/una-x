"""Direction-3 probe: per-phase device-region walls + K-replay residency variant.

Decomposes the G10 full-service region (which the G11 crossover measured as
one 0.13535 s wall at rep4000) into its phases — per-array H2D, launch,
sync, per-array D2H — and measures the residency shape dossier 08 specifies:
upload immutable inputs ONCE, then K launches with per-replay sync + D2H
outputs (per-row semantics). Walls only: this is not a route claim and never
feeds the CPU 10x gate (hardware addition, recorded separately).

Shape: the G11 battery's rep4000 case (4000 origins, 238387 nnz), generated
deterministically (seed 20261012) with the Phase-B laws respected
(strictly increasing ids within an origin, ascending distances).

Without a CUDA device this exits with a typed ``device_unavailable`` status.

Run (device node):
    PYTHONPATH=src python probes/gpu_phase_decomposition.py --out <out.json>
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np


def _rep4000(seed: int = 20261012):
    rng = np.random.default_rng(seed)
    origins, target_nnz = 4000, 238387
    base = target_nnz // origins
    extra = target_nnz - base * origins
    counts = np.full(origins, base, dtype=np.int64)
    counts[:extra] += 1
    indptr = np.zeros(origins + 1, dtype=np.int64)
    np.cumsum(counts, out=indptr[1:])
    nnz = int(indptr[-1])
    ids = np.empty(nnz, dtype=np.int64)
    dist = np.empty(nnz, dtype=np.float64)
    for o in range(origins):
        lo, hi = int(indptr[o]), int(indptr[o + 1])
        ids[lo:hi] = np.sort(rng.choice(20000, hi - lo, replace=False))
        dist[lo:hi] = np.sort(rng.uniform(1.0, 2000.0, hi - lo))
    d_weights = rng.uniform(0.0, 1.0, 20000)
    return indptr, ids, dist, d_weights


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--replays", type=int, default=32)
    args = ap.parse_args()

    result = {"schema": "una10x-gpu-phase-decomposition-1",
              "purpose": "Direction 3 evidence (PLAN_GPU.md §5); walls only",
              "python": sys.executable}
    sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "src"))

    indptr, ids, dist, d_weights = _rep4000()
    knn_weights = np.array([1.0], dtype=np.float64)
    cutoff = 1500.0
    result["shape"] = {"origins": int(indptr.size - 1),
                       "nnz": int(ids.size), "cutoff": cutoff,
                       "replays": args.replays}

    from urban_network_analysis.Engines import _gpu_fold as g

    if not g.cuda_available():
        result["status"] = "device_unavailable"
        args.out.write_text(json.dumps(result, indent=1))
        print(json.dumps({"status": "device_unavailable"}))
        return 0
    g.require_device()
    from numba import cuda

    dev = cuda.get_current_device()
    result["device"] = {"name": dev.name.decode()
                        if isinstance(dev.name, bytes) else str(dev.name)}

    # reference full-service wall via the G10 bridge (3 alternating pairs)
    full = []
    for _ in range(6):
        t0 = time.perf_counter()
        g.gpu_replay_fold(indptr, ids, dist, d_weights, cutoff,
                          0.001, 0.0, 500.0, 0.01, knn_weights, "logistic")
        full.append(time.perf_counter() - t0)
    full.sort()
    result["full_service_bridge_s"] = {"median": full[len(full) // 2],
                                       "min": full[0], "max": full[-1]}

    # --- phase decomposition ---------------------------------------------
    kernel = g._kernel()
    origin_count = indptr.size - 1
    nnz = ids.size
    phases = {"h2d": {}, "d2h": {}}
    t0 = time.perf_counter()
    d_indptr = cuda.to_device(indptr)
    t1 = time.perf_counter()
    d_dist = cuda.to_device(dist)
    t2 = time.perf_counter()
    d_ids = cuda.to_device(ids)
    t3 = time.perf_counter()
    d_w = cuda.to_device(d_weights)
    t4 = time.perf_counter()
    d_knn = cuda.to_device(knn_weights)
    t5 = time.perf_counter()
    d_reach = cuda.device_array(origin_count, dtype=np.float64)
    d_ge = cuda.device_array(origin_count, dtype=np.float64)
    d_gl = cuda.device_array(origin_count, dtype=np.float64)
    d_kn = cuda.device_array(origin_count, dtype=np.float64)
    d_kept = cuda.device_array(origin_count, dtype=np.int64)
    d_pairs_d = cuda.device_array(nnz, dtype=np.float64)
    d_pairs_i = cuda.device_array(nnz, dtype=np.int64)
    d_scd = cuda.device_array(nnz, dtype=np.float64)
    d_scw = cuda.device_array(nnz, dtype=np.float64)
    t6 = time.perf_counter()
    blocks = (origin_count + 127) // 128
    kernel[blocks, 128](d_indptr, d_dist, d_ids, d_w, d_knn,
                        cutoff, 0.001, 0.0, 500.0, 0.01, 1,
                        d_reach, d_ge, d_gl, d_kn, d_kept,
                        d_pairs_d, d_pairs_i, d_scd, d_scw)
    t7 = time.perf_counter()
    cuda.synchronize()
    t8 = time.perf_counter()
    r_reach = d_reach.copy_to_host()
    r_ge = d_ge.copy_to_host()
    r_gl = d_gl.copy_to_host()
    r_kn = d_kn.copy_to_host()
    r_kept = d_kept.copy_to_host()
    t9 = time.perf_counter()
    phases["h2d"] = {"indptr_s": t1 - t0, "dist_s": t2 - t1,
                     "ids_s": t3 - t2, "weights_s": t4 - t3,
                     "knn_s": t5 - t4, "alloc_s": t6 - t5}
    phases["launch_s"] = t7 - t6
    phases["sync_s"] = t8 - t7
    phases["d2h_outputs_s"] = t9 - t8
    phases["total_s"] = t9 - t0
    result["phase_decomposition"] = phases

    # --- residency variant: upload once, K launches ----------------------
    K = args.replays
    t0 = time.perf_counter()
    per_replay = []
    for _ in range(K):
        ta = time.perf_counter()
        kernel[blocks, 128](d_indptr, d_dist, d_ids, d_w, d_knn,
                            cutoff, 0.001, 0.0, 500.0, 0.01, 1,
                            d_reach, d_ge, d_gl, d_kn, d_kept,
                            d_pairs_d, d_pairs_i, d_scd, d_scw)
        cuda.synchronize()
        _ = d_reach.copy_to_host(), d_ge.copy_to_host(), \
            d_gl.copy_to_host(), d_kn.copy_to_host(), \
            d_kept.copy_to_host()
        per_replay.append(time.perf_counter() - ta)
    total = time.perf_counter() - t0
    per_replay.sort()
    result["residency_upload_once"] = {
        "replays": K,
        "total_s": total,
        "per_replay_median_s": per_replay[len(per_replay) // 2],
        "per_replay_min_s": per_replay[0],
    }
    result["law"] = ("walls only; hardware addition recorded separately; "
                     "never fed to the CPU 10x gate; not a route claim")

    args.out.write_text(json.dumps(result, indent=1))
    print(json.dumps({"out": str(args.out),
                      "full_service_median_s":
                          result["full_service_bridge_s"]["median"],
                      "per_replay_median_s":
                          result["residency_upload_once"]
                          ["per_replay_median_s"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
