"""P01F: worker-side backpressure telemetry on the parallel prepared boundary
(P04/R2 gate input on current main).

P01E (trace_coordinator_ticks.py) measured the coordinator's serial slice
(active_total ≈ 6.3 s of a 15.8 s wall; 3 fsync/row durable publication) but
the coordinator thread cannot say whether workers actually STARVE during it.
This probe measures the worker side directly through the landed P01M
telemetry seam (batch/telemetry.py, design 01 — opt-in, disabled recorders
inert, diagnostic runs never comparators):

  - UNA_PHASE_TELEMETRY=1, one measured pass, w4t1, warm prepared store;
  - workerpool JSONL spans per process: ``inbox_get`` (waiting for a job),
    ``result_put`` (synchronous put);
  - coordinator JSONL spans (plan + commit-side phases).

Verdict quantities derived from the workerpool timelines:
  - per-worker mid-run idle: inbox_get time strictly inside
    [first job arrival, last result put] of that worker — end-of-run
    straggler wait and the initial wait are excluded by construction;
  - aggregate starved worker-seconds = sum of mid-run idle (the P04/R2
    backpressure evidence, or its refutation);
  - per-worker busy share of its own window;
  - coordinator span inventory for the commit-side view.

No speedup claim; telemetry only.  Run (alan interpreter, worktree root):
    PYTHONPATH=src python campaigns/una_10x/next/probes/trace_worker_backpressure.py \
        --manifest <cohort_manifest.csv> --out-dir <run dir>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--numba-cache-dir", type=Path, default=Path(
        "/storage/scratch1/1/dsynn6/campaign_data/numba_cache_p00_cand"))
    args = ap.parse_args()

    manifest = args.manifest.resolve()
    manifest_sha = hashlib.sha256(manifest.read_bytes()).hexdigest()
    runs_dir = args.out_dir / "runs"
    phase_dir = runs_dir / "phase_events"
    runs_dir.mkdir(parents=True, exist_ok=True)
    phase_dir.mkdir(parents=True, exist_ok=True)

    os.environ["NUMBA_CACHE_DIR"] = str(args.numba_cache_dir)
    os.environ["NUMBA_NUM_THREADS"] = "1"  # w4t1 gate shape
    # P01M opt-in diagnostics (design 01): enabled run is a diagnostic run,
    # never a comparator.
    os.environ["UNA_PHASE_TELEMETRY"] = "1"
    os.environ["UNA_PHASE_DIR"] = str(phase_dir)
    os.environ["UNA_PHASE_RUN_ID"] = "p01f-backpressure"
    sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "src"))

    import urban_network_analysis as una_pkg
    from urban_network_analysis import UNA as UnaClass

    result = {
        "schema": "una10x-next-p01f-backpressure-1",
        "purpose": ("worker-side mid-run idle (starvation) measurement on "
                    "the parallel prepared boundary; P04/R2 gate input; "
                    "no claim"),
        "python": sys.executable,
        "package": str(Path(una_pkg.__file__).resolve()),
        "manifest": str(manifest),
        "manifest_sha256": manifest_sha,
        "manifest_sha_matches_p00_pin": manifest_sha ==
        "1161f4857a3d11640b883cbd25dac212ffda84c1157c1dba9d2cfd18ee95daaf",
        "numba_num_threads_env": os.environ.get("NUMBA_NUM_THREADS"),
        "workers": args.workers,
        "regime": ("parallel public call w4t1; pass 1 untimed warm (telemetry "
                   "already on), pass 2 measured (diagnostic — telemetry "
                   "overhead included, never a comparator)"),
        "phase_dir": str(phase_dir),
        "cpuset_cpus": len(os.sched_getaffinity(0)),
    }

    una = UnaClass()
    t0 = time.perf_counter()
    una.RunBatch("accessibility", pairing_file=str(manifest),
                 parallel=True, workers=args.workers)
    result["warm_pass_wall_s"] = time.perf_counter() - t0

    una2 = UnaClass()
    t1 = time.perf_counter()
    una2.RunBatch("accessibility", pairing_file=str(manifest),
                  parallel=True, workers=args.workers)
    result["telemetry_pass_wall_s"] = time.perf_counter() - t1

    # ── post-process the JSONL timelines ────────────────────────────────
    pool_files = sorted(phase_dir.glob("phase_events.workerpool.*.jsonl"))
    coord_files = sorted(phase_dir.glob("phase_events.coordinator.*.jsonl"))
    result["workerpool_jsonl_files"] = [p.name for p in pool_files]
    result["coordinator_jsonl_files"] = [p.name for p in coord_files]

    workers = []
    for path in pool_files:
        spans = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    spans.append(json.loads(line))
        spans.sort(key=lambda e: e["start_mono_ns"])
        gets = [e for e in spans if e["phase"] == "inbox_get"]
        puts = [e for e in spans if e["phase"] == "result_put"]
        # mid-run window: first job arrival .. last result put
        if not gets or not puts:
            workers.append({"file": path.name, "spans": len(spans),
                            "note": "no inbox_get/result_put spans"})
            continue
        win_start = gets[0]["end_mono_ns"]
        win_end = puts[-1]["end_mono_ns"]
        mid = [g for g in gets
               if g["start_mono_ns"] >= win_start
               and g["end_mono_ns"] <= win_end]
        mid_idle_s = sum(g["end_mono_ns"] - g["start_mono_ns"]
                         for g in mid) / 1e9
        put_s = sum(p["end_mono_ns"] - p["start_mono_ns"]
                    for p in puts) / 1e9
        window_s = (win_end - win_start) / 1e9
        workers.append({
            "file": path.name, "pid": spans[0].get("pid"),
            "window_s": round(window_s, 4),
            "jobs_done": len(puts),
            "midrun_idle_s": round(mid_idle_s, 4),
            "midrun_idle_spans": len(mid),
            "result_put_total_s": round(put_s, 4),
            "busy_share_of_window": round(
                (window_s - mid_idle_s - put_s) / window_s, 4)
            if window_s > 0 else None,
            "midrun_idle_top": sorted(
                ({"s": round((g["end_mono_ns"] - g["start_mono_ns"]) / 1e9, 4),
                  "t_rel_s": round((g["start_mono_ns"] - win_start) / 1e9, 3)}
                 for g in mid),
                key=lambda d: -d["s"])[:8],
        })

    done = [w for w in workers if "window_s" in w]
    result["workerpool"] = workers
    result["aggregate"] = {
        "workers_measured": len(done),
        "makespan_s": round(max(w["window_s"] for w in done), 4)
        if done else None,
        "starved_worker_seconds_midrun": round(
            sum(w["midrun_idle_s"] for w in done), 4) if done else None,
        "total_jobs": sum(w["jobs_done"] for w in done) if done else 0,
    }

    coord = []
    for path in coord_files:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    coord.append(json.loads(line))
    by_phase = {}
    for e in coord:
        agg = by_phase.setdefault(e["phase"], {
            "count": 0, "wall_s_sum": 0.0, "max_wall_s": 0.0})
        wall = (e["end_mono_ns"] - e["start_mono_ns"]) / 1e9
        agg["count"] += 1
        agg["wall_s_sum"] = round(agg["wall_s_sum"] + wall, 4)
        agg["max_wall_s"] = round(max(agg["max_wall_s"], wall), 4)
    result["coordinator_spans"] = by_phase

    out = runs_dir / "worker_backpressure.json"
    out.write_text(json.dumps(result, indent=1))
    print(json.dumps({"out": str(out),
                      "warm_s": result["warm_pass_wall_s"],
                      "telemetry_wall_s": result["telemetry_pass_wall_s"],
                      "aggregate": result["aggregate"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
