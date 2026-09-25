"""Campaign benchmark runner (T05 harness).

Orchestrates single-job and batch measurements for one arm under a frozen
workload and resource policy. Raw per-job records are appended to
<out>/raw.jsonl; a session summary is written to <out>/session.json.

Modes:
  single: N sequential fresh-process jobs (subprocess per job).
  batch:  W persistent worker processes consuming a bounded queue of N
          jobs; per-worker effective Numba threads H; sum(W*H) must be
          within the admitted CPU budget passed via --cpu-budget.

The benchmark owner must hold the exclusive machine lease while this runs.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]


def _resource_snapshot():
    import psutil
    vm = psutil.virtual_memory()
    return {
        "loadavg": os.getloadavg(),
        "mem_total_bytes": vm.total,
        "mem_available_bytes": vm.available,
        "cpu_percent_interval": psutil.cpu_percent(interval=0.2),
    }


def _make_job_spec(spec, out_root, job_id, assert_mode, source_root):
    job_out = Path(out_root) / f"job_{job_id:05d}"
    job_out.mkdir(parents=True, exist_ok=True)
    settings = dict(spec["settings"])
    settings["output_folder"] = str(job_out)
    return {
        "job_id": job_id,
        "analysis": spec["analysis"],
        "settings": settings,
        "assert_mode": assert_mode,
        "source_root": str(source_root) if source_root else None,
        "output_root": str(job_out),
    }


def run_single(args, spec, out_dir):
    results = []
    for i in range(args.jobs):
        job_spec = _make_job_spec(spec, out_dir / "outputs", i,
                                  args.arm, args.src)
        spec_path = out_dir / f"spec_{i:05d}.json"
        result_path = out_dir / f"result_{i:05d}.json"
        with open(spec_path, "w") as f:
            json.dump(job_spec, f)

        env = os.environ.copy()
        env.pop("PYTHONPATH", None)
        if args.arm == "source":
            env["PYTHONPATH"] = str(args.src)
        env["NUMBA_NUM_THREADS"] = str(args.threads)
        env.setdefault("OMP_NUM_THREADS", "1")
        env.setdefault("OPENBLAS_NUM_THREADS", "1")
        env.setdefault("MKL_NUM_THREADS", "1")
        # Launch from outside any checkout: neutral scratch cwd.
        cwd = str(out_dir)

        t_spawn = time.perf_counter_ns()
        proc = subprocess.run(
            [sys.executable, str(HERE / "job.py"),
             "--spec", str(spec_path), "--result", str(result_path)],
            capture_output=True, text=True, env=env, cwd=cwd, timeout=3600)
        t_done = time.perf_counter_ns()
        rec = {"mode": "single", "job_id": i,
               "spawn_to_done_ns": t_done - t_spawn,
               "returncode": proc.returncode,
               "loadavg_at_start": os.getloadavg()}
        if proc.returncode == 0 and result_path.exists():
            with open(result_path) as f:
                rec.update(json.load(f))
        else:
            rec["stderr_tail"] = proc.stderr[-2000:]
        results.append(rec)
        with open(out_dir / "raw.jsonl", "a") as f:
            f.write(json.dumps(rec, default=str) + "\n")
    return results


def run_batch(args, spec, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    ctx = mp.get_context("spawn")
    spec_queue = ctx.Queue(maxsize=args.queue)
    result_queue = ctx.Queue()

    # Warm-up spec: tiny W1 accessibility job into a disposable dir.
    warm_root = out_dir / "warmup"
    warm_root.mkdir(parents=True, exist_ok=True)
    from workload_gen import settings_for
    w1_dir = args.workload_dir / "W1_small_genuine"
    warm_settings = settings_for("W1_small_genuine", w1_dir,
                                 warm_root, "accessibility")
    warm_root.mkdir(parents=True, exist_ok=True)
    warm_spec_path = out_dir / "warmup_spec.json"
    with open(warm_spec_path, "w") as f:
        json.dump({"job_id": "warmup", "analysis": "accessibility",
                   "settings": warm_settings,
                   "assert_mode": args.arm,
                   "source_root": str(args.src) if args.src else None,
                   "output_root": str(warm_root)}, f)

    env_note = {"NUMBA_NUM_THREADS": str(args.threads),
                "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1"}
    if args.arm == "source":
        env_note["PYTHONPATH"] = str(args.src)

    def _make_worker_env():
        e = os.environ.copy()
        e.pop("PYTHONPATH", None)
        e.update(env_note)
        return e

    procs = []
    for w in range(args.workers):
        p = ctx.Process(target=_worker_entry,
                        args=(w, spec_queue, result_queue,
                              str(warm_spec_path), _make_worker_env()))
        p.start()
        procs.append(p)

    readies = {}
    deadline = time.time() + 1800
    while len(readies) < args.workers and time.time() < deadline:
        msg = result_queue.get(timeout=1800)
        if msg["type"] == "worker_ready":
            readies[msg["worker_id"]] = msg
    if len(readies) < args.workers:
        raise RuntimeError(f"only {len(readies)}/{args.workers} workers became ready")

    with open(out_dir / "workers.json", "w") as f:
        json.dump(readies, f, indent=1, default=str)

    for i in range(args.jobs):
        job_spec = _make_job_spec(spec, out_dir / "outputs", i,
                                  args.arm, args.src)
        spec_queue.put(job_spec)

    t0 = time.perf_counter_ns()
    done = 0
    failed = 0
    while done + failed < args.jobs:
        msg = result_queue.get(timeout=3600)
        typ = msg["type"]
        if typ == "job_done":
            done += 1
            with open(out_dir / "raw.jsonl", "a") as f:
                f.write(json.dumps({"mode": "batch", **msg,
                                    "loadavg_after_job": os.getloadavg()},
                                   default=str) + "\n")
        elif typ == "job_failed":
            failed += 1
            with open(out_dir / "raw.jsonl", "a") as f:
                f.write(json.dumps({"mode": "batch", **msg}, default=str) + "\n")
    t1 = time.perf_counter_ns()

    for _ in range(args.workers):
        spec_queue.put(None)
    exits = {}
    deadline = time.time() + 600
    while len(exits) < args.workers and time.time() < deadline:
        msg = result_queue.get(timeout=600)
        if msg["type"] == "worker_exit":
            exits[msg["worker_id"]] = msg
    for p in procs:
        p.join(timeout=60)
        if p.is_alive():
            p.terminate()

    summary = {
        "mode": "batch", "workers": args.workers, "threads": args.threads,
        "queue": args.queue, "jobs": args.jobs, "jobs_done": done,
        "jobs_failed": failed,
        "batch_wall_ns": t1 - t0,
        "throughput_jobs_per_s": args.jobs / ((t1 - t0) / 1e9),
        "worker_exits": exits,
        "cpu_budget_check": {
            "admitted_C": args.cpu_budget,
            "workers_x_threads": args.workers * args.threads,
            "within_budget": args.workers * args.threads <= args.cpu_budget,
        },
        "resource_at_end": _resource_snapshot(),
    }
    with open(out_dir / "session.json", "w") as f:
        json.dump(summary, f, indent=1, default=str)
    return summary


def _worker_entry(worker_id, spec_queue, result_queue, warmup_spec, env):
    """spawn target: rebuild environment then enter the worker loop."""
    os.environ.clear()
    os.environ.update(env)
    import batch_worker
    batch_worker.worker_loop(worker_id, spec_queue, result_queue, warmup_spec)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["source", "wheel"], required=True)
    ap.add_argument("--src", type=Path, default=None,
                    help="source root (…/src) for --arm source")
    ap.add_argument("--workload", default="W3_medium_proxy")
    ap.add_argument("--workload-dir", type=Path, required=True,
                    help="directory containing (or to receive) fixture files")
    ap.add_argument("--analysis", choices=["accessibility", "flow"],
                    default="accessibility")
    ap.add_argument("--mode", choices=["single", "batch"], default="single")
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--queue", type=int, default=4)
    ap.add_argument("--cpu-budget", type=int, default=8)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--label", default="")
    args = ap.parse_args()

    if args.arm == "source":
        assert args.src and args.src.is_dir(), "--src required for source arm"
    assert args.workers * args.threads <= args.cpu_budget, (
        f"workers*threads={args.workers * args.threads} exceeds admitted "
        f"budget C={args.cpu_budget}")

    args.out.mkdir(parents=True, exist_ok=True)

    # Workload: generate once if absent, then freeze hashes into manifest.
    # Each workload owns a fixture subdirectory (file names overlap).
    from workload_gen import (generate_w1, generate_w3, manifest_for,
                              settings_for)
    fixture_dir = args.workload_dir / args.workload
    fixture_dir.mkdir(parents=True, exist_ok=True)
    marker = fixture_dir / "manifest.json"
    if not marker.exists():
        if args.workload == "W1_small_genuine":
            spec = generate_w1(fixture_dir)
        elif args.workload == "W3_medium_proxy":
            spec = generate_w3(fixture_dir)
        else:
            raise SystemExit(f"unknown workload {args.workload}")
        with open(marker, "w") as f:
            json.dump(manifest_for(fixture_dir, spec), f, indent=1)
    with open(marker) as f:
        workload_manifest = json.load(f)

    settings = settings_for(args.workload, fixture_dir,
                            args.out / "outputs", args.analysis)
    run_spec = {"workload": args.workload, "analysis": args.analysis,
                "settings": settings}

    with open(args.out / "session_config.json", "w") as f:
        json.dump({
            "arm": args.arm, "src": str(args.src) if args.src else None,
            "workload_manifest": workload_manifest,
            "analysis": args.analysis, "mode": args.mode,
            "jobs": args.jobs, "workers": args.workers,
            "threads": args.threads, "queue": args.queue,
            "cpu_budget": args.cpu_budget, "label": args.label,
            "settings": settings,
            "resource_at_start": _resource_snapshot(),
            "python": sys.version.split()[0],
        }, f, indent=1, default=str)

    if args.mode == "single":
        run_single(args, run_spec, args.out)
    else:
        run_batch(args, run_spec, args.out)


if __name__ == "__main__":
    main()
