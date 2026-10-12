"""P01E: coordinator tick trace — does coordinator work OVERLAP the worker
window, or SERIALIZE with it? (R2 design input on current main.)

The P01P parent cProfile (profile_parallel_prepared.py) measured shares but not
the TIMELINE: posix.fsync 2.14 s + publication syscalls ~1.1 s could be hiding
inside the select.poll wait (already overlapped with worker execution — cheap
to keep) or sit between poll returns (per-row publication ON the critical
path — R2 batching recovers it). cProfile cannot distinguish the two; a
timestamped C-call trace can.

Instrument: sys.setprofile c_call/c_return events in the COORDINATOR only
(workers are separate processes; the declared parent-only asymmetry of P01P
applies). Recorded C calls, grouped by name:

  wait side:    poll                      (select.poll.poll — the coordinator
                                           waiting on worker result fds)
  durable side: fsync, fdatasync, replace, rename, rmdir, unlink, link, mkdir
  cheap census: open, close, stat, fstat

Intervals are EXACT: a c_call/name push and its matching c_return pop bound
each C call on the coordinator thread (LIFO holds — C calls made directly
from Python frames return before Python continues; C→Python→C nesting pops
in order). Each interval is therefore that call's true service time, and the
durable-vs-wait overlap split is measured, not bounded.

No speedup claim; telemetry only. Run (alan interpreter, worktree root):
    PYTHONPATH=src python campaigns/una_10x/next/probes/trace_coordinator_ticks.py \
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

TRACE_NAMES = {
    # wait side
    "poll": "wait",
    # durable publication side
    "fsync": "durable", "fdatasync": "durable", "replace": "durable",
    "rename": "durable", "rmdir": "durable", "unlink": "durable",
    "link": "durable", "mkdir": "durable",
    # cheap census
    "open": "census", "close": "census", "stat": "census", "fstat": "census",
}

EVENT_RECORD_CAP = 400_000  # hard bound; the coordinator makes far fewer


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
    args.out_dir.mkdir(parents=True, exist_ok=True)
    os.environ["NUMBA_CACHE_DIR"] = str(args.numba_cache_dir)
    os.environ["NUMBA_NUM_THREADS"] = "1"  # w4t1 gate shape
    sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "src"))

    import urban_network_analysis as una_pkg
    from urban_network_analysis import UNA as UnaClass

    result = {
        "schema": "una10x-next-p01e-coordinator-ticks-1",
        "purpose": ("coordinator tick trace: overlap-vs-serialization of the "
                    "coordinator's durable/publication work vs the worker "
                    "window; R2 design input; no claim"),
        "python": sys.executable,
        "package": str(Path(una_pkg.__file__).resolve()),
        "manifest": str(manifest),
        "manifest_sha256": manifest_sha,
        "manifest_sha_matches_p00_pin": manifest_sha ==
        "1161f4857a3d11640b883cbd25dac212ffda84c1157c1dba9d2cfd18ee95daaf",
        "numba_num_threads_env": os.environ.get("NUMBA_NUM_THREADS"),
        "workers": args.workers,
        "trace_names": dict(TRACE_NAMES),
        "interval_convention": ("[c_call, matching c_return) — exact per-call "
                                "service time on the coordinator thread"),
        "cpuset_cpus": len(os.sched_getaffinity(0)),
    }

    una = UnaClass()
    t0 = time.perf_counter()
    una.RunBatch("accessibility", pairing_file=str(manifest),
                 parallel=True, workers=args.workers)
    result["warm_pass_wall_s"] = time.perf_counter() - t0

    # ── traced pass ──────────────────────────────────────────────────────
    iv_events: list = []   # (name, t_enter_ns, t_exit_ns), exact intervals
    stack: list = []       # (name, t_enter_ns) — open C calls, LIFO
    cap_hit = {"flag": False}

    def _prof(frame, event, arg):
        if event == "c_call":
            name = arg.__name__ if arg is not None else ""
            stack.append((name, time.perf_counter_ns()))
        elif event in ("c_return", "c_exception"):
            if stack:
                name, t0n = stack.pop()
                if name in TRACE_NAMES:
                    if len(iv_events) < EVENT_RECORD_CAP:
                        t1n = time.perf_counter_ns()
                        if t1n > t0n:
                            iv_events.append((name, t0n, t1n))
                    else:
                        cap_hit["flag"] = True

    sys.setprofile(_prof)
    una2 = UnaClass()
    t1 = time.perf_counter()
    una2.RunBatch("accessibility", pairing_file=str(manifest),
                  parallel=True, workers=args.workers)
    wall_s = time.perf_counter() - t1
    sys.setprofile(None)
    result["profiled_pass_wall_s"] = wall_s
    result["event_cap_hit"] = cap_hit["flag"]

    # ── analysis ─────────────────────────────────────────────────────────
    by_name: dict[str, list] = {}
    for name, t0n, t1n in iv_events:
        if t1n > t0n:  # drop zero/negative durations
            by_name.setdefault(name, []).append((t0n, t1n))

    def total_s(ivs):
        return sum((b - a) for a, b in ivs) / 1e9

    result["interval_summary"] = {
        name: {"count": len(ivs), "total_s": round(total_s(ivs), 4)}
        for name, ivs in sorted(by_name.items())}

    poll_ivs = sorted(by_name.get("poll", []))
    poll_total = total_s(poll_ivs)
    result["poll_total_s"] = round(poll_total, 4)
    result["active_total_s"] = round(max(wall_s - poll_total, 0.0), 4)

    def overlap_s(ivs, ref_ivs):
        if not ivs or not ref_ivs:
            return 0.0
        tot = 0
        for a, b in ivs:
            for ra, rb in ref_ivs:
                lo = max(a, ra)
                hi = min(b, rb)
                if hi > lo:
                    tot += hi - lo
        return tot / 1e9

    durable = {}
    for name in ("fsync", "fdatasync", "replace", "rename", "rmdir",
                 "unlink", "link", "mkdir"):
        ivs = by_name.get(name, [])
        if ivs:
            dur = total_s(ivs)
            ov = overlap_s(ivs, poll_ivs)
            durable[name] = {
                "count": len(ivs),
                "total_s": round(dur, 4),
                "overlap_with_worker_wait_s": round(ov, 4),
                "outside_worker_wait_s": round(dur - ov, 4),
            }
    result["durable_split"] = durable

    # completion ticks: poll intervals ended by a handler call (short waits
    # where a result fd was already readable)
    ticks = [b for a, b in poll_ivs if (b - a) / 1e9 <= 0.05]
    result["completion_tick_count"] = len(ticks)
    result["poll_call_count"] = len(poll_ivs)
    if len(ticks) >= 2:
        gaps = sorted((ticks[i + 1] - ticks[i]) / 1e9
                      for i in range(len(ticks) - 1))
        result["inter_completion_gap_s"] = {
            "n": len(gaps),
            "median": round(gaps[len(gaps) // 2], 4),
            "p90": round(gaps[int(len(gaps) * 0.9)], 4),
            "max": round(gaps[-1], 4)}

    if iv_events:
        t_base = iv_events[0][1]
        result["intervals_first_2000"] = [
            [name, round((a - t_base) / 1e6, 3), round((b - a) / 1e6, 3)]
            for name, a, b in iv_events[:2000]]

    out = args.out_dir / "coordinator_ticks.json"
    out.write_text(json.dumps(result, indent=1))
    print(json.dumps({"out": str(out),
                      "warm_s": result["warm_pass_wall_s"],
                      "wall_s": wall_s,
                      "poll_total_s": result["poll_total_s"],
                      "active_total_s": result["active_total_s"],
                      "durable_split": result["durable_split"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
