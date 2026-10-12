"""P03C precondition telemetry: worker/numba-thread topology matrix (plan §6.3).

Answers the one P01 hypothesis left unresolved (H3): how does the parallel
cohort run's wall time depend on the (workers, numba-threads-per-worker)
topology on the CURRENT tree (post compiled-engine merge)?  Telemetry only —
no speedup claim, no gate; P03C design input.

Regime:

  - the SAME frozen cohort manifest (sha256 pinned below; asserted per leg),
  - the on-disk numba cache owner directory from the P00 pins, shared by
    every leg, preceded by ONE untimed warm pass (config W4T1) so JIT/cache
    cold cost does not pollute timed legs,
  - each leg is its own SUBPROCESS because the numba thread budget is read
    from the environment at import time.  For parallel legs the child ALSO
    passes ``execution=ExecutionOptions(threads_per_worker=T)``: the batch
    runtime sets worker thread budgets from that option (spawn shim
    ``una_batch_worker_main.pool_main`` overwrites NUMBA_NUM_THREADS et al.
    before the heavy import), so the parent's env alone would not reach the
    workers.  The env is still set on every child for the coordinator side
    (and governs the serial leg), and OMP/BLAS budgets follow the declared
    config so no leg oversubscribes beyond its declared topology,
  - serial reference leg: ``parallel=False`` with ambient thread env
    (the P01R declared serial regime),
  - ``perf_counter`` brackets EXACTLY the public ``RunBatch`` call in the
    child; the parent collects one ``RESULT_JSON: {...}`` line per leg,
  - each leg is bounded: a pass exceeding ``PASS_TIMEOUT_S`` is recorded
    honestly (timeout) and the remaining reps of that config are aborted,
  - whole matrix on the login node under ``nice 10`` (probe-scale precedent,
    not charged to the sbatch budget); cpuset/nproc/loadavg recorded at
    start and end for co-tenant noise disclosure.

Run (alan interpreter, repo root of the branch worktree):
    PYTHONPATH=src nice -n 10 python campaigns/una_10x/next/probes/replay_thread_matrix.py \
        --manifest <cohort_manifest.csv> --out-dir <run dir>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

MANIFEST_PIN = ("1161f4857a3d11640b883cbd25dac212ffda84c1157c1dba9d2cf"
                "d18ee95daaf")
PASS_TIMEOUT_S = 400.0
REPS = (1, 2)
RESULT_PREFIX = "RESULT_JSON: "
THREAD_ENV_KEYS = ("NUMBA_NUM_THREADS", "OMP_NUM_THREADS",
                   "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                   "VECLIB_MAXIMUM_THREADS")

# (label, workers, threads_per_worker); workers=None -> serial reference.
MATRIX = [
    ("serial", None, None),
    ("W1T4", 1, 4),
    ("W2T2", 2, 2),
    ("W4T1", 4, 1),
    ("W4T4", 4, 4),
]


def _host_snapshot() -> dict:
    """cpuset / nproc / loadavg — co-tenant noise disclosure."""
    snap = {
        "utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "sched_getaffinity": sorted(os.sched_getaffinity(0)),
        "nproc_affinity": len(os.sched_getaffinity(0)),
        "cpu_count": os.cpu_count(),
        "loadavg_1_5_15": [round(x, 2) for x in os.getloadavg()],
    }
    try:
        snap["cpuset_cpus_effective"] = (
            Path("/sys/fs/cgroup/cpuset.cpus.effective").read_text().strip())
    except OSError:
        snap["cpuset_cpus_effective"] = None
    return snap


def _child(args) -> int:
    """One timed pass: construct UNA, bracket exactly the RunBatch call."""
    os.environ["NUMBA_CACHE_DIR"] = str(args.numba_cache_dir)
    if args.threads is not None:
        for key in THREAD_ENV_KEYS:
            os.environ[key] = str(args.threads)
    sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "src"))

    import urban_network_analysis as una_pkg
    from urban_network_analysis import UNA as UnaClass
    from urban_network_analysis.Execution import ExecutionOptions

    manifest = args.manifest.resolve()
    manifest_sha = hashlib.sha256(manifest.read_bytes()).hexdigest()
    if manifest_sha != MANIFEST_PIN:
        print(f"manifest sha {manifest_sha} != pin {MANIFEST_PIN}",
              file=sys.stderr)
        return 2

    una = UnaClass()
    t0 = time.perf_counter()
    if args.workers is None:
        una.RunBatch("accessibility", pairing_file=str(manifest),
                     parallel=False)
    else:
        una.RunBatch("accessibility", pairing_file=str(manifest),
                     parallel=True, workers=args.workers,
                     execution=ExecutionOptions(
                         threads_per_worker=args.threads))
    wall_s = time.perf_counter() - t0

    line = {
        "config": args.label,
        "wall_s": round(wall_s, 3),
        "rep": args.rep,
        "manifest_sha_matches_p00_pin": manifest_sha == MANIFEST_PIN,
        "package": str(Path(una_pkg.__file__).resolve()),
    }
    try:
        import numba
        line["numba_threads_seen"] = numba.get_num_threads()
        try:
            line["numba_threading_layer"] = numba.threading_layer()
        except ValueError:      # no parallel target compiled in this leg
            pass
    except ImportError:
        pass
    report = getattr(una, "batch_report", None)
    if report is not None:
        line["workers_effective"] = getattr(report, "workers_effective", None)
        line["pool_notes"] = [n for n in
                              getattr(report, "notes_runtime", ())
                              if "row-worker pool" in n]
        line["rows_committed"] = len(getattr(report, "commit_order", ()))
    print(RESULT_PREFIX + json.dumps(line), flush=True)
    return 0


def _leg(args, label: str, workers, threads, rep: int,
         legs_dir: Path) -> dict:
    """Run one child subprocess; collect its RESULT_JSON line."""
    cmd = [str(args.python), str(Path(__file__).resolve()), "--child",
           "--manifest", str(args.manifest),
           "--numba-cache-dir", str(args.numba_cache_dir),
           "--label", label, "--rep", str(rep)]
    if workers is not None:
        cmd += ["--workers", str(workers), "--threads", str(threads)]
    env = os.environ.copy()
    env["NUMBA_CACHE_DIR"] = str(args.numba_cache_dir)
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[4] / "src")
    if threads is None:     # serial leg: ambient env, the P01R regime
        for key in THREAD_ENV_KEYS:
            env.pop(key, None)
    else:
        for key in THREAD_ENV_KEYS:
            env[key] = str(threads)
    leg = {"config": label, "rep": rep, "parallel": workers is not None}
    if workers is not None:
        leg["workers"] = workers
        leg["threads_per_worker"] = threads
    t0 = time.monotonic()
    try:
        proc = subprocess.run(cmd, env=env, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True,
                              timeout=PASS_TIMEOUT_S)
        leg["rc"] = proc.returncode
        leg_log = proc.stdout
    except subprocess.TimeoutExpired as exc:
        leg["timeout"] = True
        leg["timeout_s"] = PASS_TIMEOUT_S
        leg["parent_wait_s"] = round(time.monotonic() - t0, 3)
        leg_log = exc.stdout if isinstance(exc.stdout, str) else \
            (exc.stdout or b"").decode(errors="replace")
    (legs_dir / f"{label}_rep{rep}.log").write_text(leg_log or "")
    for row in (leg_log or "").splitlines():
        if row.startswith(RESULT_PREFIX):
            leg.update(json.loads(row[len(RESULT_PREFIX):]))
            break
    else:
        leg["child_error"] = True    # no RESULT_JSON line: child died early
    return leg


def _parent(args) -> int:
    manifest = args.manifest.resolve()
    manifest_sha = hashlib.sha256(manifest.read_bytes()).hexdigest()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    legs_dir = args.out_dir / "legs"
    legs_dir.mkdir(exist_ok=True)

    result = {
        "schema": "una10x-next-p03c-thread-matrix-1",
        "purpose": "P03C precondition telemetry (plan 6.3, P01 H3): "
                   "worker/numba-thread topology vs wall on the current "
                   "tree; telemetry only, no speedup claim, no gate",
        "python": sys.executable,
        "python_version": sys.version.split()[0],
        "child_interpreter": str(args.python),
        "manifest": str(manifest),
        "manifest_sha256": manifest_sha,
        "manifest_sha_matches_p00_pin": manifest_sha == MANIFEST_PIN,
        "numba_cache_dir": str(args.numba_cache_dir),
        "pass_timeout_s": PASS_TIMEOUT_S,
        "reps": list(REPS),
        "matrix": [{"config": label, "parallel": workers is not None,
                    "workers": workers, "threads_per_worker": threads}
                   for label, workers, threads in MATRIX],
        "regime_notes": [
            "one subprocess per leg: numba reads NUMBA_NUM_THREADS at "
            "import time; OMP/BLAS budgets follow the declared config",
            "parallel legs pass execution=ExecutionOptions("
            "threads_per_worker=T): the spawn shim overwrites the worker "
            "thread env from that option, so env alone would not reach "
            "workers (runtime.py _Pool -> una_batch_worker_main.pool_main)",
            "serial reference leg: parallel=False, ambient thread env "
            "(the P01R declared serial regime)",
            "one untimed warm pass (W4T1) precedes timed legs; shared "
            "numba cache dir makes every later leg warm",
            "perf_counter brackets exactly the public RunBatch call in "
            "each child; parent collects RESULT_JSON lines",
            "whole matrix under nice 10 on the login node (probe-scale "
            "precedent, not charged to the sbatch budget); host snapshots "
            "at start/end disclose co-tenant noise",
        ],
        "host_start": _host_snapshot(),
        "results": [],
        "summary": {},
    }
    try:
        import importlib.metadata as md
        result["versions"] = {pkg: md.version(pkg) for pkg in
                              ("numba", "numpy", "llvmlite", "geopandas",
                               "pyogrio")}
    except Exception as exc:    # version strings only; never fail a leg
        result["versions"] = {"error": repr(exc)}

    # Untimed warm pass first (config W4T1) so JIT/cache cold cost does not
    # pollute timed legs.
    warm = _leg(args, "warm", 4, 1, 0, legs_dir)
    warm["untimed"] = True
    result["warm_pass"] = warm

    for label, workers, threads in MATRIX:
        walls = []
        aborted = False
        for rep in REPS:
            leg = _leg(args, label, workers, threads, rep, legs_dir)
            result["results"].append(leg)
            if leg.get("timeout"):
                aborted = True     # record honestly; abort remaining reps
                break
            if "wall_s" in leg:
                walls.append(leg["wall_s"])
        entry = {"walls_s": walls, "aborted": aborted}
        if walls:
            entry["median_wall_s"] = round(statistics.median(walls), 3)
        result["summary"][label] = entry

    result["host_end"] = _host_snapshot()
    medians = {label: entry["median_wall_s"]
               for label, entry in result["summary"].items()
               if "median_wall_s" in entry}
    result["observed_ordering"] = [
        label for label, _ in sorted(medians.items(), key=lambda kv: kv[1])]
    result["reading_notes"] = (
        "Mechanical ordering only (median wall, ascending): "
        + ", ".join(f"{k}={v}s" for k, v in sorted(medians.items(),
                                                   key=lambda kv: kv[1]))
        + ". Interpretation (H3 oversubscription reading) is in the "
          "receipt next-p03c; this file establishes no claim.")

    out = args.out_dir / "thread_matrix.json"
    out.write_text(json.dumps(result, indent=1))
    print(json.dumps({"out": str(out),
                      "manifest_sha_matches_p00_pin":
                          result["manifest_sha_matches_p00_pin"],
                      "observed_ordering": result["observed_ordering"]}))
    # A matrix in which no leg produced a wall is a systemic failure, not a
    # result — fail loudly instead of writing an all-empty summary as exit 0.
    return 0 if medians else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path,
                    help="matrix output dir (parent mode only)")
    ap.add_argument("--numba-cache-dir", type=Path, default=Path(
        "/storage/scratch1/1/dsynn6/campaign_data/numba_cache_p00_cand"))
    ap.add_argument("--python", type=Path, default=Path(sys.executable),
                    help="interpreter for child legs (parent mode)")
    ap.add_argument("--child", action="store_true")
    ap.add_argument("--label")
    ap.add_argument("--rep", type=int)
    ap.add_argument("--workers", type=int)
    ap.add_argument("--threads", type=int)
    args = ap.parse_args()
    if args.child:
        return _child(args)
    if args.out_dir is None:
        ap.error("--out-dir is required in parent mode")
    return _parent(args)


if __name__ == "__main__":
    raise SystemExit(main())
