"""N00 measured-leg runner (una_10x, C-arm on the control cohort).

Mirrors the P05S leg runner's structure and asserts (the matched-arm
law, designs/07 + BENCHMARKS.md) with the N00 deltas only:

  * ``--arm native|numba``: the native arm installs the OPT-IN cohort
    hook (trace_build_native.install()) after imports and snapshots the
    engagement counters around the public call; the numba arm touches
    nothing.  Both arms run under the SAME interpreter (venv-n00).
  * load average is sampled on the same 0.5 s clock as RSS (the
    in-run co-tenancy record; the driver separately gates on the
    pre-leg loadavg and records it in the leg console).
  * engagement asserts: a native leg MUST show dispatches >= 1 and
    admission_refusals unchanged-by-error (fail the leg, never pass
    silently); a numba leg MUST show engagement untouched.
  * schema una-10x-n00-leg-run-1; ``parent_runner`` pins the P05S
    runner's bytes this file was derived from.

The timer brackets ONLY the public RunBatch call; import and process
overhead are recorded separately.  Shakedown runs are labelled by the
caller via --tag and are never campaign figures.
"""
import argparse
import dataclasses
import faulthandler
import hashlib
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git(tree, *args):
    return subprocess.run(["git", "-C", str(tree), *args],
                          capture_output=True, text=True).stdout.strip()


class _TreeSampler(threading.Thread):
    """Peak RSS (single + tree) + load average at a fixed interval."""

    # 0.5 s (the p05s runner used 0.05 s on a quiet host; psutil
    # children() snapshots the whole system process table on macOS, at
    # 20 Hz under co-tenancy that Python loop starves the leg's own
    # workers - measured in the aborted first shakedown, 2026-10-07).
    def __init__(self, interval_s=0.5):
        super().__init__(daemon=True)
        import psutil
        self._psutil = psutil
        self._proc = psutil.Process()
        self._interval = interval_s
        self.peak_single = 0
        self.peak_tree = 0
        self.load_samples = []
        self._stop_evt = threading.Event()

    def run(self):
        while not self._stop_evt.is_set():
            try:
                mem = self._proc.memory_info().rss
                self.peak_single = max(self.peak_single, mem)
                tree = mem
                for c in self._proc.children(recursive=True):
                    try:
                        tree += c.memory_info().rss
                    except self._psutil.Error:
                        pass
                self.peak_tree = max(self.peak_tree, tree)
                self.load_samples.append(os.getloadavg()[0])
            except self._psutil.Error:
                pass
            self._stop_evt.wait(self._interval)

    def stop(self):
        self._stop_evt.set()
        self.join(timeout=5)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree", required=True)
    ap.add_argument("--pairing", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--arm", choices=["native", "numba"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--parallel", action="store_true")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--threads", type=int, default=1,
                    help="ExecutionOptions.threads_per_worker")
    ap.add_argument("--queue-depth", type=int, default=64,
                    help="ExecutionOptions.queue_depth")
    ap.add_argument("--cache-mode", choices=["off", "disk"], default="off")
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--data-folder", default=None,
                    help="settings.data_folder for pairing file resolution")
    args = ap.parse_args(argv)

    assert os.environ.get("NUMBA_CACHE_DIR"), \
        "NUMBA_CACHE_DIR must be pinned to a per-arm fresh dir"
    assert os.environ.get("PYTHONDONTWRITEBYTECODE") == "1", \
        "PYTHONDONTWRITEBYTECODE=1 required (compile-check law)"

    tree = Path(args.tree).resolve()
    pairing = Path(args.pairing).resolve()
    outdir = Path(args.outdir).resolve()
    outdir_existed_before = outdir.exists()
    if args.cache_mode == "disk":
        assert args.cache_dir, "--cache-dir required with --cache-mode disk"
        cache_dir = Path(args.cache_dir).resolve()
        cache_dir_existed_before = cache_dir.exists()
    else:
        cache_dir = None
        cache_dir_existed_before = None

    t_import0 = time.perf_counter()
    from urban_network_analysis import UNA
    from urban_network_analysis.Execution import CacheOptions, ExecutionOptions
    import_s = time.perf_counter() - t_import0

    # Wedge triage (2026-10-07 shakedown): when the batch loop hits any
    # exception, runtime._cancel joins the inbox feeder thread - which
    # deadlocks forever when the feeder is blocked on a backlogged inbox
    # with no live worker readers - so the ORIGINAL exception never
    # propagates and the leg wedges silently.  The pending exception is
    # visible on this thread NOW: print it before delegating.  Print
    # only, no behavior change (the wrapper fires solely on a path that
    # already fails the leg).
    from urban_network_analysis.batch import runtime as _rt
    _orig_cancel = _rt._cancel

    def _cancel_with_exc_trace(pool, staging):
        exc = sys.exc_info()[1]
        print(f"[n00-triage] exception entering _cancel: "
              f"{type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.stderr.flush()
        return _orig_cancel(pool, staging)

    _rt._cancel = _cancel_with_exc_trace

    # N00 delta: opt-in native install (the ONLY production wiring) and
    # the engagement snapshot frame around the public call.
    engagement_before = None
    engagement_after = None
    if args.arm == "native":
        import trace_build_native as tbn
        tbn.install()
        from urban_network_analysis.batch import cohort as cohort_mod
        assert cohort_mod._NATIVE_BUILD_HOOK is tbn.native_build_trace
        engagement_before = tbn.engagement()

    pins = {
        "tree_head": _git(tree, "rev-parse", "HEAD"),
        "tree_dirty": bool(_git(tree, "status", "--porcelain")),
        "manifest_sha256": _sha256(pairing),
        "manifest_bytes": pairing.stat().st_size,
        "manifest_path": str(pairing),
        "interpreter": sys.version,
        "executable": sys.executable,
        "numba_cache_dir": os.environ["NUMBA_CACHE_DIR"],
        "numba_version": __import__("numba").__version__,
        "numpy_version": __import__("numpy").__version__,
        "host_ncpu": os.cpu_count(),
        "loadavg_at_import_done": os.getloadavg(),
        "parent_runner": ("campaign_runs/una_10x/tenx-20261005T2331Z/"
                          "p05s/run_screen_leg.py"),
    }

    p = UNA(verbosity=0)
    p.settings.output_folder = str(outdir)
    if args.data_folder:
        p.settings.data_folder = str(Path(args.data_folder).resolve())

    execution = ExecutionOptions(
        threads_per_worker=args.threads,
        queue_depth=args.queue_depth,
        cache=(CacheOptions(mode="disk", directory=str(cache_dir))
               if args.cache_mode == "disk" else CacheOptions()))

    # Wedge forensics (first shakedown wedged 2026-10-07 in an opaque
    # write-block that `sample` could not attribute to a line): if this
    # leg ever wedges, faulthandler dumps EVERY thread's Python stack to
    # the sidecar every WEDGE_DUMP_S so the driver's leg timeout leaves
    # a named artifact instead of a silent pipe.  Cancelled on success;
    # the sidecar then carries the explicit clean marker (so absence of
    # a dump is never an inference - the file always says which).
    stacks_path = Path(str(args.out) + ".stacks.txt")
    WEDGE_DUMP_S = 300
    with stacks_path.open("w") as stacks_file:
        faulthandler.dump_traceback_later(WEDGE_DUMP_S, repeat=True,
                                          file=stacks_file)
        sampler = _TreeSampler()
        sampler.start()
        t0 = time.perf_counter()
        try:
            p.RunBatch("accessibility", str(pairing),
                       parallel=args.parallel, workers=args.workers,
                       execution=execution)
        finally:
            public_call_wall_s = time.perf_counter() - t0
            sampler.stop()
            faulthandler.cancel_dump_traceback_later()
        stacks_file.write("clean: no wedge dump fired (watchdog %ds, "
                          "cancelled after %.4fs)\n"
                          % (WEDGE_DUMP_S, public_call_wall_s))

    if args.arm == "native":
        import trace_build_native as tbn
        engagement_after = tbn.engagement()
        tbn.uninstall()

    report = p.batch_report
    rows = [{"index": r.index, "name": r.name, "status": r.status,
             "backend": getattr(r, "backend", None),
             "detail": getattr(r, "detail", None)}
            for r in report.rows]

    out_files = []
    if outdir.exists():
        for f in sorted(outdir.rglob("*")):
            if f.is_file():
                out_files.append({"rel": str(f.relative_to(outdir)),
                                  "bytes": f.stat().st_size,
                                  "sha256": _sha256(f)})

    cache_entries = None
    if cache_dir is not None and cache_dir.exists():
        cache_entries = sum(1 for f in cache_dir.rglob("*") if f.is_file())

    zero_failed = all(r["status"] == "ran" for r in rows)
    loads = sampler.load_samples
    stacks_text = stacks_path.read_text()
    wedge_clean = stacks_text.startswith("clean:")

    out = {
        "schema": "una-10x-n00-leg-run-1",
        "tag": args.tag,
        "arm": args.arm,
        "task": "N00 C-arm measured leg (control cohort, designs/07 "
                "matched-window law)",
        "pins": pins,
        "config": {
            "parallel": args.parallel,
            "workers": args.workers,
            "threads_per_worker": args.threads,
            "queue_depth": args.queue_depth,
            "cache_mode": args.cache_mode,
            "cache_dir": str(cache_dir) if cache_dir else None,
            "cache_dir_existed_before": cache_dir_existed_before,
            "writer_concurrency": 1,
        },
        "outdir": str(outdir),
        "outdir_existed_before": outdir_existed_before,
        "timing": {
            "import_s": round(import_s, 4),
            "public_call_wall_s": round(public_call_wall_s, 4),
            "timer_note": "wall brackets ONLY the public RunBatch call; "
                          "import recorded separately (BENCHMARKS.md)",
        },
        "sampler": {
            "interval_s": 0.5,
            "peak_single_rss_bytes": sampler.peak_single,
            "peak_tree_rss_bytes": sampler.peak_tree,
            "loadavg_1m_min": round(min(loads), 2) if loads else None,
            "loadavg_1m_max": round(max(loads), 2) if loads else None,
            "loadavg_1m_mean": (round(sum(loads) / len(loads), 2)
                                if loads else None),
            "load_samples_n": len(loads),
        },
        "wedge_watchdog": {
            "dump_interval_s": 300,
            "sidecar": str(stacks_path),
            "clean": wedge_clean,
        },
        "engagement": {
            "before": engagement_before,
            "after": engagement_after,
        },
        "row_count": len(rows),
        "zero_failed_jobs": zero_failed,
        "rows": rows,
        "effective_snapshot": (
            dataclasses.asdict(report.effective)
            if getattr(report, "effective", None) is not None else None),
        "output_files": out_files,
        "output_file_count": len(out_files),
        "output_bytes_total": sum(f["bytes"] for f in out_files),
        "cache_entries_after": cache_entries,
    }

    # engagement asserts (fail the leg, never silently pass)
    assert zero_failed, [r for r in rows if r["status"] != "ran"][:5]
    if args.cache_mode == "off":
        assert not cache_entries, "cache dir populated under mode 'off'"
    if args.arm == "native":
        assert engagement_after is not None and \
            engagement_before is not None
        assert engagement_after["dispatches"] > \
            engagement_before["dispatches"], \
            "native arm dispatched nothing - install() wiring dead"
        assert engagement_after["errors"] == engagement_before["errors"]
    else:
        assert engagement_before is None and engagement_after is None

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1) + "\n")
    print(f"LEG OK tag={args.tag} arm={args.arm} rows={len(rows)} "
          f"wall={public_call_wall_s:.4f}s files={len(out_files)} "
          f"bytes={sum(f['bytes'] for f in out_files)} "
          f"load1m[start={pins['loadavg_at_import_done'][0]:.2f} "
          f"max={out['sampler']['loadavg_1m_max']}] -> {args.out}")


if __name__ == "__main__":
    main()
