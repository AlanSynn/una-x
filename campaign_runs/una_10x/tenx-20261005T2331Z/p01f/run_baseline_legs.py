"""P01F measured-leg runner (una_10x post_p00, baseline policy audit).

Runs ONE public RunBatch leg under the interpreter that executes this
script and records a P00-shaped run JSON.  The timer brackets ONLY the
public RunBatch call (BENCHMARKS.md); import and process overhead are
recorded separately.  No campaign figure may be claimed from a
shakedown run on warmup2.csv — functional shake-downs are labelled
"shakedown" by the caller via --tag.

Runner rules (matched-arm law, designs/07 + BENCHMARKS.md):
  * fresh output dir, existence recorded BEFORE the call;
  * NUMBA_CACHE_DIR must be provided and is recorded (per-arm fresh dir);
  * cache arms: mode 'off' (fixed policy) or 'disk' with an explicit
    per-arm directory (WS02 strongest-policy control) — never mixed
    across arms;
  * row statuses, output census (bytes + sha256 per file) and cache-dir
    engagement are asserted after the call;
  * tree identity (HEAD sha + dirty state) recorded from --tree.

Usage (repo-root-free; run under the arm's own venv):

    NUMBA_CACHE_DIR=<fresh dir> python run_baseline_legs.py \
        --tree <repo root> --pairing <cohort.csv> --outdir <fresh dir> \
        --tag <name> --out <run.json> \
        [--parallel --workers N] [--cache-mode disk --cache-dir D]
"""
import argparse
import dataclasses
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
    """Peak RSS (single + tree) at a fixed interval; daemon."""

    def __init__(self, interval_s=0.05):
        super().__init__(daemon=True)
        import psutil
        self._psutil = psutil
        self._proc = psutil.Process()
        self._interval = interval_s
        self.peak_single = 0
        self.peak_tree = 0
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
    ap.add_argument("--out", required=True)
    ap.add_argument("--parallel", action="store_true")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--cache-mode", choices=["off", "disk"], default="off")
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--row-cache-dir", default=None,
                    help="cache directory declared to ROWS via the pairing "
                         "CSV's execution.cache.directory column (row-level "
                         "policy); censused and asserted after the run")
    ap.add_argument("--data-folder", default=None,
                    help="settings.data_folder for pairing file resolution")
    args = ap.parse_args(argv)

    assert os.environ.get("NUMBA_CACHE_DIR"), \
        "NUMBA_CACHE_DIR must be pinned to a per-arm fresh dir"

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
    row_cache_dir = (Path(args.row_cache_dir).resolve()
                     if args.row_cache_dir else None)
    row_cache_entries_before = (
        sum(1 for f in row_cache_dir.rglob("*") if f.is_file())
        if row_cache_dir and row_cache_dir.exists() else 0)

    t_import0 = time.perf_counter()
    from urban_network_analysis import UNA
    from urban_network_analysis.Execution import CacheOptions, ExecutionOptions
    import_s = time.perf_counter() - t_import0

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
    }

    p = UNA(verbosity=0)
    p.settings.output_folder = str(outdir)
    if args.data_folder:
        p.settings.data_folder = str(Path(args.data_folder).resolve())

    execution = None
    if args.cache_mode == "disk":
        execution = ExecutionOptions(
            cache=CacheOptions(mode="disk", directory=str(cache_dir)))

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
    row_cache_entries_after = None
    if row_cache_dir is not None:
        assert row_cache_dir.exists(), \
            f"row-declared cache dir never created: {row_cache_dir}"
        row_cache_entries_after = sum(
            1 for f in row_cache_dir.rglob("*") if f.is_file())

    zero_failed = all(r["status"] == "ran" for r in rows)

    out = {
        "schema": "una-10x-p01f-leg-run-1",
        "tag": args.tag,
        "task": "P01F baseline policy audit measured leg",
        "pins": pins,
        "config": {
            "parallel": args.parallel,
            "workers": args.workers,
            "threads_per_worker": 1,
            "cache_mode": args.cache_mode,
            "cache_dir": str(cache_dir) if cache_dir else None,
            "cache_dir_existed_before": cache_dir_existed_before,
            "row_cache_dir": str(row_cache_dir) if row_cache_dir else None,
            "row_cache_entries_before": row_cache_entries_before,
            "queue_depth": 64,
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
            "interval_s": 0.05,
            "peak_single_rss_bytes": sampler.peak_single,
            "peak_tree_rss_bytes": sampler.peak_tree,
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
        "row_cache_entries_after": row_cache_entries_after,
    }

    # engagement asserts (fail the leg, never silently pass)
    assert zero_failed, [r for r in rows if r["status"] != "ran"][:5]
    if args.cache_mode == "off" and row_cache_dir is None:
        assert not cache_entries, "cache dir populated under mode 'off'"
    elif args.cache_mode == "disk":
        assert cache_entries, "disk cache dir empty after disk-mode run"
    if row_cache_dir is not None:
        if row_cache_entries_before == 0:
            assert row_cache_entries_after > 0, \
                "row-declared cache dir empty after a cold policy run"
        else:
            assert row_cache_entries_after >= row_cache_entries_before, \
                "row-declared cache dir shrank across a warm run"

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1) + "\n")
    print(f"LEG OK tag={args.tag} rows={len(rows)} "
          f"wall={public_call_wall_s:.4f}s files={len(out_files)} "
          f"-> {args.out}")


if __name__ == "__main__":
    main()
