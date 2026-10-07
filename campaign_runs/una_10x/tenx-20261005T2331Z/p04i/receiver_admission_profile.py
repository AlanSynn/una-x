"""P04I admission leg (una_10x post_p00, designs/05): re-profile the
RESIDUAL on the current tree — post P02I prepared single-flight, P03C
compiled replay, P03M-B metric reuse (rider cb7acd5) — over the
P01F-registered anchor-verified 32-row control cohort with P01M phase
telemetry ON, and decompose where the wall time now goes.

Designs/05 admission question this leg answers WITH MEASUREMENT: does
post-P02/P03 telemetry show MATERIAL result-put/commit backpressure
(coordinator recv-wait + commit-side spans) — or is the residual
compute-bound, so the receiver change is NOT admitted?

Run discipline (matched-arm law, BENCHMARKS.md):
  * timer brackets ONLY the public RunBatch call;
  * fresh output dir, existence recorded BEFORE the call;
  * NUMBA_CACHE_DIR pinned to a fresh per-leg dir (asserted);
  * UNA_BATCH_PHASES=1 (P01M telemetry, sampler overhead measured in
    P01M — diagnostics only, never primary timing);
  * w4 parallel, cache mode off (fixed policy) — the same shape as the
    P01F off_par_w4 control legs for phase-fraction comparability;
  * full row census + output census + counters after the call;
  * tree identity pinned from --tree.

This script records FACTS only (spans, totals, counters, fractions).
The admission VERDICT is a separate object (receiver_admission.json).
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
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


_REUSE_KEYS = ("reuse_bundle_hits", "reuse_split_rows",
               "reuse_split_copies", "reuse_split_folds",
               "reuse_collapse_full_folds", "reuse_registrations",
               "reuse_refusals", "reuse_split_domain_refused")


def _reuse_engagement(cohort_summary):
    """Run-level reuse engagement aggregated from the per-row deltas
    (the runtime summary predates the P03M counters; deltas carry
    them).  Empty dict when the cohort route did not engage."""
    if not isinstance(cohort_summary, dict):
        return {}
    per_row = cohort_summary.get("per_row") or {}
    agg = {k: 0 for k in _REUSE_KEYS}
    seen = False
    modes = {}
    for stats in per_row.values():
        row_modes = stats.get("reuse_mode")
        for k in _REUSE_KEYS:
            if k in stats:
                seen = True
                agg[k] += stats[k] or 0
        if row_modes:
            modes[str(row_modes)] = modes.get(str(row_modes), 0) + 1
    if not seen:
        return {}
    agg["reuse_modes_seen"] = modes
    return agg


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree", required=True)
    ap.add_argument("--pairing", required=True)
    ap.add_argument("--data-folder", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    assert os.environ.get("NUMBA_CACHE_DIR"), \
        "NUMBA_CACHE_DIR must be pinned to a fresh per-leg dir"
    assert os.environ.get("UNA_BATCH_PHASES") == "1", \
        "UNA_BATCH_PHASES=1 required for the admission profile"

    tree = Path(args.tree).resolve()
    pairing = Path(args.pairing).resolve()
    outdir = Path(args.outdir).resolve()
    outdir_existed_before = outdir.exists()

    t_import0 = time.perf_counter()
    from urban_network_analysis import UNA
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
        "una_batch_phases": os.environ["UNA_BATCH_PHASES"],
        "numba_version": __import__("numba").__version__,
        "numpy_version": __import__("numpy").__version__,
        "host_ncpu": os.cpu_count(),
    }

    p = UNA(verbosity=0)
    p.settings.output_folder = str(outdir)
    p.settings.data_folder = str(Path(args.data_folder).resolve())

    t0 = time.perf_counter()
    try:
        p.RunBatch("accessibility", str(pairing), parallel=True,
                   workers=args.workers)
    finally:
        public_call_wall_s = time.perf_counter() - t0

    report = p.batch_report
    assert report is not None, "batch call aborted; no report"
    rows = [{"index": r.index, "name": r.name, "status": r.status,
             "backend": getattr(r, "backend", None),
             "worker_pid": getattr(r, "worker_pid", None)}
            for r in report.rows]
    zero_failed = all(r["status"] == "ran" for r in rows)

    extra = report.extra
    coord_phases = list(extra.get("coordinator_phases", []))
    phase_totals = dict(extra.get("phase_totals", {}))
    row_transport = extra.get("row_transport", {})
    cohort_summary = extra.get("cohort", {})
    prepared_summary = extra.get("prepared", {})

    # Worker-side spans: aggregate wall_ns by phase across all rows.
    worker_phase_totals = {}
    for r in report.rows:
        for span in (getattr(r, "phases", None) or ()):
            agg = worker_phase_totals.setdefault(
                span["phase"], {"count": 0, "wall_ns": 0})
            agg["count"] += 1
            agg["wall_ns"] += int(span["end_mono_ns"]) - \
                int(span["start_mono_ns"])

    # Coordinator-side span aggregation.
    coord_phase_totals = {}
    for span in coord_phases:
        agg = coord_phase_totals.setdefault(
            span["phase"], {"count": 0, "wall_ns": 0})
        agg["count"] += 1
        agg["wall_ns"] += int(span["end_mono_ns"]) - \
            int(span["start_mono_ns"])

    # Per-row transport latency (dispatch -> recv, coordinator epoch).
    transport = []
    for key, stamps in sorted(row_transport.items(),
                              key=lambda kv: int(kv[0])):
        d, v = stamps.get("dispatch_mono_ns"), stamps.get("recv_mono_ns")
        transport.append({"row": int(key),
                          "dispatch_to_recv_s":
                              (v - d) / 1e9 if d and v else None})

    # Output census from the report's own manifest.
    out_files = 0
    out_bytes = 0
    for rels in report.output_manifest.values():
        for rp in rels:
            f = Path(rp)
            if f.exists():
                out_files += 1
                out_bytes += f.stat().st_size

    recv_wait_s = phase_totals.get("result_recv_wait_ns", 0) / 1e9
    commit_side_phases = ("staged_verify", "publish", "state_delta",
                          "composite", "journal")
    commit_side_s = sum(
        coord_phase_totals.get(ph, {"wall_ns": 0})["wall_ns"]
        for ph in commit_side_phases) / 1e9
    worker_run_s = worker_phase_totals.get("run", {}).get("wall_ns", 0) / 1e9
    worker_bootstrap_s = worker_phase_totals.get(
        "bootstrap_import", {}).get("wall_ns", 0) / 1e9

    doc = {
        "schema": "una-10x-p04i-admission-profile-1",
        "task": "P04I residual re-profile leg (designs/05 admission "
                "question) — FACTS ONLY",
        "generated": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                   time.gmtime()),
        "pins": pins,
        "config": {
            "parallel": True, "workers": args.workers,
            "threads_per_worker": 1, "cache_mode": "off",
            "outdir": str(outdir),
            "outdir_existed_before": outdir_existed_before,
            "cohort_note": "P01F anchor-verified 32-row control cohort "
                           "(local_audit_manifest.csv, sha 235405de); "
                           "w4 matches the P01F off_par_w4 control arm "
                           "for phase-fraction comparability",
        },
        "timing": {
            "import_s": import_s,
            "public_call_wall_s": public_call_wall_s,
            "timer_note": "wall brackets ONLY the public RunBatch call "
                          "(BENCHMARKS.md); import recorded separately",
        },
        "rows": rows,
        "row_count": len(rows),
        "zero_failed_jobs": zero_failed,
        "workers_effective": report.workers_effective,
        "cancelled": report.cancelled,
        "output_census": {"files": out_files, "bytes": out_bytes,
                          "note": "from report.output_manifest, live "
                                  "stat at collection time"},
        "telemetry_integrity": {
            "dropped_events": extra.get("telemetry_dropped_events"),
            "buffer_eviction_caveat":
                "the bounded span buffer evicted late-run spans "
                "(surviving counts per phase are in "
                "coordinator_phase_totals); commit_side_spans_s is "
                "therefore a LOWER BOUND from surviving spans — "
                "project the surviving per-row means across all rows "
                "before drawing conclusions; result_recv_wait_ns is a "
                "bounded accumulator (no eviction) and is reliable",
            "note": "P01M telemetry is diagnostics-only (designs/01); "
                    "fractions below are diagnostic decomposition, not "
                    "primary timing evidence",
        },
        "decomposition": {
            "recv_wait_s_coordinator_total": recv_wait_s,
            "commit_side_spans_s": commit_side_s,
            "commit_side_phases": commit_side_phases,
            "worker_run_span_s_sum_over_workers": worker_run_s,
            "worker_bootstrap_import_s_sum_over_workers":
                worker_bootstrap_s,
            "recv_wait_fraction_of_wall":
                recv_wait_s / public_call_wall_s,
            "commit_side_fraction_of_wall":
                commit_side_s / public_call_wall_s,
        },
        "coordinator_phase_totals": coord_phase_totals,
        "worker_phase_totals": worker_phase_totals,
        "phase_totals_raw": phase_totals,
        "row_transport": transport,
        "cohort_summary": {
            k: cohort_summary.get(k) for k in
            ("rows", "builds", "faultins", "resident_hits",
             "singleflight_lost", "replays", "folds_done",
             "build_s", "faultin_s", "replay_s",
             "distinct_trace_identities", "refusals", "store_root")
            if isinstance(cohort_summary, dict) and k in cohort_summary},
        "cohort_per_row": {
            str(i): {k: stats.get(k) for k in
                     ("trace_id", "metric_id", "reuse_mode",
                      "builds", "faultins", "resident_hits", "replays",
                      "folds_done", "reuse_bundle_hits",
                      "reuse_split_rows", "reuse_split_folds",
                      "reuse_split_copies",
                      "reuse_collapse_full_folds", "reuse_refusals",
                      "reuse_registrations",
                      "reuse_split_domain_refused")}
            for i, stats in sorted(
                (cohort_summary or {}).get("per_row", {}).items(),
                key=lambda kv: int(kv[0]))},
        "cohort_reuse_engagement": _reuse_engagement(cohort_summary),
        "prepared_summary": prepared_summary,
    }
    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=2) + "\n")
    print("wrote", out)
    print("rows:", len(rows), "all ran:", zero_failed,
          "| wall:", round(public_call_wall_s, 4), "s",
          "| workers:", report.workers_effective)
    print("recv_wait_s:", round(recv_wait_s, 4),
          "(", round(100 * recv_wait_s / public_call_wall_s, 2), "% wall )")
    print("commit_side_s:", round(commit_side_s, 4),
          "(", round(100 * commit_side_s / public_call_wall_s, 2), "% wall )")
    print("worker run sum:", round(worker_run_s, 2), "s",
          "| bootstrap sum:", round(worker_bootstrap_s, 2), "s")


if __name__ == "__main__":
    main()
