"""R3 diagnostic: are the P03M typed metric routes LIVE on the frozen
beta cohort?  One untimed warm pass (store + numba warm) then ONE
measured pass whose cohort counters are dumped verbatim:

  extra["cohort"]["metric_split_folds"]  admitted ge-only split folds
  extra["cohort"]["metric_fallbacks"]    split refused -> whole-row path
  extra["cohort"]["metric_errors"]       internal identity/store fault
  extra["cohort"]["metric_field_hits"]   typed metric fields served
  per-row deltas in extra["cohort"]["per_row"]

This is a DIAGNOSTIC probe: no timing claim, no gate; it settles which
metric-reuse machinery engages on cohort_manifest.csv (gravity_beta
varies, knn_decay=logistic).  Run from the branch root:
    PYTHONPATH=src python campaigns/una_10x/next/probes/r3_cohort_counters.py \
        --manifest <cohort_manifest.csv> --out <runs.json>
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
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    manifest = args.manifest.resolve()
    manifest_sha = hashlib.sha256(manifest.read_bytes()).hexdigest()
    os.environ["NUMBA_NUM_THREADS"] = "1"  # w4t1 gate shape
    sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "src"))

    import urban_network_analysis as una_pkg
    from urban_network_analysis import UNA as UnaClass
    from urban_network_analysis.Engines import _trace_access_scratch as tas

    result = {
        "schema": "una10x-next-r3-cohort-counters-1",
        "purpose": ("diagnostic: which P03M metric-reuse routes engage "
                    "on the frozen beta cohort; no timing claim"),
        "python": sys.executable,
        "package": str(Path(una_pkg.__file__).resolve()),
        "manifest": str(manifest),
        "manifest_sha256": manifest_sha,
        "manifest_sha_matches_p00_pin": manifest_sha ==
        "1161f4857a3d11640b883cbd25dac212ffda84c1157c1dba9d2cfd18ee95daaf",
        "replay_ge_split_admitted": bool(
            getattr(tas, "_REPLAY_GE_SPLIT_ADMITTED", False)),
        "workers": args.workers,
        "numba_num_threads_env": os.environ.get("NUMBA_NUM_THREADS"),
    }

    una = UnaClass()
    una.RunBatch("accessibility", pairing_file=str(manifest),
                 parallel=True, workers=args.workers)  # untimed warm

    una2 = UnaClass()
    t0 = time.perf_counter()
    una2.RunBatch("accessibility", pairing_file=str(manifest),
                  parallel=True, workers=args.workers)
    result["measured_pass_wall_s"] = round(time.perf_counter() - t0, 4)

    extra = getattr(una2.batch_report, "extra", None) or {}
    cohort = dict(extra.get("cohort") or {})
    per_row = cohort.pop("per_row", None)
    result["cohort_totals"] = cohort
    if per_row is not None:
        result["per_row_metric"] = {
            i: {k: v for k, v in row.items() if k.startswith("metric")}
            for i, row in per_row.items()}
    result["preparation"] = {
        k: (extra.get("preparation") or {}).get(k)
        for k in ("rows", "cold_build", "verified_hit", "lease_hits",
                  "decoded_nofit", "fallback_compute", "refusals")}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps({k: result[k] for k in (
        "cohort_totals", "measured_pass_wall_s",
        "replay_ge_split_admitted")}, indent=1))
    print(json.dumps({"out": str(out)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
