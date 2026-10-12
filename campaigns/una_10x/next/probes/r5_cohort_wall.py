"""R5 diagnostic: clean cohort-wall probe on the branch bytes (no telemetry,
no claim).  Two measured warm passes after one untimed warm pass; per-pass
wall + preparation counters (lease engagement truth).  Run from the branch
root:
    PYTHONPATH=src python campaigns/una_10x/next/probes/r5_cohort_wall.py \
        --manifest <cohort_manifest.csv> --out <runs.json> [--passes 2]
This is a DIAGNOSTIC run: the tenx-law paired gate is the only throughput
authority; this probe informs the commit-bound model, nothing else.
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
    ap.add_argument("--passes", type=int, default=2)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    manifest = args.manifest.resolve()
    manifest_sha = hashlib.sha256(manifest.read_bytes()).hexdigest()
    os.environ["NUMBA_NUM_THREADS"] = "1"  # w4t1 gate shape
    sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "src"))

    import urban_network_analysis as una_pkg
    from urban_network_analysis import UNA as UnaClass

    result = {
        "schema": "una10x-next-r5-cohort-wall-1",
        "purpose": ("diagnostic warm cohort wall on R5 branch bytes; "
                    "no speedup claim; not a gate"),
        "python": sys.executable,
        "package": str(Path(una_pkg.__file__).resolve()),
        "manifest": str(manifest),
        "manifest_sha256": manifest_sha,
        "manifest_sha_matches_p00_pin": manifest_sha ==
        "1161f4857a3d11640b883cbd25dac212ffda84c1157c1dba9d2cfd18ee95daaf",
        "workers": args.workers,
        "numba_num_threads_env": os.environ.get("NUMBA_NUM_THREADS"),
        "cpuset_cpus": len(os.sched_getaffinity(0)),
        "passes": [],
    }

    una = UnaClass()
    # untimed warm pass: builds the prepared store + numba caches
    una.RunBatch("accessibility", pairing_file=str(manifest),
                 parallel=True, workers=args.workers)
    for i in range(args.passes):
        una2 = UnaClass()
        t0 = time.perf_counter()
        una2.RunBatch("accessibility", pairing_file=str(manifest),
                      parallel=True, workers=args.workers)
        wall = time.perf_counter() - t0
        extra = getattr(una2.batch_report, "extra", None) or {}
        prep = extra.get("preparation") or {}
        result["passes"].append({
            "pass": i + 1,
            "wall_s": wall,
            "prep_rows": prep.get("rows"),
            "cold_build": prep.get("cold_build"),
            "verified_hit": prep.get("verified_hit"),
            "lease_hits": prep.get("lease_hits"),
            "decoded_nofit": prep.get("decoded_nofit"),
            "fallback_compute": prep.get("fallback_compute"),
            "refusals": prep.get("refusals"),
        })
        print(json.dumps(result["passes"][-1]))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1))
    print(json.dumps({"out": str(out)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
