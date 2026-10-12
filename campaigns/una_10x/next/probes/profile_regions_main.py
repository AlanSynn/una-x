"""Region-share telemetry on the CURRENT main tree (P01-refresh, plan §6.1).

Mirrors the declared P00 serial regime so the bucket shares are comparable
with campaign_runs/una_10x/tenx-20261003T1212Z/p00_profile/ (stage_model.json,
P00_ANALYSIS_BRIEF.md §3.2):

  - the SAME cohort manifest (sha256 pinned in the receipt; expected
    1161f4857a3d11640b883cbd25dac212ffda84c1157c1dba9d2cfd18ee95daaf),
  - the on-disk numba cache owner directory from the P00 pins (reused; the
    current tree's signatures differ, so pass 1 below is the declared
    untimed warm pass and pass 2 is the profiled pass),
  - the SERIAL public call only: RunBatch("accessibility", parallel=False)
    with the timer/profiler bracketing exactly that call.

The profiler is heuristic bucketing (the original P00 profiler script is not
tracked); the regex mapping is recorded in the output JSON so the shares are
auditable and re-derivable from the saved pstats. This probe measures where
time goes; it establishes no speedup.

Run (alan interpreter, repo root of the branch worktree):
    PYTHONPATH=src python campaigns/una_10x/next/probes/profile_regions_main.py \
        --manifest <cohort_manifest.csv> --out-dir <run dir>
"""
from __future__ import annotations

import argparse
import cProfile
import hashlib
import json
import os
import pstats
import re
import sys
import time
from pathlib import Path

# Bucket families per P00_ANALYSIS_BRIEF.md §3.2 / stage_model.json names.
# Mapping revision 2 (2026-10-12): the first pass left two large families
# in "other" — `builtins.next` is the pyogrio arrow record-batch iterator
# (the P00 brief itself groups "pyogrio read_arrow + next + from_wkb" as
# one load family) and pyogrio's `raw.py:write` + feather cache writes are
# the export family. First matching regex wins; the rest falls to "other".
BUCKETS = [
    ("load_gis",
     r"pyogrio|read_arrow|from_wkb|builtins\.next|geoio|geopandas|fiona"),
    ("metrics_fold",
     r"fold|reach|gravity|knn|scope|integrated_scope_access"),
    ("export_io",
     r"export|to_csv|to_file|writer|_write|write_feather|raw\.py.*:write$"),
    ("topology", r"topology|networkx|csr|adjacency|strtree|BuildTopology"),
    ("trace_build", r"trace"),
    ("search_heap", r"heap|dijkstra|_search"),
    ("cohort_store", r"cohort_store|checkpoint|journal"),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--numba-cache-dir", type=Path, default=Path(
        "/storage/scratch1/1/dsynn6/campaign_data/numba_cache_p00_cand"))
    args = ap.parse_args()

    manifest = args.manifest.resolve()
    manifest_sha = hashlib.sha256(manifest.read_bytes()).hexdigest()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    os.environ["NUMBA_CACHE_DIR"] = str(args.numba_cache_dir)
    sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "src"))

    import urban_network_analysis as una_pkg
    from urban_network_analysis import UNA as UnaClass

    result = {
        "schema": "una10x-next-p01r-region-share-1",
        "purpose": "refresh region shares on current main; no speedup claim",
        "python": sys.executable,
        "package": str(Path(una_pkg.__file__).resolve()),
        "manifest": str(manifest),
        "manifest_sha256": manifest_sha,
        "manifest_sha_matches_p00_pin": manifest_sha ==
        "1161f4857a3d11640b883cbd25dac212ffda84c1157c1dba9d2cfd18ee95daaf",
        "numba_cache_dir": str(args.numba_cache_dir),
        "regime": "serial public call; pass 1 untimed warm, pass 2 profiled",
        "bucket_regexes": BUCKETS,
    }

    una = UnaClass()
    t0 = time.perf_counter()
    una.RunBatch("accessibility", pairing_file=str(manifest),
                 parallel=False)
    warm_s = time.perf_counter() - t0
    result["warm_pass_wall_s"] = warm_s

    una2 = UnaClass()
    prof = cProfile.Profile()
    t1 = time.perf_counter()
    prof.enable()
    una2.RunBatch("accessibility", pairing_file=str(manifest),
                  parallel=False)
    prof.disable()
    wall_s = time.perf_counter() - t1
    result["profiled_pass_wall_s"] = wall_s

    pstats_path = args.out_dir / "prof_main_serial.pstats"
    prof.dump_stats(str(pstats_path))
    stats = pstats.Stats(str(pstats_path))

    buckets = {name: {"tottime_s": 0.0, "cumtime_max_s": 0.0}
               for name, _ in BUCKETS}
    buckets["other"] = {"tottime_s": 0.0, "cumtime_max_s": 0.0}
    total_self = 0.0
    for (_file, _line, func_name), (_cc, _nc, tt, ct, _callers) in \
            stats.stats.items():
        total_self += tt
        hit = "other"
        for name, pattern in BUCKETS:
            if re.search(pattern, func_name, re.IGNORECASE) or \
                    re.search(pattern, _file, re.IGNORECASE):
                hit = name
                break
        buckets[hit]["tottime_s"] += tt
        buckets[hit]["cumtime_max_s"] = max(
            buckets[hit]["cumtime_max_s"], ct)
    for name, b in buckets.items():
        b["share_of_profiled_self"] = (
            b["tottime_s"] / total_self) if total_self else None
    result["total_profiled_self_s"] = total_self
    result["buckets"] = buckets
    result["pstats"] = str(pstats_path)

    out = args.out_dir / "region_share_main.json"
    out.write_text(json.dumps(result, indent=1))
    print(json.dumps({"out": str(out), "wall_s": wall_s,
                      "manifest_sha_matches_p00_pin":
                          result["manifest_sha_matches_p00_pin"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
