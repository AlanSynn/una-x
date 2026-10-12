"""P01-parallel: region telemetry of the PARALLEL PREPARED boundary on current main.

The serial region-share probe (profile_regions_main.py) measures a boundary that
structurally never arms the landed P02 prepared route (NR-6: worker-route-only).
This probe measures the actual tenx gate boundary instead:

  - the PARALLEL public call: RunBatch("accessibility", parallel=True, workers=4)
    with NUMBA_NUM_THREADS=1 (the w4t1 gate shape),
  - on the SAME frozen cohort manifest (sha256 asserted),
  - pass 1 untimed warm (JIT + prepared-store build-or-hit), pass 2 parent-cProfiled.

The cProfile sees ONLY the coordinator process: dispatch, pickling/IPC, joins/waits,
commit/journal/hashing, publication. Worker-internal time is invisible except as wait
buckets — that asymmetry is itself the measurement (coordinator-vs-wait split). The
prepared-store regime (files/bytes before/after) is recorded so cold-build vs warm-hit
is declared, never assumed.

Bucket mapping revision 3 (parallel-aware; first matching regex wins; the rest falls
to "other"). No speedup claim; telemetry only.

Run (alan interpreter, repo root of the branch worktree):
    PYTHONPATH=src python campaigns/una_10x/next/probes/profile_parallel_prepared.py \
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

# Parallel-boundary families. Mapping revision 3: coordinator/IPC/verify split out
# (the P01 H1/H2 axes), prepared/restore separated from load (prepared hits skip
# pyogrio), raw.py:write correctly under export (the mapping-v2 lesson).
BUCKETS = [
    ("coordinator_commit",
     r"commit|journal|checkpoint|_publish|cohort_store|finalize|_commit"),
    ("ipc", r"pickle|queue|_send|_recv|_pack|_unpack|feed|put\(|get\(|pipe"),
    ("verify_hash", r"sha256|hashlib|verify|digest|_file_identity"),
    ("prepared_route",
     r"prepared|prep_session|flight|arena|PrepSession|_prepared_stage_cache"),
    ("restore_decode", r"from_wkb|_restore|sindex|strtree|STRtree|_geoms_from"),
    ("load_gis", r"pyogrio|read_arrow|geoio|geopandas|read_file|read_feather"),
    ("metrics_fold", r"fold|reach|gravity|knn|scope|integrated_scope_access"),
    ("export_io", r"export|to_csv|to_file|writer|_write|write_feather"),
    ("pool_wait", r"join|wait|result\(|apply_async|map\(|monitor|drain"),
    ("topology", r"topology|networkx|csr|adjacency|BuildTopology"),
]


def _store_state(root: Path) -> dict:
    if not root.is_dir():
        return {"exists": False, "files": 0, "bytes": 0}
    n = 0
    total = 0
    for p in root.rglob("*"):
        if p.is_file():
            try:
                total += p.stat().st_size
                n += 1
            except OSError:
                pass
    return {"exists": True, "files": n, "bytes": total}


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

    # The prepared store root resolves under the first dispatched row's public
    # folder; for this manifest that is the Results/ dir next to the inputs.
    store_root = manifest.parent / "Results" / ".una-batch-prepared"
    store_before = _store_state(store_root)

    result = {
        "schema": "una10x-next-p01p-parallel-region-1",
        "purpose": "parallel prepared-boundary telemetry on current main; no claim",
        "python": sys.executable,
        "package": str(Path(una_pkg.__file__).resolve()),
        "manifest": str(manifest),
        "manifest_sha256": manifest_sha,
        "manifest_sha_matches_p00_pin": manifest_sha ==
        "1161f4857a3d11640b883cbd25dac212ffda84c1157c1dba9d2cfd18ee95daaf",
        "numba_cache_dir": str(args.numba_cache_dir),
        "numba_num_threads_env": os.environ.get("NUMBA_NUM_THREADS"),
        "workers": args.workers,
        "regime": ("parallel public call w4t1; pass 1 untimed warm, "
                   "pass 2 parent-cProfiled; worker-internal time invisible "
                   "except as wait buckets (declared asymmetry)"),
        "bucket_regexes": BUCKETS,
        "prepared_store_root": str(store_root),
        "prepared_store_before": store_before,
        "cpuset_cpus": len(os.sched_getaffinity(0)),
        "loadavg_start": os.getloadavg(),
    }

    una = UnaClass()
    t0 = time.perf_counter()
    una.RunBatch("accessibility", pairing_file=str(manifest),
                 parallel=True, workers=args.workers)
    warm_s = time.perf_counter() - t0
    result["warm_pass_wall_s"] = warm_s

    una2 = UnaClass()
    prof = cProfile.Profile()
    t1 = time.perf_counter()
    prof.enable()
    rb = una2.RunBatch("accessibility", pairing_file=str(manifest),
                       parallel=True, workers=args.workers)
    prof.disable()
    wall_s = time.perf_counter() - t1
    result["profiled_pass_wall_s"] = wall_s
    if isinstance(rb, dict):
        result["runbatch_return_keys"] = sorted(rb.keys())[:40]

    store_after = _store_state(store_root)
    result["prepared_store_after"] = store_after
    result["prepared_store_regime_observed"] = (
        "cold_build_in_pass1" if store_after["files"] > store_before["files"]
        else "warm_hits" if store_before["files"] > 0 else "no_store_seen")
    result["loadavg_end"] = os.getloadavg()

    pstats_path = args.out_dir / "prof_parent_w4.pstats"
    prof.dump_stats(str(pstats_path))
    stats = pstats.Stats(str(pstats_path))

    buckets = {name: {"tottime_s": 0.0, "cumtime_max_s": 0.0}
               for name, _ in BUCKETS}
    buckets["other"] = {"tottime_s": 0.0, "cumtime_max_s": 0.0}
    total_self = 0.0
    top = []
    for (_file, _line, func_name), (_cc, _nc, tt, ct, _callers) in \
            stats.stats.items():
        total_self += tt
        top.append((tt, ct, f"{_file}:{_line}:{func_name}"))
        hit = "other"
        for name, pattern in BUCKETS:
            if re.search(pattern, func_name, re.IGNORECASE) or \
                    re.search(pattern, _file, re.IGNORECASE):
                hit = name
                break
        buckets[hit]["tottime_s"] += tt
        buckets[hit]["cumtime_max_s"] = max(buckets[hit]["cumtime_max_s"], ct)
    for name, b in buckets.items():
        b["share_of_profiled_self"] = (
            b["tottime_s"] / total_self) if total_self else None
    result["total_profiled_parent_self_s"] = total_self
    result["parent_buckets"] = buckets
    top.sort(reverse=True)
    result["parent_top_self"] = [
        {"tottime_s": round(tt, 3), "cumtime_s": round(ct, 3), "loc": loc}
        for tt, ct, loc in top[:25]]
    result["pstats"] = str(pstats_path)

    out = args.out_dir / "parallel_region_share.json"
    out.write_text(json.dumps(result, indent=1))
    print(json.dumps({"out": str(out), "warm_s": warm_s, "wall_s": wall_s,
                      "store_regime": result["prepared_store_regime_observed"],
                      "manifest_sha_matches_p00_pin":
                          result["manifest_sha_matches_p00_pin"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
