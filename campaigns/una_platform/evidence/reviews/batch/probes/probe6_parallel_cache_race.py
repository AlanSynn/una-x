"""BATCH_REVIEW probe 6: parallel RunBatch x disk cache, same run.

Neither reviewed suite runs the BATCH_EXEC parallel path with the
CACHE_GRAPH disk store COLD, where several spawned workers contend for
identical stage keys in the same run (identical rows share the network,
snap and engine stages; cross-process safety rests on content-addressed
writes, not on any tested lock).

Row cache options are set BEFORE SaveSettingsToProject on every row
(review v1 set them on the live Settings after the snapshots and
silently measured an uncached batch — recorded in the probe log).

(a) serial reference: 3 identical rows, no cache, row artifacts kept.
(b) parallel cold cache: same rows, disk store cold, workers=2 ->
    no crash, every row artifact byte-identical to the serial
    reference, store holds exactly one entry per content key with no
    tmp/lock leftovers.
(c) parallel warm cache: rerun -> row artifacts byte-identical again
    and no store entry rewritten or added (workers actually reuse
    across processes).
"""
import hashlib
import os
import shutil
import sys
from dataclasses import replace
from pathlib import Path

REPO = Path(__file__).resolve().parents[6]
sys.path.insert(0, str(REPO / "src"))

WORK = Path("/tmp/batch_review_probe6")
SMOKE = REPO / "tests" / "execution" / "fixtures"
ARTS = (".feather", ".geojson")


def make(out, cache_dir=None, n=3):
    """Rows carry their disk-cache options BEFORE the snapshot."""
    from urban_network_analysis import UNA
    from urban_network_analysis.Execution import CacheOptions
    p = UNA(verbosity=0)
    p.settings.data_folder = str(SMOKE)
    p.settings.network_file = "network.geojson"
    p.settings.origins_file = "origins.geojson"
    p.settings.destinations_file = "destinations.geojson"
    p.settings.output_folder = str(out)
    p.settings.output_wStamp = False
    for i in range(n):
        p.settings.name = f"row{i}"
        p.settings.search_radius = 95.0   # identical rows -> shared stages
        if cache_dir is not None:
            p.settings.execution = replace(
                p.execution,
                cache=CacheOptions(mode="disk", directory=str(cache_dir)))
        p.SaveSettingsToProject()
    return p


def row_digests(out):
    return {
        f: hashlib.sha256((out / f).read_bytes()).hexdigest()
        for f in sorted(os.listdir(out))
        if f.startswith("row") and f.endswith(ARTS)
    }


def store_state(cache_dir):
    entries, tmps, mtimes = 0, 0, {}
    for root, _dirs, files in os.walk(cache_dir):
        for f in files:
            fp = Path(root) / f
            if f.endswith((".tmp", ".part", ".lock")):
                tmps += 1
                continue
            entries += 1
            mtimes[str(fp)] = fp.stat().st_mtime_ns
    return entries, tmps, mtimes


def main():
    if WORK.exists():
        shutil.rmtree(WORK)
    WORK.mkdir(parents=True)
    cache = WORK / "cache"
    ok = True

    # ---- (a) serial reference, no cache ----------------------------------
    ref = WORK / "ref"
    make(ref).RunBatch("accessibility")
    ref_rows = row_digests(ref)
    print(f"(a) serial reference: {len(ref_rows)} row artifacts")
    a_ok = len(ref_rows) == 6   # 3 rows x (feather + geojson)
    print("(a) VERDICT:", "PASS" if a_ok else "FAIL")
    ok &= a_ok

    # ---- (b) parallel, cold disk cache, contending workers ---------------
    out_b = WORK / "out_b"
    pb = make(out_b, cache_dir=cache)
    pb.RunBatch("accessibility", parallel=True, workers=2)
    par_rows = row_digests(out_b)
    entries, tmps, _ = store_state(cache)
    n_bad = sum(1 for f, h in par_rows.items() if ref_rows.get(f) != h)
    print(f"(b) parallel cold: {len(par_rows)} artifacts, {n_bad} differ "
          f"from serial bytes | store: {entries} entries, {tmps} tmp/lock")
    b_ok = n_bad == 0 and len(par_rows) == 6 and tmps == 0 and entries > 0
    print("(b) VERDICT:", "PASS" if b_ok else "FAIL")
    ok &= b_ok

    # ---- (c) parallel warm: cross-process reuse ---------------------------
    _, _, m1 = store_state(cache)
    out_c = WORK / "out_c"
    pc = make(out_c, cache_dir=cache)
    pc.RunBatch("accessibility", parallel=True, workers=2)
    warm_rows = row_digests(out_c)
    _, _, m2 = store_state(cache)
    rewritten = sum(1 for k, v in m2.items() if m1.get(k) != v)
    added = len(m2) - len(m1)
    warm_bad = sum(1 for f, h in warm_rows.items() if ref_rows.get(f) != h)
    print(f"(c) parallel warm: {warm_bad} artifacts differ from serial | "
          f"store rewritten: {rewritten}, added: {added}")
    c_ok = warm_bad == 0 and rewritten == 0 and added == 0
    print("(c) VERDICT:", "PASS" if c_ok else "FAIL")
    ok &= c_ok

    print("PROBE6 VERDICT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
