"""BATCH_REVIEW probe 3: stage-cache identity and invalidation, end to end.

Independent replication of the dossier-06 invalidation matrix on the
reviewed bytes (e2d790b), with a producer counter on the o_access stage
(the MAJOR-1 seam: a cold cached run must produce exactly once; a warm
run must produce zero times and serve from the store).

Scenarios (smoke grid, accessibility non-turns route ->
AccessibilityWElevation):
  A1 cold / warm / again producer counts + byte parity
  A2 mtime change alone (bytes pristine)   -> reuse (producer 0)
  B  same path, different bytes            -> producer miss
  C  different bytes, mtime pinned to the pristine file's -> miss
  D  knn_weights float32 vs float64        -> miss (dtype closure)
  E  semantic_profile una_legacy vs madina_legacy -> miss
  F  output-only change (row name)         -> reuse, identical arrays
  G  cache mode 'off'                      -> zero key computations,
                                              no cache dir, identical bytes
"""
import hashlib
import os
import shutil
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[6]
sys.path.insert(0, str(REPO / "src"))

WORK = Path("/tmp/batch_review_probe3")
SMOKE = REPO / "tests" / "execution" / "fixtures"

from urban_network_analysis import UNA  # noqa: E402
from urban_network_analysis.Engines import (  # noqa: E402
    AccessibilityWElevation as AWE,
)
from urban_network_analysis.cache import identity as ckid  # noqa: E402
from urban_network_analysis.Execution import CacheOptions  # noqa: E402

NC = {"n": 0}   # o_access producer calls
NK = {"n": 0}   # identity.stage_key calls

_orig_produce = AWE.AccessibilityWElevation._compute_o_access
_orig_key = ckid.stage_key


def install():
    def counted_key(*a, **kw):
        NK["n"] += 1
        return _orig_key(*a, **kw)

    ckid.stage_key = counted_key
    # The single ground-truth producer of the o_access stage is the
    # graph engine's o_access: the cached path reaches it through
    # _compute_o_access, the uncached path calls it directly.  Count it
    # once for both.  The graph engine is an inner class of the module —
    # find it by shape rather than by name.
    import urban_network_analysis.Engines.AccessibilityWElevation as mod
    candidates = []
    for obj in vars(mod).values():
        if isinstance(obj, type):
            candidates.append(obj)
            candidates += [v for v in vars(obj).values()
                           if isinstance(v, type)]   # nested classes
    for cls in candidates:
        if "o_access" in vars(cls):
            orig = cls.__dict__["o_access"]

            def counted_o(self, o_idx, settings, d_weights,
                          _orig=orig, _cls=cls):
                NC["n"] += 1
                return _orig(self, o_idx, settings, d_weights)

            cls.o_access = counted_o
            _patched.append((cls, orig))
    if not _patched:
        raise RuntimeError("probe bug: no o_access producer found")


_patched = []


def reach_digest(una):
    return hashlib.sha256(
        una.accessibility.reach.tobytes()).hexdigest()[:16]


def run(cache_opts, data_dir=None, name="probe", knn=None):
    p = UNA(verbosity=0)
    p.settings.data_folder = str(data_dir or SMOKE)
    p.settings.network_file = "network.geojson"
    p.settings.origins_file = "origins.geojson"
    p.settings.destinations_file = "destinations.geojson"
    p.settings.output_folder = str(WORK / "out")
    p.settings.output_wStamp = False
    p.settings.name = name
    p.settings.search_radius = 95.0
    if knn is not None:
        p.settings.knn_weights = knn
    p.settings.execution = replace(p.execution, cache=cache_opts)
    p.RunAccessibility()
    return p


def main():
    install()
    if WORK.exists():
        shutil.rmtree(WORK)
    WORK.mkdir(parents=True)
    disk = CacheOptions(mode="disk", directory=str(WORK / "cache"))
    out_rows = []

    # A1: cold / warm / again --------------------------------------------
    NC["n"] = 0
    p = run(disk, name="A")
    cold_digest = reach_digest(p)
    c_old, c_warm, c_again = NC["n"], 0, 0
    p = run(disk, name="A")
    c_warm = NC["n"] - c_old
    d_warm = reach_digest(p)
    p = run(disk, name="A")
    c_again = NC["n"] - c_old - c_warm
    ok = (c_old, c_warm, c_again) == (1, 0, 0) \
        and d_warm == cold_digest
    out_rows.append(("A1 producers cold/warm/again",
                     (c_old, c_warm, c_again), (1, 0, 0), ok,
                     f"digest {cold_digest} stable"))

    # A2: mtime bump alone, bytes pristine -> reuse -----------------------
    before = NC["n"]
    os.utime(SMOKE / "network.geojson", None)   # mtime = now
    run(disk, name="A")
    reuse = NC["n"] - before
    out_rows.append(("A2 mtime change only", reuse, 0, reuse == 0,
                     "bytes unchanged; mtime is not identity"))

    # B: same path, different bytes -> miss --------------------------------
    import geopandas as gpd
    data = WORK / "data"
    data.mkdir()
    for f in ("network.geojson", "origins.geojson",
              "destinations.geojson"):
        shutil.copy(SMOKE / f, data / f)
    net = gpd.read_file(data / "network.geojson")
    net.geometry = net.geometry.translate(0.0005, 0.0005)
    net.to_file(data / "network.geojson", driver="GeoJSON")
    before = NC["n"]
    run(disk, data_dir=data, name="A")
    miss = NC["n"] - before
    out_rows.append(("B same path new bytes", miss, 1, miss == 1,
                     "content-addressed, path-free"))

    # C: different bytes, mtime pinned to the pristine file's -> miss -----
    pristine_stat = (SMOKE / "network.geojson").stat()
    net2 = gpd.read_file(data / "network.geojson")
    net2.geometry = net2.geometry.translate(0.0007, 0.0)
    net2.to_file(data / "network.geojson", driver="GeoJSON")
    os.utime(data / "network.geojson",
             (pristine_stat.st_atime, pristine_stat.st_mtime))
    before = NC["n"]
    run(disk, data_dir=data, name="A")
    miss = NC["n"] - before
    out_rows.append(("C new bytes + restored mtime", miss, 1, miss == 1,
                     "mtime matches nothing; bytes decide"))

    # D: knn_weights dtype flip -> miss ------------------------------------
    before = NC["n"]
    run(disk, name="A", knn=np.array([1.0, 0.5], dtype=np.float32))
    miss = NC["n"] - before
    out_rows.append(("D knn_weights float32", miss, 1, miss == 1,
                     "dtype is in the closure"))

    # E: semantic profile flip -> miss -------------------------------------
    before = NC["n"]
    p_e = run(disk, name="A")
    p_e.settings.execution = replace(
        p_e.settings.execution, semantic_profile="madina_legacy",
        cache=disk)
    p_e.RunAccessibility()
    miss = NC["n"] - before
    out_rows.append(("E profile una_legacy->madina_legacy", miss, 1,
                     miss == 1, "profile inside the env fingerprint"))

    # F: output-only change -> reuse ----------------------------------------
    before = NC["n"]
    p_f = run(disk, name="F_other_name")
    reuse = NC["n"] - before
    same = reach_digest(p_f) == cold_digest
    out_rows.append(("F output-only change (row name)", reuse, 0,
                     reuse == 0 and same,
                     f"arrays identical: {same}"))

    # G: mode 'off' fully inert ---------------------------------------------
    never = WORK / "never"
    NK["n"] = 0
    before = NC["n"]
    p_g = run(CacheOptions(mode="off", directory=str(never)), name="A")
    ok = NK["n"] == 0 and (NC["n"] - before) == 1 \
        and not never.exists() and reach_digest(p_g) == cold_digest
    out_rows.append(("G mode=off inert",
                     (NK["n"], NC["n"] - before, not never.exists()),
                     (0, 1, True), ok,
                     f"bytes equal cold run: "
                     f"{reach_digest(p_g) == cold_digest}"))

    ckid.stage_key = _orig_key
    for cls, orig in _patched:
        cls.o_access = orig

    print(f"{'scenario':<40} {'got':<26} {'want':<14} verdict")
    all_ok = True
    for name, got, want, ok, note in out_rows:
        all_ok &= ok
        print(f"{name:<40} {str(got):<26} {str(want):<14} "
              f"{'PASS' if ok else 'FAIL'}  # {note}")
    print("PROBE3 VERDICT:", "PASS" if all_ok else "FAIL")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
