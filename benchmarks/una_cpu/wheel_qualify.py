"""T10 installed-wheel qualification (dossier 07).

Runs INSIDE a clean venv whose only relevant content is the installed wheel,
with CWD outside any repository. Proves, for that installed copy:

  1. import path is the venv's site-packages (not a source tree)
  2. importlib.metadata version == package __version__; about() works
  3. --expect-csr present: the ordered-CSR helper is packaged and the
     engine's construction engages it on genuine int32 input (counter via
     monkeypatch, non-performance test) and does NOT on crafted unsupported
     input (int16 endpoints -> original fallback, engine still works)
     --expect-csr absent: helper import must fail (baseline wheel)
  4. CSV / GeoJSON / Feather writers work through the public export path
  5. no packet/reference module is imported by the runtime

Output: one JSON record at --out.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import sys
import tempfile
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--expect-csr", choices=["present", "absent"],
                    required=True)
    ap.add_argument("--tests-dir", required=True,
                    help="tests/perf_contract dir providing stub_topology")
    ap.add_argument("--wheel-sha256", default="")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rec: dict = {"expect_csr": args.expect_csr,
                 "wheel_sha256": args.wheel_sha256,
                 "python": sys.version.split()[0],
                 "cwd": os.getcwd()}

    # --- 1. import path ---------------------------------------------------
    import urban_network_analysis as una
    pkg_path = Path(una.__file__).resolve()
    rec["package_file"] = str(pkg_path)
    rec["in_site_packages"] = "site-packages" in str(pkg_path)
    assert rec["in_site_packages"], f"import not from site-packages: {pkg_path}"
    assert "src" not in pkg_path.parts[:len(pkg_path.parts) - 4], \
        "unexpected source-tree path in import"

    # --- 2. version / about ----------------------------------------------
    rec["version_metadata"] = importlib.metadata.version("urban-network-analysis")
    rec["version_package"] = una.__version__
    assert rec["version_metadata"] == rec["version_package"]
    rec["about"] = str(una.about())

    # --- 3. helper presence + path selection counters ----------------------
    sys.path.insert(0, str(Path(args.tests_dir).resolve()))
    from stub_topology import StubTopology  # noqa: E402

    import numpy as np  # noqa: E402

    from urban_network_analysis import Settings  # noqa: E402
    from urban_network_analysis.Engines.Accessibility import (  # noqa: E402
        Accessibility,
    )
    import urban_network_analysis.Engines.Accessibility as acc_mod  # noqa: E402

    counts = {"calls": 0, "returned_csr": 0}
    real_builder = getattr(acc_mod, "_try_build_ordered_csr", None)

    def counting(*a, **k):
        counts["calls"] += 1
        out = real_builder(*a, **k)
        if out is not None:
            counts["returned_csr"] += 1
        return out

    if args.expect_csr == "present":
        assert real_builder is not None, "helper missing from wheel"
        rec["helper_module"] = str(sys.modules[
            "urban_network_analysis.Engines.Accessibility"].__file__)

        def make_topo(endpoint_dtype):
            rng = np.random.default_rng(20260925)
            ne = 40
            edges = rng.integers(0, 12, size=(ne, 2))
            w = rng.random(ne) * 10.0 + 0.5
            z = rng.random(12) * 20.0
            o = (rng.integers(0, 12, 4), rng.integers(0, 12, 4),
                 rng.random(4) + 0.5, rng.random(4), None)
            d = (rng.integers(0, 12, 6), rng.integers(0, 12, 6),
                 rng.random(6), rng.random(6), rng.random(6) * 2.0)
            return StubTopology(
                edges[:, 0].copy(), edges[:, 1].copy(), w, 12,
                o[0], o[1], o[2], o[3], d[0], d[1], d[2], d[3], d[4],
                z=z, start_dtype=endpoint_dtype, end_dtype=endpoint_dtype)

        s = Settings()
        s.elevation = True
        s.elevation_penalty = 0.3
        s.search_radius = 40.0

        acc_mod._try_build_ordered_csr = counting
        try:
            # genuine real-topology representation: int32 endpoints on BOTH
            # columns (measured representation of real network files)
            acc32 = Accessibility(make_topo(np.int32))
            rec["genuine_int32"] = dict(counts,
                pointer_last=int(acc32.graph_engine.adjacency_pointer[-1]),
                incidences=int(acc32.graph_engine.adjacency_vector.shape[0]))
            assert counts["calls"] == 1 and counts["returned_csr"] == 1, \
                f"fast path not engaged on genuine input: {counts}"
            assert rec["genuine_int32"]["pointer_last"] == 2 * 40
            counts.update({"calls": 0, "returned_csr": 0})
            # crafted unsupported representation: int16 endpoints -> the
            # engine consults the helper once, the helper REFUSES (None), and
            # the original construction executes (fallback semantics)
            acc16 = Accessibility(make_topo(np.int16))
            rec["crafted_int16"] = dict(counts,
                pointer_last=int(acc16.graph_engine.adjacency_pointer[-1]),
                incidences=int(acc16.graph_engine.adjacency_vector.shape[0]))
            assert counts["calls"] == 1 and counts["returned_csr"] == 0, \
                f"int16 must be refused by the guard (fallback): {counts}"
            assert rec["crafted_int16"]["pointer_last"] == 2 * 40
        finally:
            acc_mod._try_build_ordered_csr = real_builder
    else:
        assert real_builder is None, \
            "baseline wheel must not contain the helper"
        rec["helper_module"] = None

    # --- 4. writers through the public export path ------------------------
    from urban_network_analysis import UNA  # noqa: E402
    from urban_network_analysis.Topology import Topology  # noqa: E402
    from urban_network_analysis.Engines.AccessibilityWElevation import (  # noqa: E402
        AccessibilityWElevation,
    )

    with tempfile.TemporaryDirectory() as td:
        # minimal genuine GeoJSON network via the package's own reader
        import geopandas as gpd
        from shapely.geometry import LineString
        rng = np.random.default_rng(7)
        rows = []
        for i in range(30):
            a = (float(rng.integers(0, 10)), float(rng.integers(0, 10)))
            b = (a[0] + float(rng.integers(1, 4)), a[1] + float(rng.integers(0, 3)))
            rows.append({"geometry": LineString([a, b]), "id": i})
        net = gpd.GeoDataFrame(rows, crs="EPSG:32631")
        net_path = Path(td) / "network.geojson"
        net.to_file(net_path, driver="GeoJSON")
        # origin/destination point layers snapped by the public Topology API
        def points_gdf(n, seed):
            r = np.random.default_rng(seed)
            return gpd.GeoDataFrame(
                {"geometry": [Point(float(r.integers(0, 12)),
                                    float(r.integers(0, 12)))
                              for _ in range(n)]},
                crs="EPSG:32631")
        from shapely.geometry import Point
        for name, n, seed in (("origins", 6, 11), ("destinations", 8, 12)):
            points_gdf(n, seed).to_file(Path(td) / f"{name}.geojson",
                                        driver="GeoJSON")

        s = Settings()
        s.network_file = str(net_path)
        s.origins_file = str(Path(td) / "origins.geojson")
        s.destinations_file = str(Path(td) / "destinations.geojson")
        s.output_folder = td
        s.output_file_name = "Results"
        s.output_csv = True
        s.output_geojson = True
        s.output_feather = True
        s.csv_delimiter = ","
        s.result_prefix = "acc_"
        s.knn_decay = "none"
        s.elevation = False
        s.elevation_penalty = 0.3
        s.search_radius = 50.0

        topo = Topology(verbosity=0)
        topo.AddNetwork(s)
        topo.AddOrigins(s)
        topo.AddDestinations(s)
        acc = AccessibilityWElevation(topo, s)
        acc.Centrality(s)
        acc.ExportAccessibilityResults(s, folder_prefix="accessibility_",
                                       file_name=s.output_file_name)
        made = sorted(p.name for p in Path(td).glob("accessibility_*/Results.*"))
        rec["export_files"] = made
        for ext in (".csv", ".geojson", ".feather"):
            assert any(f.endswith(ext) for f in made), f"missing {ext}: {made}"
        for p in Path(td).glob("accessibility_*/Results.*"):
            assert p.stat().st_size > 0, f"empty export {p}"

    # --- 5. no packet/reference module in the runtime ---------------------
    rec["packet_modules_loaded"] = sorted(
        m for m in sys.modules
        if m.split(".")[0] in ("optimization_una_cpu", "prior"))
    assert not rec["packet_modules_loaded"], rec["packet_modules_loaded"]

    rec["ok"] = True
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rec, indent=1, default=str))
    print(json.dumps(rec, indent=1, default=str))


if __name__ == "__main__":
    main()
