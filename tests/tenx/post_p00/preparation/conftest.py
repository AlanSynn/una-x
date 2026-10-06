"""Fixtures for the P02 preparation suite (increment A: reader/state
equivalence evidence — NO dispatch change).

The parent post_p00 conftest already pins sys.path and the smoke grid;
this package adds the stage-cache settings builder and the format
materialization helper shared by the PP01 reader-equivalence and the
Add*-state-equivalence fixtures.  Evidence-only: nothing here may change
produced-path dispatch (P02I increment A boundary, contract increments.A).
"""
from __future__ import annotations

from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[4]
SMOKE = REPO / "tests" / "execution" / "fixtures"

FORMATS = ("geojson", "feather", "parquet")

# Import the CLASS from the submodule explicitly: any earlier submodule
# import elsewhere in the session (e.g. the batch worker chain) rebinds
# urban_network_analysis.Settings to the module object, so the lazy
# package-level export is not stable across test sessions.
from urban_network_analysis.Settings import Settings as _Settings  # noqa: E402


@pytest.fixture(scope="session")
def smoke_layers():
    """The three frozen smoke layers, read once through the original
    public reader stack (the PP01 comparison baseline for format
    materialization)."""
    gpd = pytest.importorskip("geopandas")
    try:
        import pyarrow  # noqa: F401
    except ImportError:
        pytest.skip("pyarrow required for the read_arrow engine path")
    return {
        name: gpd.read_file(str(SMOKE / f"{name}.geojson"),
                            engine="pyogrio", use_arrow=True)
        for name in ("network", "origins", "destinations")
    }


@pytest.fixture
def format_dir(tmp_path, smoke_layers):
    """Materialize the smoke grid in a requested admitted format so every
    route comparison is internally consistent (geojson | feather |
    parquet)."""
    gpd = pytest.importorskip("geopandas")

    def build(fmt):
        if fmt not in FORMATS:
            raise ValueError(f"unsupported format {fmt!r}")
        d = tmp_path / f"data_{fmt}"
        d.mkdir(parents=True, exist_ok=True)
        for name, gdf in smoke_layers.items():
            out = d / f"{name}.{fmt}"
            if fmt == "geojson":
                gdf.to_file(out, driver="GeoJSON")
            elif fmt == "feather":
                gdf.to_feather(out)
            else:
                gdf.to_parquet(out)
        return d

    return build


@pytest.fixture
def make_prep_settings():
    """Settings builder over a data dir; cache policy goes through the
    four serialized execution dotted keys exactly like the invalidation
    suite (tests/cache/invalidation/conftest.py)."""
    def build(data_dir, *, cache_mode=None, cache_dir=None, network=None,
              origins=None, destinations=None, **over):
        s = _Settings()
        s.data_folder = str(data_dir)
        s.output_folder = str(Path(data_dir).parent / "out")
        s.network_file = network or "network.geojson"
        s.origins_file = origins or "origins.geojson"
        s.destinations_file = destinations or "destinations.geojson"
        s.progressbar = False
        s.logger_verbosity = 0
        for k, v in over.items():
            if k.startswith("execution."):
                s._apply_execution_key(k, v)
            else:
                setattr(s, k, v)
        if cache_dir is not None:
            s._apply_execution_key(
                "execution.cache.mode", cache_mode or "disk")
            s._apply_execution_key("execution.cache.directory", cache_dir)
        return s

    return build


@pytest.fixture
def stage_stats():
    """Hit/miss accounting for one cache root via the public stage-cache
    factory."""
    from urban_network_analysis.cache.stages import stage_cache_for

    def stats(settings):
        st = stage_cache_for(settings.execution.cache).stats()
        return {"hits": st["hits"], "misses": st["misses"]}

    return stats
