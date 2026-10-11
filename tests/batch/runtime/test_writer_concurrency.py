"""Writer-concurrency export overlap (post-closure candidate expw-20261011).

Laws under test:
  * default path (no ``_writer_concurrency`` transport attribute): the
    exact historical inline write path — no thread pool is ever created
    and the bytes equal direct serial writes;
  * wc > 1: per-format writes overlap on a bounded pool, the bytes are
    IDENTICAL to the serial path (GDAL's GeoJSON "name" member embeds the
    basename — final staging relpaths are used, never temp names), the
    Saved log lines are emitted after the join in the serial order
    feather -> csv -> geojson, and the pool is fully shut down (no
    lingering 'una-writer' threads);
  * the worker's env transport parses strictly and fails loudly;
  * a parallel RunBatch row run with wc > 1 leaves no
    ``_writer_concurrency`` residue on any transported Settings snapshot.
"""
from __future__ import annotations

import threading
from pathlib import Path

import geopandas as gpd
import pytest

from urban_network_analysis.Engines import Base as EnginesBase
from urban_network_analysis.batch import worker as worker_mod


@pytest.fixture
def routes_gdf():
    """Small routes-like frame: geometry + attribute columns."""
    from shapely.geometry import Point

    return gpd.GeoDataFrame(
        {"vid": [1, 2, 3], "cost": [1.5, 2.5, 3.5],
         "geometry": [Point(0, 0), Point(1, 1), Point(2, 2)]},
        crs="EPSG:26986",
    )


@pytest.fixture
def out_flags():
    """feather + geojson on, csv off (the cohort legs' flag set)."""
    import urban_network_analysis.Settings as _settings_mod

    s = _settings_mod.Settings()
    return s


class _RecordingLogger:
    def __init__(self):
        self.calls = []

    def log(self, label, msg, v=0):
        self.calls.append((label, msg, v))


@pytest.fixture
def rec_logger():
    return _RecordingLogger()


def _manual_serial_writes(gdf, folder, stem, settings):
    """The historical inline writes, performed directly (byte reference)."""
    import os

    folder = Path(folder)
    written = {}
    if settings.output_feather:
        p = folder / (stem + ".feather")
        gdf.to_feather(p)
        written["feather"] = p.read_bytes()
    if settings.output_csv:
        p = folder / (stem + ".csv")
        gdf.drop(columns="geometry").to_csv(
            p, sep=settings.csv_delimiter, index=False)
        written["csv"] = p.read_bytes()
    if settings.output_geojson and "geometry" in gdf.columns:
        p = folder / (stem + ".geojson")
        try:
            import pyogrio
            pyogrio.write_dataframe(
                gdf, str(p), driver="GeoJSON",
                layer_options={"COORDINATE_PRECISION": "6"})
        except ImportError:
            gdf.to_file(str(p), driver="GeoJSON")
        written["geojson"] = p.read_bytes()
    return written


def _writer_threads():
    return [t for t in threading.enumerate()
            if t.name.startswith("una-writer")]


# ---------------------------------------------------------------------------
# write_gdf_outputs
# ---------------------------------------------------------------------------

def test_default_path_creates_no_thread_pool(
        routes_gdf, out_flags, tmp_path, rec_logger, monkeypatch):
    """No transport attribute -> the historical inline path, no pool."""
    out = tmp_path / "out"
    out.mkdir()
    import os
    real_tpe = EnginesBase.ThreadPoolExecutor

    def _no_pool(*a, **k):
        raise AssertionError(
            "ThreadPoolExecutor created on the default (wc=1) path")

    monkeypatch.setattr(EnginesBase, "ThreadPoolExecutor", _no_pool)
    wrote = EnginesBase.write_gdf_outputs(
        routes_gdf, out_flags, str(out), "r0", rec_logger, "T", desc="routes")
    assert wrote is True
    assert [c[1] for c in rec_logger.calls] == [
        f"Saved routes feather → {os.path.join(str(out), 'r0.feather')}",
        f"Saved routes GeoJSON → {os.path.join(str(out), 'r0.geojson')}",
    ]
    ref = _manual_serial_writes(routes_gdf, out, "r0", out_flags)
    assert (out / "r0.feather").read_bytes() == ref["feather"]
    assert (out / "r0.geojson").read_bytes() == ref["geojson"]
    assert real_tpe is not None  # (silence linters; patch above did the work)


def test_wc2_bytes_identical_to_serial(
        routes_gdf, out_flags, tmp_path, rec_logger):
    """wc=2 output bytes == manual serial bytes, per format."""
    out = tmp_path / "out"
    out.mkdir()
    EnginesBase.write_gdf_outputs(
        routes_gdf, out_flags, str(out), "r0", rec_logger, "T", desc="routes",
        writer_concurrency=2)
    ref = _manual_serial_writes(routes_gdf, out, "r0", out_flags)
    assert (out / "r0.feather").read_bytes() == ref["feather"]
    assert (out / "r0.geojson").read_bytes() == ref["geojson"]
    assert _writer_threads() == [], "writer pool not fully shut down"


def test_wc3_csv_included_and_log_order_serial(
        routes_gdf, out_flags, tmp_path, rec_logger, monkeypatch):
    """3 formats, wc=3: csv written, log order feather->csv->geojson even
    when the feather write is the slowest."""
    import os

    out = tmp_path / "out"
    out.mkdir()
    out_flags.output_csv = True

    real_to_feather = gpd.GeoDataFrame.to_feather
    calls = []

    def slow_to_feather(self, path, *a, **k):
        calls.append("feather_start")
        import time
        time.sleep(0.3)
        real_to_feather(self, path, *a, **k)
        calls.append("feather_end")

    monkeypatch.setattr(gpd.GeoDataFrame, "to_feather", slow_to_feather)
    EnginesBase.write_gdf_outputs(
        routes_gdf, out_flags, str(out), "r0", rec_logger, "T", desc="routes",
        writer_concurrency=3)
    # all three writes completed (feather_end precedes every log line)
    assert calls == ["feather_start", "feather_end"]
    msgs = [c[1] for c in rec_logger.calls]
    assert msgs == [
        f"Saved routes feather → {os.path.join(str(out), 'r0.feather')}",
        f"Saved routes CSV → {os.path.join(str(out), 'r0.csv')}",
        f"Saved routes GeoJSON → {os.path.join(str(out), 'r0.geojson')}",
    ]
    ref = _manual_serial_writes(routes_gdf, out, "r0", out_flags)
    assert (out / "r0.csv").read_bytes() == ref["csv"]
    assert (out / "r0.feather").read_bytes() == ref["feather"]
    assert (out / "r0.geojson").read_bytes() == ref["geojson"]
    assert _writer_threads() == []


def test_single_format_wc4_takes_inline_path(
        routes_gdf, out_flags, tmp_path, rec_logger, monkeypatch):
    """One enabled format + wc=4 -> nothing to overlap, inline path."""
    out = tmp_path / "out"
    out.mkdir()
    out_flags.output_geojson = False

    def _no_pool(*a, **k):
        raise AssertionError("pool created for a single enabled format")

    monkeypatch.setattr(EnginesBase, "ThreadPoolExecutor", _no_pool)
    wrote = EnginesBase.write_gdf_outputs(
        routes_gdf, out_flags, str(out), "r0", rec_logger, "T",
        writer_concurrency=4)
    assert wrote is True
    ref = _manual_serial_writes(routes_gdf, out, "r0", out_flags)
    assert (out / "r0.feather").read_bytes() == ref["feather"]


def test_write_failure_raises_and_pool_drains(
        routes_gdf, out_flags, tmp_path, rec_logger):
    """First-in-order failure is re-raised after the join; no threads."""
    out = tmp_path / "out"
    out.mkdir()

    def boom(self, path, *a, **k):
        raise RuntimeError("injected feather failure")

    orig = gpd.GeoDataFrame.to_feather
    gpd.GeoDataFrame.to_feather = boom
    try:
        with pytest.raises(RuntimeError, match="injected feather failure"):
            EnginesBase.write_gdf_outputs(
                routes_gdf, out_flags, str(out), "r0", rec_logger, "T",
                writer_concurrency=2)
    finally:
        gpd.GeoDataFrame.to_feather = orig
    assert _writer_threads() == []
    # no log line may precede the failure (logs come after the join)
    assert rec_logger.calls == []


# ---------------------------------------------------------------------------
# ExportAccessibilityResults
# ---------------------------------------------------------------------------

class _FakeTopology:
    def __init__(self, origins_gdf, writer_concurrency=None):
        self.origins = origins_gdf
        if writer_concurrency is not None:
            self._writer_concurrency = writer_concurrency


class _FakeEngine:
    def __init__(self, origins_gdf, logger):
        self.topology = _FakeTopology(origins_gdf)
        self.reach = [1.0, 2.0]
        self.gravity_logistic = None
        self.gravity_exponential = None
        self.knn_access = None
        self.logger = logger


def _origins_gdf():
    from shapely.geometry import Point

    return gpd.GeoDataFrame(
        {"geometry": [Point(0, 0), Point(1, 1)]}, crs="EPSG:26986")


def _make_settings(tmp_path, wc):
    import urban_network_analysis.Settings as _settings_mod

    s = _settings_mod.Settings()
    s.output_folder = str(tmp_path / f"out_{wc}")
    s.output_wStamp = False
    return s


def _exercise_export(tmp_path, wc):
    """Run ExportAccessibilityResults once at wc (None = default path)."""
    from urban_network_analysis.Engines import Base as B

    out = tmp_path / f"out_{wc}"
    out.mkdir()
    logger = _RecordingLogger()
    engine = _FakeEngine(
        _FakeTopology(_origins_gdf(), writer_concurrency=wc), logger)
    s = _make_settings(tmp_path, wc)
    B.Base.ExportAccessibilityResults(engine, s, file_name="cen")
    files = {}
    for fmt in ("feather", "geojson"):
        p = out / f"cen.{fmt}"
        files[fmt] = p.read_bytes() if p.exists() else None
    return files, [c for c in logger.calls if c[0] == "Export Results"]


def test_export_accessibility_results_wc2_identical(tmp_path):
    """wc=2 export bytes + log lines == the default-path export."""
    default_files, default_logs = _exercise_export(tmp_path, None)
    wc_files, wc_logs = _exercise_export(tmp_path, 2)
    assert default_files["feather"] is not None
    assert default_files["feather"] == wc_files["feather"]
    assert default_files["geojson"] == wc_files["geojson"]
    norm = lambda msgs: [m.split("exported to ")[0] for m in msgs]  # noqa: E731
    assert norm([c[1] for c in default_logs]) == norm([c[1] for c in wc_logs])
    assert _writer_threads() == []


# ---------------------------------------------------------------------------
# worker env transport
# ---------------------------------------------------------------------------

def test_worker_env_parse(monkeypatch):
    monkeypatch.delenv(worker_mod._WRITER_CONCURRENCY_ENV, raising=False)
    assert worker_mod._load_writer_concurrency() == 1
    monkeypatch.setenv(worker_mod._WRITER_CONCURRENCY_ENV, "4")
    assert worker_mod._load_writer_concurrency() == 4
    monkeypatch.setenv(worker_mod._WRITER_CONCURRENCY_ENV, " 2 ")
    assert worker_mod._load_writer_concurrency() == 2
    monkeypatch.setenv(worker_mod._WRITER_CONCURRENCY_ENV, "0")
    with pytest.raises(ValueError, match="must be >= 1"):
        worker_mod._load_writer_concurrency()
    monkeypatch.setenv(worker_mod._WRITER_CONCURRENCY_ENV, "x")
    with pytest.raises(ValueError, match="not an integer"):
        worker_mod._load_writer_concurrency()


# ---------------------------------------------------------------------------
# end-to-end: parallel RunBatch with wc > 1
# ---------------------------------------------------------------------------

def test_runbatch_parallel_wc4_no_settings_leak(
        make_batch, tmp_path, monkeypatch):
    """A wc=4 parallel run: rows ran, and no ``_writer_concurrency``
    residue on any post-run Settings snapshot (runtime-only transport)."""
    out_a = tmp_path / "wc4"
    monkeypatch.setenv(worker_mod._WRITER_CONCURRENCY_ENV, "4")
    p = make_batch(2, out_folder=out_a)
    p.RunBatch("accessibility", parallel=True, workers=2)
    assert all(r.status == "ran" for r in p.batch_report.rows)
    for s in p.projects:
        assert not hasattr(s, "_writer_concurrency"), \
            "runtime-only writer_concurrency leaked into Settings state"
