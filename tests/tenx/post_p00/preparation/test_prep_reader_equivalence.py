"""PP01 (P02 increment A): original path vs snapshot reader, per admitted
format.

TEST_MATRIX PP01: "Same values/schema/order/UIDs/CRS/dtypes/geometry and
required warnings/errors."  Increment A proves the two in-repo reader
routes are byte-equivalent per format BEFORE any prepared layer may reuse
them (contract layers 1-2; engagement stays .geojson-first until these
fixtures pass for the other formats).

The corrupt-bytes case pins only "both routes fail loudly" — the exception
CLASSES are recorded as observation in prepared_L0_L2.json, not asserted,
because BytesIO vs path decode can surface different wrappers on the same
underlying stack.
"""
from __future__ import annotations

import hashlib

import pytest

from urban_network_analysis.Topology import (
    Topology,
    _decode_snapshot_gdf,
    _read_snapshot,
)
from urban_network_analysis.cache.identity import file_digest

from geopandas.testing import assert_geodataframe_equal

LAYERS = ("network", "origins", "destinations")


@pytest.mark.parametrize("fmt", ("geojson", "feather", "parquet"))
def test_pp01_reader_routes_equivalent(format_dir, fmt):
    """The uncached _get_gdf route and the snapshot/decode route produce
    identical frames for every admitted format.  Exactness is pinned
    directly (geometry WKB bytes, column dtypes) so it does not depend on
    the geopandas helper's comparison options."""
    d = format_dir(fmt)
    for name in LAYERS:
        path = d / f"{name}.{fmt}"
        direct = Topology._get_gdf(str(path), None)
        snap = _decode_snapshot_gdf(str(path), _read_snapshot(str(path)),
                                    None)
        assert_geodataframe_equal(direct, snap, check_crs=True)
        assert list(direct.columns) == list(snap.columns)
        assert (direct.dtypes == snap.dtypes).all()
        assert [g.wkb for g in direct.geometry] == \
            [g.wkb for g in snap.geometry]


@pytest.mark.parametrize("fmt", ("geojson", "feather", "parquet"))
def test_pp01_snapshot_bytes_are_file_bytes(format_dir, fmt):
    """TOCTOU pin (contract toctou_law): _read_snapshot returns exactly the
    file bytes, and file_digest over the snapshot equals an independent
    sha256 of the file — the same bytes are hashed and decoded."""
    path = format_dir(fmt) / f"network.{fmt}"
    raw = path.read_bytes()
    snapshot = _read_snapshot(str(path))
    assert snapshot == raw
    assert file_digest(snapshot) == hashlib.sha256(raw).hexdigest()


@pytest.mark.parametrize("fmt", ("geojson", "feather", "parquet"))
def test_pp01_corrupt_bytes_fail_both_routes(format_dir, fmt):
    """Truncated bytes fail loudly on BOTH routes — a prepared admission may
    never silently succeed on partial input."""
    path = format_dir(fmt) / f"network.{fmt}"
    raw = path.read_bytes()
    path.write_bytes(raw[: len(raw) // 2])
    with pytest.raises(Exception):
        Topology._get_gdf(str(path), None)
    with pytest.raises(Exception):
        _decode_snapshot_gdf(str(path), _read_snapshot(str(path)), None)
