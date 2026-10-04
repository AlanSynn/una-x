"""Fixtures for the post-P00 suites (campaign una_10x, P01M onward).

Same self-contained public-API style as the cohort island: the batch is
exercised through the PUBLIC RunBatch API over the frozen 3x3 smoke
grid.  Telemetry tests toggle the opt-in environment (UNA_PHASE_*)
through monkeypatch so spawned workers inherit it at p.start().
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
for p in (str(REPO / "src"),):
    if p not in sys.path:
        sys.path.insert(0, p)

SMOKE = REPO / "tests" / "execution" / "fixtures"


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "tenx_post_p00: post-P00 program tests (una_10x)")


@pytest.fixture(scope="session")
def smoke_fixture() -> Path:
    for f in ("network.geojson", "origins.geojson", "destinations.geojson"):
        if not (SMOKE / f).is_file():
            pytest.skip(f"smoke-grid fixture missing: {f}")
    return SMOKE


@pytest.fixture
def make_batch(smoke_fixture, tmp_path):
    """UNA instance with N independent accessibility rows over the frozen
    smoke grid (same shape as the cohort island's fixture)."""
    from urban_network_analysis import UNA

    def make(n=2, *, out_folder=None, extra=None):
        p = UNA(verbosity=0)
        p.settings.data_folder = str(smoke_fixture)
        p.settings.network_file = "network.geojson"
        p.settings.origins_file = "origins.geojson"
        p.settings.destinations_file = "destinations.geojson"
        p.settings.output_folder = str(out_folder or tmp_path / "out")
        p.settings.output_wStamp = False
        for i in range(n):
            p.settings.name = f"row{i}"
            p.settings.search_radius = 90 + 10 * i
            if extra:
                extra(i, p.settings)
            p.SaveSettingsToProject()
        return p

    return make


@pytest.fixture
def telemetry_extra():
    """The extra['telemetry'] block of a run's batch report (or None)."""
    def get(una):
        extra = getattr(una.batch_report, "extra", None) or {}
        return extra.get("telemetry")
    return get


def tree_bytes(root, skip_hidden=False):
    """relpath -> bytes for every file under root (hidden runtime entries
    excluded on both sides when skip_hidden)."""
    out = {}
    for dirpath, _dirs, names in os.walk(root):
        for name in names:
            if skip_hidden and name.startswith("."):
                continue
            path = os.path.join(dirpath, name)
            rel = os.path.relpath(path, root)
            if skip_hidden and (rel.startswith(".") or "/." in rel):
                continue
            out[rel] = open(path, "rb").read()
    return out
