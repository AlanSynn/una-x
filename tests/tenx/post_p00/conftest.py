"""Fixtures for the post-P00 P01M telemetry suite (campaign una_10x).

Same public-API exercise pattern as the batch runtime / cohort suites:
parallel RunBatch over the frozen 3x3 smoke grid, assertions watching the
P01M diagnostics surface (RowExecutionOutcome.phases /
dropped_events and BatchExecutionReport.extra coordinator telemetry).
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

# Telemetry must be OFF unless a test explicitly enables it: an exported
# shell variable must never silently flip the whole suite's schema.
@pytest.fixture(autouse=True)
def _telemetry_default_off(monkeypatch):
    monkeypatch.delenv("UNA_BATCH_PHASES", raising=False)


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "tenx_p01m: post-P00 phase-telemetry tests")


@pytest.fixture(scope="session")
def smoke_fixture() -> Path:
    for f in ("network.geojson", "origins.geojson", "destinations.geojson"):
        if not (SMOKE / f).is_file():
            pytest.skip(f"smoke-grid fixture missing: {f}")
    return SMOKE


@pytest.fixture
def make_batch(smoke_fixture, tmp_path):
    """UNA instance with N independent accessibility rows over the frozen
    smoke grid (planner-proven independent, worker-admissible)."""
    from urban_network_analysis import UNA

    def make(n=2, *, out_folder=None):
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
            p.SaveSettingsToProject()
        return p

    return make
