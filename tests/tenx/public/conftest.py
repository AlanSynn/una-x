"""Fixtures for the una_10x public-chronology suite."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
for p in (str(REPO / "src"),):
    if p not in sys.path:
        sys.path.insert(0, p)

# The campaign's frozen 3x3 smoke grid (single provenance copy, referenced
# never copied — same discipline as tests/batch/runtime).
SMOKE = REPO / "tests" / "execution" / "fixtures"

pytest_plugins = []


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "chronology: una_10x public chronology tests")


@pytest.fixture(scope="session")
def smoke_fixture() -> Path:
    for f in ("network.geojson", "origins.geojson", "destinations.geojson"):
        if not (SMOKE / f).is_file():
            pytest.skip(f"smoke-grid fixture missing: {f}")
    return SMOKE
