"""EXECUTION-suite fixtures (campaign task EXECUTION, contract §6).

The suite is self-contained: it needs only the repo's ``src`` tree and the
committed 3x3 smoke-grid fixture in ``tests/execution/fixtures/`` — no
reference venv, no compiled backends, no GPU.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
for _p in (str(REPO), str(HERE), str(REPO / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

FIXTURES = HERE / "fixtures"


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "execution: public execution/cache/capability contract "
        "(contract §6) and kernel ABI (dossier 10)")


@pytest.fixture(scope="session")
def smoke_fixture() -> Path:
    """The frozen 3x3 smoke grid fixture directory (see PROVENANCE.md)."""
    for f in ("network.geojson", "origins.geojson", "destinations.geojson"):
        if not (FIXTURES / f).is_file():
            pytest.skip(f"smoke-grid fixture missing: {f}")
    return FIXTURES
