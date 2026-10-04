"""Fixtures for the C30 cohort-route suite (dossier 06, campaign una_10x).

Self-contained on purpose: the cohort route is exercised through the
PUBLIC RunBatch/RunAccessibility APIs against the frozen 3x3 smoke grid,
exactly like the batch runtime suite — but every assertion here watches
the cohort session's own counters (extra["cohort"]) rather than the
runtime's scheduling internals.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[3]
for p in (str(REPO / "src"),):
    if p not in sys.path:
        sys.path.insert(0, p)

SMOKE = REPO / "tests" / "execution" / "fixtures"


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "tenx_cohort: shared routing-trace cohort route tests")


@pytest.fixture(scope="session")
def smoke_fixture() -> Path:
    for f in ("network.geojson", "origins.geojson", "destinations.geojson"):
        if not (SMOKE / f).is_file():
            pytest.skip(f"smoke-grid fixture missing: {f}")
    return SMOKE


@pytest.fixture
def make_batch(smoke_fixture, tmp_path):
    """UNA instance with N independent accessibility rows over the frozen
    smoke grid.  Rows differ in name and search radius by default; pass
    ``extra(i, settings)`` to override per row."""
    from urban_network_analysis import UNA

    def make(n=2, *, out_folder=None, extra=None, data_dir=None):
        p = UNA(verbosity=0)
        p.settings.data_folder = str(data_dir or smoke_fixture)
        p.settings.network_file = "network.geojson"
        p.settings.origins_file = "origins.geojson"
        p.settings.destinations_file = "destinations.geojson"
        p.settings.output_folder = str(out_folder or tmp_path / "out")
        # Timestamped subfolders would make serial/parallel artifact-tree
        # comparisons flaky across a minute boundary (the stamp is in the
        # folder NAME only, never file content) — pin it off.
        p.settings.output_wStamp = False
        for i in range(n):
            p.settings.name = f"row{i}"
            p.settings.search_radius = 90 + 10 * i
            if extra:
                extra(i, p.settings)
            p.SaveSettingsToProject()
        return p

    return make


def tree_bytes(root, skip_hidden=False):
    """relpath -> bytes for every file under root.  ``skip_hidden``
    drops dot-prefixed entries on BOTH sides of a comparison — runtime
    state (journal, staging, the cohort store root) is not a row
    artifact; its presence/persistence is asserted explicitly by the
    tests that own it."""
    out = {}
    for dirpath, _dirs, names in os.walk(root):
        for name in names:
            if skip_hidden and name.startswith("."):
                continue
            path = os.path.join(dirpath, name)
            rel = os.path.relpath(path, root)
            if skip_hidden and (rel.startswith(".") or "/." in rel
                                or "\\." in rel):
                continue
            out[rel] = open(path, "rb").read()
    return out


@pytest.fixture
def cohort_summary():
    """The aggregated cohort block a parallel run reported (or None)."""
    def get(una):
        extra = getattr(una.batch_report, "extra", None) or {}
        return extra.get("cohort")
    return get
