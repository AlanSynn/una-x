"""Fixtures for the batch planner suite (dossier 04)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
for p in (str(REPO / "src"),):
    if p not in sys.path:
        sys.path.insert(0, p)

from urban_network_analysis.Settings import Settings  # noqa: E402


def pytest_configure(config):
    config.addinivalue_line("markers",
                            "batch_plan: dossier-04 batch planner tests")


@pytest.fixture
def make_settings(tmp_path):
    """Build a realistic run-able row: files under tmp_path/<name>/data,
    outputs to their own folder, timestamped subfolders off so output
    artifacts are literal predictable paths (planner tests pass wstamp=True
    explicitly where the flag matters)."""
    def make(name="row", *, stem=None, folder=None, data_folder=None,
             inputs=("net.geojson", "origins.geojson", "dests.geojson"),
             wstamp=False, **overrides):
        s = Settings()
        s.name = name
        s.data_folder = str(data_folder if data_folder is not None
                            else tmp_path / name / "data")
        s.network_file, s.origins_file, s.destinations_file = inputs
        if stem is not None:
            s.output_file_name = stem
        if folder is not None:
            s.output_folder = str(folder)
        s.output_wStamp = wstamp
        for k, v in overrides.items():
            setattr(s, k, v)
        return s
    return make
