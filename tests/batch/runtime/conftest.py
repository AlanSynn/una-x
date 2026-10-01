"""Fixtures for the parallel RunBatch runtime suite (dossiers 04/05)."""
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

# The campaign's frozen 3x3 smoke grid (single provenance copy lives with
# the EXECUTION suite; runtime tests reference, never copy, it).
SMOKE = REPO / "tests" / "execution" / "fixtures"


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "batch_runtime: parallel RunBatch runtime tests")


@pytest.fixture(scope="session")
def smoke_fixture() -> Path:
    for f in ("network.geojson", "origins.geojson", "destinations.geojson"):
        if not (SMOKE / f).is_file():
            pytest.skip(f"smoke-grid fixture missing: {f}")
    return SMOKE


@pytest.fixture
def make_batch(smoke_fixture, tmp_path):
    """Build a UNA instance with N independent accessibility rows.

    Rows differ in name (→ output stem) and search radius, write to a
    shared output folder under tmp_path, and read only the frozen smoke
    inputs — so they are planner-proven independent and worker-admissible.
    """
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


def tree_bytes(root):
    """relpath -> bytes for every file under root (artifact byte-diffs)."""
    out = {}
    for dirpath, _dirs, names in os.walk(root):
        for name in names:
            path = os.path.join(dirpath, name)
            out[os.path.relpath(path, root)] = open(path, "rb").read()
    return out


def values_equal(a, b):
    """Equality that tolerates numpy arrays inside Settings snapshots."""
    try:
        eq = a == b
    except Exception:
        return False
    if isinstance(eq, np.ndarray):
        return bool(eq.all()) if eq.shape else False
    return bool(eq)


def install_inject_specs(monkeypatch):
    """Teach the coordinator's job builder to honour per-row
    ``Settings.inject_specs`` (test-only fault-injection specs consumed by
    the worker's ``_inject`` seam).  Returns the captured jobs."""
    from urban_network_analysis.batch import runtime as rt

    captured = []
    orig_job_for = rt._job_for

    def job_for(una, plan, i, staging_dir, analysis, script_output_folder,
                need_state):
        job = orig_job_for(una, plan, i, staging_dir, analysis,
                           script_output_folder, need_state)
        specs = getattr(una.projects[i], "inject_specs", None)
        if specs:
            object.__setattr__(job, "inject", specs)  # frozen dataclass
        captured.append(job)
        return job

    monkeypatch.setattr(rt, "_job_for", job_for)
    return captured


@pytest.fixture(autouse=True)
def _inject_specs_honored(monkeypatch):
    """Every runtime test may set ``inject_specs`` on any row's Settings;
    the coordinator's job builder picks them up (installed before test
    bodies so later per-test ``_job_for`` patches chain on top)."""
    install_inject_specs(monkeypatch)
