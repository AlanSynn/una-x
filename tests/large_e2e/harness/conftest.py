"""Shared fixtures for the large-e2e harness self-tests (H02).

Every fixture is tiny and synthetic: stub engine, fake venv layout, tiny
manifests.  No large jobs are launched anywhere in this suite.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

TESTS_DIR = Path(__file__).resolve().parent
if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))

# The harness package under benchmarks/large_e2e is importable in tests so
# coordinator-side modules (sampler, dispatch, ...) can be exercised
# directly; run.py subprocesses do their own path setup.
HARNESS_PARENT = TESTS_DIR.parents[2] / "benchmarks" / "large_e2e"
if str(HARNESS_PARENT) not in sys.path:
    sys.path.insert(0, str(HARNESS_PARENT))

import stubkit  # noqa: E402


@pytest.fixture(scope="session")
def stub_venv(tmp_path_factory) -> dict:
    """A fake arm venv: site-packages + stub package + dist-info."""
    return stubkit.build_stub_venv(tmp_path_factory.mktemp("stubvenv"))


@pytest.fixture(scope="session")
def stub_identity(stub_venv, tmp_path_factory) -> Path:
    """identity_kind=installed_wheel identity generated in a subprocess."""
    out = tmp_path_factory.mktemp("identity") / "stub_identity.json"
    stubkit.generate_identity(
        stubkit.CAMPAIGN_PYTHON, arm="stub", kind="installed_wheel",
        out_path=out, site_packages=[stub_venv["site_packages"]])
    return out


@pytest.fixture
def stub_manifest(tmp_path) -> Path:
    """Tiny accessibility manifest whose inputs live under tmp."""
    data_dir = tmp_path / "inputs"
    return stubkit.make_manifest(tmp_path / "workload.json", data_dir,
                                 analysis="accessibility")


@pytest.fixture
def run_dir(tmp_path) -> Path:
    out = tmp_path / "run_out"
    out.mkdir()
    return out


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "real_engine: smoke tests against the REAL source engine")
