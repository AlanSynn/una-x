"""Shared fixtures for the platform-harness suite (HARNESS).

Campaign-owned paths only: all outputs, caches and temp state live under
pytest tmp dirs (outside the repository).  The reference interpreter is the
installed una_legacy wheel venv; its absence skips (with an explicit
environment reason) rather than failing — hardware/environment absence
blocks qualification, it does not fake a pass.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
PY_ALAN = sys.executable
PY_LEGACY = REPO / ".refs" / "venv_una_legacy" / "bin" / "python"

# Frozen contract (campaigns/una_platform/evidence/contract/*):
WHEEL_FILE_SHA256 = "323c27f3ace8daafcc6d025596a605e8263cc14c8c4fe602c427d51069448ac4"
WHEEL_TREE_SHA256 = "6b9903bebbb4eaeb371361e7211d68526b7d3da0a6a7aa6f18806a49b57b85e8"
EXPECTED_IDENTITY = {
    "package": "urban_network_analysis",
    "version": "2.6.0",
    "wheel_tree_sha256": WHEEL_TREE_SHA256,
}

sys.path.insert(0, str(REPO))

from benchmarks.platform.fixtures import build_smoke_fixture, smoke_manifest  # noqa: E402


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "harness_heavy: executes real installed-wheel jobs (minutes; fresh "
        "JIT caches per leg)")


@pytest.fixture(scope="session")
def legacy_python() -> str:
    if not PY_LEGACY.is_file():
        pytest.skip("una_legacy reference venv not present (.refs/venv_una_legacy); "
                    "environment absence blocks, never fakes", allow_module_level=False)
        raise AssertionError("unreachable")
    return str(PY_LEGACY)


@pytest.fixture(scope="session")
def legacy_wheel_verified(legacy_python) -> str:
    """Installed-tree identity anchor: the venv install must match the
    frozen tree digest recorded in profiles.json (original wheel file was
    not retained post-purge; the install is the artifact of record)."""
    from benchmarks.platform.identity import resolve_identity, assert_installed
    info = resolve_identity(legacy_python)
    assert_installed(info, "urban_network_analysis",
                     expected_wheel_tree_sha256=WHEEL_TREE_SHA256)
    return info["installed_tree_sha256"]


@pytest.fixture(scope="session")
def smoke_root(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("smoke_fixture")
    return Path(build_smoke_fixture(root)["network"]).parent


@pytest.fixture(scope="session")
def smoke_manifest_frozen(smoke_root, legacy_wheel_verified) -> dict:
    """Frozen smoke manifest with the verified installed identity."""
    man_path = smoke_root / "manifest.json"
    doc = smoke_manifest(man_path, smoke_root,
                         expected_identity=dict(EXPECTED_IDENTITY))
    # obligations pinned from the executed pilot (HARNESS receipt):
    doc["output_obligations"] = {"files": {
        "Results.geojson": {"format": "geojson", "expect_features": 2,
                            "min_bytes": 200},
        "Results.feather": {"format": "feather", "expect_rows": 2,
                            "expect_columns": ["geometry", "reach",
                                               "gravity_logistic",
                                               "gravity_exponential",
                                               "knn_logistic"]},
    }}
    man_path.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")
    return json.loads(man_path.read_text())


@pytest.fixture(scope="session")
def base_plan(smoke_manifest_frozen, smoke_root) -> dict:
    """Child plan used by engagement-leg negative controls."""
    return {
        "harness": {"repo": str(REPO),
                    "package": EXPECTED_IDENTITY["package"]},
        "model": smoke_manifest_frozen["model"],
        "settings_patch": smoke_manifest_frozen["settings_patch"],
        "backend_reported": smoke_manifest_frozen["backend_requested"],
        "run": {"out_dir": str(smoke_root / "leg_engagement")},
    }


@pytest.fixture(scope="session")
def frozen_constants() -> dict:
    return {"repo": REPO, "wheel_file_sha256": WHEEL_FILE_SHA256,
            "wheel_tree_sha256": WHEEL_TREE_SHA256,
            "expected_identity": dict(EXPECTED_IDENTITY),
            "src_root": str(REPO / "src")}


@pytest.fixture()
def run_out(tmp_path) -> Path:
    return tmp_path / "run_out"
