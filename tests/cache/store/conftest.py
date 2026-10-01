"""CACHE_STORE suite fixtures (campaign task CACHE_STORE, dossier 06).

Self-contained: exercises the store primitive directly with raw byte
payloads.  No analysis-stack import, no reference venv, no network — the
store depends only on stdlib plus ``Execution.CacheOptions``.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
for _p in (str(REPO), str(REPO / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from urban_network_analysis.Execution import CacheOptions  # noqa: E402
from urban_network_analysis.cache import ContentStore  # noqa: E402


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "cache_store: bounded non-executable content store primitive "
        "(dossier 06 store protocol, contract §6 CacheOptions)")


@pytest.fixture
def cache_dir(tmp_path):
    return tmp_path / "una-cache"


@pytest.fixture
def make_store(cache_dir):
    """Factory: ``make_store(**CacheOptions_overrides) -> ContentStore``."""
    def _make(**over):
        defaults = dict(mode="disk", directory=str(cache_dir))
        defaults.update(over)
        return ContentStore(CacheOptions(**defaults))
    return _make


@pytest.fixture
def store(make_store):
    return make_store()
