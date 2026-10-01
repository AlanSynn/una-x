"""Facade environment pins: the compat una package imports WITHOUT any
upstream madina module, exposes the full delivered betweenness surface,
and never pulls a ``madina*`` module into sys.modules."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.madina_api

REPO = Path(__file__).resolve().parents[3]

FACADE_ENV_PROBE = r"""
import sys

sys.path.insert(0, "src")

from urban_network_analysis.compat.madina import una
from urban_network_analysis.compat.madina.una import tools

BETWEENNESS = ("parallel_betweenness", "one_betweenness_2",
               "clockwiseangle_and_distance", "betweenness_exposure",
               "paralell_betweenness_exposure", "get_origin_properties",
               "one_access", "parallel_access")
TOOLS_FNS = ("validate_zonal_ready", "accessibility", "service_area",
             "alternative_paths", "betweenness")
ENGINE = ("turn_o_scope", "path_generator", "bfs_subgraph_generation",
          "wandering_messenger")

missing = [n for n in BETWEENNESS + ENGINE if not hasattr(una, n)]
missing += [n for n in TOOLS_FNS if not hasattr(tools, n)]
print("missing:", missing)
assert not missing

import importlib
btd = importlib.import_module("urban_network_analysis.compat.madina.una.betweenness")
assert all(hasattr(btd, n) for n in BETWEENNESS)

madina_loaded = [m for m in sys.modules if m.startswith("madina")]
print("madina modules:", madina_loaded)
assert not madina_loaded
print("ok")
"""


def test_facade_imports_without_upstream():
    proc = subprocess.run([sys.executable, "-c", FACADE_ENV_PROBE],
                          capture_output=True, text=True, cwd=str(REPO),
                          timeout=300)
    assert proc.returncode == 0, (
        f"facade env probe failed\nstdout:\n{proc.stdout[-2000:]}\n"
        f"stderr:\n{proc.stderr[-2000:]}")
    assert "missing: []" in proc.stdout
    assert "madina modules: []" in proc.stdout
    assert "ok" in proc.stdout
