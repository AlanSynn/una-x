"""Import-hygiene probe (review NOTE-8c): the cache package must not pull
the analysis stack or heavy third-party dependencies — subprocess probe so
the CURRENT interpreter's sys.modules cannot leak in."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]

PROBE = """
import sys
import urban_network_analysis.cache  # noqa: F401
loaded_una = sorted(m for m in sys.modules
                    if m.startswith("urban_network_analysis"))
heavy = sorted(m for m in ("numpy", "pandas", "geopandas", "numba",
                           "shapely", "networkx")
               if m in sys.modules)
print(loaded_una)
print(heavy)
"""


@pytest.mark.cache_store
def test_cache_package_imports_no_analysis_stack():
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO / "src")
    env.setdefault("PYTHONHASHSEED", "0")
    res = subprocess.run([sys.executable, "-c", PROBE], capture_output=True,
                         text=True, env=env, timeout=120)
    assert res.returncode == 0, res.stderr
    import ast
    lines = res.stdout.strip().splitlines()
    loaded_una, heavy = (ast.literal_eval(line) for line in lines[:2])
    assert loaded_una == [
        "urban_network_analysis",
        "urban_network_analysis.Execution",
        "urban_network_analysis.cache",
        "urban_network_analysis.cache.store",
    ]
    assert heavy == []
