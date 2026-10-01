"""Import hygiene: planning a batch must not load the analysis stack or
heavy third-party dependencies — subprocess probe so the CURRENT
interpreter's sys.modules cannot leak in."""
from __future__ import annotations

import ast
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]

PROBE = """
import sys
import urban_network_analysis.batch  # noqa: F401
loaded_una = sorted(m for m in sys.modules
                    if m.startswith("urban_network_analysis"))
heavy = sorted(m for m in ("pandas", "geopandas", "shapely", "networkx",
                           "numba", "pyproj")
               if m in sys.modules)
print(loaded_una)
print(heavy)
print(sorted(m for m in ("numpy",) if m in sys.modules))
"""


@pytest.mark.batch_plan
def test_batch_package_imports_no_analysis_stack():
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO / "src")
    env.setdefault("PYTHONHASHSEED", "0")
    res = subprocess.run([sys.executable, "-c", PROBE], capture_output=True,
                         text=True, env=env, timeout=120)
    assert res.returncode == 0, res.stderr
    lines = res.stdout.strip().splitlines()
    loaded_una, heavy, numpy_loaded = (ast.literal_eval(line)
                                       for line in lines[:3])
    assert loaded_una == [
        "urban_network_analysis",
        "urban_network_analysis.Execution",
        "urban_network_analysis.Settings",
        "urban_network_analysis.batch",
        "urban_network_analysis.batch.plan",
    ]
    assert heavy == []
    assert numpy_loaded == ["numpy"]   # Settings is stdlib+numpy; nothing heavier
