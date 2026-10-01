"""Facade-environment pins that need NO reference venv: the D3/D4
deltas and the documented namespace-identity decision, verified in the
facade-only environment (no pydeck installed).

These run under the alan interpreter; the parity arms live in
test_zonal_surface_parity.py / test_zonal_reflection.py (skipped when the
reference venv is absent — this module is not).
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.madina_api

REPO = Path(__file__).resolve().parents[3]


def _run(code: str) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env.pop("USE_PYGEOS", None)          # D3 precondition: not set by us
    return subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True,
        env=env, cwd=str(REPO), timeout=300)


def test_d3_import_does_not_mutate_environ():
    """D3: importing the facade must not set os.environ['USE_PYGEOS']
    (upstream pinned zonal.py sets it to '0' at import)."""
    proc = _run(
        "import os, sys\n"
        "sys.path.insert(0, 'src')\n"
        "os.environ.pop('USE_PYGEOS', None)\n"
        "import urban_network_analysis.compat.madina.zonal\n"
        "print('USE_PYGEOS' in os.environ)")
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert proc.stdout.strip() == "False"


def test_d4_create_map_importerror_without_pydeck():
    """D4: pydeck absent here; create_map must raise an actionable
    ImportError naming the missing dependency and the install command
    (not a bare ImportError from a module-level import)."""
    proc = _run(
        "import sys\n"
        "sys.path.insert(0, 'src')\n"
        "assert 'pydeck' not in sys.modules\n"
        "try:\n"
        "    import pydeck\n"
        "    raise SystemExit('pydeck unexpectedly present')\n"
        "except ImportError:\n"
        "    pass\n"
        "import geopandas as gpd\n"
        "from shapely.geometry import LineString\n"
        "from urban_network_analysis.compat.madina import Zonal\n"
        "z = Zonal()\n"
        "z.load_layer('s', gpd.GeoDataFrame(\n"
        "    {'Geometric': [1.0]},\n"
        "    geometry=[LineString([(0, 0), (1, 1)])], crs='EPSG:3857'))\n"
        "try:\n"
        "    z.create_map()\n"
        "except ImportError as exc:\n"
        "    text = str(exc)\n"
        "    assert 'pydeck' in text and 'pip install pydeck' in text, text\n"
        "    print('actionable')\n"
        "else:\n"
        "    raise SystemExit('create_map did not raise')\n")
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert proc.stdout.strip() == "actionable"


def test_namespace_identity_package_restored():
    """Documented identity decision (PACKAGE handoff I6): upstream's
    star-import shadowing binds the attribute ``madina.zonal`` to the
    inner zonal.py module; the compat namespace deliberately restores
    the package.  sys.modules and the attribute are both the package."""
    proc = _run(
        "import sys\n"
        "sys.path.insert(0, 'src')\n"
        "from urban_network_analysis.compat.madina import zonal\n"
        "import urban_network_analysis.compat.madina as m\n"
        "print(hasattr(zonal, '__path__'), zonal is m.zonal,\n"
        "      sys.modules['urban_network_analysis.compat.madina.zonal'] is zonal)")
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert proc.stdout.split() == ["True", "True", "True"]


def test_facade_importable_and_versioned():
    """The supported surface is present in a facade-only environment
    with the pinned version constants."""
    proc = _run(
        "import sys\n"
        "sys.path.insert(0, 'src')\n"
        "from urban_network_analysis.compat.madina import (\n"
        "    Zonal, Layer, Layers, Network, VERSION, RELEASE_DATE,\n"
        "    DEFAULT_COLORS, prepare_geometry, color_gdf,\n"
        "    create_deckGL_map, node_edge_builder, efficient_node_insertion)\n"
        "assert VERSION == '0.0.15' and RELEASE_DATE == '2023-02-16'\n"
        "print('ok', len(DEFAULT_COLORS))\n")
    assert proc.returncode == 0, proc.stderr[-2000:]
    # pinned: DEFAULT_COLORS has exactly 5 entries (streets, blocks,
    # parcels, network_edges, network_nodes) — the count is asserted,
    # not decoration
    assert proc.stdout.strip() == "ok 5"
