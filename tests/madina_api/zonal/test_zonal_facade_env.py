"""Facade-environment pins that need NO reference venv: the D3/D4
deltas, the documented namespace-identity decision, and the top-level
una reexport surface (API_REVIEW R1 closure), verified in the
facade-only environment (no pydeck installed).

These run under the alan interpreter; the parity arms live in
test_zonal_surface_parity.py / test_zonal_reflection.py /
test_top_level_reflection.py (skipped when the reference venv is
absent — this module is not).
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


def test_d3_zonal_port_has_no_env_write():
    """D3: the compat zonal PORT carries no ``os.environ['USE_PYGEOS']``
    write (upstream pinned zonal.py:3 sets it to '0' at import).  Pinned
    at AST level: importing ``...compat.madina.zonal`` executes the
    compat/madina parent __init__, whose R1 una reexport legitimately
    carries the pinned upstream una-chain write (betweenness.py:3,
    workflows.py:6 — upstream ``import madina`` mutates the environment
    the same way), so an import-time environ probe can no longer isolate
    the zonal site.  See
    test_top_level_import_carries_pinned_una_env_write below."""
    import ast
    zonal_dir = REPO / "src" / "urban_network_analysis" / "compat" / "madina" / "zonal"
    offenders = []
    for path in sorted(zonal_dir.glob("*.py")):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            # any subscript write/lookup keyed 'USE_PYGEOS' ...
            if (isinstance(node, ast.Subscript)
                    and isinstance(node.slice, ast.Constant)
                    and node.slice.value == "USE_PYGEOS"):
                offenders.append(f"{path.name}:subscript")
            # ... or a setdefault/get-style call keyed 'USE_PYGEOS'
            if (isinstance(node, ast.Call) and node.args
                    and isinstance(node.args[0], ast.Constant)
                    and node.args[0].value == "USE_PYGEOS"):
                offenders.append(f"{path.name}:call")
    assert not offenders, offenders


def test_top_level_env_write_fires_at_first_una_load():
    """D3-LAZY (cd3-20261012T0030Z): the R1 eager reexport's import-time
    una-chain write moved to the FIRST una-surface load (PEP 562
    __getattr__/__dir__ in compat/madina/__init__.py).  A bare facade
    import — and therefore the D3 zonal import, which executes the
    parent __init__ — leaves os.environ untouched; the pinned upstream
    write ``os.environ['USE_PYGEOS'] = '0'`` (betweenness.py:105,
    carried verbatim; upstream ``import madina`` writes the same value
    at zonal.py:3) fires exactly once at its upstream site when the una
    surface is first touched.  Supersedes R1's
    test_top_level_import_carries_pinned_una_env_write, which pinned the
    eager de2872f trigger.  D3 itself is unchanged — see
    test_d3_zonal_port_has_no_env_write."""
    proc = _run(
        "import os, sys\n"
        "sys.path.insert(0, 'src')\n"
        "os.environ.pop('USE_PYGEOS', None)\n"
        "import urban_network_analysis.compat.madina\n"
        "assert 'USE_PYGEOS' not in os.environ, (\n"
        "    'bare facade import wrote the environment')\n"
        "import urban_network_analysis.compat.madina as m\n"
        "assert callable(m.parallel_betweenness)\n"
        "assert os.environ.get('USE_PYGEOS') == '0', (\n"
        "    'una surface load did not carry the pinned write')\n"
        "print('lazy-trigger ok')")
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert proc.stdout.strip() == "lazy-trigger ok"


def test_d3_lazy_facade_and_zonal_imports_leave_env_pristine():
    """D3 at the facade surface, fresh interpreter: importing the
    compat.madina package or the compat.madina.zonal package (the exact
    tests/platform_geometry D3 import shape) writes no environment
    variable, and the una surface stays unexecuted until a una name is
    touched."""
    proc = _run(
        "import os, sys\n"
        "sys.path.insert(0, 'src')\n"
        "os.environ.pop('USE_PYGEOS', None)\n"
        "import urban_network_analysis.compat.madina as m\n"
        "import urban_network_analysis.compat.madina.zonal as zp\n"
        "assert 'USE_PYGEOS' not in os.environ\n"
        "assert not any('compat.madina.una' in k for k in sys.modules)\n"
        "print('pristine', zp.Zonal.__name__)\n")
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert proc.stdout.strip() == "pristine Zonal"


def test_star_import_materializes_the_full_surface():
    """``from ...compat.madina import *`` mirrors upstream's star-import
    (no ``__all__`` defined, mirroring upstream, whose __init__ defines
    none): the machinery's ``__all__`` lookup routes through the lazy
    trigger, so the landed public set is the materialized surface —
    spot-pinned by the 17 functional engine/path names plus the
    una-chain module imports (``os`` lands at the top level exactly as
    upstream, via betweenness.py's module-top ``import os``).  The live
    full-set authority stays test_top_level_reflection.py."""
    proc = _run(
        "import sys, types\n"
        "sys.path.insert(0, 'src')\n"
        "ns = {}\n"
        "exec('from urban_network_analysis.compat.madina import *', ns)\n"
        "fns = " + repr(TOP_LEVEL_FUNCTIONS) + "\n"
        "missing = [f for f in fns if not callable(ns.get(f))]\n"
        "assert not missing, missing\n"
        "assert isinstance(ns['betweenness'], types.ModuleType)\n"
        "assert 'os' in ns\n"
        "print('star ok')")
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert proc.stdout.strip() == "star ok"


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


# Top-level una reexport (API_REVIEW R1 closure, mda-20261001T1215Z).
# Functional names are pinned here so the live-digest comparison in
# test_top_level_reflection.py can never silently pass on a stub; the
# authoritative name SET is enumerated live from .refs there.
#
# Generator (pinned upstream 8b5c3bd3, .refs/venv_madina_legacy):
#   python -c "import sys; sys.path.insert(0,'.refs/madina_ref/src');\
# import madina; print(sorted(n for n in dir(madina) if not n.startswith('_')))"
# minus the module-level imports (concurrent, deque, futures, getsizeof,
# heappop, heappush, math, mp, os, psutil, time) and the submodule
# bindings (una, zonal, betweenness, paths).  16 functions + the
# betweenness engine module = the 17 functional engine/path names of R1.
TOP_LEVEL_FUNCTIONS = [
    # una/betweenness.py engine (8)
    "get_origin_properties", "one_access", "parallel_access",
    "parallel_betweenness", "one_betweenness_2",
    "clockwiseangle_and_distance", "betweenness_exposure",
    "paralell_betweenness_exposure",
    # una/paths.py (8)
    "path_generator", "bfs_subgraph_generation", "wandering_messenger",
    "bfs_paths_many_targets_iterative", "bfs_path_edges_many_targets_iterative",
    "turn_o_scope", "turn_penalty_value", "angle_deviation_between_two_lines",
]


def test_top_level_una_reexport_and_attribute_chain():
    """R1: the upstream ``from .una import *`` reexport lands.  The 17
    functional engine/path names are top-level and referencable, the
    ``m.una`` attribute-chain pattern works (m.una is the una package),
    and the zonal package restore survives the second star import."""
    proc = _run(
        "import sys, types\n"
        "sys.path.insert(0, 'src')\n"
        "import urban_network_analysis.compat.madina as m\n"
        "fns = " + repr(TOP_LEVEL_FUNCTIONS) + "\n"
        "not_callable = [f for f in fns if not callable(getattr(m, f))]\n"
        "assert not not_callable, not_callable\n"
        "assert isinstance(m.betweenness, types.ModuleType)\n"
        "assert callable(m.betweenness.parallel_betweenness)\n"
        "assert hasattr(m.una, '__path__'), 'm.una is not the una package'\n"
        "assert m.una is sys.modules[\n"
        "    'urban_network_analysis.compat.madina.una']\n"
        "assert hasattr(m.zonal, '__path__'), 'zonal restore lost'\n"
        "print('ok', len(fns) + 1)\n")
    assert proc.returncode == 0, proc.stderr[-2000:]
    # 16 functions + the betweenness engine module binding
    assert proc.stdout.strip() == "ok 17"


def test_top_level_tools_workflows_stay_subpath_only():
    """The una.tools / una.workflows helpers stay una-subpath-only,
    exactly as upstream: plain ``import madina`` does not bind
    ``una.tools`` there either (AttributeError until the submodule is
    imported), and no tools/workflows helper becomes a top-level name."""
    proc = _run(
        "import sys\n"
        "sys.path.insert(0, 'src')\n"
        "import urban_network_analysis.compat.madina as m\n"
        "assert not any(hasattr(m, h) for h in (\n"
        "    'validate_zonal_ready', 'accessibility', 'service_area',\n"
        "    'alternative_paths', 'Logger', 'betweenness_flow_simulation',\n"
        "    'KNN_accessibility'))\n"
        "assert not hasattr(m.una, 'tools') and not hasattr(m.una, 'workflows')\n"
        "import urban_network_analysis.compat.madina.una.tools\n"
        "import urban_network_analysis.compat.madina.una.workflows\n"
        "assert callable(m.una.tools.accessibility)\n"
        "assert callable(m.una.workflows.KNN_accessibility)\n"
        "print('subpath-only ok')\n")
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert proc.stdout.strip() == "subpath-only ok"
