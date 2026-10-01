"""Facade-environment pins that need NO reference venv: the una
package imports under the plain alan interpreter (no upstream madina,
no reference tree), carries the full paths surface, and does NOT
import upstream madina at any point.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.madina_api

REPO = Path(__file__).resolve().parents[3]


def _run(code: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True,
        cwd=str(REPO), timeout=300)


def test_facade_una_importable_without_upstream():
    """The compat una package imports in an environment where upstream
    madina is NOT importable (no .refs on sys.path) and exposes the
    complete paths surface + interim tools surface."""
    proc = _run(
        "import sys\n"
        "sys.path.insert(0, 'src')\n"
        "assert 'madina' not in sys.modules\n"
        "from urban_network_analysis.compat.madina import una\n"
        "fns = ['path_generator', 'bfs_subgraph_generation',\n"
        "       'bfs_paths_many_targets_iterative', 'wandering_messenger',\n"
        "       'bfs_path_edges_many_targets_iterative', 'turn_o_scope',\n"
        "       'turn_penalty_value', 'angle_deviation_between_two_lines']\n"
        "missing = [f for f in fns if not hasattr(una, f)]\n"
        "assert not missing, missing\n"
        "from urban_network_analysis.compat.madina.una import tools\n"
        "assert hasattr(tools, 'alternative_paths')\n"
        "print('ok', len(fns))\n")
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert proc.stdout.strip() == "ok 8"


def test_facade_una_never_imports_upstream_madina():
    """The port must not depend on upstream madina at runtime: after
    importing the una package, no module whose name starts with
    'madina' may be in sys.modules, and no facade file may reference
    the upstream tree."""
    proc = _run(
        "import sys\n"
        "sys.path.insert(0, 'src')\n"
        "import urban_network_analysis.compat.madina.una\n"
        "import urban_network_analysis.compat.madina.una.tools\n"
        "bad = [m for m in sys.modules if m == 'madina' or m.startswith('madina.')]\n"
        "assert not bad, bad\n"
        "print('clean')\n")
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert proc.stdout.strip() == "clean"
