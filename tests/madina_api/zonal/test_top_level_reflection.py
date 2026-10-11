"""Top-level compat.madina namespace parity against the pinned upstream
(API_REVIEW row R1, returned to the compat namespace owner; closed by
the ``from .una import *`` reexport in compat/madina/__init__.py).

The upstream digest is enumerated LIVE from ``.refs/madina_ref`` under
the reference venv — no name list is hardcoded.  Ledgered deltas here:

module namespace (users' ``import madina`` / ``import ...compat.madina
as m`` view; underscore names filtered from BOTH digests)
  reference-only names: pdk  (D4: pydeck lazily imported in the facade;
  upstream's zonal star-import copies the pydeck binding to the top
  level.  ``os`` is NOT ledgered here: it returns through the una star
  import — pinned upstream betweenness.py imports os at module top).

identity
  m.zonal: upstream users see the inner zonal.py MODULE (star-import
  shadowing quirk); the compat namespace deliberately restores the
  package (compat/madina/__init__.py) — recorded, ledgered, PACKAGE I6,
  same decision as the zonal-package reflection test.  m.una is the una
  PACKAGE in both arms: the una namespace carries no ``una``/``zonal``
  name, so the una star import does not shadow it.  m.betweenness /
  m.paths are the una submodules in both arms.
"""
from __future__ import annotations

import functools
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.madina_api

HERE = Path(__file__).resolve().parent
REPO = Path(__file__).resolve().parents[3]
for _p in (str(REPO), str(HERE), str(REPO / "tests" / "large_e2e")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from _legacy_paths import ARTIFACTS_OUT  # portable-path fix (HARNESS 2026-09-30)

TOP_REFLECT_SCRIPT = HERE / "_top_reflection.py"

# ---- ledgered deltas (see module docstring) -------------------------
REFERENCE_ONLY_TOP_NAMES = {"pdk"}


def _run_top_reflection(madina_python, arm) -> dict:
    out_dir = (Path(str(ARTIFACTS_OUT)) / "madina_zonal" / "top_reflection").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{arm}.json"
    if out.exists():
        out.unlink()
    cmd = [madina_python, str(TOP_REFLECT_SCRIPT), "--arm", arm, "--out", str(out)]
    env = dict(os.environ, PYTHONHASHSEED="0")
    proc = subprocess.run(cmd, capture_output=True, text=True,
                          timeout=300, env=env, cwd=str(REPO))
    if not out.exists():
        raise AssertionError(
            f"top_reflection/{arm} produced no digest (rc={proc.returncode})\n"
            f"stdout:\n{proc.stdout[-2000:]}\nstderr:\n{proc.stderr[-2000:]}")
    digest = json.loads(out.read_text())
    if "probe_error" in digest:
        raise AssertionError(
            f"top_reflection/{arm} failed:\n{digest['probe_error'][-2000:]}")
    return digest


@pytest.fixture(scope="session")
def top_reflection_runner(madina_python):
    return functools.partial(_run_top_reflection, madina_python)


def test_top_level_namespace_matches_pinned_reference(top_reflection_runner):
    """The top level carries the complete upstream una reexport: the
    facade name set must equal the upstream set minus exactly the
    ledgered {pdk} — in BOTH directions (no silent additions)."""
    ref = top_reflection_runner("reference")
    facade = top_reflection_runner("facade")
    ref_names = set(ref["top_public"])
    facade_names = set(facade["top_public"])
    only_ref = ref_names - facade_names
    only_facade = facade_names - ref_names
    assert only_ref == REFERENCE_ONLY_TOP_NAMES, (
        f"top-level names still missing from the facade (beyond the D4 "
        f"pdk ledger): {sorted(only_ref - REFERENCE_ONLY_TOP_NAMES)}")
    assert not only_facade, f"facade-only top-level names: {sorted(only_facade)}"
    # the una namespace itself is the reexport source: must be identical
    assert facade["una_public"] == ref["una_public"], (
        "una namespace diverged: "
        f"only-ref={sorted(set(ref['una_public']) - set(facade['una_public']))} "
        f"only-facade={sorted(set(facade['una_public']) - set(ref['una_public']))}")


def test_top_level_submodule_identity(top_reflection_runner):
    """m.una must be the una PACKAGE in both arms (attribute-chain
    pattern works: ``import ...compat.madina as m; m.una.<sub>`` after
    the submodule import — upstream raises the same AttributeError when
    the submodule was never imported).  m.betweenness / m.paths are the
    una submodules in both arms; the m.zonal package restore stays the
    ledgered I6 identity delta."""
    ref = top_reflection_runner("reference")
    facade = top_reflection_runner("facade")
    assert ref["identity"]["una"]["is_package"] is True
    assert facade["identity"]["una"]["is_package"] is True
    assert facade["identity"]["una"]["module"].endswith(".una")
    assert ref["identity"]["betweenness"]["is_package"] is False
    assert facade["identity"]["betweenness"]["is_package"] is False
    assert ref["identity"]["paths"]["is_package"] is False
    assert facade["identity"]["paths"]["is_package"] is False
    # ledgered I6: upstream attribute = inner zonal.py module, compat = package
    assert ref["identity"]["zonal"]["is_package"] is False
    assert facade["identity"]["zonal"]["is_package"] is True
