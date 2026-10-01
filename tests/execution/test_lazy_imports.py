"""Import hygiene of the public execution surface.

Importing the package (or the Execution module) must not pull the analysis
stack; constructing options in a fresh interpreter must work without numba.
Documented pre-existing quirk (unchanged, order-dependent, applies to
UNA/Settings/Topology alike): a from-import of a name that collides with a
submodule (``from urban_network_analysis import Settings``) binds the
SUBMODULE once the submodule has been imported; attribute access
(``una_pkg.Settings``) always binds the class via PEP 562.  New execution
export names deliberately avoid that collision.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.execution

REPO_SRC = str(Path(__file__).resolve().parents[2] / "src")


def _fresh(code: str) -> subprocess.CompletedProcess:
    env = {**os.environ, "PYTHONPATH": REPO_SRC}
    return subprocess.run([sys.executable, "-c", code, REPO_SRC],
                          capture_output=True, text=True, env=env)


def test_package_import_does_not_pull_the_analysis_stack():
    r = _fresh(
        "import sys; import urban_network_analysis as p; "
        "pulled = [m for m in ('numba', 'sklearn', 'geopandas', 'pandas') "
        "if m in sys.modules]; assert not pulled, pulled; "
        "assert hasattr(p, 'ExecutionOptions'); print('CLEAN')")
    assert "CLEAN" in r.stdout, r.stderr[-400:]


def test_execution_module_standalone_import_is_dependency_free():
    r = _fresh(
        "import sys; "
        "from urban_network_analysis.Execution import ExecutionOptions; "
        "assert not any(m in sys.modules for m in ('numba', 'numpy', "
        "'pandas', 'sklearn')), [m for m in ('numba','numpy','pandas',"
        "'sklearn') if m in sys.modules]; "
        "assert ExecutionOptions().backend == 'auto'; print('CLEAN')")
    assert "CLEAN" in r.stdout, r.stderr[-400:]


def test_from_import_of_execution_names_binds_classes_not_modules():
    """The new export names must never hit the from-import submodule-
    shadowing quirk that Settings/UNA/Topology have (pre-existing)."""
    r = _fresh(
        "import urban_network_analysis as p; "
        "from urban_network_analysis import ExecutionOptions, CacheOptions, "
        "CapabilityError, BackendNotAvailableError, ExecutionNotAdmittedError; "
        "import urban_network_analysis.UNA as _una_mod  # trigger shadowing\n"
        "assert ExecutionOptions is p.ExecutionOptions\n"
        "assert CacheOptions is p.CacheOptions\n"
        "assert CapabilityError is p.CapabilityError\n"
        "assert hasattr(ExecutionOptions(), 'queue_depth'); print('CLEAN')")
    assert "CLEAN" in r.stdout, r.stderr[-400:]


def test_settings_module_shadowing_quirk_is_documented_not_regressed():
    """Pre-existing order-dependent quirk (v2.6 lazy exports, unchanged by
    EXECUTION, applies to UNA/Settings/Topology alike): once a submodule is
    imported, Python binds it as a package attribute and it shadows the
    PEP 562 lazy class on EVERY access path — ``p.Settings`` and
    ``from ... import Settings`` both yield the module; the class stays
    reachable at ``urban_network_analysis.Settings.Settings``.  Recorded so
    a future cleanup consciously decides about it; new execution export
    names deliberately avoid same-named modules and never hit this."""
    r = _fresh(
        "import urban_network_analysis as p; "
        "import urban_network_analysis.UNA\n"
        "import types\n"
        "assert isinstance(p.Settings, types.ModuleType)  # shadowed\n"
        "from urban_network_analysis import Settings as from_imported\n"
        "assert from_imported is p.Settings            # same shadowing\n"
        "from urban_network_analysis.Settings import Settings as cls\n"
        "assert type(cls).__name__ == 'type'          # class still reachable\n"
        "print('CLEAN')")
    assert "CLEAN" in r.stdout, r.stderr[-400:]
