"""Error-contract parity and registered-unsupported pinned paths.

Invalid arguments must raise the SAME exception type with the SAME
message through the facade as through the bridged reference.
Registered unsupported pinned paths (bug registry) keep their pinned
failure behavior — a documented quirk, not a feature to invent.
"""
from __future__ import annotations

import pytest

from conftest import ERROR_SCENARIOS

pytestmark = pytest.mark.platform_geometry


@pytest.mark.parametrize("scenario", ERROR_SCENARIOS)
def test_error_contract_matches_reference(scenario, arm_runner):
    ref = arm_runner(f"error:{scenario}", "reference")
    facade = arm_runner(f"error:{scenario}", "facade")
    assert ref["raised"] is not None, (
        f"{scenario}: expected the reference to raise; the scenario is stale")
    assert facade == ref, (
        f"{scenario}: facade error contract diverged: {facade} != {ref}")


def test_registered_unsupported_paths_match_reference(arm_runner):
    # visualize_graph -> NotImplementedError (pinned, documented unsupported)
    ref = arm_runner("error:visualize_graph", "reference")
    facade = arm_runner("error:visualize_graph", "facade")
    assert ref == facade and ref["raised"] == "NotImplementedError", ref


def test_facade_import_writes_no_environment_variables(monkeypatch):
    """D3: the facade must not mutate os.environ at import time (the
    pinned zonal.py set USE_PYGEOS; unnecessary since geopandas 1.0 and
    a library must not touch the user's environment)."""
    import os
    import subprocess
    import sys
    from conftest import REPO
    code = (
        "import os, sys\n"
        f"sys.path.insert(0, {str(REPO / 'src')!r})\n"
        "import urban_network_analysis.compat.madina.zonal as zp\n"
        "assert 'USE_PYGEOS' not in os.environ, 'USE_PYGEOS was set'\n"
        "print('env-clean', zp.Zonal.__name__)\n"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True,
                          text=True, timeout=120)
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert "env-clean Zonal" in proc.stdout


def test_facade_lazily_imports_without_pydeck_usage_paths(monkeypatch):
    """D4: module import must not need pydeck; create_deckGL_map raises
    an actionable ImportError when pydeck is absent."""
    import builtins
    import importlib
    import sys
    from conftest import REPO
    sys.path.insert(0, str(REPO / "src"))
    real_import = builtins.__import__

    def no_pydeck(name, *a, **k):
        if name == "pydeck" or name.startswith("pydeck."):
            raise ImportError("No module named 'pydeck' (blocked by test)")
        return real_import(name, *a, **k)

    # reload the facade's utils module with pydeck blocked: a module-level
    # pydeck import would fail the reload itself
    monkeypatch.setattr(builtins, "__import__", no_pydeck)
    import urban_network_analysis.compat.madina.zonal.utils as zutils
    importlib.reload(zutils)
    try:
        with pytest.raises(ImportError, match="pip install pydeck"):
            zutils.create_deckGL_map(gdf_list=[])
    finally:
        monkeypatch.undo()
        importlib.reload(zutils)
