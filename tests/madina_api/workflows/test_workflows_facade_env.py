"""Facade-environment probes for the workflows module (mirrors the
zonal/flow facade-env tests): in THIS interpreter (the alan env,
without pydeck) the facade must import cleanly with no pydeck leak and
no upstream ``madina`` import, Logger must be constructible, and map
rendering must raise the D4 actionable ImportError.  The real parity
arms run under the reference venv (conftest)."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]

pytestmark = pytest.mark.madina_api

PROBE = r'''
import json, sys
sys.path.insert(0, "src")
result = {}
try:
    import urban_network_analysis.compat.madina.una.workflows as wf
    result["import_ok"] = True
    result["pydeck_leaked"] = "pydeck" in sys.modules
    result["madina_leaked"] = any(
        m == "madina" or m.startswith("madina.") for m in sys.modules)
    result["has_logger"] = hasattr(wf, "Logger")
    result["has_flow"] = hasattr(wf, "betweenness_flow_simulation")
    result["has_knn"] = hasattr(wf, "KNN_accessibility")
    result["una_exports_tools"] = hasattr(
        sys.modules.get(
            "urban_network_analysis.compat.madina.una"),
        "betweenness") if "urban_network_analysis.compat.madina.una" in \
        sys.modules else None
    # Logger constructible without pydeck
    logger = wf.Logger.__new__(wf.Logger)
    import pandas as pd
    logger.__init__(output_folder="/tmp", pairing_table=pd.DataFrame())
    result["logger_constructible"] = isinstance(
        logger.log_df, pd.DataFrame)
    # D4: map rendering without pydeck raises the actionable ImportError
    try:
        logger.flow_map_template_1(flow_gdf=None, flow_parameter="bt")
        result["lazy_import_error"] = None
    except ImportError as exc:
        result["lazy_import_error"] = str(exc)
except Exception as exc:  # noqa: BLE001
    import traceback
    result["probe_error"] = traceback.format_exc()
print("FACADE_ENV_JSON<<<")
print(json.dumps(result))
print(">>>")
'''


def _run_probe() -> dict:
    proc = subprocess.run(
        [sys.executable, "-c", PROBE],
        capture_output=True, text=True, timeout=300, cwd=str(REPO))
    out = proc.stdout
    if "FACADE_ENV_JSON<<<" not in out:
        raise AssertionError(
            f"probe produced no JSON (rc={proc.returncode})\n"
            f"stdout:\n{out[-1500:]}\nstderr:\n{proc.stderr[-1500:]}")
    payload = out.split("FACADE_ENV_JSON<<<", 1)[1].split(">>>", 1)[0]
    return json.loads(payload)


def test_facade_imports_without_pydeck_and_no_upstream_leak():
    r = _run_probe()
    assert "probe_error" not in r, r.get("probe_error")
    assert r["import_ok"] is True
    assert r["pydeck_leaked"] is False
    assert r["madina_leaked"] is False
    assert r["has_logger"] and r["has_flow"] and r["has_knn"]


def test_logger_constructible_without_pydeck():
    r = _run_probe()
    assert "probe_error" not in r, r.get("probe_error")
    assert r["logger_constructible"] is True


def test_d4_map_rendering_raises_actionable_importerror():
    r = _run_probe()
    assert "probe_error" not in r, r.get("probe_error")
    err = r["lazy_import_error"]
    assert err is not None
    assert "pydeck" in err and "pip install pydeck" in err


@pytest.mark.parametrize("symbol", [
    "Logger", "betweenness_flow_simulation", "KNN_accessibility"])
def test_una_package_lazy_surface(symbol):
    """The una package does NOT import workflows eagerly (upstream
    __init__ star-imports only betweenness+paths) — importing the
    package must not pull pydeck, and the workflow module must still be
    reachable as a submodule."""
    probe = (
        "import sys; sys.path.insert(0, 'src');\n"
        "import urban_network_analysis.compat.madina.una as una\n"
        "print('pydeck_leaked', 'pydeck' in sys.modules)\n"
        "import importlib\n"
        f"mod = importlib.import_module("
        f"'urban_network_analysis.compat.madina.una.workflows')\n"
        f"print('symbol_ok', hasattr(mod, {symbol!r}))\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True, text=True, timeout=300, cwd=str(REPO))
    assert proc.returncode == 0, proc.stderr[-1500:]
    assert "pydeck_leaked False" in proc.stdout
    assert "symbol_ok True" in proc.stdout
