"""In-process candidate-tree loader for the F3 tests.

Mirrors tests/large_e2e/F1/cand_load.py.  The B0 side of every
comparison comes from the oracle package's `support.b0()` singleton
(import root wt-b0/src, module name `urban_network_analysis`).  The
candidate tree cannot share that module name in-process, so it is
imported ONCE under the fixed alias `una_f3_cand` with its real file
paths preserved.

The returned namespace binds the F3-relevant surfaces: the
AggregateFlow CLASS and MODULE (the module, so tests can interpose the
module-global `_scipy_dijkstra` call-sequence instrumentation) and the
private `_large_flow_workspace` policy module (so tests can patch
DEFAULT_CAP_BYTES per the frozen {64,128,256} MiB set and inspect the
selector), plus Settings/Logger and resolved module files for evidence.
"""
from __future__ import annotations

import importlib
import importlib.util
import os
import sys
import types

from _legacy_paths import LEGACY_WS  # portable-path fix (HARNESS 2026-09-30)
ALIAS = "una_f3_cand"

CANDIDATE_ROOT = os.environ.get(
    "UNA_F3_CANDIDATE_ROOT",
    str(LEGACY_WS) + "/wt-large-e2e/src",
)

_CAND = None


def load_candidate(root=None):
    """Import the candidate tree under the fixed alias; return the
    bound namespace."""
    if root is None:
        root = CANDIDATE_ROOT
    pkg_dir = os.path.join(root, "urban_network_analysis")
    if not os.path.isdir(pkg_dir):
        raise RuntimeError(f"candidate package dir missing: {pkg_dir}")
    if ALIAS in sys.modules:
        raise RuntimeError(f"{ALIAS} already imported; refusing to rebind")

    init_file = os.path.join(pkg_dir, "__init__.py")
    spec = importlib.util.spec_from_file_location(
        ALIAS, init_file, submodule_search_locations=[pkg_dir])
    pkg = importlib.util.module_from_spec(spec)
    sys.modules[ALIAS] = pkg
    spec.loader.exec_module(pkg)

    mod_flow = importlib.import_module(f"{ALIAS}.Engines.AggregateFlow")
    mod_lfws = importlib.import_module(f"{ALIAS}.Engines._large_flow_workspace")
    mod_settings = importlib.import_module(f"{ALIAS}.Settings")
    mod_logger = importlib.import_module(f"{ALIAS}.Logger")

    ns = types.SimpleNamespace()
    ns.root = root
    ns.alias = ALIAS
    ns.AggregateFlow = mod_flow.AggregateFlow
    ns.flow_module = mod_flow
    ns.lfws = mod_lfws
    ns.Settings = mod_settings.Settings
    ns.Logger = mod_logger.Logger
    ns.module_files = {
        "Engines.AggregateFlow": mod_flow.__file__,
        "Engines._large_flow_workspace": mod_lfws.__file__,
        "Settings": mod_settings.__file__,
        "Logger": mod_logger.__file__,
    }
    return ns


def cand():
    """Lazy singleton candidate namespace for this pytest session."""
    global _CAND
    if _CAND is None:
        _CAND = load_candidate()
    return _CAND
