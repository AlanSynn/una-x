"""In-process candidate-tree loader for the F1 tests.

The B0 side of every comparison comes from the oracle package's
`support.b0()` singleton (import root wt-b0/src, module name
`urban_network_analysis`). The candidate tree cannot share that module
name in-process, so it is imported ONCE under the fixed alias
`una_f1_cand` with its real file paths preserved (numba cache keys use
the file paths, so the candidate's cache entries stay distinct from
B0's and persist across processes).

The returned namespace mirrors b0_import.load_arm's bindings for the
F1-relevant surfaces (the AggregateFlow engine, its kernel family and
the Python-side numerical helpers), and records the resolved module
files for evidence.
"""
from __future__ import annotations

import importlib
import importlib.util
import os
import sys
import types

from _legacy_paths import LEGACY_WS  # portable-path fix (HARNESS 2026-09-30)
ALIAS = "una_f1_cand"

CANDIDATE_ROOT = os.environ.get(
    "UNA_F1_CANDIDATE_ROOT",
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
    mod_pkg = importlib.import_module(ALIAS)
    mod_settings = importlib.import_module(f"{ALIAS}.Settings")
    mod_logger = importlib.import_module(f"{ALIAS}.Logger")

    ns = types.SimpleNamespace()
    ns.root = root
    ns.alias = ALIAS
    ns.AggregateFlow = mod_flow.AggregateFlow
    ns.Settings = mod_settings.Settings
    ns.Logger = mod_logger.Logger

    # AggregateFlow kernel family and Python-side numerical helpers
    # (same binding set b0_import.load_arm exposes for the flow arm)
    ns._accumulate_od_flow = mod_flow._accumulate_od_flow
    ns._accumulate_od_flow_turns = getattr(
        mod_flow, "_accumulate_od_flow_turns", None)
    ns._find_arc = mod_flow._find_arc
    ns._decay = mod_flow._decay
    ns._compute_trip_volumes = mod_flow._compute_trip_volumes
    ns._cutoff_for_shortest = mod_flow._cutoff_for_shortest
    ns._DECAY_EQUAL = mod_flow._DECAY_EQUAL
    ns._DECAY_EXPONENTIAL = mod_flow._DECAY_EXPONENTIAL
    ns._DECAY_LOGISTIC = mod_flow._DECAY_LOGISTIC

    ns.module_files = {
        "urban_network_analysis": mod_pkg.__file__,
        "Engines.AggregateFlow": mod_flow.__file__,
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
