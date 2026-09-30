"""In-process composed-candidate loader for the I00 composition tests.

Same discipline as the A3/F3 loaders: the B0 side of every comparison
comes from the oracle package's `support.b0()` singleton (import root
wt-b0/src, module name `urban_network_analysis`); the COMPOSED
candidate tree — all five screened tracks in one source (A1+A3 dispatch
in the Accessibility families, F1 striping + F2 local overlap + F3
gradient chunks in AggregateFlow) — is imported ONCE under the fixed
alias `una_composition_cand` with its real file paths preserved (numba
cache keys use the file paths, so the candidate's cache entries stay
distinct from B0's and persist across processes).

The namespace binds both engine families, the flow engine, both private
scratch modules, and the module objects needed for source-byte
assertions, and records the resolved module files for evidence.
"""
from __future__ import annotations

import importlib
import importlib.util
import os
import sys
import types

from _legacy_paths import LEGACY_WS  # portable-path fix (HARNESS 2026-09-30)
ALIAS = "una_composition_cand"

CANDIDATE_ROOT = os.environ.get(
    "UNA_COMPOSITION_CANDIDATE_ROOT",
    str(LEGACY_WS) + "/wt-large-e2e/src",
)

_CAND = None


def load_candidate(root=None):
    """Import the composed candidate tree under the fixed alias; return
    the bound namespace."""
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

    mod_acc = importlib.import_module(f"{ALIAS}.Engines.Accessibility")
    mod_acce = importlib.import_module(
        f"{ALIAS}.Engines.AccessibilityWElevation")
    mod_wturns = importlib.import_module(f"{ALIAS}.Engines.AccessibilityWTurns")
    mod_flow = importlib.import_module(f"{ALIAS}.Engines.AggregateFlow")
    mod_helper = importlib.import_module(
        f"{ALIAS}.Engines._large_access_scratch")
    mod_lfws = importlib.import_module(
        f"{ALIAS}.Engines._large_flow_workspace")
    mod_settings = importlib.import_module(f"{ALIAS}.Settings")
    mod_logger = importlib.import_module(f"{ALIAS}.Logger")

    ns = types.SimpleNamespace()
    ns.root = root
    ns.alias = ALIAS

    # Accessibility engine family (candidate; carries A1+A3 dispatch)
    ns.compact_vector_node_view_scope = mod_acc.compact_vector_node_view_scope
    ns.adjust_destination_distances = mod_acc.adjust_destination_distances
    ns.reach_gravity_knn_access = mod_acc.reach_gravity_knn_access
    ns.od_compact_vector_node_view_scope = mod_acc.od_compact_vector_node_view_scope
    ns.integrated_scope_access = mod_acc.integrated_scope_access

    # AccessibilityWElevation engine family (candidate; the family the
    # observed O2 pass exercises; carries the mirrored A1+A3 dispatch)
    ns.integrated_scope_access_elevation = mod_acce.integrated_scope_access
    ns.od_compact_vector_node_view_scope_elevation = (
        mod_acce.od_compact_vector_node_view_scope)
    ns.compact_vector_node_view_scope_elevation = (
        mod_acce.compact_vector_node_view_scope)

    # A1/A3 private helpers (the sibling dispatch routes)
    ns._a1_scope_search = mod_helper._a1_scope_search
    ns._a1_scope_admits = mod_helper._a1_scope_admits
    ns._a3_scope_search_tailless = mod_helper._a3_scope_search_tailless
    ns._a3_tail_admits = mod_helper._a3_tail_admits
    ns.helper_module = mod_helper

    # Flow engine + F3 workspace selector + module (for source-byte and
    # attribute-level assertions)
    ns.AggregateFlow = mod_flow.AggregateFlow
    ns.flow_module = mod_flow
    ns.lfws = mod_lfws
    ns.Settings = mod_settings.Settings
    ns.Logger = mod_logger.Logger
    ns.wturns_module = mod_wturns

    ns.module_files = {
        "Engines.Accessibility": mod_acc.__file__,
        "Engines.AccessibilityWElevation": mod_acce.__file__,
        "Engines.AccessibilityWTurns": mod_wturns.__file__,
        "Engines.AggregateFlow": mod_flow.__file__,
        "Engines._large_access_scratch": mod_helper.__file__,
        "Engines._large_flow_workspace": mod_lfws.__file__,
        "Settings": mod_settings.__file__,
    }
    return ns


def cand():
    """Session-wide singleton composed-candidate namespace."""
    global _CAND
    if _CAND is None:
        _CAND = load_candidate()
    return _CAND
