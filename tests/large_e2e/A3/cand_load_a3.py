"""In-process candidate-tree loader for the A3 tests.

Same discipline as the A1 loader: the B0 side of every comparison
comes from the oracle package's `support.b0()` singleton (import root
wt-b0/src, module name `urban_network_analysis`); the candidate tree
is imported ONCE under the fixed alias `una_a3_cand` with its real
file paths preserved (numba cache keys use the file paths, so the
candidate's cache entries stay distinct from B0's and persist across
processes).

The namespace binds both private A1/A3 helpers plus the public kernel
and driver surfaces of both engine families, and records the resolved
module files for evidence.
"""
from __future__ import annotations

import importlib
import importlib.util
import os
import sys
import types

ALIAS = "una_a3_cand"

CANDIDATE_ROOT = os.environ.get(
    "UNA_A3_CANDIDATE_ROOT",
    "/Users/alansynn/orca/workspaces/una-x/wt-large-e2e/src",
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

    mod_acc = importlib.import_module(f"{ALIAS}.Engines.Accessibility")
    mod_acce = importlib.import_module(
        f"{ALIAS}.Engines.AccessibilityWElevation")
    mod_helper = importlib.import_module(
        f"{ALIAS}.Engines._large_access_scratch")

    ns = types.SimpleNamespace()
    ns.root = root
    ns.alias = ALIAS

    # A1 private helper (the route A3 falls back to on refusal)
    ns._a1_scope_search = mod_helper._a1_scope_search
    ns._a1_scope_admits = mod_helper._a1_scope_admits

    # A3 private helper (task A3I)
    ns._a3_scope_search_tailless = mod_helper._a3_scope_search_tailless
    ns._a3_tail_admits = mod_helper._a3_tail_admits

    # Accessibility kernel family (candidate)
    ns.compact_vector_node_view_scope = mod_acc.compact_vector_node_view_scope
    ns.adjust_destination_distances = mod_acc.adjust_destination_distances
    ns.reach_gravity_knn_access = mod_acc.reach_gravity_knn_access
    ns.od_compact_vector_node_view_scope = mod_acc.od_compact_vector_node_view_scope
    ns.integrated_scope_access = mod_acc.integrated_scope_access

    # AccessibilityWElevation kernel family (candidate) — the family the
    # observed O2 pass exercises
    ns.integrated_scope_access_elevation = mod_acce.integrated_scope_access
    ns.od_compact_vector_node_view_scope_elevation = (
        mod_acce.od_compact_vector_node_view_scope)
    ns.compact_vector_node_view_scope_elevation = (
        mod_acce.compact_vector_node_view_scope)

    ns.module_files = {
        "Engines.Accessibility": mod_acc.__file__,
        "Engines.AccessibilityWElevation": mod_acce.__file__,
        "Engines._large_access_scratch": mod_helper.__file__,
    }
    return ns


def cand():
    """Lazy singleton candidate namespace for this pytest session."""
    global _CAND
    if _CAND is None:
        _CAND = load_candidate()
    return _CAND
