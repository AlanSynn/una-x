"""Two-import-root loader for the H03 oracles.

Every B0-family import in the oracle package goes through here so the
exact import paths are recordable in evidence. Two named roots are
supported (arm_a / arm_b) so later candidate comparisons reuse this
package without modification:

    UNA_ORACLE_ARM_A  import root for arm A (default: B0)
    UNA_ORACLE_ARM_B  import root for arm B (default: B0)

H03 itself only ever loads B0-vs-B0 and B0-vs-trace; no candidate code
is imported.

In-process loads use ONE root (the in-process interpreter cannot hold
two versions of the same module name at once). Two-root comparisons run
this package's runner in a subprocess per root -- see
benchmarks/large_e2e/oracle/compare_arms.py.
"""
from __future__ import annotations

import importlib
import os
import sys
import types

B0_ROOT = "/Users/alansynn/orca/workspaces/una-x/wt-b0/src"
B0_COMMIT = "361928e4ba38f34622cafe065b0025244db61368"


def arm_roots():
    """Return (root_a, root_b) honoring the env overrides."""
    return (
        os.environ.get("UNA_ORACLE_ARM_A", B0_ROOT),
        os.environ.get("UNA_ORACLE_ARM_B", B0_ROOT),
    )


def load_arm(root=None):
    """Import the B0 kernel/engine modules bound to ONE import root.

    Returns a namespace with the kernels, engines and helper functions
    the oracles compare, plus the resolved module paths for evidence.
    """
    if root is None:
        root = arm_roots()[0]
    if root not in sys.path:
        sys.path.insert(0, root)

    ns = types.SimpleNamespace()
    ns.root = root

    mod_acc = importlib.import_module("urban_network_analysis.Engines.Accessibility")
    mod_flow = importlib.import_module("urban_network_analysis.Engines.AggregateFlow")
    mod_pkg = importlib.import_module("urban_network_analysis")
    mod_settings = importlib.import_module("urban_network_analysis.Settings")
    mod_logger = importlib.import_module("urban_network_analysis.Logger")

    ns.Accessibility = mod_acc.Accessibility
    ns.AggregateFlow = mod_flow.AggregateFlow
    ns.Settings = mod_settings.Settings
    ns.Logger = mod_logger.Logger

    # Accessibility kernel family
    ns.reach_gravity_knn_access = mod_acc.reach_gravity_knn_access
    ns.compact_vector_node_view_scope = mod_acc.compact_vector_node_view_scope
    ns.adjust_destination_distances = mod_acc.adjust_destination_distances
    ns.od_compact_vector_node_view_scope = mod_acc.od_compact_vector_node_view_scope
    ns.integrated_scope_access = mod_acc.integrated_scope_access

    # AggregateFlow kernel family and Python-side numerical helpers
    ns._accumulate_od_flow = mod_flow._accumulate_od_flow
    ns._find_arc = mod_flow._find_arc
    ns._decay = mod_flow._decay
    ns._compute_trip_volumes = mod_flow._compute_trip_volumes
    ns._cutoff_for_shortest = mod_flow._cutoff_for_shortest
    ns._DECAY_EQUAL = mod_flow._DECAY_EQUAL
    ns._DECAY_EXPONENTIAL = mod_flow._DECAY_EXPONENTIAL
    ns._DECAY_LOGISTIC = mod_flow._DECAY_LOGISTIC

    # Module identities for the evidence record
    ns.module_files = {
        "urban_network_analysis": mod_pkg.__file__,
        "Engines.Accessibility": mod_acc.__file__,
        "Engines.AggregateFlow": mod_flow.__file__,
        "Settings": mod_settings.__file__,
        "Logger": mod_logger.__file__,
    }
    return ns


def root_identity(root=None):
    """Exact import-path record for one root (no package import needed)."""
    if root is None:
        root = arm_roots()[0]
    probe = os.path.join(root, "urban_network_analysis", "Engines", "Accessibility.py")
    return {"root": root, "b0_commit": B0_COMMIT, "accessibility_py": probe}
