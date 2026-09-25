"""Duck-typed Topology/Settings stubs for kernel-level exactness tests.

These stubs exercise ONLY the attributes consumed by
Accessibility._build_compact_graph_from_topology and
AccessibilityWElevation._build_compact_graph_from_topology. Genuine GIS
loading is covered separately by the L2 public-API trials; kernel tests use
these stubs so the baseline package can run unmodified from its own source
tree and the candidate from its own.

This module is test infrastructure, imported by the per-arm runner process;
it is NOT part of the production package.
"""
from __future__ import annotations

import numpy as np


class _StubLogger:
    def __init__(self, verbosity: int = 0):
        self.verbosity = verbosity
        self.records = []

    def log(self, tag, message, v=1):
        self.records.append((tag, message, v))


class _StubNetwork:
    def __init__(self, start_nodes, end_nodes, weights, node_count, z=None):
        self.start_nodes = start_nodes
        self.end_nodes = end_nodes
        self.weights = weights
        # node_points only needs a .shape whose [0] is the node count.
        self.node_points = np.zeros((node_count, 2), dtype=np.float64)
        self.z = z


class _StubAccessPoints:
    def __init__(self, edge_start_node, edge_end_node, weight_to_start,
                 weight_to_end, node_weight=None):
        self.edge_start_node = edge_start_node
        self.edge_end_node = edge_end_node
        self.weight_to_start = weight_to_start
        self.weight_to_end = weight_to_end
        # Destinations carry node_weight; origins leave it None.
        self.node_weight = (node_weight if node_weight is not None
                            else np.ones(len(edge_start_node), dtype=np.float64))


class StubTopology:
    """Minimal topology surface for both non-turn accessibility builders."""

    def __init__(self, start_nodes, end_nodes, weights, node_count,
                 o_start, o_end, o_w_start, o_w_end,
                 d_start, d_end, d_w_start, d_w_end, d_node_weight,
                 z=None, start_dtype=None, end_dtype=None):
        sn = np.asarray(start_nodes, dtype=start_dtype or np.int64)
        en = np.asarray(end_nodes, dtype=end_dtype or np.int64)
        self.network = _StubNetwork(sn, en, np.asarray(weights, dtype=np.float64),
                                    node_count, z=z)
        self.origins = _StubAccessPoints(
            np.asarray(o_start, dtype=np.int64), np.asarray(o_end, dtype=np.int64),
            np.asarray(o_w_start, dtype=np.float64),
            np.asarray(o_w_end, dtype=np.float64))
        self.destinations = _StubAccessPoints(
            np.asarray(d_start, dtype=np.int64), np.asarray(d_end, dtype=np.int64),
            np.asarray(d_w_start, dtype=np.float64),
            np.asarray(d_w_end, dtype=np.float64),
            node_weight=np.asarray(d_node_weight, dtype=np.float64))
        self.logger = _StubLogger()
        # Consumed by Engines.Base.__init__
        self.num_threads = 1
        self.num_clusters = None
        self._obstacle_penalties = (None, None)
        self._partial_corrections = (None, None)

    def get_obstacle_arc_penalties(self):
        return self._obstacle_penalties

    def get_partial_edge_corrections(self, access_points, for_origins: bool = False):
        return self._partial_corrections


class StubSettings:
    """Minimal Settings surface for AccessibilityWElevation / Centrality."""

    def __init__(self, search_radius=100.0, elevation=True, elevation_penalty=0.3,
                 gravity_beta=0.5, gravity_plateau=10.0,
                 gravity_logistic_midpoint=40.0, gravity_decay_constant=18.0,
                 knn_decay="exponential", knn_weights=(1.0, 0.5, 0.25),
                 calculate_reach=True, calculate_exponential_gravity=True,
                 calculate_logistic_gravity=True, calculate_knn_access=True):
        self.search_radius = search_radius
        self.elevation = elevation
        self.elevation_penalty = elevation_penalty
        self.gravity_beta = gravity_beta
        self.gravity_plateau = gravity_plateau
        self.gravity_logistic_midpoint = gravity_logistic_midpoint
        self.gravity_decay_constant = gravity_decay_constant
        self.knn_decay = knn_decay
        self.knn_weights = np.asarray(knn_weights, dtype=np.float64)
        self.calculate_reach = calculate_reach
        self.calculate_exponential_gravity = calculate_exponential_gravity
        self.calculate_logistic_gravity = calculate_logistic_gravity
        self.calculate_knn_access = calculate_knn_access
