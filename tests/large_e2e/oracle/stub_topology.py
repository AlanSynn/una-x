"""Minimal topology/settings shims so the REAL B0 engines can be
constructed on oracle fixtures.

This mirrors the role of tests/perf_contract/stub_topology.py but only
implements what Accessibility and AggregateFlow actually read. No B0
code is duplicated — the engines, kernels and Settings class are the
genuine imports; only the container objects are stubs.
"""
from __future__ import annotations

import numpy as np


class StubLogger:
    """Silent, allocation-free logger matching the B0 Logger interface."""

    def __init__(self):
        self.log_list = []

    def log(self, event, details="", v=0):
        self.log_list.append((event, details))

    def warn(self, warning_text):
        self.log_list.append(("warn", warning_text))


class StubNetwork:
    def __init__(self, case):
        self.start_nodes = case.start
        self.end_nodes = case.end
        self.weights = case.weights
        self.node_points = np.zeros((case.node_count, 3), dtype=np.float64)
        self.z = None
        self.geometry = np.zeros(len(case.weights), dtype=object)  # len == n_edges


class StubAccessPoints:
    def __init__(self, start, end, w_start, w_end, node_weight, edge_ids):
        self.edge_start_node = np.asarray(start, dtype=np.int64)
        self.edge_end_node = np.asarray(end, dtype=np.int64)
        self.weight_to_start = np.asarray(w_start, dtype=np.float64)
        self.weight_to_end = np.asarray(w_end, dtype=np.float64)
        self.node_weight = np.asarray(node_weight, dtype=np.float64)
        self.nearest_edge_id = np.asarray(edge_ids, dtype=np.int64)
        self.geometry = None


class StubTopology:
    """Container with every attribute the two engines' builders read."""

    def __init__(self, case):
        self.network = StubNetwork(case)
        n_o = len(case.o_start)
        n_d = len(case.d_start)
        self.origins = StubAccessPoints(
            case.o_start, case.o_end, case.o_ws, case.o_we,
            np.ones(n_o, dtype=np.float64), np.zeros(n_o, dtype=np.int64),
        )
        self.destinations = StubAccessPoints(
            case.d_start, case.d_end, case.d_ws, case.d_we,
            case.d_node_weight, np.zeros(n_d, dtype=np.int64),
        )
        self.logger = StubLogger()
        self.num_threads = 1
        self.num_clusters = 0
        self.has_clusters = False
        self.cluster_buffer_radius = None
        self.obstacles = None
        self.observer_points = None

    def get_partial_edge_corrections(self, access_points, for_origins=True):
        return None, None


class StubFlowTopology(StubTopology):
    """StubTopology built from flow_network_spec() with edge ids."""

    def __init__(self, spec, origin_weights=None):
        edges = spec["edges"]
        self.node_count = spec["node_count"]
        self.edge_specs = edges
        start = np.array([e[0] for e in edges], dtype=np.int64)
        end = np.array([e[1] for e in edges], dtype=np.int64)
        weights = np.array([e[2] for e in edges], dtype=np.float64)

        class _Net:
            pass

        net = _Net()
        net.start_nodes = start
        net.end_nodes = end
        net.weights = weights
        net.node_points = np.zeros((self.node_count, 3), dtype=np.float64)
        net.z = None
        net.geometry = np.zeros(len(edges), dtype=object)
        self.network = net

        o_edge = np.array([o[0] for o in spec["origins"]], dtype=np.int64)
        o = spec["origins"]
        ow = (np.array([x[5] for x in o], dtype=np.float64)
              if origin_weights is None else np.asarray(origin_weights, dtype=np.float64))
        self.origins = StubAccessPoints(
            [x[1] for x in o], [x[2] for x in o], [x[3] for x in o],
            [x[4] for x in o], ow, o_edge,
        )
        d = spec["destinations"]
        self.destinations = StubAccessPoints(
            [x[1] for x in d], [x[2] for x in d], [x[3] for x in d],
            [x[4] for x in d], [x[5] for x in d],
            [x[0] for x in d],
        )
        self.logger = StubLogger()
        self.num_threads = 1
        self.num_clusters = 0
        self.has_clusters = False
        self.cluster_buffer_radius = None
        self.obstacles = None
        self.observer_points = None

    def get_partial_edge_corrections(self, access_points, for_origins=True):
        return None, None


def make_settings(settings_cls, accessibility=True, **overrides):
    """A real B0 Settings instance with oracle-controlled fields.

    Accessibility flag: True prepares accessibility-relevant fields
    (float search_radius, knn_weights converted to an ndarray exactly as
    Settings.Validation does); False prepares the flow fields.
    """
    s = settings_cls()
    if accessibility:
        s.search_radius = float(overrides.pop("search_radius", 10.0))
        knn = overrides.pop("knn_weights", (1.0, 1.0, 0.5))
        s.knn_weights = np.asarray(knn, dtype=np.float64)
        if isinstance(s.knn_weights, np.ndarray):
            s.knn_weights = np.ascontiguousarray(s.knn_weights)
        s.knn_decay = overrides.pop("knn_decay", "logistic")
        s.gravity_beta = float(overrides.pop("gravity_beta", 0.001))
        s.gravity_plateau = float(overrides.pop("gravity_plateau", 0.0))
        s.gravity_logistic_midpoint = float(
            overrides.pop("gravity_logistic_midpoint", 500.0))
        s.gravity_decay_constant = float(
            overrides.pop("gravity_decay_constant", np.log(99.0)))
        s.calculate_reach = True
        s.calculate_exponential_gravity = True
        s.calculate_logistic_gravity = True
        s.calculate_knn_access = True
    for key, value in overrides.items():
        setattr(s, key, value)
    return s
