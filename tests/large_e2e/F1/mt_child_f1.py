"""T4 bounded child: K=3 flow engine runs, twice with fresh engine
objects, asserting in-process bit-identity, saving all outputs for
cross-arm/cross-run byte comparison in the parent. One arm per child
(one import root each), like A1's mt_child.

Usage: mt_child_f1.py ROOT OUT_NPZ [K]

Self-contained: the fixture spec and topology stub are inline (no
oracle package import — the child imports ONLY the arm named by ROOT).
"""
from __future__ import annotations

import os
import sys

import numpy as np

root, out_npz = sys.argv[1], sys.argv[2]
K = int(sys.argv[3]) if len(sys.argv) > 3 else 3
sys.path.insert(0, root)

from urban_network_analysis.Engines.AggregateFlow import AggregateFlow
from urban_network_analysis.Settings import Settings


# ---- inline adversarial fixture (oracle flow_network_spec values) ----
EDGES = [
    (0, 1, 1.0), (1, 2, 1.0), (2, 3, 1.0), (3, 4, 1.0), (4, 8, 1.5),
    (1, 5, 2.0), (5, 6, 1.0), (6, 2, 1.0), (0, 6, 3.0), (2, 6, 0.5),
    (7, 7, 0.5), (3, 5, 2.0),
]
ORIGINS = [   # (edge_id, start, end, w_start, w_end, weight)
    (0, 0, 1, 0.1, 0.9, 3.0), (5, 1, 5, 0.2, 0.2, 2.0),
    (8, 0, 6, 0.3, 0.3, 0.0), (6, 5, 6, 0.4, 0.6, 1.0),
    (2, 2, 3, 0.5, 0.5, 1.0), (7, 6, 2, 0.5, 0.5, 0.75),
]
DESTINATIONS = [
    (0, 0, 1, 0.3, 0.7, 2.0), (3, 3, 4, 0.5, 0.5, 1.0),
    (10, 7, 7, 0.5, 0.5, 5.0),
]


class _StubAccessPoints:
    def __init__(self, rows):
        self.edge_start_node = np.array([r[1] for r in rows], dtype=np.int64)
        self.edge_end_node = np.array([r[2] for r in rows], dtype=np.int64)
        self.weight_to_start = np.array([r[3] for r in rows], dtype=np.float64)
        self.weight_to_end = np.array([r[4] for r in rows], dtype=np.float64)
        self.node_weight = np.array([r[5] for r in rows], dtype=np.float64)
        self.nearest_edge_id = np.array([r[0] for r in rows], dtype=np.int64)
        self.geometry = None


class _StubLogger:
    def __init__(self):
        self.log_list = []

    def log(self, event, details="", v=0):
        self.log_list.append((event, details))

    def warn(self, warning_text):
        self.log_list.append(("warn", warning_text))


class StubTopology:
    def __init__(self):
        self.node_count = 9
        start = np.array([e[0] for e in EDGES], dtype=np.int64)
        end = np.array([e[1] for e in EDGES], dtype=np.int64)
        weights = np.array([e[2] for e in EDGES], dtype=np.float64)

        class _Net:
            pass

        net = _Net()
        net.start_nodes = start
        net.end_nodes = end
        net.weights = weights
        net.node_points = np.zeros((self.node_count, 3), dtype=np.float64)
        net.z = None
        net.geometry = np.zeros(len(EDGES), dtype=object)
        self.network = net
        self.origins = _StubAccessPoints(ORIGINS)
        self.destinations = _StubAccessPoints(DESTINATIONS)
        self.logger = _StubLogger()
        self.num_threads = 1
        self.num_clusters = 0
        self.has_clusters = False
        self.cluster_buffer_radius = None
        self.obstacles = None
        self.observer_points = None

    def get_partial_edge_corrections(self, access_points, for_origins=True):
        return None, None


def make_settings():
    s = Settings()
    s.search_radius = 6.0
    s.flow_detour_mode = "ratio"
    s.flow_detour_ratio = 1.5
    s.flow_detour_buffer = 1.0
    s.flow_decay = False
    s.flow_decay_curve = "exponential"
    s.flow_decay_method = "gravity_cap"
    s.flow_gravity_cap = 100.0
    s.flow_path_detour_penalty = "exponential"
    s.flow_route_enumeration_beta = 0.3
    s.flow_route_enumeration_logistic_midpoint = 100.0
    s.flow_origin_weights = True
    s.flow_destination_weights = True
    s.flow_compute_node_flow = True
    s.turns = False
    return s


def run_once():
    eng = AggregateFlow(StubTopology())
    eng.num_threads = K
    eng.Centrality(make_settings())
    return eng


first = run_once()
payload = {
    "edge_flow_AB": first.edge_flow_AB,
    "edge_flow_BA": first.edge_flow_BA,
    "edge_flow": first.edge_flow,
    "node_flow": first.node_flow,
}

second = run_once()
assert second.edge_flow_AB.tobytes() == payload["edge_flow_AB"].tobytes()
assert second.edge_flow_BA.tobytes() == payload["edge_flow_BA"].tobytes()
assert second.edge_flow.tobytes() == payload["edge_flow"].tobytes()
assert second.node_flow.tobytes() == payload["node_flow"].tobytes()

np.savez(out_npz, **payload)
print("F1_MT_CHILD_OK", os.environ.get("NUMBA_NUM_THREADS"), K, out_npz)
