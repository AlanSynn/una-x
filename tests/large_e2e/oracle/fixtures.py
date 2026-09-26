"""Programmatic, seeded L0 fixture corpus for the H03 oracles.

Two families:

* Engine cases (`engine_cases`) — raw network/origin/destination arrays
  accepted by the real engine constructors. From these the oracles build
  a StubTopology (tests) and replicate the fallback adjacency builder
  row order (kernel-level calls): forward incidences in input-edge
  order, then reverse incidences in input-edge order, per row
  (identical in both B0 construction paths — verified against the live
  engine's arrays in the golden tests).

* Kernel-only cases (`kernel_scope_cases`) — hand-built CSR rows that
  the kernel contract must handle (flags=False incidences, parallel
  incidences in both orders, stale queue entries, cutoff/nextafter
  boundaries, subnormals, signed zeros, near-overflow finite weights,
  the R+1==R sentinel domain, inf incidence weights).

Hazard cases (negative/NaN weights — documented nontermination domains)
are NOT fixtures here; they run only in bounded child processes from
child_hazard.py.

All randomness flows through np.random.default_rng(FIXTURE_SEED) with a
single draw order, so the corpus is reproducible from this file alone.
"""
from __future__ import annotations

import numpy as np

FIXTURE_SEED = 20260925

MAX_FLOAT = np.finfo(np.float64).max
MIN_SUBNORMAL = np.float64(5e-324)
MAX_SUBNORMAL = np.nextafter(np.float64(2.2250738585072014e-308), np.float64(-1.0))
TWO_POW_53 = np.float64(2.0 ** 53)


class EngineCase:
    """Raw arrays for one engine-level L0 graph."""

    def __init__(self, name, node_count, edges, origins, destinations,
                 cutoff, knn_decay="logistic", knn_weights=(1.0, 1.0, 0.5),
                 start_dtype=np.int64, gravity_beta=0.001, metric_plateau=0.0,
                 metric_midpoint=500.0):
        self.name = name
        self.node_count = node_count
        # edges: list of (u, v, w)
        self.start = np.array([e[0] for e in edges], dtype=start_dtype)
        self.end = np.array([e[1] for e in edges], dtype=start_dtype)
        self.weights = np.array([e[2] for e in edges], dtype=np.float64)
        # origins/destinations: list of (start_node, end_node, w_start, w_end[, d_weight])
        self.o_start = np.array([o[0] for o in origins], dtype=np.int64)
        self.o_end = np.array([o[1] for o in origins], dtype=np.int64)
        self.o_ws = np.array([o[2] for o in origins], dtype=np.float64)
        self.o_we = np.array([o[3] for o in origins], dtype=np.float64)
        self.d_start = np.array([d[0] for d in destinations], dtype=np.int64)
        self.d_end = np.array([d[1] for d in destinations], dtype=np.int64)
        self.d_ws = np.array([d[2] for d in destinations], dtype=np.float64)
        self.d_we = np.array([d[3] for d in destinations], dtype=np.float64)
        self.d_node_weight = np.array(
            [d[4] if len(d) > 4 else 1.0 for d in destinations], dtype=np.float64
        )
        self.cutoff = float(cutoff)
        self.knn_decay = knn_decay
        self.knn_weights = np.array(knn_weights, dtype=np.float64)
        self.gravity_beta = float(gravity_beta)
        self.metric_plateau = float(metric_plateau)
        self.metric_midpoint = float(metric_midpoint)

    def kernel_arrays(self):
        """Replicate B0's fallback adjacency row order: per row, forward
        incidences in input-edge order then reverse incidences in
        input-edge order. Returns the kernel signature arrays.
        """
        n = self.node_count
        start = self.start.astype(np.int64)
        end = self.end.astype(np.int64)
        w = self.weights.astype(np.float64)
        pointer = np.zeros(n + 1, dtype=np.int64)
        for node in range(n):
            count = int(np.sum(start == node) + np.sum(end == node))
            pointer[node + 1] = pointer[node] + count
        nbrs, ws, flags = [], [], []
        for node in range(n):
            m_start = start == node
            nbrs.extend(end[m_start]); ws.extend(w[m_start]); flags.extend([True] * int(m_start.sum()))
            m_end = end == node
            nbrs.extend(start[m_end]); ws.extend(w[m_end]); flags.extend([True] * int(m_end.sum()))
        return dict(
            adjacency_pointer=pointer,
            adjacency_vector=np.array(nbrs, dtype=np.int64),
            adjacency_vector_weights=np.array(ws, dtype=np.float64),
            adjacynct_vector_network_node=np.array(flags, dtype=np.bool_),
        )

    def terminal_arrays(self):
        return dict(
            o_terminal_idxs=np.array([self.o_start, self.o_end], dtype=np.int64).T,
            o_terminal_weights=np.array([self.o_ws, self.o_we], dtype=np.float64).T,
            d_count=len(self.d_node_weight),
            d_terminal_idxs=np.array([self.d_start, self.d_end], dtype=np.int64).T,
            d_terminal_weights=np.array([self.d_ws, self.d_we], dtype=np.float64).T,
            d_weights=self.d_node_weight,
        )


def engine_cases():
    """The L0 engine/graph battery. Order is part of the corpus identity."""
    rng = np.random.default_rng(FIXTURE_SEED)
    cases = []

    cases.append(EngineCase(
        "l0_empty_edges", 2, [],
        origins=[(0, 1, 0.0, 5.0)], destinations=[(0, 1, 1.0, 2.0, 3.0)],
        cutoff=10.0))

    cases.append(EngineCase(
        "l0_duplicate_directed", 3, [(0, 1, 2.0), (0, 1, 2.0), (1, 2, 1.0)],
        origins=[(0, 0, 0.0, 0.0)], destinations=[(2, 2, 0.5, 0.5, 2.0)],
        cutoff=10.0))

    cases.append(EngineCase(
        "l0_parallel_cheap_first", 2, [(0, 1, 2.0), (0, 1, 5.0)],
        origins=[(0, 0, 0.0, 0.0)], destinations=[(1, 1, 1.0, 1.0, 1.0)],
        cutoff=10.0))

    cases.append(EngineCase(
        "l0_parallel_cheap_last", 2, [(0, 1, 5.0), (0, 1, 2.0)],
        origins=[(0, 0, 0.0, 0.0)], destinations=[(1, 1, 1.0, 1.0, 1.0)],
        cutoff=10.0))

    cases.append(EngineCase(
        "l0_self_loop", 3, [(1, 1, 3.0), (0, 1, 2.0)],
        origins=[(0, 0, 0.0, 0.0)], destinations=[(2, 1, 0.25, 0.75, 1.5)],
        cutoff=10.0))

    cases.append(EngineCase(
        "l0_degree_one_terminal", 3, [(0, 1, 2.0), (1, 2, 3.0)],
        origins=[(2, 2, 0.5, 7.5)], destinations=[(0, 0, 1.0, 2.0, 1.0)],
        cutoff=100.0))

    cases.append(EngineCase(
        "l0_disconnected_components", 4, [(0, 1, 2.0), (2, 3, 2.0)],
        origins=[(0, 1, 0.1, 0.2)], destinations=[(2, 3, 0.3, 0.4, 2.5)],
        cutoff=10.0))

    cases.append(EngineCase(
        "l0_isolated_origin_destination", 3, [(0, 1, 1.0)],
        origins=[(0, 0, 0.0, 0.0)], destinations=[(2, 2, 0.5, 0.5, 4.0)],
        cutoff=10.0))

    cases.append(EngineCase(
        "l0_same_terminal_unequal_costs", 2, [(0, 1, 2.5)],
        origins=[(0, 0, 0.5, 7.5)], destinations=[(1, 1, 0.25, 3.75, 1.0)],
        cutoff=10.0))

    cases.append(EngineCase(
        "l0_overlapping_host_edges", 3, [(0, 1, 2.0), (1, 2, 3.0)],
        origins=[(0, 1, 0.2, 0.8)], destinations=[(0, 1, 0.3, 0.7, 2.0)],
        cutoff=10.0))

    # Seeded random micro-graph with duplicates and a self-loop.
    edges = []
    seen = {(0, 1)}
    edges.append((0, 1, 2.0))
    while len(edges) < 9:
        u = int(rng.integers(0, 6))
        v = int(rng.integers(0, 6))
        edges.append((u, v, float(rng.random() * 5.0) + 0.5))
    cases.append(EngineCase(
        "l0_tiny_random_dup_loop", 6, edges,
        origins=[(0, 2, 0.3, 0.6), (3, 5, 0.1, 0.9)],
        destinations=[(1, 4, 0.2, 0.7, 1.25), (5, 5, 0.5, 0.5, 0.75),
                      (0, 3, 0.15, 0.85, 2.5)],
        cutoff=8.0))

    # int32 endpoints: the representation observed topologies feed in.
    cases.append(EngineCase(
        "l1_int32_endpoints", 5,
        [(0, 1, 1.5), (1, 2, 1.5), (2, 3, 1.0), (3, 4, 1.0), (0, 4, 6.0)],
        origins=[(0, 1, 0.2, 0.2)], destinations=[(3, 4, 0.4, 0.4, 2.0)],
        cutoff=9.0, start_dtype=np.int32))

    # Float-domain battery ------------------------------------------------
    cases.append(EngineCase(
        "flt_nextafter_cutoff", 2,
        [(0, 1, 5.0), (0, 1, float(np.nextafter(np.float64(5.0), np.float64(0)))),
         (0, 1, float(np.nextafter(np.float64(5.0), np.float64(1))))],
        origins=[(0, 0, 0.0, 0.0)], destinations=[(1, 1, 0.5, 0.5, 1.0)],
        cutoff=5.0))

    cases.append(EngineCase(
        "flt_subnormals", 3,
        [(0, 1, float(MIN_SUBNORMAL)), (1, 2, float(MAX_SUBNORMAL))],
        origins=[(0, 0, 0.0, 0.0)], destinations=[(2, 1, 0.0, 0.0, 1.0)],
        cutoff=1.0))

    cases.append(EngineCase(
        "flt_signed_zero", 3,
        [(0, 1, -0.0), (0, 2, 0.0)],
        # The END terminal assignment happens last, so weight_to_end
        # -0.0 is what lands in scope[0]; the -0.0 edge keeps label
        # -0.0 on node 1 (c = -0.0 + -0.0), while node 2 gets +0.0
        # (c = +0.0 + -0.0 = +0.0).
        origins=[(0, 0, 0.0, -0.0)],
        destinations=[(1, 1, -0.0, 0.0, 1.0), (1, 1, 0.0, -0.0, 1.0),
                      (2, 2, -0.0, 0.0, 1.0)],
        cutoff=5.0))

    cases.append(EngineCase(
        "flt_near_overflow_finite", 3,
        [(0, 1, 1.5e308), (1, 2, 1.5e308)],
        origins=[(0, 0, 0.0, 0.0)], destinations=[(1, 2, 1e308, 1e308, 1.0)],
        cutoff=1e307))

    # Sentinel domain: b = 1 + 2**53 rounds back to R. Destination d0
    # sits on DISCONNECTED node 2, so its adjusted distance is
    # min(b+w_s, b+w_e) == R and it PASSES the <= R filter in B0 — the
    # A2 b<=R hazard, recorded as golden behavior. d1 is reachable.
    # Equal adjusted distances with unequal weights cover the KNN tie.
    cases.append(EngineCase(
        "flt_sentinel_two_pow_53", 3, [(0, 1, 2.0)],
        origins=[(0, 0, 0.0, 0.0)],
        destinations=[(2, 2, 0.5, 0.5, 1.0), (1, 1, 0.5, 0.5, 3.0)],
        cutoff=float(TWO_POW_53)))

    cases.append(EngineCase(
        "flt_maxfloat_cutoff", 2, [(0, 1, 2.0)],
        origins=[(0, 0, 0.0, 0.0)], destinations=[(1, 1, 0.5, 0.5, 1.0)],
        cutoff=float(MAX_FLOAT)))

    cases.append(EngineCase(
        "flt_inf_incidence_weight", 3,
        [(0, 1, float(np.inf)), (1, 2, 1.0)],
        origins=[(0, 0, 0.0, 0.0)], destinations=[(2, 1, 0.5, 0.5, 1.0)],
        cutoff=10.0))

    # Metric variants on a common graph -----------------------------------
    base = [(0, 1, 1.0), (1, 2, 1.0), (0, 2, 1.5), (2, 3, 2.0)]
    common_origins = [(0, 1, 0.25, 0.5), (2, 3, 0.75, 0.25)]
    common_dests = [(1, 2, 0.5, 0.5, 1.5), (3, 0, 0.25, 0.75, 2.5),
                    (0, 2, 0.1, 0.9, 0.5)]
    cases.append(EngineCase(
        "metric_knn_exponential_k1", 4, base,
        origins=common_origins, destinations=common_dests,
        cutoff=4.0, knn_decay="exponential", knn_weights=(1.0,)))
    cases.append(EngineCase(
        "metric_knn_logistic_k3", 4, base,
        origins=common_origins, destinations=common_dests,
        cutoff=4.0, knn_decay="logistic", knn_weights=(1.0, 1.0, 0.5)))
    cases.append(EngineCase(
        "metric_knn_none_long_weights", 4, base,
        origins=common_origins, destinations=common_dests,
        cutoff=4.0, knn_decay="none",
        knn_weights=(1.0, 0.8, 0.6, 0.4, 0.2)))
    cases.append(EngineCase(
        "metric_empty_knn_weights", 4, base,
        origins=common_origins, destinations=common_dests,
        cutoff=4.0, knn_decay="exponential", knn_weights=()))
    cases.append(EngineCase(
        "metric_zero_beta_plateau", 4, base,
        origins=common_origins, destinations=common_dests,
        cutoff=4.0, knn_decay="logistic", knn_weights=(1.0, 1.0),
        gravity_beta=0.0, metric_plateau=1.0, metric_midpoint=2.0))
    return cases


class KernelScopeCase:
    """Hand-built CSR row battery for compact_vector_node_view_scope."""

    def __init__(self, name, pointer, nbrs, weights, flags, v_count, d_count,
                 o_idx, o_w, cutoff):
        self.name = name
        self.adjacency_pointer = np.asarray(pointer, dtype=np.int64)
        self.adjacency_vector = np.asarray(nbrs, dtype=np.int64)
        self.adjacency_vector_weights = np.asarray(weights, dtype=np.float64)
        self.adjacynct_vector_network_node = np.asarray(flags, dtype=np.bool_)
        self.v_count = int(v_count)
        self.d_count = int(d_count)
        self.o_terminal_idxs = np.asarray(o_idx, dtype=np.int64)
        self.o_terminal_weights = np.asarray(o_w, dtype=np.float64)
        self.cutoff = float(cutoff)


def kernel_scope_cases():
    cases = []
    # K1: flags False incidence — eligible but never pushed (virtual-like).
    cases.append(KernelScopeCase(
        "k_flags_false_row",
        pointer=[0, 2, 3, 3], nbrs=[1, 2, 0], weights=[1.0, 1.5, 1.0],
        flags=[True, False, True], v_count=3, d_count=1,
        o_idx=[[0, 0]], o_w=[[0.0, 0.0]], cutoff=10.0))
    # K2/K3: duplicate destinations in both orders (A1 staged-overwrite).
    cases.append(KernelScopeCase(
        "k_dup_dest_cheap_first",
        pointer=[0, 2, 2], nbrs=[1, 1], weights=[2.0, 5.0],
        flags=[True, True], v_count=2, d_count=1,
        o_idx=[[0, 0]], o_w=[[0.0, 0.0]], cutoff=10.0))
    cases.append(KernelScopeCase(
        "k_dup_dest_cheap_last",
        pointer=[0, 2, 2], nbrs=[1, 1], weights=[5.0, 2.0],
        flags=[True, True], v_count=2, d_count=1,
        o_idx=[[0, 0]], o_w=[[0.0, 0.0]], cutoff=10.0))
    # K4: stale queue entry — direct 0->2 weight 2.5 is pushed (deg(2)>1),
    # later improved to 2.0 via node 1; the (2.5, 2) entry is then popped
    # and re-scanned against current labels (no longer eligible).
    cases.append(KernelScopeCase(
        "k_stale_queue",
        pointer=[0, 2, 4, 6, 7], nbrs=[1, 2, 2, 3, 0, 3, 2],
        weights=[1.0, 2.5, 1.0, 1.0, 1.0, 1.0, 1.0],
        flags=[True] * 7, v_count=4, d_count=1,
        o_idx=[[0, 0]], o_w=[[0.0, 0.0]], cutoff=10.0))
    # K5: zero cutoff — eligibility is c <= 0.
    cases.append(KernelScopeCase(
        "k_zero_cutoff",
        pointer=[0, 2, 3, 3], nbrs=[1, 2, 1], weights=[0.0, 1.0, 0.0],
        flags=[True] * 3, v_count=3, d_count=1,
        o_idx=[[0, 1]], o_w=[[0.0, 0.0]], cutoff=0.0))
    # K6: heap tie ordering by node id (equal keys); node 3 is a real
    # fourth network node with its own row. NOTE: the first draft of
    # this case pointed node 3's incidence at the DESTINATION TAIL
    # (v_count) with no row of its own; the compiled kernel read past
    # the end of adjacency_pointer there (numba does not bounds-check)
    # — a production-unrepresentable shape, fixed here to a well-formed
    # graph and recorded in oracle_manifest.json.
    cases.append(KernelScopeCase(
        "k_equal_weight_ties",
        pointer=[0, 3, 4, 5, 6], nbrs=[2, 1, 3, 0, 0, 0],
        weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        flags=[True] * 6, v_count=4, d_count=1,
        o_idx=[[0, 0]], o_w=[[0.0, 0.0]], cutoff=10.0))
    # K7: degree-one neighbor network flag True — assignment without push.
    cases.append(KernelScopeCase(
        "k_degree_one_no_push",
        pointer=[0, 2, 3, 3], nbrs=[1, 2, 0], weights=[1.0, 2.0, 1.0],
        flags=[True, True, True], v_count=3, d_count=1,
        o_idx=[[0, 0]], o_w=[[0.0, 0.0]], cutoff=10.0))
    # K8: seed exactly equal to the sentinel value b = 1 + cutoff.
    cases.append(KernelScopeCase(
        "k_seed_equals_sentinel",
        pointer=[0, 1, 2], nbrs=[1, 0], weights=[1.0, 1.0],
        flags=[True, True], v_count=2, d_count=1,
        o_idx=[[0, 0]], o_w=[[6.0, 0.5]], cutoff=5.0))
    # K9: origin start == end terminal, unequal weights (overwrite order).
    cases.append(KernelScopeCase(
        "k_same_terminal_overwrite",
        pointer=[0, 1, 2], nbrs=[1, 0], weights=[2.0, 1.0],
        flags=[True, True], v_count=2, d_count=1,
        o_idx=[[0, 0]], o_w=[[7.5, 0.5]], cutoff=10.0))
    return cases


def flow_network_spec():
    """Tiny adversarial flow network (shared by flow F1/F2/F3 oracles).

    n_net=9. Properties: two parallel routes, a cycle, a U-turn
    dead-end candidate, a disconnected self-loop node (7), a zero-weight
    origin (o2), an origin sharing its host edge with a destination
    (o0/d0), an unreachable destination (d2), tied distances (e9 cross
    arc 0.5). edge ids are the list positions.
    """
    edges = [
        (0, 1, 1.0),   # e0  host edge of d0 and o0
        (1, 2, 1.0),   # e1
        (2, 3, 1.0),   # e2
        (3, 4, 1.0),   # e3  host edge of d1 (leaf node 4 → U-turn domain)
        (4, 8, 1.5),   # e4  degree-one node 8
        (1, 5, 2.0),   # e5  host edge of o1
        (5, 6, 1.0),   # e6
        (6, 2, 1.0),   # e7  cycle 1-5-6-2
        (0, 6, 3.0),   # e8  host edge of o2
        (2, 6, 0.5),   # e9  cross arc, creates tied distances
        (7, 7, 0.5),   # e10 disconnected self-loop; host edge of d2
        (3, 5, 2.0),   # e11 cross link
    ]
    origins = [  # (edge_id, start_node, end_node, w_start, w_end, origin_weight)
        (0, 0, 1, 0.1, 0.9, 3.0),   # o0 — same host edge as d0
        (5, 1, 5, 0.2, 0.2, 2.0),   # o1
        (8, 0, 6, 0.3, 0.3, 0.0),   # o2 — zero weight (skipped when use_o_weights)
        (6, 5, 6, 0.4, 0.6, 1.0),   # o3 — cycle edge e6
        (2, 2, 3, 0.5, 0.5, 1.0),   # o4 — edge e2
        (7, 6, 2, 0.5, 0.5, 0.75),  # o5 — cycle edge e7
    ]
    # five ACTIVE origins so every K in {1,2,3} has all three stripes
    # non-empty — the F1 fold-order mutant (final sum K-1..0) is only
    # byte-detectable when an edge receives >= 3 nonzero partials.
    destinations = [  # (edge_id, start_node, end_node, w_start, w_end, dest_weight)
        (0, 0, 1, 0.3, 0.7, 2.0),   # d0 — overlapping host edge with o0
        (3, 3, 4, 0.5, 0.5, 1.0),   # d1
        (10, 7, 7, 0.5, 0.5, 5.0),  # d2 — unreachable
    ]
    return dict(node_count=9, edges=edges, origins=origins,
                destinations=destinations)


def flow_settings_overrides(node_flow=True):
    return dict(
        search_radius=6.0,
        flow_detour_mode="ratio",
        flow_detour_ratio=1.5,
        flow_detour_buffer=1.0,
        flow_decay=False,
        flow_decay_curve="exponential",
        flow_decay_method="gravity_cap",
        flow_gravity_cap=100.0,
        flow_path_detour_penalty="exponential",
        flow_route_enumeration_beta=0.3,
        flow_route_enumeration_logistic_midpoint=100.0,
        flow_origin_weights=True,
        flow_destination_weights=True,
        flow_compute_node_flow=node_flow,
        turns=False,
    )
