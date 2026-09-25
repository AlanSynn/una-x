"""Per-arm engine runner for L1 baseline-vs-candidate comparisons.

Executed as a subprocess with PYTHONPATH bound to ONE source tree
(baseline fork source or candidate integration source). For every seeded
case it builds both non-turn accessibility engines from a stub topology,
then records: CSR arrays, per-origin Dijkstra search state, centrality
outputs and the OD matrix, into one NPZ. The pytest driver compares the two
arms' NPZs bit-for-bit.

Usage: python _arm_runner.py --out /path/to/arm_out.npz [--no-jit-parallel]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

TESTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TESTS_DIR))

from stub_topology import StubTopology, StubSettings  # noqa: E402


def build_cases():
    """Deterministic adversarial cases. Returns list of (name, dict)."""
    cases = []
    rng = np.random.default_rng(20260925)

    def rand_graph(node_count, n_edges, dup_prob=0.3, loop_prob=0.15):
        edges = []
        while len(edges) < n_edges:
            if edges and rng.random() < dup_prob:
                a, b = edges[int(rng.integers(0, len(edges)))]
            elif rng.random() < loop_prob:
                v = int(rng.integers(0, node_count))
                a, b = v, v
            else:
                a = int(rng.integers(0, node_count))
                b = int(rng.integers(0, node_count))
            edges.append((a, b))
        arr = np.array(edges, dtype=np.int64)
        return arr[:, 0].copy(), arr[:, 1].copy()

    def access_points(node_count, n_points, seed_rng, with_node_weight=None):
        s = seed_rng.integers(0, node_count, n_points)
        e = seed_rng.integers(0, node_count, n_points)
        ws = seed_rng.random(n_points) * 5.0
        we = seed_rng.random(n_points) * 5.0
        nw = with_node_weight(seed_rng, n_points) if with_node_weight else None
        return s, e, ws, we, nw

    # 1: random medium tiny with elevation
    s, e = rand_graph(30, 90)
    z = rng.random(30) * 20.0
    o = access_points(30, 8, rng, lambda r, n: r.random(n) + 0.5)
    d = access_points(30, 12, rng, lambda r, n: r.random(n) * 3.0 + 0.25)
    cases.append(("random_elev", dict(
        start=s, end=e, weights=rng.random(90) * 50.0 + 1.0, node_count=30,
        z=z, o=o, d=d, settings=dict(elevation=True, elevation_penalty=0.3,
                                     search_radius=80.0))))

    # 2: same graph, elevation settings off (symmetric)
    cases.append(("same_symmetric", dict(
        start=s, end=e, weights=rng.random(0), node_count=30,
        z=z, o=o, d=d, settings=dict(elevation=False, elevation_penalty=0.3,
                                     search_radius=80.0), reuse_from="random_elev")))

    # 3: no z, elevation settings on (falls back to symmetric weights)
    cases.append(("no_z_elev_on", dict(
        start=s, end=e, weights=rng.random(0), node_count=30,
        z=None, o=o, d=d, settings=dict(elevation=True, elevation_penalty=0.3,
                                        search_radius=80.0), reuse_from="random_elev")))

    # 4: parallel/self-loop heavy micro graph
    s4 = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4], dtype=np.int64)
    e4 = np.array([1, 1, 2, 0, 3, 2, 4, 4, 3], dtype=np.int64)
    w4 = np.array([2.0, 2.5, 1.0, 1.25, 3.0, 0.5, 4.0, 4.25, 0.75])
    o4 = (np.array([0, 1, 2]), np.array([2, 3, 4]),
          np.array([0.5, 1.0, 1.5]), np.array([0.25, 0.75, 1.25]), None)
    d4 = (np.array([0, 2, 4]), np.array([1, 3, 0]),
          np.array([1.0, 2.0, 0.5]), np.array([0.5, 1.5, 0.25]),
          np.array([1.5, 0.5, 2.5]))
    cases.append(("micro_parallel", dict(
        start=s4, end=e4, weights=w4, node_count=5,
        z=np.array([0.0, 5.0, 2.5, 10.0, 1.0]), o=o4, d=d4,
        settings=dict(elevation=True, elevation_penalty=1.0,
                      search_radius=15.0))))

    # 5: near-overflow finite weight pins (engine-level equality at the
    # float64 boundary). NaN/inf pins are excluded here: the upstream scope
    # kernel enters unbounded queue growth on NaN edge weights in BOTH trees
    # (see evidence/integration/csr/nonfinite_weight_hazard.md).
    s5, e5 = rand_graph(12, 30, dup_prob=0.2, loop_prob=0.2)
    w5 = np.random.default_rng(7).random(30) * 20.0 + 0.5
    w5[3] = 1e300
    w5[11] = 5e300
    o5 = access_points(12, 4, rng, lambda r, n: r.random(n) + 0.5)
    d5 = access_points(12, 6, rng, lambda r, n: r.random(n))
    cases.append(("extreme_finite_weights", dict(
        start=s5, end=e5, weights=w5, node_count=12,
        z=None, o=o5, d=d5, settings=dict(elevation=True, elevation_penalty=0.3,
                                          search_radius=40.0))))

    # 6: int32 endpoints -> helper refuses, both arms use original path
    s6, e6 = rand_graph(10, 24)
    o6 = access_points(10, 3, rng, lambda r, n: r.random(n))
    d6 = access_points(10, 5, rng, lambda r, n: r.random(n))
    cases.append(("int32_fallback", dict(
        start=s6, end=e6, weights=rng.random(24) * 10.0, node_count=10,
        z=None, o=o6, d=d6, start_dtype=np.int32,
        settings=dict(elevation=True, elevation_penalty=0.3,
                      search_radius=30.0))))
    return cases


def run_arm(out_path, only_case=None):
    from urban_network_analysis.Engines.Accessibility import Accessibility
    from urban_network_analysis.Engines.AccessibilityWElevation import (
        AccessibilityWElevation,
    )
    from urban_network_analysis.Engines.AccessibilityWElevation import (
        compact_vector_node_view_scope,
    )

    store = {}
    base_cases = dict(build_cases())
    for name, spec in base_cases.items():
        if only_case is not None and name != only_case:
            continue
        if "reuse_from" in spec:
            src = base_cases[spec["reuse_from"]]
            weights = src["weights"]
            o, d = src["o"], src["d"]
        else:
            weights = spec["weights"]
            o, d = spec["o"], spec["d"]

        topo = StubTopology(
            spec["start"], spec["end"], weights, spec["node_count"],
            o[0], o[1], o[2], o[3],
            d[0], d[1], d[2], d[3], d[4],
            z=spec["z"],
            start_dtype=spec.get("start_dtype"),
        )
        st = StubSettings(**spec["settings"])

        eng_a = Accessibility(topo)
        ga = eng_a.graph_engine
        store[f"{name}/acc/pointer"] = np.asarray(ga.adjacency_pointer)
        store[f"{name}/acc/neighbors"] = np.asarray(ga.adjacency_vector)
        store[f"{name}/acc/weights"] = np.asarray(ga.adjacency_vector_weights)
        store[f"{name}/acc/flags"] = np.asarray(ga.adjacynct_vector_network_node)

        eng_e = AccessibilityWElevation(topo, st)
        ge = eng_e.graph_engine
        store[f"{name}/awe/pointer"] = np.asarray(ge.adjacency_pointer)
        store[f"{name}/awe/neighbors"] = np.asarray(ge.adjacency_vector)
        store[f"{name}/awe/weights"] = np.asarray(ge.adjacency_vector_weights)
        store[f"{name}/awe/flags"] = np.asarray(ge.adjacynct_vector_network_node)

        # per-origin search state (Dijkstra labels) on the AWE CSR
        for oi in range(min(3, len(ge.o_terminal_idxs))):
            sw, _pred = compact_vector_node_view_scope(
                ge.o_terminal_idxs[oi], ge.o_terminal_weights[oi],
                ge.adjacency_pointer, ge.adjacency_vector,
                ge.adjacency_vector_weights, ge.adjacynct_vector_network_node,
                st.search_radius, ge.d_count,
            )
            store[f"{name}/scope/{oi}"] = np.asarray(sw)

        # full centrality + OD through the public engine methods
        eng_e.Centrality(st)
        store[f"{name}/cent/reach"] = np.asarray(eng_e.reach, dtype=np.float64)
        store[f"{name}/cent/gravity_exponential"] = np.asarray(eng_e.gravity_exponential)
        store[f"{name}/cent/gravity_logistic"] = np.asarray(eng_e.gravity_logistic)
        store[f"{name}/cent/knn_access"] = np.asarray(eng_e.knn_access)
        od = eng_e.OD_Matrix(search_radius=st.search_radius)
        store[f"{name}/od"] = np.asarray(od)

    np.savez(out_path, **store)
    print(f"wrote {out_path} with {len(store)} arrays")


if __name__ == "__main__":
    import resource

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--case", default=None,
                    help="run a single case into <out> (short-lived process; "
                    "limits per-process compile memory)")
    args = ap.parse_args()
    run_arm(args.out, only_case=args.case)
    peak_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print(f"peak_rss_kb={peak_kb}")
