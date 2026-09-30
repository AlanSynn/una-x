"""SCIENCE confirmation probes (bug registry hypotheses -> executed evidence).

One probe per registry hypothesis.  Each returns a JSON-serializable
observations dict; the harness runs every probe in a bounded subprocess
(hard time + RSS limits, NUMERICS.md validation procedure) and retains
the capsule.  Probes CHARACTERIZE baseline behavior — they do not pass
judgement; the assertions that promote/refute registry entries live in
test_bug_probes.py against the recorded capsule.

Probes (registry IDs):
  frac_weight_trunc    BUG-FRAC-WEIGHT-TRUNC   reach dtype vs fractional weights
  same_edge_od         BUG-SAME-EDGE-OD        endpoint-route vs direct partial impedance
  coincident_seeds     BUG-COINCIDENT-SEEDS    seed overwrite vs minimum at equal terminal ids
  nonfinite_validation BUG-NONFINITE-NOVALIDATION  NaN/negative/zero-cost/sentinel kernel behavior
  prep_zero_coord      BUG-PREP-ZERO-COORD-DROP    pinned filter(None) Z-drop eats 0.0 planar coords

BUG-SVC-GEOMCOLL is promoted on the executed FAILURES INC03 capsule
(no duplicate probe here).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]


def _csr(edges):
    """Edge list [(u, v, w)] -> (pointer, vector, weights) sorted CSR."""
    edges = sorted(edges)
    node_count = max(max(u, v) for u, v, _ in edges) + 1
    pointer = np.zeros(node_count + 1, dtype=np.int64)
    for u, v, _ in edges:
        pointer[u + 1] += 1
        pointer[v + 1] += 1
    pointer = np.cumsum(pointer)
    vector = np.zeros(pointer[-1], dtype=np.int64)
    weights = np.zeros(pointer[-1], dtype=np.float64)
    fill = pointer[:-1].copy()
    for u, v, w in edges:
        vector[fill[u]] = v
        weights[fill[u]] = w
        fill[u] += 1
        vector[fill[v]] = u
        weights[fill[v]] = w
        fill[v] += 1
    return pointer, vector, weights


def _terminal(row):
    return (np.array([row[0]], dtype=np.int64).reshape(1, 2),
            np.array([row[1]], dtype=np.float64).reshape(1, 2))


# ----------------------------------------------------------------------
# baseline-engine probes (import the REAL source tree)
# ----------------------------------------------------------------------

def _baseline_imports():
    sys.path.insert(0, str(REPO / "src"))
    from urban_network_analysis.Engines import Accessibility as A
    return A


def probe_frac_weight_trunc():
    """Registry probe: destinations with weights 0.25/0.50 inside the
    radius -> weighted reach must be 0.75 binary64; the driver allocates
    the reach vector with o_terminal_idxs.dtype (int64)."""
    A = _baseline_imports()
    pointer, vector, weights = _csr([(0, 1, 10.0)])
    net_node = np.array([True], dtype=np.bool_)
    o_idxs, o_weights = _terminal(((0, 0), (0.0, 0.0)))
    d_idxs = np.array([[0, 0], [1, 1]], dtype=np.int64)
    d_terminal_weights = np.array([[0.0, 0.0], [0.0, 0.0]], dtype=np.float64)
    d_weights = np.array([0.25, 0.50], dtype=np.float64)
    reach, gravity_exp, gravity_log, knn = A.integrated_scope_access(
        o_idxs, o_weights,
        pointer, vector, weights, net_node,
        d_idxs, d_terminal_weights, d_weights,
        0.0,          # gravity_beta
        0.0,          # gravity_plateau
        1.0,          # gravity_logistic_midpoint
        1.0,          # gravity_growth_rate
        "none",       # knn_decay (kernel's else branch: coefficients * weights)
        np.array([1.0, 0.5], dtype=np.float64),   # knn_weights (array, per-kNN)
        100.0,        # cutoff
    )
    exact = float(d_weights.sum())
    return {
        "reach_dtype": str(reach.dtype),
        "reach_value": int(reach[0]),
        "exact_weighted_sum_binary64": exact,
        "truncated": int(reach[0]) != exact,
        "gravity_exponential_dtype": str(gravity_exp.dtype),
        "gravity_exponential_value": float(gravity_exp[0]),
    }


def probe_same_edge_od():
    """Registry probe: origin at 10 m and destination at 90 m on the same
    100 m edge.  Direct partial impedance = 80; the kernel assembles
    min(endpoint routes) = 100 (origin split seeds route through both
    endpoints)."""
    A = _baseline_imports()
    pointer, vector, weights = _csr([(0, 1, 100.0)])
    net_node = np.array([True], dtype=np.bool_)
    o_idxs, o_weights = _terminal(((0, 1), (10.0, 90.0)))
    d_idxs, d_weights = _terminal(((0, 1), (90.0, 10.0)))
    od = A.od_compact_vector_node_view_scope(
        o_idxs, o_weights,
        pointer, vector, weights, net_node,
        1000.0,       # cutoff (wide open; geometry, not radius, is under test)
        1,            # d_count
        d_idxs, d_weights,
    )
    # scope label cross-check for the same configuration
    scope, _ = A.compact_vector_node_view_scope(
        o_idxs[0], o_weights[0], pointer, vector, weights, net_node,
        1000.0, 0)
    direct_partial = 80.0
    return {
        "od_distance": float(od[0, 0]),
        "direct_partial_impedance": direct_partial,
        "endpoint_route_value": 100.0,
        "scope_node_labels": [float(x) for x in scope[:2]],
        "matches_endpoint_routes_not_direct":
            float(od[0, 0]) == 100.0 and float(od[0, 0]) != direct_partial,
    }


def probe_coincident_seeds():
    """Registry probe: origin terminal pair collapses to one node
    (idxs (0,0)) with asymmetric split weights (0, 5).  Seed init
    assigns start then end, so 5 overwrites 0 instead of min(0,5)=0;
    the downstream label of the neighbor node shifts by the same
    delta."""
    A = _baseline_imports()
    pointer, vector, weights = _csr([(0, 1, 10.0)])
    net_node = np.array([True], dtype=np.bool_)
    o_idxs, o_weights = _terminal(((0, 0), (0.0, 5.0)))
    scope, _ = A.compact_vector_node_view_scope(
        o_idxs[0], o_weights[0], pointer, vector, weights, net_node,
        1000.0, 0)
    seed_label = float(scope[0])
    neighbor_label = float(scope[1])
    # kernel mechanism: line 137 sets node 0 seed = 0 (start weight),
    # line 138 overwrites it with 5 (end weight); the stale (0, 0) queue
    # entry still relaxes the neighbor first, so the neighbor label
    # comes from the DISCARDED start weight, while the origin node
    # keeps the overwritten 5 — the overwrite rule, not min.
    return {
        "coincident_node_label": seed_label,
        "neighbor_label": neighbor_label,
        "min_rule_label": 0.0,
        "overwrite_rule_label": 5.0,
        "neighbor_via_stale_start_entry": 10.0,
        "matches_overwrite_not_min": seed_label == 5.0,
    }


def probe_nonfinite_validation():
    """Registry probe: unvalidated kernels under NaN cost, negative
    cycle, zero-cost cycle, and sentinel-radius overflow.  Run inside
    the bounded child only (negative cycles may not terminate)."""
    A = _baseline_imports()
    out = {}

    # (a) NaN edge cost: NaN fails both comparison guards, so the
    # neighbor is silently unreachable — no error, no marker.
    pointer, vector, weights = _csr([(0, 1, 10.0)])
    weights[0] = np.nan            # directed edge slot 0 -> 1
    net_node = np.array([True], dtype=np.bool_)
    o_idxs, o_weights = _terminal(((0, 0), (0.0, 0.0)))
    scope, _ = A.compact_vector_node_view_scope(
        o_idxs[0], o_weights[0], pointer, vector, weights, net_node,
        100.0, 0)
    out["nan_edge_neighbor_label"] = float(scope[1])
    out["nan_edge_silently_unreachable"] = bool(scope[1] > 100.0)

    # (b) zero-cost cycle: 2-node ring with 0.0 costs — termination and
    # labels under a label-correcting scan without a visited set.
    pointer, vector, weights = _csr([(0, 1, 0.0), (1, 0, 0.0)])
    net_node = np.array([True, True], dtype=np.bool_)
    o_idxs, o_weights = _terminal(((0, 0), (0.0, 0.0)))
    scope, _ = A.compact_vector_node_view_scope(
        o_idxs[0], o_weights[0], pointer, vector, weights, net_node,
        100.0, 0)
    out["zero_cost_cycle_labels"] = [float(x) for x in scope[:2]]
    out["zero_cost_cycle_terminated"] = True

    # (c) sentinel initialization at extreme radius: labels are
    # ones(...) + cutoff; with cutoff near max float the sentinel
    # itself overflows to inf.
    pointer, vector, weights = _csr([(0, 1, 10.0)])
    net_node = np.array([True], dtype=np.bool_)
    scope, _ = A.compact_vector_node_view_scope(
        o_idxs[0], o_weights[0], pointer, vector, weights, net_node,
        np.nextafter(np.float64(np.finfo(np.float64).max), 0), 0)
    out["sentinel_overflow_label_init"] = bool(scope[1] > np.finfo(np.float64).max)
    out["extreme_cutoff_reachable_label"] = float(scope[1])

    # (d) negative cycle: supervised hang/corruption check.  A 2-node
    # ring with -5 costs drives label-correcting forever if the kernel
    # re-relaxes; bound THIS sub-probe with an internal iteration guard
    # so the capsule records the mechanism without hanging the child.
    pointer, vector, weights = _csr([(0, 1, 10.0)])
    weights[:] = -5.0              # both directed slots: -5 ring
    net_node = np.array([True], dtype=np.bool_)
    try:
        import signal

        def _alarm(signum, frame):
            raise TimeoutError("negative-cycle relaxation did not settle")

        old = signal.signal(signal.SIGALRM, _alarm)
        signal.setitimer(signal.ITIMER_REAL, 2.0)
        try:
            scope, _ = A.compact_vector_node_view_scope(
                o_idxs[0], o_weights[0], pointer, vector, weights, net_node,
                100.0, 0)
            out["negative_cycle"] = "terminated"
            out["negative_cycle_labels"] = [float(x) for x in scope[:2]]
        except TimeoutError as exc:
            out["negative_cycle"] = f"nonterminating ({exc})"
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, old)
    except Exception as exc:                      # noqa: BLE001 - capsule records all
        out["negative_cycle"] = f"error: {type(exc).__name__}: {exc}"

    return out


# ----------------------------------------------------------------------
# pinned-madina probe (runs under venv_madina_legacy in its own arm)
# ----------------------------------------------------------------------

def probe_prep_zero_coord():
    """Registry probe: pinned prepare_geometry strips Z with
    tuple(filter(None, [x, y])) — filter(None) drops 0.0, so any vertex
    with x=0 or y=0 loses a planar coordinate once the layer contains a
    has_z row."""
    import geopandas as gpd
    from shapely.geometry import LineString
    sys.path.insert(0, str(REPO / ".refs" / "madina_ref" / "src"))
    from madina.zonal.utils import prepare_geometry

    # ALL rows carry Z (a mixed layer hits the separate BUG-PREP-MIXED-Z
    # TypeError first); this row's first vertex has x=0.0 AND y=0.0 as
    # legitimate planar coordinates, plus a control row without zeros.
    gdf = gpd.GeoDataFrame(
        geometry=[
            LineString([(50.0, 50.0, 0.0), (60.0, 60.0, 0.0)]),
            LineString([(0.0, 0.0, 1.0), (10.0, 10.0, 2.0)]),
        ],
        crs="EPSG:32633")
    out = prepare_geometry(gdf)
    coords = [list(g.coords) for g in out.geometry]
    return {
        "result_coords": [list(map(tuple, c)) for c in coords],
        "control_row_first_vertex": tuple(coords[0][0]),
        "zero_row_first_vertex": tuple(coords[1][0]),
        "expected_first_vertex": (0.0, 0.0),
        "planar_zeros_dropped": tuple(coords[1][0]) != (0.0, 0.0),
    }


PROBES = {
    "frac_weight_trunc": probe_frac_weight_trunc,
    "same_edge_od": probe_same_edge_od,
    "coincident_seeds": probe_coincident_seeds,
    "nonfinite_validation": probe_nonfinite_validation,
    "prep_zero_coord": probe_prep_zero_coord,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", required=True, choices=sorted(PROBES))
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    try:
        observations = PROBES[args.probe]()
        capsule = {"probe": args.probe, "ok": True,
                   "observations": observations}
    except Exception as exc:                      # noqa: BLE001 - capsule records all
        import traceback
        capsule = {"probe": args.probe, "ok": False,
                   "error": f"{type(exc).__name__}: {exc}",
                   "traceback": traceback.format_exc()[-4000:]}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(capsule, sort_keys=True, indent=1))
    print(f"capsule: {out}")


if __name__ == "__main__":
    main()
