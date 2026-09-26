"""Replay helpers for the OBSERVED kernel-boundary fixtures captured
from the B0 O2 pass (tests/large_e2e/oracle/golden/observed/).

This is the comparator extension H04-N4 requires: any B0-family
namespace (B0 itself, or a candidate loaded via b0_import.load_arm with
UNA_ORACLE_ARM_A/ARM_B pointed at the candidate tree) can be run
against the captured typed arrays; outputs are compared with
comparator.assert_array_bytes_equal (dtype+shape+bytes, no tolerance).

Boundaries (see capture_observed.py for the capture contract):
  - integrated_scope_access (AccessibilityWElevation family — the O2
    accessibility kernel; called twice in the pass: RunAccessibility
    and the auto-gravity-cap derivation inside RunFlow)
  - _accumulate_od_flow (per-OD via-arc loading kernel; fixtures store
    the ZERO-BUFFER per-call delta, delivered return bit-equal to the
    live call)
  - _compute_trip_volumes (per-origin trip generation)
  - _precompute_dest_gradients (scipy-based reverse gradient assembly,
    replicated by trace_gradients.trace_gradient_chunks)
"""
from __future__ import annotations

import json

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra as scipy_dijkstra

from support import GOLDEN_DIR
from trace_gradients import trace_gradient_chunks

OBS_DIR = GOLDEN_DIR / "observed"
NPZ_PATH = OBS_DIR / "observed_o2_kernels.npz"
SIDECAR_PATH = OBS_DIR / "observed_o2_kernels.hashes.json"


def load_observed():
    """(sidecar_dict, lazy npz handle). The npz is opened lazily; each
    array is materialized only on access, keeping suite RAM bounded."""
    with open(SIDECAR_PATH) as fh:
        sidecar = json.load(fh)
    return sidecar, np.load(NPZ_PATH)


def selected_od_records(sidecar):
    return [r for r in sidecar["calls"]["accumulate_od_flow"] if r["selected"]]


def replay_integrated_scope_access(b, npz, rec):
    """Run one captured integrated_scope_access call through namespace
    `b` (which must carry the AccessibilityWElevation-family kernel for
    the observed records). Returns (outputs list, args dict)."""
    args = {name: np.asarray(npz[key])
            for name, key in rec["args"].items()}
    if "Elevation" in rec["module"]:
        kernel = b.integrated_scope_access_elevation
    else:
        kernel = b.integrated_scope_access
    outs = kernel(**args, **rec["scalars"])
    return [np.asarray(o) for o in outs], args


def replay_gradient(npz, rec, chunk=None):
    """Rebuild the reverse CSR from the captured arrays and replicate
    the engine's chunked gradient assembly. Returns the assembled
    (indptr, nodes, dist, pred) dict.

    chunk=None uses the engine's rule (1e8 // n_total sources per
    scipy call, a ~1.2 GB dense transient at O2 scale). A smaller chunk
    bounds only that transient: scipy sources are independent and the
    assembly concatenates in source order either way, so the assembled
    bytes are chunk-invariant — the replay tests PROVE that by matching
    the engine-chunk captured outputs with a bounded chunk."""
    indptr = np.asarray(npz[rec["args"]["_csr_rev_indptr"]])
    indices = np.asarray(npz[rec["args"]["_csr_rev_indices"]])
    data = np.asarray(npz[rec["args"]["_csr_rev_data"]])
    n_total = indptr.shape[0] - 1
    csr_rev = csr_matrix((data, indices, indptr), shape=(n_total, n_total))
    dest_nodes = np.asarray(npz[rec["args"]["dest_nodes"]])
    n_dest = int(rec["scalars"]["n_dest"])
    limit = float(rec["scalars"]["limit"])
    if chunk is None:
        chunk = max(1, int(1e8 // max(n_total, 1)))
    return trace_gradient_chunks(scipy_dijkstra, csr_rev, dest_nodes,
                                 limit, n_dest, chunk)


def replay_od_flow(b, sidecar, npz, rec, budget_override=None):
    """Run one captured _accumulate_od_flow call with FRESH ZERO output
    buffers (the fixture's canonical per-call delta form). Returns
    (out_AB, out_BA, out_node_flow, delivered). budget_override injects
    a mutated budget for negative checks.

    Buffer shapes are part of the captured boundary contract: the out
    arrays are EDGE-indexed (n_edges = 69957 at O2), not arc-indexed
    (the kernel writes via edge_id_of_arc), so their shapes are taken
    from the captured outputs, not derived from the CSR."""
    s = rec["scalars"]
    arrs = sidecar["arrays"]
    out_AB = np.zeros(arrs[rec["outputs"]["out_AB"]]["shape"], dtype=np.float64)
    out_BA = np.zeros(arrs[rec["outputs"]["out_BA"]]["shape"], dtype=np.float64)
    out_node = np.zeros(arrs[rec["outputs"]["out_node_flow"]]["shape"],
                        dtype=np.float64)
    budget = float(s["budget"]) if budget_override is None \
        else float(budget_override)
    delivered = b._accumulate_od_flow(
        np.asarray(npz[rec["args"]["indptr"]]),
        np.asarray(npz[rec["args"]["indices"]]),
        np.asarray(npz[rec["args"]["weights"]]),
        np.asarray(npz[rec["args"]["edge_id_of_arc"]]),
        np.asarray(npz[rec["args"]["dir_of_arc"]]),
        np.asarray(npz[rec["args"]["d_o"]]),
        np.asarray(npz[rec["args"]["d_d"]]),
        np.asarray(npz[rec["args"]["pred_o"]]),
        np.asarray(npz[rec["args"]["pred_d"]]),
        int(s["origin_virtual_node"]),
        int(s["dest_virtual_node"]),
        int(s["o_edge_id"]),
        int(s["d_edge_id"]),
        float(s["d_shortest"]),
        budget,
        int(s["decay_curve_id"]),
        float(s["decay_beta"]),
        float(s["decay_midpoint"]),
        float(s["trip_volume"]),
        int(s["n_net"]),
        out_AB, out_BA, out_node,
    )
    return out_AB, out_BA, out_node, delivered


def replay_trip_volumes(b, npz, rec):
    args = {name: np.asarray(npz[key])
            for name, key in rec["args"].items()}
    return np.asarray(b._compute_trip_volumes(**args, **rec["scalars"]))
