"""T16: observed-fixture replay (H04-N4 capture, O2 scale).

The captured integrated_scope_access records (module
AccessibilityWElevation, 16 origins, the real observed CSR) replayed
through the candidate namespace must reproduce every stored output
byte-exactly, and the B0 side is re-asserted in the same session so
the triangle candidate==B0==stored holds in one place.

The capture contains ZERO od_compact_vector_node_view_scope records
(the observed O2 pass never called the OD driver), so the OD-driver
observed-scale coverage here replays the captured integrated inputs
through the OD drivers of BOTH arms and requires candidate==B0 byte
equality (recorded in result.json as a T16 pin correction vs the
proof's assumed 'captured OD driver records').
"""
from __future__ import annotations

import numpy as np

import observed_replay as obsrep
from comparator import assert_array_bytes_equal

from harness_a1 import b0ns, cns

INames = ("reach", "gravity_exponential", "gravity_logistic", "knn_access")


def test_t16_observed_integrated_replay_bytes():
    sidecar, npz = obsrep.load_observed()
    b = b0ns()
    c = cns()
    for i, rec in enumerate(sidecar["calls"]["integrated_scope_access"]):
        outs_b0, _ = obsrep.replay_integrated_scope_access(b, npz, rec)
        outs_c, _ = obsrep.replay_integrated_scope_access(c, npz, rec)
        for j, name in enumerate(INames):
            stored = np.asarray(npz[rec["outputs"][name]])
            assert_array_bytes_equal(outs_b0[j], stored,
                                     f"t16/isa[{i}]/b0/{name}")
            assert_array_bytes_equal(outs_c[j], stored,
                                     f"t16/isa[{i}]/cand/{name}")


def test_t16_observed_scale_od_driver_candidate_equals_b0():
    sidecar, npz = obsrep.load_observed()
    b = b0ns()
    c = cns()
    rec = sidecar["calls"]["integrated_scope_access"][0]
    args = {name: np.asarray(npz[key]) for name, key in rec["args"].items()}
    od_args = (
        args["o_terminal_idxs"], args["o_terminal_weights"],
        args["adjacency_pointer"], args["adjacency_vector"],
        args["adjacency_vector_weights"],
        args["adjacynct_vector_network_node"],
        float(rec["scalars"]["cutoff"]),
        args["d_terminal_idxs"].shape[0],
        args["d_terminal_idxs"], args["d_terminal_weights"],
    )
    out_b0 = np.asarray(b.od_compact_vector_node_view_scope_elevation(*od_args))
    out_c = np.asarray(c.od_compact_vector_node_view_scope_elevation(*od_args))
    assert_array_bytes_equal(out_c, out_b0, "t16/od_observed_scale")
    # non-degenerate: the observed radius reaches destinations
    assert np.count_nonzero(out_b0) > 0
