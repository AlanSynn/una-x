"""T11 bounded child: per-call NRT allocation census for the compiled
_accumulate_od_flow on ONE selected observed O2 record
(NUMBA_NRT_STATS=1, set before numba import). One arm per child.

Usage: nrt_child_f1.py ROOT OUT_JSON

Replays the first selected captured call R times with fresh zero
buffers and records per-call NRT (alloc, free) deltas. The parent
asserts alloc==free per call (no leak), arm-equality of the per-call
counts, and the H05 rate (48 allocs/call).
"""
from __future__ import annotations

import json
import os

os.environ["NUMBA_NRT_STATS"] = "1"  # must precede numba import

import sys

import numpy as np

root, out_json = sys.argv[1], sys.argv[2]
sys.path.insert(0, root)

from urban_network_analysis.Engines.AggregateFlow import _accumulate_od_flow
from numba.core.runtime import rtsys

OBS_DIR = ("/Users/alansynn/orca/workspaces/una-x/wt-large-e2e"
           "/tests/large_e2e/oracle/golden/observed")
SIDECAR_PATH = os.path.join(OBS_DIR, "observed_o2_kernels.hashes.json")
NPZ_PATH = os.path.join(OBS_DIR, "observed_o2_kernels.npz")

with open(SIDECAR_PATH) as fh:
    sidecar = json.load(fh)
rec = next(r for r in sidecar["calls"]["accumulate_od_flow"]
           if r.get("selected"))
npz = np.load(NPZ_PATH)

s = rec["scalars"]
arrs = sidecar["arrays"]
out_ab = np.zeros(arrs[rec["outputs"]["out_AB"]]["shape"], dtype=np.float64)
out_ba = np.zeros(arrs[rec["outputs"]["out_BA"]]["shape"], dtype=np.float64)
out_node = np.zeros(arrs[rec["outputs"]["out_node_flow"]]["shape"],
                    dtype=np.float64)

ARGS = (
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
    float(s["budget"]),
    int(s["decay_curve_id"]),
    float(s["decay_beta"]),
    float(s["decay_midpoint"]),
    float(s["trip_volume"]),
    int(s["n_net"]),
    out_ab, out_ba, out_node,
)


def fresh_call():
    ab = np.zeros(out_ab.shape, dtype=np.float64)
    ba = np.zeros(out_ba.shape, dtype=np.float64)
    nd = np.zeros(out_node.shape, dtype=np.float64)
    return _accumulate_od_flow(*ARGS[:20], ab, ba, nd)


# warm-up first: NRT stats become readable after the NRT initializes
fresh_call()

per_call = []
for _ in range(8):
    b = rtsys.get_allocation_stats()
    fresh_call()
    a = rtsys.get_allocation_stats()
    per_call.append({"alloc": int(a.alloc - b.alloc),
                     "free": int(a.free - b.free)})

with open(out_json, "w") as fh:
    json.dump({"root": root, "record_origin_virtual":
               int(s["origin_virtual_node"]),
               "per_call": per_call}, fh, indent=2)
print("F1_NRT_CHILD_OK", out_json)
