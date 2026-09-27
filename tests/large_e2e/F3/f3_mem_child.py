"""Margin-gap measurement child (proof section 3(d)(ii), test_f3_cleanup).

Runs INSIDE a fresh child process so ru_maxrss is a clean high-water
for one budgeted gradient call on a representative sparse-row fixture
(the observed cell's regime: ~60 finite entries per destination row,
not the parts-dominating regime — dossier stop condition line 36).

Reports JSON on stdout:
  {v_prime, n_dest, schedule, payload_bytes, retained_bytes,
   rss_before, rss_after, rss_delta, gap_bytes, margin_half,
   gate_pass, ru_maxrss_units}

gate: gap_bytes = rss_delta - payload_bytes - retained_bytes
      <= MARGIN_BYTES // 2  (32 MiB)  — realized-transient-minus-payload
per slice within half the margin; violation = stop-and-re-derive the
margin with reviewer approval BEFORE the implementation commit freezes
the proof (the caps themselves never change).
"""
from __future__ import annotations

import json
import resource
import sys
import types

import numpy as np
from scipy.sparse import csr_matrix

SRC_ROOT = sys.argv[1]

sys.path.insert(0, SRC_ROOT)
from urban_network_analysis.Engines.AggregateFlow import (  # noqa: E402
    AggregateFlow)
from urban_network_analysis.Engines import (  # noqa: E402
    _large_flow_workspace as lfws)


def rss():
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # darwin reports BYTES; Linux reports KiB
    return raw * (1 if sys.platform == "darwin" else 1024)


def build_stub():
    v_net, n_dest = 20000, 300
    arcs = []
    for i in range(v_net):
        arcs.append((i, (i + 1) % v_net, 1.0))
        arcs.append((i, (i + 7) % v_net, 2.0))
    # each destination virtual attached to a local neighborhood anchor;
    # limit bounds the backward reach to a sparse row
    for d in range(n_dest):
        arcs.append(((17 * d + 11) % v_net, v_net + d, 1.0))
    n_total = v_net + n_dest
    order = sorted(range(len(arcs)), key=lambda i: (arcs[i][0], arcs[i][1]))
    indptr = np.zeros(n_total + 1, dtype=np.int64)
    indices = np.empty(len(arcs), dtype=np.int64)
    weights = np.empty(len(arcs), dtype=np.float64)
    for pos, i in enumerate(order):
        u, v, w = arcs[i]
        indices[pos] = v
        weights[pos] = w
        indptr[u + 1] += 1
    indptr = np.cumsum(indptr)
    fwd = csr_matrix((weights, indices, indptr), shape=(n_total, n_total))
    stub = types.SimpleNamespace()
    stub._n_network_nodes = v_net
    stub._n_destinations = n_dest
    stub._n_origins = 0
    stub._csr_indptr = np.asarray(fwd.indptr)
    stub._csr_indices = np.asarray(fwd.indices)
    stub._csr_weights = fwd.data
    stub._csr_fwd = fwd
    stub._csr_rev = fwd.T.tocsr()
    stub._gradient_limit = lambda ns: 30.0
    captured = []
    stub.logger = types.SimpleNamespace(
        log=lambda *a, **k: captured.append((a, k)))
    stub._captured = captured
    return stub


def main():
    import gc
    stub = build_stub()
    gc.collect()
    r0 = rss()
    out = AggregateFlow._precompute_dest_gradients(stub, {})
    r1 = rss()
    gc.collect()
    line = stub._captured[0][0][1]
    sched = line.split("schedule=")[1].split("]")[0]
    widths = [int(x) for x in sched[1:].split(",") if x]
    v_prime = stub._csr_indptr.shape[0] - 1
    payload = sum(w * v_prime * 13 for w in widths)  # dist f8 + pred i4 + mask b1
    retained = out[0].nbytes + out[1].nbytes + out[2].nbytes + out[3].nbytes
    delta = r1 - r0
    gap = delta - payload - retained
    print(json.dumps({
        "v_prime": int(v_prime),
        "n_dest": int(stub._n_destinations),
        "schedule": widths,
        "payload_bytes": int(payload),
        "retained_bytes": int(retained),
        "rss_before": int(r0),
        "rss_after": int(r1),
        "rss_delta": int(delta),
        "gap_bytes": int(gap),
        "margin_half": lfws.MARGIN_BYTES // 2,
        "gate_pass": bool(gap <= lfws.MARGIN_BYTES // 2),
        "ru_maxrss_units": "bytes (darwin) / KiB*1024 (linux)",
        "log_line": line,
    }))


if __name__ == "__main__":
    main()
