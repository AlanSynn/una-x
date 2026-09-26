"""F3 explanatory trace: byte-budgeted chunked destination-gradient
assembly, replicating AggregateFlow._precompute_dest_gradients with a
PARAMETRIZED chunk size (B0's default chunk is max(1, 1e8 // n_total);
the numerical result must be independent of the chunk choice).

Per-chunk records expose the dist+pred row extracts and the per-row
np.where(finite) order. The final concatenated indptr/nodes/dist/pred
must be identical across chunk sizes — verified bit-for-bit by the
tests against both this trace's variants and the compiled B0 engine
method.

reverse_where=True is the negative mutant (per-row retained order
reversed before the sparse append).
"""
from __future__ import annotations

import numpy as np


def trace_gradient_chunks(scipy_dijkstra, csr_rev, dest_nodes, limit,
                          n_dest, chunk, reverse_where=False):
    """Replica of the chunked sparse gradient assembly.

    scipy_dijkstra must be scipy.sparse.csgraph.dijkstra from the SAME
    environment the compiled engine uses. chunk must be >= 1.
    """
    if chunk < 1:
        raise ValueError("chunk must be >= 1")
    idx_parts, dist_parts, pred_parts = [], [], []
    counts = np.zeros(n_dest, dtype=np.int64)
    chunks = []
    for s in range(0, n_dest, chunk):
        e = min(s + chunk, n_dest)
        dist, preds = scipy_dijkstra(
            csr_rev, directed=True,
            indices=dest_nodes[s:e],
            limit=limit,
            return_predecessors=True,
        )
        if dist.ndim == 1:
            dist, preds = dist[None, :], preds[None, :]
        finite = np.isfinite(dist)
        rows = []
        for k in range(e - s):
            cols = np.where(finite[k])[0]
            if reverse_where:
                cols = cols[::-1].copy()
            counts[s + k] = cols.shape[0]
            idx_parts.append(cols.astype(np.int64))
            dist_parts.append(dist[k, cols].astype(np.float64))
            pred_parts.append(preds[k, cols].astype(np.int32))
            rows.append({
                "chunk_start": int(s),
                "chunk_end": int(e),
                "row_in_chunk": int(k),
                "global_dest_index": int(s + k),
                "cols": cols.astype(np.int64).copy(),
                "dist_values": dist[k, cols].astype(np.float64).copy(),
                "pred_values": preds[k, cols].astype(np.int32).copy(),
            })
        chunks.append({
            "start": int(s), "end": int(e),
            "dist_shape": list(dist.shape),
            "dist_dtype": str(dist.dtype),
            "pred_dtype": str(preds.dtype),
            "rows": rows,
        })
    indptr = np.zeros(n_dest + 1, dtype=np.int64)
    np.cumsum(counts, out=indptr[1:])
    nodes = np.concatenate(idx_parts) if idx_parts else np.zeros(0, np.int64)
    dist_flat = (np.concatenate(dist_parts) if dist_parts
                 else np.zeros(0, np.float64))
    pred_flat = (np.concatenate(pred_parts) if pred_parts
                 else np.zeros(0, np.int32))
    return {
        "indptr": indptr,
        "nodes": nodes,
        "dist": dist_flat,
        "pred": pred_flat,
        "counts": counts,
        "chunks": chunks,
        "chunk_size": int(chunk),
        "n_chunks": len(chunks),
    }
