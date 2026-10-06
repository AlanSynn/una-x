"""Frozen pre-change reference for P03C bit parity (campaign una_10x).

This module is a BYTE-FAITHFUL copy of ``replay_trace`` as it existed on
commit a7281c8 (P02R amendment tree), captured BEFORE the compiled outer
driver replaced its Python origin loop.  The P03C dossier (designs/03)
requires bit parity against BOTH the current replay and the original
integrated engine; this frozen copy IS the "current replay" arm — the
actual pre-change bytes, not a re-typing from memory.

DO NOT EDIT: any change here voids the parity receipt.  The parity
fixtures import ``legacy_replay_trace`` and compare its outputs, bit for
bit, against the live ``urban_network_analysis...replay_trace``.
"""

import hashlib

import numpy as np

from urban_network_analysis.Engines._trace_access_scratch import (
    TRACE_SCHEMA_VERSION,
    TraceRecord,
    TraceReplayResult,
    reach_gravity_knn_access,
)


def legacy_replay_trace(trace, d_weights, gravity_beta, gravity_plateau,
                 gravity_logistic_midpoint, gravity_growth_rate,
                 knn_decay, knn_weights, cutoff):
    """Replay metric jobs from a COMPLETE trace.

    Fresh per-job arrays (the C10 fresh-copy law).  Guards run BEFORE
    any reuse: an incomplete trace is never a cache hit; an invalid
    metric input is refused even though a trace exists (no invalid
    input becomes successful because an earlier trace hit avoided its
    validation); the cutoff must be typed- and bit-exact against the
    trace identity (-0.0 is not +0.0); the destination weight vector
    must match the trace's
    recorded destination count and the dossier-02 value domain.
    """
    def _refuse(reason):
        return TraceReplayResult(
            reach=np.zeros(0), gravity_exponential=np.zeros(0),
            gravity_logistic=np.zeros(0), knn_access=np.zeros(0),
            route="trace_replay_refused", refusal=reason,
            trace_id=trace.trace_id if isinstance(trace, TraceRecord)
            else "", folds_done=0)

    if not isinstance(trace, TraceRecord):
        return _refuse("not_a_trace_record")
    if trace.schema_version != TRACE_SCHEMA_VERSION:
        return _refuse("trace_schema_mismatch")
    if not trace.complete:
        return _refuse("incomplete_trace_not_a_cache_hit")

    cutoff_value = float(cutoff)
    cutoff_dtype = str(np.dtype(getattr(cutoff, "dtype", type(cutoff))))
    # C11-NOTE-3 repair: the gate is typed- AND bit-exact — IEEE-754
    # value equality conflates -0.0 with +0.0 (and distinct NaN
    # payloads), so the cutoff enters by its 8 bytes, never by ==.
    if np.float64(cutoff_value).tobytes() \
            != np.float64(trace.cutoff_value).tobytes() or \
            cutoff_dtype != trace.cutoff_dtype:
        return _refuse("cutoff_mismatch_with_trace_identity")

    if not (isinstance(d_weights, np.ndarray)
            and d_weights.ndim == 1):
        return _refuse("dest_weights_layout")
    if d_weights.shape[0] != trace.dest_count:
        return _refuse("dest_count_mismatch")
    if not np.all(np.isfinite(d_weights)) or \
            np.any(d_weights < 0.0):
        return _refuse("dest_weights_value_domain")

    origin_count = trace.origin_count
    reach = np.empty(origin_count, dtype=np.dtype(trace.reach_dtype))
    ge = np.empty(origin_count, dtype=np.dtype(trace.metric_dtype))
    gl = np.empty(origin_count, dtype=np.dtype(trace.metric_dtype))
    kn = np.empty(origin_count, dtype=np.dtype(trace.metric_dtype))
    folds_done = 0
    indptr = trace.indptr
    ids_all = trace.destination_ids
    dist_all = trace.distance_bits

    # Runtime gates on the trace ITSELF (C10-MINOR-1 behavioral
    # enforcement): a corrupt or structurally impossible record is
    # refused before any fold, never replayed as a cache hit.  The
    # packed layout must be exactly the dossier-04 law: indptr[O+1]
    # from 0, non-negative counts, nnz agreement, int64 ids in range,
    # float64 distances, and every slice strictly ascending (the
    # producer's contract order).
    if indptr.dtype != np.int64 or ids_all.dtype != np.int64 \
            or dist_all.dtype != np.float64 \
            or indptr.shape[0] != origin_count + 1 \
            or int(indptr[0]) != 0 \
            or dist_all.shape[0] != ids_all.shape[0] \
            or int(indptr[-1]) != ids_all.shape[0] \
            or int(np.diff(indptr).min(initial=0)) < 0:
        return _refuse("corrupt_trace_structure")
    if ids_all.shape[0] and (int(ids_all.min()) < 0
                             or int(ids_all.max()) >= trace.dest_count):
        return _refuse("corrupt_trace_destination_ids")
    # C11-NOTE-4: the packed bytes must still hash to the digest bound
    # at build — a forged or bit-flipped retained record is refused
    # before any fold, never replayed as a cache hit.
    if hashlib.sha256(indptr.tobytes() + ids_all.tobytes()
                      + dist_all.tobytes()).hexdigest() \
            != trace.content_digest:
        return _refuse("corrupt_trace_content_digest")

    for origin in range(origin_count):
        sl = slice(int(indptr[origin]), int(indptr[origin + 1]))
        cand_ids = ids_all[sl]
        if cand_ids.shape[0] > 1 and int(np.diff(cand_ids).min()) <= 0:
            return _refuse("corrupt_trace_slice_order")
        # Writable working copies before the compiled fold: the fold
        # sees exactly the S11 argument domain (fresh C-contiguous
        # writable arrays), and the immutable trace is never aliased
        # into a callee.
        cand_dist = np.array(dist_all[sl])
        cand_wts = np.array(d_weights[cand_ids])
        reach[origin], ge[origin], gl[origin], kn[origin] = \
            reach_gravity_knn_access(
                d_distance=cand_dist,
                d_weights=cand_wts,
                cutoff=cutoff,
                gravity_beta=gravity_beta,
                gravity_plateau=gravity_plateau,
                gravity_logistic_midpoint=gravity_logistic_midpoint,
                gravity_growth_rate=gravity_growth_rate,
                knn_gravity_plateau=gravity_plateau,
                knn_weights=knn_weights,
                knn_gravity_beta=gravity_beta,
                knn_decay=knn_decay,
                knn_gravity_logistic_midpoint=(
                    gravity_logistic_midpoint),
                knn_gravity_growth_rate=gravity_growth_rate,
            )
        folds_done += 1
    return TraceReplayResult(
        reach=reach, gravity_exponential=ge, gravity_logistic=gl,
        knn_access=kn, route="trace_replay", refusal="",
        trace_id=trace.trace_id, folds_done=folds_done)
