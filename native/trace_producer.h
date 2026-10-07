/* N00 native trace producer core (campaign una_10x, dossier 08).
 *
 * Exact AOT port of the numba kernel
 * `Engines._trace_access_scratch._tr_epoch_scope_search_tile` and the
 * per-tile driver inside `build_trace` (the trace producer region).
 *
 * Parity law (verified by tests/native, bit-exact):
 * - NO floating-point reduction exists in this kernel: every
 *   accumulator is int64.  FP work is single adds, comparisons and the
 *   Python-semantics min (b < a) ? b : a -- none of it is fastmath-
 *   sensitive (no reassociation or contraction opportunity).
 * - The heap orders (float64 weight, int64 node) pairs lexicographic-
 * ally.  Node ids are unique, so the pop sequence is a total order;
 * duplicate identical pairs are indistinguishable.
 * - Per-origin candidate sorts run over UNIQUE int64 destination ids:
 *   any correct ascending sort yields the identical sequence.
 * - The caller (Python) runs the verbatim S11 admission gate before
 *   this code: edge costs and cutoff are finite nonnegative, dtypes
 *   are exactly int64/float64/bool_ and layouts are C-contiguous.
 *   NaN/inf never enter the heap.
 *
 * Dossier 08 obligations implemented here:
 * - immutable input buffers (all input pointers are const; nothing is
 *   written through them),
 * - explicit owners (every output buffer is owned by ProducerOutput
 *   and released by producer_output_free; no global mutable state --
 *   all scratch is per-call),
 * - bounded scratch (declared, measured high-water in
 *   ProducerOutput.scratch_peak_bytes),
 * - bounded and cancellable work (cancel flag checked at every tile
 *   barrier; budget checked per tile with build_trace's exact
 *   formula and boundary),
 * - no unsafe FP flags (built without -ffast-math and with
 *   -ffp-contract=off; see native/build.py).
 */
#ifndef UNA_N00_TRACE_PRODUCER_H
#define UNA_N00_TRACE_PRODUCER_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Status codes (mapped to Python exceptions by the binding AFTER the
 * GIL is re-acquired -- precise exception mapping happens in the
 * binding, never inside the released section). */
#define UNA_N00_OK 0
#define UNA_N00_ERR_ALLOC 1
#define UNA_N00_ERR_ARG 2
#define UNA_N00_CANCELED 3

struct una_n00_input {
    /* --- immutable input buffers (caller-owned, const) --- */
    const int64_t *o_idx;   /* [O, 2] origin terminal nodes        */
    const double *o_w;      /* [O, 2] origin seed weights          */
    const int64_t *adj_ptr; /* [N + 1] CSR pointer                 */
    const int64_t *adj_vec; /* [E] CSR neighbor ids                */
    const double *adj_w;    /* [E] CSR edge costs (finite >= 0)    */
    const uint8_t *net;     /* [E] flag column (np.bool_ bytes)    */
    const int64_t *d_idx;   /* [D, 2] destination terminals        */
    const double *d_w;      /* [D, 2] destination terminal weights */
    const int64_t *rev_ptr; /* [N + 1] reverse incidence pointer   */
    const int64_t *rev_vec; /* [2D] reverse incidence (dups kept)  */
    /* --- dimensions --- */
    int64_t O, N, D, E;
    /* --- scalars (same values build_trace computes) --- */
    double cutoff;
    int64_t epoch_rollover;
    int64_t tile_origins; /* already clamped: max(1, min(req, O))   */
    int32_t lanes;        /* max(1, min(threads_bound, tile_origins)) */
    int64_t budget_cand_bytes;
    /* 16 * candidates_total ceiling derived from the caller's trace
     * byte budget: budget_bytes - metadata_bytes - indptr_bytes.  The
     * per-tile budget check reproduces build_trace's formula
     * ((O+1)*8 + 16*cand_total + metadata > budget) exactly; passing
     * the precomputed ceiling keeps the arithmetic identical while
     * the binding records the original budget in evidence. */
    int64_t max_degree; /* from the admission gate (>= 0) */
};

struct una_n00_output {
    /* --- packed trace stream (origin-major, ascending dest id per
     * origin; the caller publishes these bytes verbatim) --- */
    int64_t *out_ids;   /* [nnz] */
    double *out_dist;   /* [nnz] */
    int64_t nnz;
    /* --- per-origin candidate counts (caller cumsums into indptr) --- */
    int64_t *counts; /* [O] */
    /* --- measured counters (same law as TraceCounters) --- */
    int64_t cand_total;
    int64_t touched_total;
    int64_t tiles_done;      /* completed tiles (incl. budget tile) */
    int64_t budget_hit;      /* origin index one past the last
                              * attempted origin, or -1 */
    int64_t scratch_peak_bytes; /* measured scratch high-water */
    /* --- lane counters, [lanes, 2]: touched, candidates --- */
    int64_t *lane_counters;
    int32_t lanes_written;
    /* --- status --- */
    int32_t status;
};

/* Run the whole bounded-tile producer stream.  Thread policy: one
 * worker per lane (input->lanes), joined before return; scratch is
 * per-call.  `cancel` may be NULL; when non-NULL it is polled at each
 * tile barrier and stops the stream at a tile boundary. */
int una_n00_run_stream(const struct una_n00_input *in,
                       struct una_n00_output *out,
                       const volatile int *cancel);

/* Free every buffer owned by `out` (idempotent; zeroes the struct). */
void una_n00_output_free(struct una_n00_output *out);

#ifdef __cplusplus
}
#endif

#endif /* UNA_N00_TRACE_PRODUCER_H */
