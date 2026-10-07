/* N00 native trace producer core -- exact port of the numba producer
 * `_tr_epoch_scope_search_tile` + `build_trace`'s tile loop (campaign
 * una_10x N00, dossier 08).  See trace_producer.h for the parity law
 * and the dossier obligations this file implements.
 *
 * Iteration structure mirrors the numba kernel exactly:
 * - lanes stride each tile: t_origin in {lane, lane+lanes, ...},
 * - per-lane arenas (stamp/label/touched/row pairs/candidate capture),
 * - epoch-labeled two-phase a3-tailless search with the dummy heap
 *   init/pop and end-then-start push gates,
 * - reverse-index candidate discovery over the per-epoch touched set,
 * - per-origin ascending-id emission with the Python-semantics min
 *   (b < a) ? b : a applied to (lw0 + tw0, lw1 + tw1).
 *
 * The only structural difference is allocation shape: the numba driver
 * allocates a fresh (tile_len x d_count) capture per tile and the
 * Python loop copies per-origin prefixes into growable part lists;
 * here capture is per-lane growable scratch appended once and assembled
 * into the packed origin-major stream after the join.  The published
 * byte layout is identical (origin-major, ascending ids per origin).
 */
#include "trace_producer.h"

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <new>
#include <thread>
#include <utility>
#include <vector>

namespace {

/* Python min semantics: min(a, b) == (b < a) ? b : a. */
inline double py_min(double a, double b) { return (b < a) ? b : a; }

/* Per-lane arena and capture scratch.  Every buffer is reused across
 * the lane's origins (bounded scratch); out_ids/out_dist grow to this
 * lane's candidate total (capacity doubling). */
struct LaneScratch {
    std::vector<int64_t> stamp;    /* [N] epoch stamps (zeroed)      */
    std::vector<double> label;     /* [N] labels (garbage until set) */
    std::vector<int64_t> touched;  /* [N] per-epoch touched nodes    */
    std::vector<int64_t> row_off;  /* [max_degree] phase-1 offsets   */
    std::vector<double> row_cost;  /* [max_degree] phase-1 costs     */
    std::vector<int64_t> cand_ids; /* [D] candidate capture          */
    std::vector<int64_t> mark;     /* [D] destination marks (zeroed) */
    std::vector<double> cand_dist; /* [D] candidate distances        */
    /* This lane's capture output, appended per origin in the lane's
     * execution order; block starts recorded in origin_off. */
    std::vector<int64_t> out_ids;
    std::vector<double> out_dist;
    /* (weight, node) heap, cleared per origin. */
    std::vector<std::pair<double, int64_t>> heap;

    int64_t arena_bytes() const {
        int64_t i8 = (int64_t)(stamp.size() + touched.size() +
                               row_off.size() + cand_ids.size() +
                               mark.size()) * 8;
        int64_t f8 = (int64_t)(label.size() + row_cost.size() +
                               cand_dist.size()) * 8;
        int64_t o8 = (int64_t)(out_ids.capacity() +
                               out_dist.capacity()) * 8 +
                     (int64_t)heap.capacity() * 16;
        return i8 + f8 + o8;
    }
};

/* Shared per-stream state. */
struct StreamState {
    std::atomic<int64_t> cand_total{0};
    std::atomic<int64_t> touched_total{0};
    /* Barrier: arrivals accumulate across tiles; the tile t barrier
     * completes when arrivals reaches (t+1) * lanes. */
    std::atomic<int64_t> arrivals{0};
    std::atomic<int64_t> generation{0};
    /* Stop reasons at a tile boundary: 1 = budget, 2 = cancel. */
    std::atomic<int> stop{0};
    std::atomic<int64_t> tiles_done{0};
    std::atomic<int64_t> scratch_peak{0};
    std::atomic<int64_t> budget_hit{-1};
    int32_t lanes = 1;
    const volatile int *cancel = nullptr;
};

void heap_push(std::vector<std::pair<double, int64_t>> &h,
               double w, int64_t n) {
    h.emplace_back(w, n);
    std::push_heap(h.begin(), h.end(), std::greater<>());
}

/* Pop the minimal (weight, node) pair. */
std::pair<double, int64_t> heap_pop(
    std::vector<std::pair<double, int64_t>> &h) {
    std::pop_heap(h.begin(), h.end(), std::greater<>());
    std::pair<double, int64_t> top = h.back();
    h.pop_back();
    return top;
}

/* One origin of the epoch search + candidate capture.  Fills
 * s.cand_ids[0..cc) with the origin's distinct candidate destinations
 * in capture order; returns the candidate and touched counts.
 * Bit-exact port of the numba body: same statement order, same
 * comparison semantics, same heap push order. */
void run_origin(const una_n00_input &in, LaneScratch &s, double b,
                int64_t e, int64_t origin, int64_t *cand_count_out,
                int64_t *touched_count_out) {
    int64_t touched_count = 0;

    int64_t o_idx_start = in.o_idx[2 * origin];
    int64_t o_idx_end = in.o_idx[2 * origin + 1];
    double o_idx_start_weight = in.o_w[2 * origin];
    double o_idx_end_weight = in.o_w[2 * origin + 1];

    /* Unconditional seed writes (start then end; same-node last-wins),
     * each touch recorded exactly once in this epoch. */
    if (s.stamp[o_idx_start] != e) {
        s.stamp[o_idx_start] = e;
        s.touched[touched_count++] = o_idx_start;
    }
    s.label[o_idx_start] = o_idx_start_weight;
    if (s.stamp[o_idx_end] != e) {
        s.stamp[o_idx_end] = e;
        s.touched[touched_count++] = o_idx_end;
    }
    s.label[o_idx_end] = o_idx_end_weight;

    /* Original dummy heap init/pop and end-then-start strict < push
     * gates (NaN weights cannot occur: the admission gate refuses
     * that domain upstream -- it is refused, not repaired). */
    s.heap.clear();
    heap_push(s.heap, o_idx_start_weight, o_idx_start);
    (void)heap_pop(s.heap);

    if (o_idx_end_weight < in.cutoff) {
        heap_push(s.heap, o_idx_end_weight, o_idx_end);
    }
    if (o_idx_start_weight < in.cutoff) {
        heap_push(s.heap, o_idx_start_weight, o_idx_start);
    }

    while (!s.heap.empty()) {
        std::pair<double, int64_t> wn = heap_pop(s.heap);
        double weight = wn.first;
        int64_t node = wn.second;

        int64_t p0 = in.adj_ptr[node];
        int64_t p1 = in.adj_ptr[node + 1];

        /* Phase one: eligibility against the pre-row logical label
         * snapshot (projection reads; no label writes), collected in
         * ascending row order. */
        int64_t n_eligible = 0;
        for (int64_t i = 0; i < p1 - p0; ++i) {
            double nb_w = in.adj_w[p0 + i] + weight;
            int64_t nb_n = in.adj_vec[p0 + i];
            double lbl = (s.stamp[nb_n] == e) ? s.label[nb_n] : b;
            if (nb_w <= in.cutoff && nb_w < lbl) {
                s.row_off[n_eligible] = p0 + i;
                s.row_cost[n_eligible] = nb_w;
                ++n_eligible;
            }
        }

        /* Phase two: assignments and pushes from the stored pairs, in
         * ascending row order, gated by the flag column and the
         * leaf-node degree gate (> 1). */
        for (int64_t j = 0; j < n_eligible; ++j) {
            int64_t off = s.row_off[j];
            double w = s.row_cost[j];
            int64_t nb_n = in.adj_vec[off];
            if (s.stamp[nb_n] != e) {
                s.stamp[nb_n] = e;
                s.touched[touched_count++] = nb_n;
            }
            s.label[nb_n] = w;
            if (in.net[off]) {
                if (in.adj_ptr[nb_n + 1] - in.adj_ptr[nb_n] > 1) {
                    heap_push(s.heap, w, nb_n);
                }
            }
        }
    }

    /* Reverse-index candidate discovery over the PER-EPOCH touched
     * set, deduplicated by destination mark. */
    int64_t cand_count = 0;
    for (int64_t t = 0; t < touched_count; ++t) {
        int64_t tv = s.touched[t];
        for (int64_t k = in.rev_ptr[tv]; k < in.rev_ptr[tv + 1]; ++k) {
            int64_t d = in.rev_vec[k];
            if (s.mark[d] != e) {
                s.mark[d] = e;
                s.cand_ids[cand_count++] = d;
            }
        }
    }

    *cand_count_out = cand_count;
    *touched_count_out = touched_count;
}

/* Emit one origin's candidates in ascending ORIGINAL destination-id
 * order with build_trace's exact distance plan.  The numba kernel
 * argsorts the unique ids, writes cand_dist[j] at sorted position j,
 * then copies it to the tile row at the same position; emitting
 * directly at the sorted position produces identical bits. */
void emit_origin(const una_n00_input &in, LaneScratch &s, double b,
                 int64_t e, int64_t cand_count) {
    if (cand_count <= 0) {
        return;
    }
    /* Ascending id; ids are unique, so the sequence is total and any
     * correct ascending sort equals the numba argsort permutation. */
    std::sort(s.cand_ids.begin(), s.cand_ids.begin() + cand_count);
    for (int64_t j = 0; j < cand_count; ++j) {
        int64_t d = s.cand_ids[j];
        int64_t t0 = in.d_idx[2 * d];
        int64_t t1 = in.d_idx[2 * d + 1];
        double lw0 = (s.stamp[t0] == e) ? s.label[t0] : b;
        double lw1 = (s.stamp[t1] == e) ? s.label[t1] : b;
        double ds = lw0 + in.d_w[2 * d];
        double de = lw1 + in.d_w[2 * d + 1];
        s.out_ids.push_back(d);
        s.out_dist.push_back(py_min(ds, de));
    }
}

struct WorkerCtx {
    const una_n00_input *in;
    StreamState *st;
    LaneScratch scratch;
    int64_t *counts;     /* [O] shared, written by the owner lane */
    int64_t *origin_off; /* [O] shared, written by the owner lane */
    int32_t *origin_lane;/* [O] shared, written by the owner lane */
    int64_t lane_touched;
    int64_t lane_cand;
    int32_t lane;
};

void lane_worker(WorkerCtx *ctx) {
    const una_n00_input &in = *ctx->in;
    StreamState &st = *ctx->st;
    LaneScratch &s = ctx->scratch;
    const double b = 1.0 + in.cutoff;

    for (int64_t tile_start = 0; tile_start < in.O;
         tile_start += in.tile_origins) {
        if (st.stop.load(std::memory_order_acquire)) {
            break;
        }
        int64_t tile_len =
            in.tile_origins < in.O - tile_start ? in.tile_origins
                                                : in.O - tile_start;
        /* Per-tile contributions: the stream totals below fetch_add
         * these once per tile, so they must restart at each tile. */
        ctx->lane_touched = 0;
        ctx->lane_cand = 0;
        int64_t epoch = 0;

        /* Tile-start clearing: stale epoch stamps/marks from earlier
         * tiles must never alias this tile's epoch. */
        for (int64_t v = 0; v < in.N; ++v) {
            s.stamp[v] = 0;
        }
        for (int64_t d = 0; d < in.D; ++d) {
            s.mark[d] = 0;
        }

        for (int64_t t = ctx->lane; t < tile_len; t += in.lanes) {
            int64_t origin = tile_start + t;
            /* Checked epoch counter: clear stamps and destination
             * marks BEFORE rollover; epochs never used as float. */
            if (epoch >= in.epoch_rollover) {
                for (int64_t v = 0; v < in.N; ++v) {
                    s.stamp[v] = 0;
                }
                for (int64_t d = 0; d < in.D; ++d) {
                    s.mark[d] = 0;
                }
                epoch = 0;
            }
            epoch += 1;
            const int64_t e = epoch;

            int64_t cc = 0;
            int64_t tc = 0;
            run_origin(in, s, b, e, origin, &cc, &tc);
            ctx->counts[origin] = cc;
            ctx->origin_lane[origin] = ctx->lane;
            if (cc > 0) {
                ctx->origin_off[origin] = (int64_t)s.out_ids.size();
                emit_origin(in, s, b, e, cc);
            }
            ctx->lane_cand += cc;
            ctx->lane_touched += tc;
        }

        st.cand_total.fetch_add(ctx->lane_cand,
                                std::memory_order_acq_rel);
        st.touched_total.fetch_add(ctx->lane_touched,
                                   std::memory_order_acq_rel);
        {
            int64_t peak = s.arena_bytes() +
                           (in.O * 12 + (int64_t)st.lanes * 16);
            int64_t cur = st.scratch_peak.load(
                std::memory_order_acquire);
            while (peak > cur &&
                   !st.scratch_peak.compare_exchange_weak(
                       cur, peak, std::memory_order_acq_rel)) {
            }
        }

        /* Barrier: the last arriver runs the per-tile bookkeeping
         * with build_trace's exact budget formula and boundary. */
        int64_t tile_index = tile_start / in.tile_origins;
        int64_t prev =
            st.arrivals.fetch_add(1, std::memory_order_acq_rel);
        if (prev + 1 == (tile_index + 1) * (int64_t)st.lanes) {
            st.tiles_done.fetch_add(1, std::memory_order_acq_rel);
            if (16 * st.cand_total.load(std::memory_order_acquire) >
                in.budget_cand_bytes) {
                /* build_trace: budget_hit = start + tile_len -- the
                 * completed triggering tile counts, and the stop is a
                 * recompute/streaming selection, never a truncated
                 * success. */
                st.budget_hit.store(tile_start + tile_len,
                                    std::memory_order_release);
                st.stop.store(1, std::memory_order_release);
            } else if (st.cancel != nullptr && *st.cancel) {
                st.stop.store(2, std::memory_order_release);
            }
            st.generation.fetch_add(1, std::memory_order_release);
        } else {
            const int64_t my_gen = tile_index;
            while (st.generation.load(std::memory_order_acquire) ==
                       my_gen &&
                   st.stop.load(std::memory_order_acquire) == 0) {
                /* spin; lanes are few and tile work dominates */
            }
        }
    }
}

int64_t at_least_one(int64_t v) { return v > 0 ? v : 1; }

} /* namespace */

extern "C" {

void una_n00_output_free(struct una_n00_output *out) {
    if (out == nullptr) {
        return;
    }
    std::free(out->out_ids);
    std::free(out->out_dist);
    std::free(out->counts);
    std::free(out->lane_counters);
    std::memset(out, 0, sizeof(*out));
}

int una_n00_run_stream(const struct una_n00_input *in,
                       struct una_n00_output *out,
                       const volatile int *cancel) {
    if (in == nullptr || out == nullptr) {
        return UNA_N00_ERR_ARG;
    }
    /* Checked-interface re-validation of the invariants the binding
     * asserts (defense in depth; cheap). */
    if (in->O < 0 || in->N < 0 || in->D < 0 || in->E < 0 ||
        in->lanes < 1 || in->tile_origins < 1 || in->max_degree < 0) {
        return UNA_N00_ERR_ARG;
    }
    /* The budget check multiplies 16 * candidates; refuse frames where
     * that product could overflow int64 (absurd sizes anyway). */
    if (in->O != 0 &&
        in->D > (((int64_t)1 << 59) / in->O)) {
        return UNA_N00_ERR_ARG;
    }
    std::memset(out, 0, sizeof(*out));

    StreamState st;
    st.lanes = in->lanes;
    st.cancel = cancel;

    const int64_t O = in->O;
    int64_t *counts = (int64_t *)std::calloc(
        (size_t)at_least_one(O), sizeof(int64_t));
    int64_t *origin_off = (int64_t *)std::calloc(
        (size_t)at_least_one(O), sizeof(int64_t));
    int32_t *origin_lane = (int32_t *)std::calloc(
        (size_t)at_least_one(O), sizeof(int32_t));
    int64_t *lane_counters = (int64_t *)std::calloc(
        (size_t)in->lanes * 2, sizeof(int64_t));
    if (!counts || !origin_off || !origin_lane || !lane_counters) {
        std::free(counts);
        std::free(origin_off);
        std::free(origin_lane);
        std::free(lane_counters);
        return UNA_N00_ERR_ALLOC;
    }

    std::vector<WorkerCtx> ctxs((size_t)in->lanes);
    const int64_t deg = in->max_degree > 0 ? in->max_degree : 1;
    try {
        for (int32_t l = 0; l < in->lanes; ++l) {
            WorkerCtx &c = ctxs[(size_t)l];
            c.in = in;
            c.st = &st;
            c.counts = counts;
            c.origin_off = origin_off;
            c.origin_lane = origin_lane;
            c.lane = l;
            c.lane_touched = 0;
            c.lane_cand = 0;
            c.scratch.stamp.resize((size_t)in->N);
            c.scratch.label.resize((size_t)in->N);
            c.scratch.touched.resize((size_t)in->N);
            c.scratch.row_off.resize((size_t)deg);
            c.scratch.row_cost.resize((size_t)deg);
            c.scratch.cand_ids.resize((size_t)in->D);
            c.scratch.mark.resize((size_t)in->D);
            c.scratch.cand_dist.resize((size_t)in->D);
        }
    } catch (const std::bad_alloc &) {
        std::free(counts);
        std::free(origin_off);
        std::free(origin_lane);
        std::free(lane_counters);
        return UNA_N00_ERR_ALLOC;
    }

    if (O > 0) {
        std::vector<std::thread> threads;
        threads.reserve((size_t)in->lanes);
        for (int32_t l = 0; l < in->lanes; ++l) {
            try {
                threads.emplace_back(lane_worker, &ctxs[(size_t)l]);
            } catch (const std::system_error &) {
                /* Spawn failure: stop everything.  Waiters also watch
                 * st.stop, so no lane can block on a barrier that can
                 * no longer complete. */
                st.stop.store(2, std::memory_order_release);
                break;
            }
        }
        for (auto &th : threads) {
            th.join();
        }
    }

    /* Per-lane counters (same layout as the numba lane_counters). */
    for (int32_t l = 0; l < in->lanes; ++l) {
        lane_counters[2 * (size_t)l] = ctxs[(size_t)l].lane_touched;
        lane_counters[2 * (size_t)l + 1] = ctxs[(size_t)l].lane_cand;
    }

    /* Assemble the packed origin-major stream. */
    const int64_t nnz = st.cand_total.load(std::memory_order_acquire);
    int64_t *out_ids = (int64_t *)std::malloc(
        (size_t)at_least_one(nnz) * sizeof(int64_t));
    double *out_dist = (double *)std::malloc(
        (size_t)at_least_one(nnz) * sizeof(double));
    if (!out_ids || !out_dist) {
        std::free(out_ids);
        std::free(out_dist);
        std::free(counts);
        std::free(origin_off);
        std::free(origin_lane);
        std::free(lane_counters);
        return UNA_N00_ERR_ALLOC;
    }
    int64_t pos = 0;
    for (int64_t o = 0; o < O; ++o) {
        const int64_t c = counts[o];
        if (c > 0) {
            const int32_t lane = origin_lane[o];
            LaneScratch &s = ctxs[(size_t)lane].scratch;
            const int64_t off = origin_off[o];
            std::memcpy(out_ids + pos,
                        s.out_ids.data() + (size_t)off,
                        (size_t)c * sizeof(int64_t));
            std::memcpy(out_dist + pos,
                        s.out_dist.data() + (size_t)off,
                        (size_t)c * sizeof(double));
            pos += c;
        }
    }

    out->out_ids = out_ids;
    out->out_dist = out_dist;
    out->nnz = nnz;
    out->counts = counts;
    out->lane_counters = lane_counters;
    out->cand_total = nnz;
    out->touched_total =
        st.touched_total.load(std::memory_order_acquire);
    out->tiles_done = st.tiles_done.load(std::memory_order_acquire);
    out->budget_hit = st.budget_hit.load(std::memory_order_acquire);
    out->scratch_peak_bytes =
        st.scratch_peak.load(std::memory_order_acquire) +
        (int64_t)sizeof(int64_t) * at_least_one(nnz) +
        (int64_t)sizeof(double) * at_least_one(nnz);
    out->lanes_written = in->lanes;
    const int stop = st.stop.load(std::memory_order_acquire);
    /* Budget stop is a normal outcome (route trace_budget_exceeded);
     * only cancel is a status.  Internal scratch is freed here. */
    std::free(origin_off);
    std::free(origin_lane);
    out->status = (stop == 2) ? UNA_N00_CANCELED : UNA_N00_OK;
    return out->status;
}

} /* extern "C" */
