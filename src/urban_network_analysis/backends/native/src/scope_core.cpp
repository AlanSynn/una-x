// NATIVE_CORE implementation: see scope_core.hpp for the contract.
//
// Worker/scratch law (SPEC.md section 6): per-worker scratch only
// (labels, eligible pairs, heap); outputs are origin-partitioned (each
// origin writes exactly its own (o, :) row); there is no ordered
// cross-origin reduction, so worker count and stripe assignment cannot
// move a bit.  Cancellation is polled at origin boundaries; a cancelled
// run reports STATUS_ERR_CANCELLED plus the first incomplete origin and
// never publishes through the binding (the Python route assembles the
// public KernelOutputs only on complete runs).
#include "scope_core.hpp"

#include <algorithm>
#include <limits>
#include <new>
#include <stdexcept>
#include <system_error>
#include <thread>

namespace una_native {
namespace {

constexpr double DOUBLE_INF = std::numeric_limits<double>::infinity();
constexpr i64 NO_ORIGIN = INT64_MAX;

void cas_min(std::atomic<i64>& slot, i64 value) {
    i64 cur = slot.load(std::memory_order_relaxed);
    while (value < cur &&
           !slot.compare_exchange_weak(cur, value, std::memory_order_relaxed)) {
    }
}

// Per-worker scratch; constructed once per driver call, reused per origin.
struct WorkerBuf {
    std::vector<double> labels;
    std::vector<i64> eligible_offset;
    std::vector<double> eligible_weight;
    ItemHeap heap;
    Engagement eng;
};

}  // namespace

// ---------------------------------------------------------------------------
// Read-safety backstop (dossier 08: no out-of-bounds reads).  The legacy
// admission mirror (kernels.scope_access.scope_route_admits) already
// routed value-domain decisions; the core re-enforces everything it
// itself relies on, so a concurrent mutation of the shared read-only
// inputs between mirror and core is reported as STATUS_ERR_ABI (internal
// mismatch fails loudly per SPEC.md section 7) instead of reading out of
// bounds.  For corrected_v1 the same scan IS the typed contract (first
// failure wins, in reference/graph.py order).
//
// Recomputes max_degree from the pointer array; `*max_degree_out` receives
// it.  Returns STATUS_OK or STATUS_MALFORMED_GRAPH / STATUS_NONFINITE_COST /
// STATUS_NEGATIVE_COST / STATUS_OOB_TERMINAL / STATUS_ERR_ABI.
// ---------------------------------------------------------------------------
std::int32_t validate_search_inputs(const SearchInputs& in, bool legacy,
                                    i64* max_degree_out) {
    if (in.node_count < 0 || in.edge_count < 0 || in.o_count < 0 ||
        in.d_count < 0)
        return STATUS_ERR_ABI;

    // -- CSR structure (graph.py order: ptr[0], monotone, ptr[-1]==E) --
    if (in.ptr[0] != 0) return STATUS_MALFORMED_GRAPH;
    i64 max_degree = 0;
    for (i64 v = 0; v < in.node_count; ++v) {
        const i64 degree = in.ptr[v + 1] - in.ptr[v];
        if (degree < 0) return STATUS_MALFORMED_GRAPH;
        if (degree > max_degree) max_degree = degree;
    }
    if (in.ptr[in.node_count] != in.edge_count) return STATUS_MALFORMED_GRAPH;
    *max_degree_out = max_degree;

    // -- neighbor range / cost domain (one pass, ascending offset order) --
    for (i64 j = 0; j < in.edge_count; ++j) {
        const i64 neighbor = in.vec[j];
        if (neighbor < 0 || neighbor >= in.node_count)
            return STATUS_MALFORMED_GRAPH;
        const double cost = in.wts[j];
        if (cost != cost || cost == DOUBLE_INF) return STATUS_NONFINITE_COST;
        if (cost < 0.0) return STATUS_NEGATIVE_COST;
    }

    if (!legacy) {
        // -- cutoff (graph.py validate_cutoff) --
        const double c = in.cutoff;
        if (c != c || c == DOUBLE_INF || c == -DOUBLE_INF)
            return STATUS_NONFINITE_COST;
        if (c < 0.0) return STATUS_NEGATIVE_COST;

        // -- origin terminals (validate_terminals, what="origin") --
        for (i64 o = 0; o < in.o_count; ++o) {
            for (i64 k = 0; k < 2; ++k) {
                const i64 t = in.o_idx[2 * o + k];
                if (t < 0 || t >= in.node_count) return STATUS_OOB_TERMINAL;
                const double w = in.o_wts[2 * o + k];
                if (w != w || w == DOUBLE_INF || w == -DOUBLE_INF)
                    return STATUS_NONFINITE_COST;
                if (w < 0.0) return STATUS_NEGATIVE_COST;
            }
        }
        // -- destination terminals (validate_terminals, what="destination") --
        for (i64 j = 0; j < in.d_count; ++j) {
            for (i64 k = 0; k < 2; ++k) {
                const i64 t = in.d_idx[2 * j + k];
                if (t < 0 || t >= in.node_count) return STATUS_OOB_TERMINAL;
                const double w = in.d_wts[2 * j + k];
                if (w != w || w == DOUBLE_INF || w == -DOUBLE_INF)
                    return STATUS_NONFINITE_COST;
                if (w < 0.0) return STATUS_NEGATIVE_COST;
            }
        }
    } else {
        // Legacy: value-domain admission belongs to the mirror; the core
        // re-checks the index domains it reads (loud mismatch on race).
        for (i64 o = 0; o < in.o_count; ++o) {
            for (i64 k = 0; k < 2; ++k) {
                const i64 t = in.o_idx[2 * o + k];
                if (t < 0 || t >= in.node_count) return STATUS_ERR_ABI;
            }
        }
        // The destination read bound is route-dependent and checked by
        // the driver: A3 route [0, V); A1 route may read the sentinel
        // tail [V, V+d) exactly as the engine's A1 label domain allows.
    }
    return STATUS_OK;
}

// ---------------------------------------------------------------------------
// Legacy schedule (SPEC.md section 3)
// ---------------------------------------------------------------------------
void scope_search_legacy_one(const SearchInputs& in, i64 origin,
                             std::vector<double>& labels,
                             std::vector<i64>& eligible_offset,
                             std::vector<double>& eligible_weight,
                             ItemHeap& heap, Engagement& eng) {
    std::fill(labels.begin(), labels.end(), legacy_sentinel(in.cutoff));

    const i64 start = in.o_idx[2 * origin];
    const i64 end = in.o_idx[2 * origin + 1];
    const double w_start = in.o_wts[2 * origin];
    const double w_end = in.o_wts[2 * origin + 1];

    // Step 1: unconditional overwrite seed writes, start then end (order
    // observable when start == end: the end weight wins).
    labels[start] = w_start;
    labels[end] = w_end;

    // Steps 2-3: single-entry heap, then the discarded initial pop (the
    // engine's prologue; reproduced verbatim).
    heap.clear();
    heap.push(HeapItem{w_start, start}, legacy_item_lt);
    (void)heap.pop(legacy_item_lt);
    ++eng.heap_pops;

    // Step 4: seed push gates, end then start, STRICT < under the fast
    // law (NaN seeds pass; characterized).
    if (fast_lt(w_end, in.cutoff)) {
        heap.push(HeapItem{w_end, end}, legacy_item_lt);
        ++eng.heap_pushes;
    }
    if (fast_lt(w_start, in.cutoff)) {
        heap.push(HeapItem{w_start, start}, legacy_item_lt);
        ++eng.heap_pushes;
    }

    // Steps 5-6: two-phase row scans.
    while (!heap.empty()) {
        const HeapItem it = heap.pop(legacy_item_lt);
        ++eng.heap_pops;
        ++eng.rows_scanned;

        const i64 row_begin = in.ptr[it.node];
        const i64 row_end = in.ptr[it.node + 1];

        // Phase 1: eligibility against the PRE-ROW label snapshot; no
        // label writes.  fast_le(c, cutoff) && fast_lt(c, label).
        i64 n_eligible = 0;
        for (i64 i = row_begin; i < row_end; ++i) {
            const double candidate = in.wts[i] + it.weight;
            const i64 neighbor = in.vec[i];
            if (fast_le(candidate, in.cutoff) &&
                fast_lt(candidate, labels[neighbor])) {
                eligible_offset[n_eligible] = i;
                eligible_weight[n_eligible] = candidate;
                ++n_eligible;
            }
        }

        // Phase 2: assign + gated push, in stored row order.  Push gate:
        // flag column AND neighbor row degree > 1 (plain integers).
        for (i64 j = 0; j < n_eligible; ++j) {
            const i64 offset = eligible_offset[j];
            const double candidate = eligible_weight[j];
            const i64 neighbor = in.vec[offset];
            labels[neighbor] = candidate;
            if (in.flag[offset] != 0) {
                if (in.ptr[neighbor + 1] - in.ptr[neighbor] > 1) {
                    heap.push(HeapItem{candidate, neighbor}, legacy_item_lt);
                    ++eng.heap_pushes;
                }
            }
        }
        eng.eligible_collected += static_cast<u64>(n_eligible);
    }
}

// ---------------------------------------------------------------------------
// corrected_v1 schedule (reference/search.py::corrected_scope_labels)
// ---------------------------------------------------------------------------
void scope_search_corrected_one(const SearchInputs& in, i64 origin,
                                std::vector<double>& labels, ItemHeap& heap,
                                Engagement& eng) {
    // np.full(node_count, +inf).
    std::fill(labels.begin(), labels.end(), DOUBLE_INF);

    const i64 start = in.o_idx[2 * origin];
    const i64 end = in.o_idx[2 * origin + 1];
    const double w_start = in.o_wts[2 * origin];
    const double w_end = in.o_wts[2 * origin + 1];

    // Declared seed combine: start seeds; end replaces on strict
    // improvement only (BUG-COINCIDENT-SEEDS corrected rule; on an exact
    // tie the incumbent start value survives).
    labels[start] = w_start;
    if (w_end < labels[end]) labels[end] = w_end;

    // Heap seeding: both FINAL seed labels, closed radius, in the
    // declared (start, end) order.
    heap.clear();
    if (labels[start] <= in.cutoff) {
        heap.push(HeapItem{labels[start], start}, corrected_item_lt);
        ++eng.heap_pushes;
    }
    if (labels[end] <= in.cutoff) {
        heap.push(HeapItem{labels[end], end}, corrected_item_lt);
        ++eng.heap_pushes;
    }

    while (!heap.empty()) {
        const HeapItem it = heap.pop(corrected_item_lt);
        ++eng.heap_pops;
        ++eng.rows_scanned;
        if (it.weight > labels[it.node]) continue;  // stale entry (declared)

        const i64 row_begin = in.ptr[it.node];
        const i64 row_end = in.ptr[it.node + 1];
        for (i64 i = row_begin; i < row_end; ++i) {
            const double candidate = it.weight + in.wts[i];  // one RN add
            const i64 neighbor = in.vec[i];
            if (candidate <= in.cutoff && candidate < labels[neighbor]) {
                labels[neighbor] = candidate;
                heap.push(HeapItem{candidate, neighbor}, corrected_item_lt);
                ++eng.heap_pushes;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Destination adjust (SPEC.md section 4 / corrected declared order)
// ---------------------------------------------------------------------------
void adjust_destinations(const i64* d_idx, const double* d_wts, i64 d_count,
                         const double* labels, double* out_row) {
    for (i64 j = 0; j < d_count; ++j) {
        const double dist_to_start = labels[d_idx[2 * j]] + d_wts[2 * j];
        const double dist_to_end = labels[d_idx[2 * j + 1]] + d_wts[2 * j + 1];
        out_row[j] = adjust_min(dist_to_start, dist_to_end);
    }
}

// ---------------------------------------------------------------------------
// Integrated drivers
// ---------------------------------------------------------------------------
namespace {

template <typename SearchOne>
std::int32_t run_origins(const SearchInputs& in, std::int32_t threads,
                         const CancelFlag* cancel, double* out, i64 domain,
                         Engagement& eng, SearchOne search_one) {
    if (in.o_count == 0) return STATUS_OK;
    if (in.d_count != 0 && in.o_count > INT64_MAX / in.d_count)
        return STATUS_ERR_OVERFLOW;
    if (cancel != nullptr && cancel->cancelled()) {
        eng.first_incomplete_origin = 0;
        return STATUS_ERR_CANCELLED;
    }

    const i64 worker_count =
        std::min<i64>(threads < 1 ? 1 : threads, in.o_count);

    std::vector<WorkerBuf> bufs;
    try {
        bufs.resize(static_cast<std::size_t>(worker_count));
        const i64 scratch_len = std::max<i64>(in.max_degree, 1);
        for (auto& buf : bufs) {
            buf.labels.resize(static_cast<std::size_t>(domain));
            buf.eligible_offset.resize(static_cast<std::size_t>(scratch_len));
            buf.eligible_weight.resize(static_cast<std::size_t>(scratch_len));
        }
    } catch (const std::bad_alloc&) {
        return STATUS_ERR_MEMORY;
    }

    std::atomic<i64> first_incomplete{NO_ORIGIN};
    std::atomic<std::int32_t> worker_status{STATUS_OK};

    auto worker_body = [&](i64 wid) {
        WorkerBuf& buf = bufs[static_cast<std::size_t>(wid)];
        for (i64 origin = wid; origin < in.o_count; origin += worker_count) {
            if (cancel != nullptr && cancel->cancelled()) {
                cas_min(first_incomplete, origin);
                return;
            }
            try {
                search_one(in, origin, buf);
            } catch (const std::bad_alloc&) {
                // Unbounded heap growth (the characterized legacy
                // NaN-seed oscillation can pump the heap indefinitely —
                // the reference never returns on that input either).
                worker_status.store(STATUS_ERR_MEMORY,
                                    std::memory_order_relaxed);
                cas_min(first_incomplete, origin);
                return;
            } catch (...) {
                worker_status.store(STATUS_ERR_INTERNAL,
                                    std::memory_order_relaxed);
                cas_min(first_incomplete, origin);
                return;
            }
            adjust_destinations(in.d_idx, in.d_wts, in.d_count,
                                buf.labels.data(), out + origin * in.d_count);
        }
    };

    std::vector<std::thread> pool;
    pool.reserve(static_cast<std::size_t>(worker_count));
    for (i64 wid = 0; wid < worker_count; ++wid) {
        try {
            pool.emplace_back(worker_body, wid);
        } catch (const std::system_error&) {
            // Thread-spawn failure: run this stripe on the calling thread
            // instead of losing the origins (never a silent wrong answer).
            worker_body(wid);
        }
    }
    for (auto& t : pool) t.join();

    for (auto& buf : bufs) {
        eng.heap_pushes += buf.eng.heap_pushes;
        eng.heap_pops += buf.eng.heap_pops;
        eng.rows_scanned += buf.eng.rows_scanned;
        eng.eligible_collected += buf.eng.eligible_collected;
    }
    eng.first_incomplete_origin =
        first_incomplete.load(std::memory_order_relaxed);

    // join() ordered the workers' stores before these loads.
    const std::int32_t err = worker_status.load(std::memory_order_relaxed);
    if (err != STATUS_OK) return err;
    if (eng.first_incomplete_origin != NO_ORIGIN) return STATUS_ERR_CANCELLED;
    return STATUS_OK;
}

}  // namespace

i64 check_domain_size(i64 node_count, i64 tail) {
    if (node_count < 0 || tail < 0) throw std::overflow_error("negative size");
    if (tail > INT64_MAX - node_count)
        throw std::overflow_error("label domain overflows int64");
    return node_count + tail;
}

std::int32_t run_integrated_legacy(const SearchInputs& in, bool route_a3,
                                   std::int32_t threads,
                                   const CancelFlag* cancel, double* out,
                                   Engagement& eng) {
    // Read-safety backstop before any worker starts.
    i64 max_degree = 0;
    const std::int32_t vstatus =
        validate_search_inputs(in, /*legacy=*/true, &max_degree);
    if (vstatus != STATUS_OK) return vstatus;

    // Destination terminal read domain: [0, V) on the A3 route; [0, V+d)
    // on the A1 route (tail sentinels are readable there, matching the
    // engine's label domain).  The Python route refuses narrower (policy
    // fallback); the core enforces loudly.
    i64 d_bound = 0;
    try {
        d_bound = route_a3 ? in.node_count
                           : check_domain_size(in.node_count, in.d_count);
    } catch (const std::overflow_error&) {
        return STATUS_ERR_OVERFLOW;
    }
    for (i64 j = 0; j < in.d_count; ++j) {
        for (i64 k = 0; k < 2; ++k) {
            const i64 t = in.d_idx[2 * j + k];
            if (t < 0 || t >= d_bound) return STATUS_ERR_ABI;
        }
    }

    SearchInputs local = in;
    local.max_degree = max_degree;
    const i64 domain = route_a3 ? in.node_count : d_bound;
    return run_origins(local, threads, cancel, out, domain, eng,
                       [](const SearchInputs& inn, i64 origin, WorkerBuf& buf) {
                           scope_search_legacy_one(inn, origin, buf.labels,
                                                   buf.eligible_offset,
                                                   buf.eligible_weight,
                                                   buf.heap, buf.eng);
                       });
}

std::int32_t run_integrated_corrected(const SearchInputs& in,
                                      std::int32_t threads,
                                      const CancelFlag* cancel, double* out,
                                      Engagement& eng) {
    // Full typed validation in reference/graph.py order (first failure
    // wins) — this IS the corrected admission contract.
    i64 max_degree = 0;
    const std::int32_t vstatus =
        validate_search_inputs(in, /*legacy=*/false, &max_degree);
    if (vstatus != STATUS_OK) return vstatus;

    SearchInputs local = in;
    local.max_degree = max_degree;
    const i64 domain = in.node_count;  // corrected_v1: V domain, no tail
    return run_origins(local, threads, cancel, out, domain, eng,
                       [](const SearchInputs& inn, i64 origin, WorkerBuf& buf) {
                           scope_search_corrected_one(inn, origin, buf.labels,
                                                      buf.heap, buf.eng);
                       });
}

}  // namespace una_native
