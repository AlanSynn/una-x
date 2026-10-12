// NATIVE_CORE: typed C++17 core for the scope-access kernel family.
//
// Contract: src/urban_network_analysis/kernels/SPEC.md (frozen
// characterization, CPU_ALGORITHMS).  This core implements the TWO-PHASE
// SCOPE SEARCH + DESTINATION ADJUST for both semantic profiles:
//
// - una_legacy:  SPEC.md section 3 schedule under the characterized
//   fast-math comparison laws (fast_lt/fast_le below) and the NaN-first
//   heap tuple law; routes A3 (tail-free) and A1 (d_count tail).
// - corrected_v1: the corrected reference schedule
//   (urban_network_analysis/reference/search.py::corrected_scope_labels):
//   +inf labels, minimum seed combine, no push gate, stale-entry skips,
//   IEEE closed-radius eligibility, CPython heapq order (NaN impossible:
//   inputs are validated first, per reference/graph.py).
//
// The FOLD stage (reach/gravity/knn reductions) is deliberately NOT
// implemented here: the una_legacy fold's bits include LLVM fastmath
// reassociation of the compiled transcription (reductions.py) that no
// declared C++ operation sequence reproduces, and corrected_v1 retains
// platform-libm-via-numpy transcendentals (NUMERICS.md).  Dossier 08
// admits keeping a region on the reference; the Python route composes
// this core with the canonical per-profile fold references.  Recorded in
// backends/native/README.md.
//
// ABI law (dossier 08): fixed-width int64_t/double/uint8_t only (no C
// long); every comparison used by the legacy profile is spelled in its
// characterized two's-complement-free IEEE form; -ffp-contract=off and
// no -ffast-math at build time (see ../build.sh); allocation sizes are
// overflow-checked before allocation; std::bad_alloc is reported as a
// status code, never thrown across the binding; workers touch only raw
// pointers and their own scratch (no Python objects in released-GIL
// regions); cancellation is checked at origin boundaries only.
#ifndef UNA_NATIVE_SCOPE_CORE_HPP
#define UNA_NATIVE_SCOPE_CORE_HPP

#include <atomic>
#include <cstdint>
#include <vector>

namespace una_native {

// Fixed-width element types (dossier 08: never platform C long).
using i64 = std::int64_t;
using u64 = std::uint64_t;
using u8 = std::uint8_t;

// ABI identity of this core/binding pair; the Python loader refuses a
// compiled module whose abi_version() differs (stale-build skew).
constexpr std::int32_t ABI_VERSION = 1;

// Structured status codes translated to Python after GIL reacquire.
enum Status : std::int32_t {
    STATUS_OK = 0,
    STATUS_MALFORMED_GRAPH = 1,   // corrected: reference MalformedGraphError
    STATUS_NONFINITE_COST = 2,    // corrected: reference NonFiniteCostError
    STATUS_NEGATIVE_COST = 3,     // corrected: reference NegativeCostError
    STATUS_OOB_TERMINAL = 4,      // corrected: reference OutOfRangeTerminalError
    STATUS_ERR_ABI = 5,           // binding/layout enforcement failure (loud)
    STATUS_ERR_MEMORY = 6,        // allocation failure (bounded, never partial)
    STATUS_ERR_CANCELLED = 7,     // cancelled between origins; no partial writes
    STATUS_ERR_OVERFLOW = 8,      // size arithmetic would overflow
    STATUS_ERR_INTERNAL = 9,      // invariant violation inside the core
};

// One call's full input view.  Pointers are borrowed; the pybind11 layer
// holds the owning numpy arrays alive across the released-GIL region.
struct SearchInputs {
    const i64* ptr;             // (node_count + 1,) CSR pointer
    i64 node_count;
    const i64* vec;             // (edge_count,) neighbor ids
    i64 edge_count;
    const double* wts;          // (edge_count,) arc weights
    const u8* flag;             // (edge_count,) legacy push-gate column only
    const i64* o_idx;           // (o_count, 2) origin terminals
    i64 o_count;
    const double* o_wts;        // (o_count, 2) origin split weights
    const i64* d_idx;           // (d_count, 2) destination terminals
    const double* d_wts;        // (d_count, 2) destination split weights
    i64 d_count;
    double cutoff;
    i64 max_degree;             // from admission (largest pointer delta)
};

// Work engagement counters (SPEC.md section 6: every route reports them).
struct Engagement {
    u64 heap_pushes = 0;
    u64 heap_pops = 0;
    u64 rows_scanned = 0;       // adjacency rows processed (== expansions)
    u64 eligible_collected = 0; // snapshot-eligible pairs collected
    i64 first_incomplete_origin = INT64_MAX;  // INT64_MAX when all complete
};

// Cooperative cancellation: a plain atomic flag polled between origins.
// The pybind11 layer exposes it as a tiny class so a Python watcher
// thread can set it WITHOUT the workers ever touching Python objects.
class CancelFlag {
public:
    void cancel() { flag_.store(1, std::memory_order_release); }
    bool cancelled() const { return flag_.load(std::memory_order_acquire) != 0; }

private:
    std::atomic<std::int32_t> flag_{0};
};

// ---------------------------------------------------------------------------
// Characterized comparison laws (SPEC.md section 3; pinned by
// tests/cpu_kernels/test_schedule_pinning.py against the compiled engine).
// Compiled STRICTLY IEEE (build.sh passes -fno-fast-math
// -ffp-contract=off); the fast law is reproduced by the explicit
// complement forms, not by compiler canonicalization.
// ---------------------------------------------------------------------------
inline bool fast_lt(double a, double b) { return !(b <= a); }
inline bool fast_le(double a, double b) { return !(b < a); }

// Builtin-min bits of the legacy adjust (plan.accessibility.adjust.
// builtin_min.v1): first operand wins on equality AND on incomparability
// (min(NaN, x) = NaN, min(x, NaN) = x, signed-zero keeps the first
// operand).  The corrected declared order (reference/search.py
// _od_row_impl: keep the incumbent unless alt < best) is the SAME
// expression; on the corrected validated domain no NaN can occur.
inline double adjust_min(double a, double b) { return (b < a) ? b : a; }

// ---------------------------------------------------------------------------
// Binary heap: verbatim CPython/numba heapq transcription (the oracle's
// _heap_push/_heap_pop/_siftdown in kernels/scope_access.py, themselves
// pinned byte-exact against the engine).  Item order under `less`.
// ---------------------------------------------------------------------------
struct HeapItem {
    double weight;
    i64 node;
};

// SPEC.md section 3 tuple law: a NaN weight compares LESS than every
// entry; finite weights compare lexicographically (weight, node) with
// ascending node tie-break.
inline bool legacy_item_lt(const HeapItem& a, const HeapItem& b) {
    const double wa = a.weight;
    const double wb = b.weight;
    if (wa != wa) return true;
    if (wb != wb) return false;
    if (wa < wb) return true;
    if (wa == wb) return a.node < b.node;
    return false;
}

// corrected_v1 heap order: plain (distance, node) lexicographic tuples
// (reference/search.py uses heapq directly; NaN cannot reach the heap —
// validated inputs, +inf labels).
inline bool corrected_item_lt(const HeapItem& a, const HeapItem& b) {
    return a.weight < b.weight ||
           (a.weight == b.weight && a.node < b.node);
}

class ItemHeap {
public:
    bool empty() const { return h_.empty(); }
    void clear() { h_.clear(); }

    void push(HeapItem item, bool (*less)(const HeapItem&, const HeapItem&)) {
        // CPython heapq.heappush (verbatim algorithm).
        h_.push_back(item);
        i64 pos = static_cast<i64>(h_.size()) - 1;
        HeapItem new_item = h_[pos];
        while (pos > 0) {
            const i64 parent_pos = (pos - 1) >> 1;
            const HeapItem parent = h_[parent_pos];
            if (less(new_item, parent)) {
                h_[pos] = parent;
                pos = parent_pos;
                continue;
            }
            break;
        }
        h_[pos] = new_item;
    }

    HeapItem pop(bool (*less)(const HeapItem&, const HeapItem&)) {
        // CPython heapq.heappop (verbatim algorithm, including the
        // trailing _siftdown the CPython pop performs).
        HeapItem last = h_.back();
        h_.pop_back();
        if (h_.empty()) return last;
        const HeapItem return_item = h_[0];
        h_[0] = last;
        const i64 end_pos = static_cast<i64>(h_.size());
        const i64 start_pos = 0;
        i64 pos = 0;
        HeapItem new_item = h_[pos];
        i64 child_pos = 2 * pos + 1;
        while (child_pos < end_pos) {
            const i64 right_pos = child_pos + 1;
            if (right_pos < end_pos && !less(h_[child_pos], h_[right_pos]))
                child_pos = right_pos;
            const HeapItem child = h_[child_pos];
            if (less(new_item, child)) break;
            h_[pos] = child;
            pos = child_pos;
            child_pos = 2 * pos + 1;
        }
        h_[pos] = new_item;
        // _siftdown(h, start_pos, pos)
        new_item = h_[pos];
        while (pos > start_pos) {
            const i64 parent_pos = (pos - 1) >> 1;
            const HeapItem parent = h_[parent_pos];
            if (less(new_item, parent)) {
                h_[pos] = parent;
                pos = parent_pos;
                continue;
            }
            break;
        }
        h_[pos] = new_item;
        return return_item;
    }

private:
    std::vector<HeapItem> h_;
};

// ---------------------------------------------------------------------------
// Single-origin searches (the characterized schedules).
// ---------------------------------------------------------------------------

// Sentinel value of the legacy label init: bit-exact
// np.ones(domain, float64) + cutoff (one binary64 RN add; the same value
// fills every slot, so computing it once and filling is bit-identical).
inline double legacy_sentinel(double cutoff) { return 1.0 + cutoff; }

// SPEC.md section 3 one-origin scope search (routes A3 and A1; the A1
// destination tail [V, V+d) is allocated like the engine but never read
// on admitted inputs).  labels.size() must equal the route's domain.
void scope_search_legacy_one(const SearchInputs& in, i64 origin,
                             std::vector<double>& labels,
                             std::vector<i64>& eligible_offset,
                             std::vector<double>& eligible_weight,
                             ItemHeap& heap, Engagement& eng);

// corrected_v1 one-origin scope search
// (reference/search.py::corrected_scope_labels): +inf labels, minimum
// seed combine (end replaces only on strict improvement), closed-radius
// heap seeding of the FINAL seed labels, stale-entry skips, every
// reached node propagates (no flag/degree gate).  labels.size() must
// equal node_count.
void scope_search_corrected_one(const SearchInputs& in, i64 origin,
                                std::vector<double>& labels, ItemHeap& heap,
                                Engagement& eng);

// Destination adjust (SPEC.md section 4 / reference declared order).
// Writes d_count doubles into out_row from the label vector.
void adjust_destinations(const i64* d_idx, const double* d_wts, i64 d_count,
                         const double* labels, double* out_row);

// ---------------------------------------------------------------------------
// Integrated drivers (all origins; worker stripes; cancellation at
// origin boundaries; allocator-failure and overflow status codes).
// ---------------------------------------------------------------------------

// route_a3: true -> tail-free domain V; false -> A1 domain V + d_count.
// out: caller-allocated (o_count, d_count) row-major adjusted distances.
std::int32_t run_integrated_legacy(const SearchInputs& in, bool route_a3,
                                   std::int32_t threads, const CancelFlag* cancel,
                                   double* out, Engagement& eng);

// corrected_v1 (single search shape; V domain, no tail, no flag column).
std::int32_t run_integrated_corrected(const SearchInputs& in,
                                      std::int32_t threads,
                                      const CancelFlag* cancel, double* out,
                                      Engagement& eng);

// Overflow gate for the label domain (dossier 08: check sums/products
// before allocation).  Returns the domain size or throws
// std::overflow_error; exposed to Python for the unit gate without
// materializing giant arrays.
i64 check_domain_size(i64 node_count, i64 tail);

// Read-safety/typed validation, public for the binding's single-origin
// label paths (the integrated drivers run it internally).  Recomputes
// max_degree from the pointer array (the drivers use the recomputed
// value, so scratch sizing and row reads always agree).  Returns a
// STATUS_* code; STATUS_OK when admitted.
std::int32_t validate_search_inputs(const SearchInputs& in, bool legacy,
                                    i64* max_degree_out);

}  // namespace una_native

#endif  // UNA_NATIVE_SCOPE_CORE_HPP
