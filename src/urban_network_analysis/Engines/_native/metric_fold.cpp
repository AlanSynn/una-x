// N00 native metric fold (una_10x): strict-IEEE C++17 port of the
// per-origin reach/gravity/knn fold for the trace-replay domain
// (the fold: src/urban_network_analysis/Engines/Accessibility.py
// reach_gravity_knn_access; the packed layout: _trace_access_scratch
// _replay_rows_compiled Phase B + Phase C).
//
// Laws honored (dossier 08 / PLATFORM_CONTINUITY):
// - no -ffast-math, no FP contraction (-ffp-contract=off): every
//   numpy/numba op is reproduced in its exact order; sums are strict
//   left-to-right over the filtered arrays. Divergence vs the numba
//   lowering is a REFUSAL recorded by the bit battery - never a
//   tolerance.
// - immutable inputs (const pointers), caller-owned outputs AND
//   scratch (bounded: nnz-sized, allocated once by the bridge),
//   int64/f64 ABI only.
// - exceptions are mapped in the Python bridge BEFORE the call;
//   nothing is thrown across the ABI - failures are return codes.
// - the knn argsort tie order must match numba's np.argsort; the
//   battery decides - on a tie-order mismatch the bridge refuses the
//   duplicate-distance domain (loud, exact fallback to the numba
//   fold).
//
// Build (recorded verbatim in native_implementation.json):
//   g++ -std=c++17 -O2 -fPIC -shared -fno-fast-math -ffp-contract=off
//       -o libuna_metric_fold.so metric_fold.cpp

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <algorithm>

namespace {

struct DistIdx {
    double d;
    int64_t i;
};

// Numba's np.argsort(kind='quicksort') equivalent for the knn crop:
// ascending distance; tie order is the battery's business (the bridge
// verifies against numba and refuses the domain on divergence).
bool dist_idx_less(const DistIdx &a, const DistIdx &b) {
    if (a.d < b.d) {
        return true;
    }
    if (b.d < a.d) {
        return false;
    }
    return a.i < b.i;
}

}  // namespace

// Whole-structure replay fold over a packed CSR of ascending slices
// (the trace layout). Weights are gathered as d_weights[cand_ids] in
// slice order, exactly as the numba driver's fresh per-origin copies
// do. Returns 0 on success; 1 structural; 2 slice-order corruption;
// 3 NaN/payload refusal is NOT raised here (NaN distances simply fall
// out of the filter, as np.where(d <= cutoff) does).
extern "C" int una_replay_fold(
    const int64_t *indptr, std::size_t origin_count,
    const double *dist_all, std::size_t nnz,
    const int64_t *ids_all,
    const double *d_weights, std::size_t node_count,
    double cutoff, double beta, double plateau, double mid, double gr,
    const double *knn_weights, std::size_t knn_weights_len,
    int64_t knn_decay,  // 0 exponential, 1 logistic, 2 flat/other
    double *reach, double *ge, double *gl, double *kn,
    int64_t *kept_per_origin,   // len >= origin_count (out)
    DistIdx *scratch_pairs,      // len >= nnz (caller-owned scratch)
    double *scratch_d, double *scratch_w) {
    // Phase B: strictly ascending ids inside every slice, in-range;
    // no output is touched until the WHOLE structure has scanned.
    for (std::size_t o = 0; o < origin_count; ++o) {
        const int64_t lo = indptr[o];
        const int64_t hi = indptr[o + 1];
        if (hi < lo || lo < 0) {
            return 1;
        }
        if (static_cast<std::size_t>(hi) > nnz) {
            return 1;
        }
        for (int64_t j = lo; j < hi; ++j) {
            if (j > lo && ids_all[j] <= ids_all[j - 1]) {
                return 2;
            }
            if (ids_all[j] < 0
                || static_cast<std::size_t>(ids_all[j]) >= node_count) {
                return 1;
            }
        }
    }
    // Phase C: strict scalar fold per origin, arrays in order.
    for (std::size_t o = 0; o < origin_count; ++o) {
        const std::size_t lo = static_cast<std::size_t>(indptr[o]);
        const std::size_t hi = static_cast<std::size_t>(indptr[o + 1]);
        double s_reach = 0.0, s_ge = 0.0, s_gl = 0.0;
        std::size_t kept = 0;
        for (std::size_t j = lo; j < hi; ++j) {
            const double d = dist_all[j];
            if (!(d <= cutoff)) {
                continue;
            }
            const double w = d_weights[ids_all[j]];
            s_reach += w;
            const double pos = d - plateau;
            const double clipped = pos > 0.0 ? pos : 0.0;
            s_ge += w / std::exp(beta * clipped);
            s_gl += w * (1.0 - 1.0
                         / (1.0 + std::exp(-gr * (pos - mid))));
            scratch_pairs[kept].d = d;
            // The pair index is the FILTERED position (scratch_d/w are
            // indexed by kept), never the slice index: the knn crop
            // reads back through the same filtered arrays.
            scratch_pairs[kept].i = static_cast<int64_t>(kept);
            scratch_d[kept] = d;
            scratch_w[kept] = w;
            ++kept;
        }
        reach[o] = s_reach;
        ge[o] = s_ge;
        gl[o] = s_gl;
        kept_per_origin[o] = static_cast<int64_t>(kept);

        const std::size_t k = std::min(kept, knn_weights_len);
        if (k == 0) {
            kn[o] = 0.0;
            continue;
        }
        std::sort(scratch_pairs, scratch_pairs + kept, dist_idx_less);
        // knn = min(n_distances.shape[0], knn_weights.shape[0]); the
        // crop takes the FIRST k of the sorted arrays.
        double s_kn = 0.0;
        for (std::size_t t = 0; t < k; ++t) {
            const int64_t si = scratch_pairs[t].i;
            const double d = scratch_d[si];
            const double w = scratch_w[si];
            const double coeff = knn_weights[t];
            if (knn_decay == 0) {
                const double pos = d - plateau;
                const double clipped = pos > 0.0 ? pos : 0.0;
                s_kn += (coeff * w) * std::exp(-beta * clipped);
            } else if (knn_decay == 1) {
                const double pos = d - plateau - mid;
                s_kn += (coeff * w)
                        * (1.0 - 1.0 / (1.0 + std::exp(-gr * pos)));
            } else {
                s_kn += coeff * w;
            }
        }
        kn[o] = s_kn;
    }
    return 0;
}
