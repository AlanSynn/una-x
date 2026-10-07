// fold_kernels.cu — device kernels for the accessibility fold region (G10).
//
// Region law (campaigns/una_10x, gpu_region.json; dossier
// 08_native_gpu_regions.md):
//   * Bitwise parity per una_legacy is mandatory.  K0 is order-preserving
//     compaction and K1 is per-element arithmetic: both are association-free,
//     so their results are bit-exact by construction GIVEN IEEE-754 double
//     behavior.  No reduction lives on the device: K3 association trees are
//     certified per cell by the bit-validation gate (validate_device_bits.py)
//     and K2's numba tie-law ordering is a host-side sequential plan.
//   * DOUBLE PRECISION ONLY.  FP32 substitution is forbidden.  Compile flags
//     contract (mirrors native/build.py law):
//       -O3 -std=c++17 -prec-sqrt=true -prec-div=true -fmad=false
//     --use_fast_math (and any -ffast-math equivalent) is FORBIDDEN: device
//     exp() must be the correctly-rounded-ish libdevice double exp, and its
//     bit identity with the host numba/LLVM exp is still NOT assumed — it is
//     exactly what the per-cell gate proves before K1 output feeds anything.
//   * The closed-radius comparison `drow[i] <= cutoff` mirrors
//     np.where(d_distance <= cutoff): NaN compares False and is excluded,
//     matching the engine's unreachable-distance domain handling.
//   * Launch configuration (grid/block/stream) and the context that owns it
//     land with G11 on the qualification host; these kernels are pure
//     functions over caller-owned staging buffers.
//
// Memory law: no dense O x D tensor anywhere.  Callers stage ONE TILE of
// per-origin distance rows ([tile_origins x d_count]); d_count is the
// destination count of the row's profile (<= 6919 for the audited Boston
// cohort row family), so a tile's staging is bounded by the declared byte
// budget in gpu/runtime/buffers.py.

// -----------------------------------------------------------------------
// K0: order-preserving compaction of in-cutoff destinations, one CUDA block
// per origin row.  Output indices are ascending (order preservation is
// semantic: the compacted order fixes every downstream association).
// UNA_G10_BLOCK_THREADS must be a power of two (Hillis-Steele scan).
// -----------------------------------------------------------------------
#define UNA_G10_BLOCK_THREADS 256

extern "C" __global__ void una_g10_fold_compact(
    const double* __restrict__ dist,        // [rows][d_count] staged tile
    long long d_count,
    double cutoff,                          // closed radius (<=)
    long long* __restrict__ out_idx,        // [rows][d_count] capacity
    unsigned long long* __restrict__ out_cnt)  // [rows]
{
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const double* drow = dist + (size_t)row * (size_t)d_count;
    long long* orow = out_idx + (size_t)row * (size_t)d_count;

    __shared__ unsigned long long s_scan[UNA_G10_BLOCK_THREADS];
    __shared__ unsigned long long s_base;

    if (tid == 0) {
        s_base = 0ULL;
    }
    __syncthreads();

    for (long long base = 0; base < d_count; base += UNA_G10_BLOCK_THREADS) {
        const long long i = base + tid;
        // NaN <= cutoff is False -> excluded, exactly like the numpy prologue.
        const bool keep = (i < d_count) && (drow[i] <= cutoff);

        s_scan[tid] = keep ? 1ULL : 0ULL;
        __syncthreads();

        // Inclusive scan over flags (Hillis-Steele; block dim is a power of 2).
        for (unsigned int off = 1; off < UNA_G10_BLOCK_THREADS; off <<= 1) {
            unsigned long long v = 0ULL;
            if ((int)tid >= (int)off) {
                v = s_scan[tid - off];
            }
            __syncthreads();
            s_scan[tid] += v;
            __syncthreads();
        }

        if (keep) {
            orow[s_base + s_scan[tid] - 1ULL] = i;
        }
        __syncthreads();  // every read of s_scan for this chunk is done

        if (tid == UNA_G10_BLOCK_THREADS - 1) {
            s_base += s_scan[UNA_G10_BLOCK_THREADS - 1];
        }
        __syncthreads();  // updated s_base visible before the next chunk
    }

    if (tid == 0) {
        out_cnt[row] = s_base;
    }
}

// -----------------------------------------------------------------------
// K1: per-element coefficients (NO cross-element operation, so the results
// are association-free).  Expressions mirror the legacy fold's summands
// element-for-element:
//   exp:  wn / exp(beta * max(0, dn - plateau))
//   log:  wn * (1 - 1 / (1 + exp(-g * (dn - plateau - midpoint))))
// np.maximum(0, x) propagates NaN while fmax returns the non-NaN operand;
// the admitted domain here is finite (post-K0 in-cutoff distances), and the
// twins in host_twins.py use np.maximum — device inputs are gated to the
// same finite domain, where the two agree.  This reliance is recorded, not
// hidden.
// -----------------------------------------------------------------------
extern "C" __global__ void una_g10_fold_coefficients_exp(
    const double* __restrict__ dn,
    const double* __restrict__ wn,
    long long m,
    double beta,
    double plateau,
    double* __restrict__ c_out)
{
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < m) {
        const double t = fmax(0.0, dn[i] - plateau);
        c_out[i] = wn[i] / exp(beta * t);
    }
}

extern "C" __global__ void una_g10_fold_coefficients_log(
    const double* __restrict__ dn,
    const double* __restrict__ wn,
    long long m,
    double growth_rate,
    double plateau,
    double midpoint,
    double* __restrict__ c_out)
{
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < m) {
        const double s = -growth_rate * (dn[i] - plateau - midpoint);
        c_out[i] = wn[i] * (1.0 - 1.0 / (1.0 + exp(s)));
    }
}

// K1 knn-branch coefficients over the SORTED, CROPPED prefix.  The sort
// itself is K2 and stays host-side (numba tie law); these kernels run on
// the cropped arrays the host hands back.
extern "C" __global__ void una_g10_fold_knn_coeffs_exp(
    const double* __restrict__ dn_s,
    const double* __restrict__ wn_s,
    const double* __restrict__ kn,
    long long m,
    double beta,
    double plateau,
    double* __restrict__ c_out)
{
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < m) {
        const double t = fmax(0.0, dn_s[i] - plateau);
        c_out[i] = (kn[i] * wn_s[i]) * exp(-beta * t);
    }
}

extern "C" __global__ void una_g10_fold_knn_coeffs_log(
    const double* __restrict__ dn_s,
    const double* __restrict__ wn_s,
    const double* __restrict__ kn,
    long long m,
    double growth_rate,
    double plateau,
    double midpoint,
    double* __restrict__ c_out)
{
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < m) {
        const double s = -growth_rate * (dn_s[i] - plateau - midpoint);
        c_out[i] = (kn[i] * wn_s[i]) * (1.0 - 1.0 / (1.0 + exp(s)));
    }
}

extern "C" __global__ void una_g10_fold_knn_coeffs_linear(
    const double* __restrict__ wn_s,
    const double* __restrict__ kn,
    long long m,
    double* __restrict__ c_out)
{
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < m) {
        c_out[i] = kn[i] * wn_s[i];
    }
}

// NOTE (intentional absence): there is NO device reduction kernel here.  A
// device sum tree is a different association than the CPU compiler's
// vectorized tree; floating atomics and completion-order sums are forbidden
// by dossier 08.  Reductions run host-side under the cell-certified tree
// (host_twins.reduce_with) or stay on the trusted CPU route.
