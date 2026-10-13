"""G10 GPU metric-fold bridge (qualification instrument, device-gated).

A numba.cuda port of the strict per-origin reach/gravity/knn fold
(the same Phase C order as the N00 native fold in
``_native/metric_fold.cpp``): one thread per origin, strictly
sequential order-bearing sums inside an origin, ``(d, filtered-index)``
insertion-sorted knn crop (total key order - the sorted result is
unique), and the three decay branches mirroring the delivered fold.
The Phase-B host structural scan also has a battery-validated njit
re-expression (``_phase_b_fast``, bit-identical return codes vs the
``_phase_b`` Python oracle on every battery case - see
``probes/gpu_phase_b_battery.py``); the oracle is retained for that
battery.  The scan is a per-call host cost, so the fold's validation
chronology (Phase-B refusal BEFORE any device work) is unchanged.

ENGAGEMENT LAW (identical to the N00 native bridge): the production
batch route NEVER consults this module - it is imported lazily and
only by the tests/tenx/gpu battery and the G11 qualification scope.
The device fold enters a production route ONLY where on-device bit
validation has proven exact equality for the profile (dossier 08);
until then the unrun-gates statement in ``gpu_region.json`` binds:
no GPU bits, no GPU walls, no GPU engagement, no capability claim.

CAPABILITY LAW: without a CUDA device this module raises
:class:`GpuUnavailable` - a recorded hardware blocker, never a
simulated pass.  CPU-only tests assert laws; they never mark hardware
validation passed.  Device transcendentals go through libdevice and
are NOT assumed bit-equal to the host libm path; G11 validates per
profile before any admission.

FORBIDDEN by the region spec (``g00_gpu/gpu_region.json``) and
enforced here by construction + test: atomics or completion-order
sums, any reduction reassociation, dense O x D x jobs tensors, FP32
substitution, fastmath, unbounded scratch (per-origin scratch is the
thread's own packed slice segment, sized nnz exactly once).
"""
import math

import numpy as np

try:  # device-gated: the module must import on CPU-only hosts
    from numba import cuda as _cuda
    _CUDA_IMPORT_ERROR = None
except Exception as _e:  # pragma: no cover - depends on host
    _cuda = None
    _CUDA_IMPORT_ERROR = _e

try:  # CPU njit for the Phase-B scan; same availability class as cuda
    from numba import njit as _njit
    _NUMBA_IMPORT_ERROR = None
except Exception as _e:  # pragma: no cover - depends on host
    _njit = None
    _NUMBA_IMPORT_ERROR = _e


class GpuUnavailable(RuntimeError):
    """No usable CUDA device/toolchain (recorded hardware blocker)."""


class GpuRefused(ValueError):
    """The GPU fold refused the input (typed refusal, exact fallback
    domain)."""


_DECAY_CODES = {"exponential": 0, "logistic": 1}
_KERNEL = None


def cuda_available():
    """True only when a device is visible AND the driver initializes."""
    if _cuda is None:
        return False
    try:
        return bool(_cuda.is_available())
    except Exception:  # driver init failure is the recorded blocker
        return False


def device_fingerprint():
    """(name, compute_capability, numba_cuda_source) or None."""
    if not cuda_available():
        return None
    dev = _cuda.get_current_device()
    cc = dev.compute_capability
    name = dev.name.decode() if isinstance(dev.name, bytes) else str(dev.name)
    return name, tuple(cc), "numba.cuda"


def require_device():
    """Raise the typed blocker unless a real device initializes."""
    if _cuda is None:
        raise GpuUnavailable(
            f"GPU fold unavailable: numba.cuda import failed "
            f"({_CUDA_IMPORT_ERROR!r}); hardware validation NOT "
            f"passed - this is a blocker, not a fallback domain")
    if not _cuda.is_available():
        raise GpuUnavailable(
            "GPU fold unavailable: no CUDA device / driver init "
            "failed; hardware validation NOT passed - this is a "
            "blocker, not a fallback domain")


def _kernel():
    """Compile (once) the strict per-origin fold kernel."""
    global _KERNEL
    if _KERNEL is not None:
        return _KERNEL

    @_cuda.jit
    def _fold(indptr, dist_all, ids_all, d_weights, knn_weights,
              cutoff, beta, plateau, mid, gr, knn_decay,
              out_reach, out_ge, out_gl, out_kn, out_kept,
              pairs_d, pairs_i, scratch_d, scratch_w):
        o = _cuda.grid(1)
        n_origins = indptr.shape[0] - 1
        if o >= n_origins:
            return
        lo = indptr[o]
        hi = indptr[o + 1]
        s_reach = 0.0
        s_ge = 0.0
        s_gl = 0.0
        kept = 0
        # strict left-to-right filtered fold (Phase C order)
        for j in range(lo, hi):
            d = dist_all[j]
            if not (d <= cutoff):  # NaN falls out, as on host
                continue
            w = d_weights[ids_all[j]]
            s_reach += w
            pos = d - plateau
            clipped = pos if pos > 0.0 else 0.0
            s_ge += w / math.exp(beta * clipped)
            s_gl += w * (1.0 - 1.0 / (1.0 + math.exp(-gr * (pos - mid))))
            # thread-owned scratch: this origin's own slice segment
            pairs_d[lo + kept] = d
            pairs_i[lo + kept] = kept  # FILTERED position, as native
            scratch_d[lo + kept] = d
            scratch_w[lo + kept] = w
            kept += 1
        out_reach[o] = s_reach
        out_ge[o] = s_ge
        out_gl[o] = s_gl
        out_kept[o] = kept
        k = kept if kept < knn_weights.shape[0] \
            else knn_weights.shape[0]
        if k == 0:
            out_kn[o] = 0.0
            return
        # insertion sort of [lo, lo+kept) by (d, i): the keys are a
        # total order (i unique), so the sorted result is unique and
        # algorithm-independent - identical to the native std::sort
        for a in range(1, kept):
            da = pairs_d[lo + a]
            ia = pairs_i[lo + a]
            b = a - 1
            while b >= 0:
                db = pairs_d[lo + b]
                ib = pairs_i[lo + b]
                if db > da or (db == da and ib > ia):
                    pairs_d[lo + b + 1] = db
                    pairs_i[lo + b + 1] = ib
                    b -= 1
                else:
                    break
            pairs_d[lo + b + 1] = da
            pairs_i[lo + b + 1] = ia
        s_kn = 0.0
        for t in range(k):
            si = pairs_i[lo + t]
            d = scratch_d[lo + si]
            w = scratch_w[lo + si]
            coeff = knn_weights[t]
            if knn_decay == 0:
                pos = d - plateau
                clipped = pos if pos > 0.0 else 0.0
                s_kn += (coeff * w) * math.exp(-beta * clipped)
            elif knn_decay == 1:
                pos2 = d - plateau - mid
                s_kn += (coeff * w) * (1.0
                                       - 1.0 / (1.0
                                                + math.exp(-gr * pos2)))
            else:
                s_kn += coeff * w
        out_kn[o] = s_kn

    _KERNEL = _fold
    return _KERNEL


def _req(a, dtype, name):
    # Layout is normalized by ascontiguousarray (a copy, so caller
    # memory is never touched); the dtype/byteorder law still binds.
    if a.dtype != np.dtype(dtype):
        raise GpuRefused(f"{name} dtype {a.dtype} != {np.dtype(dtype)}")


def _phase_b(indptr, ids_all, node_count, nnz):
    """Host structural scan; same codes as the native fold.

    Retained as the ORACLE for the bit-identity battery
    (``probes/gpu_phase_b_battery.py``); the bridge calls the
    battery-validated :func:`_phase_b_fast` below.
    """
    for o in range(indptr.shape[0] - 1):
        lo = int(indptr[o])
        hi = int(indptr[o + 1])
        if hi < lo or lo < 0:
            return 1
        if hi > nnz:
            return 1
        for j in range(lo, hi):
            if j > lo and ids_all[j] <= ids_all[j - 1]:
                return 2
            if ids_all[j] < 0 or ids_all[j] >= node_count:
                return 1
    return 0


# Precedence law of the scan (verified against the oracle above): per
# origin ascending, (a) hi<lo or lo<0 -> 1, (b) hi>nnz -> 1, then per j
# ascending in [lo,hi): (c) j>lo and ids_all[j] <= ids_all[j-1] -> 2,
# (d) ids_all[j] < 0 or >= node_count -> 1.  (c) is checked BEFORE (d)
# at the same j, so at one j code 2 wins; across j the earlier j's
# class wins; origin order precedes both.  The fast form is a verbatim
# transplant of that loop under njit (parallel=False: sequential order
# keeps the precedence law trivially; early exit is NOT preserved and
# need not be - only the returned code must be identical on every
# input, which the battery asserts case by case).
if _njit is not None:
    @_njit(cache=False, nogil=True, parallel=False)
    def _phase_b_fast(indptr, ids_all, node_count, nnz):
        n_origins = indptr.shape[0] - 1
        for o in range(n_origins):
            lo = indptr[o]
            hi = indptr[o + 1]
            if hi < lo or lo < 0:
                return 1
            if hi > nnz:
                return 1
            for j in range(lo, hi):
                if j > lo and ids_all[j] <= ids_all[j - 1]:
                    return 2
                if ids_all[j] < 0 or ids_all[j] >= node_count:
                    return 1
        return 0
else:  # pragma: no cover - numba absent: bridge is device-gated anyway
    _phase_b_fast = _phase_b


def gpu_replay_fold(indptr, ids_all, dist_all, d_weights, cutoff,
                    gravity_beta, gravity_plateau,
                    gravity_logistic_midpoint, gravity_growth_rate,
                    knn_weights, knn_decay):
    """Whole-structure GPU replay fold (the packed trace layout).

    Returns ``(reach, ge, gl, kn, kept)`` fresh float64/int64 arrays
    (the caller owns them; host inputs are never aliased - the device
    receives copies).  Refusals are typed and precede any device work;
    without a device this raises :class:`GpuUnavailable` loudly.
    """
    indptr = np.ascontiguousarray(indptr)
    ids_all = np.ascontiguousarray(ids_all)
    dist_all = np.ascontiguousarray(dist_all)
    d_weights = np.ascontiguousarray(d_weights)
    knn_weights = np.ascontiguousarray(knn_weights)
    _req(indptr, np.int64, "indptr")
    _req(ids_all, np.int64, "ids_all")
    _req(dist_all, np.float64, "dist_all")
    _req(d_weights, np.float64, "d_weights")
    _req(knn_weights, np.float64, "knn_weights")
    if indptr.ndim != 1 or indptr.shape[0] < 1:
        raise GpuRefused("indptr must be a non-empty 1-D int64 array")
    if indptr[0] != 0 or indptr[-1] != ids_all.shape[0]:
        raise GpuRefused("indptr endpoints must bound ids_all")
    origin_count = indptr.shape[0] - 1
    nnz = ids_all.shape[0]
    if dist_all.shape[0] != nnz:
        raise GpuRefused("dist_all must be packed to ids_all")
    # Mirror the production fold exactly: any kind other than
    # exponential/logistic takes the flat else-branch (Accessibility
    # has no refusal there either).
    decay = _DECAY_CODES.get(knn_decay, 2)
    cutoff = float(cutoff)
    for v in (cutoff, gravity_beta, gravity_plateau,
              gravity_logistic_midpoint, gravity_growth_rate):
        fv = float(v)
        if not np.isfinite(fv):
            raise GpuRefused("fold scalars must be finite")
    rc = _phase_b_fast(indptr, ids_all, d_weights.shape[0], nnz)
    if rc == 2:
        raise GpuRefused("slice-order corruption (Phase B refusal)")
    if rc != 0:
        raise GpuRefused("structural (Phase B refusal)")
    require_device()
    if origin_count == 0:
        # no launch is possible; bits are the empty-structure law
        # (all sums +0.0, kn 0.0, kept 0) - identical to the native
        return (np.zeros(0), np.zeros(0), np.zeros(0), np.zeros(0),
                np.zeros(0, dtype=np.int64))

    kernel = _kernel()
    d_indptr = _cuda.to_device(indptr)
    d_dist = _cuda.to_device(dist_all)
    d_ids = _cuda.to_device(ids_all)
    d_w = _cuda.to_device(d_weights)
    d_knn = _cuda.to_device(knn_weights)
    d_reach = _cuda.device_array(origin_count, dtype=np.float64)
    d_ge = _cuda.device_array(origin_count, dtype=np.float64)
    d_gl = _cuda.device_array(origin_count, dtype=np.float64)
    d_kn = _cuda.device_array(origin_count, dtype=np.float64)
    d_kept = _cuda.device_array(origin_count, dtype=np.int64)
    # bounded exact-size scratch: each thread owns only its slice
    # segment [indptr[o], indptr[o+1]) - nnz total, never O*nnz
    d_pairs_d = _cuda.device_array(nnz, dtype=np.float64)
    d_pairs_i = _cuda.device_array(nnz, dtype=np.int64)
    d_scd = _cuda.device_array(nnz, dtype=np.float64)
    d_scw = _cuda.device_array(nnz, dtype=np.float64)
    tpb = 128
    blocks = (origin_count + tpb - 1) // tpb
    kernel[blocks, tpb](
        d_indptr, d_dist, d_ids, d_w, d_knn,
        cutoff, float(gravity_beta), float(gravity_plateau),
        float(gravity_logistic_midpoint), float(gravity_growth_rate),
        int(decay),
        d_reach, d_ge, d_gl, d_kn, d_kept,
        d_pairs_d, d_pairs_i, d_scd, d_scw)
    # synchronize BEFORE any result is published (region law)
    _cuda.synchronize()
    reach = d_reach.copy_to_host()
    ge = d_ge.copy_to_host()
    gl = d_gl.copy_to_host()
    kn = d_kn.copy_to_host()
    kept = d_kept.copy_to_host()
    return reach, ge, gl, kn, kept
