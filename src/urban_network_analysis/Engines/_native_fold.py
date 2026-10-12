"""N00 native metric-fold bridge (qualification instrument).

The AOT C++17 fold in ``_native/metric_fold.cpp`` reproduces the
per-origin reach/gravity/knn fold of ``Accessibility.reach_gravity_
knn_access`` in strict IEEE arithmetic (no -ffast-math, no FP
contraction).  The bit battery (tests/tenx/native) establishes where
its outputs are bit-identical to the delivered numba fastmath fold
and where they diverge.

ENGAGEMENT LAW (the N00 admission decision): the production batch
route NEVER consults this module.  The delivered numba fastmath
lowering is not a representable op order at general trip counts (the
battery maps divergence at ULP level, data- and trip-count dependent;
same hazard class as the P03M LLVM split record), so the campaign's
bit-parity law admits the native fold only where the battery has
proven exact equality.  This bridge therefore serves the battery and
the N10 qualification scope; the public fallback stays the numba
fold, which is exact by construction.

Build: :func:`ensure_built` compiles ``_native/metric_fold.cpp`` with
the recorded command into ``_native/`` (content-keyed; the .so is a
build artifact and stays out of git).  Absent toolchain or failed
build raises :class:`NativeBuildUnavailable` - a build blocker, not a
silent fallback: the battery needs the bits.
"""
import hashlib
import os
import subprocess

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_CPP = os.path.join(_HERE, "_native", "metric_fold.cpp")
_SO = os.path.join(_HERE, "_native", "libuna_metric_fold.so")

BUILD_CMD = [
    "g++", "-std=c++17", "-O2", "-fPIC", "-shared", "-fno-fast-math",
    "-ffp-contract=off", "-o", _SO, _CPP,
]

_DECAY_CODES = {"exponential": 0, "logistic": 1}


class NativeBuildUnavailable(RuntimeError):
    """The native fold could not be built (recorded blocker)."""


def so_sha256():
    h = hashlib.sha256()
    with open(_SO, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_built(force=False):
    """Build the .so with the recorded command if absent or stale.

    Returns the .so sha256.  Content-keyed on the C++ source: a stale
    artifact is rebuilt, never silently reused.
    """
    if force or not os.path.exists(_SO) \
            or os.path.getmtime(_SO) < os.path.getmtime(_CPP):
        try:
            subprocess.run(BUILD_CMD, check=True, capture_output=True,
                           text=True)
        except (OSError, subprocess.CalledProcessError) as exc:
            raise NativeBuildUnavailable(
                f"native fold build failed: {exc}") from exc
    return so_sha256()


class NativeRefused(ValueError):
    """The native fold refused the input (typed refusal, exact
    fallback domain)."""


_LIB = None


def _lib():
    global _LIB
    if _LIB is None:
        import ctypes
        ensure_built()
        lib = ctypes.CDLL(_SO)
        P64 = ctypes.POINTER(ctypes.c_int64)
        PD = ctypes.POINTER(ctypes.c_double)
        lib.una_replay_fold.restype = ctypes.c_int
        lib.una_replay_fold.argtypes = [
            P64, ctypes.c_size_t,          # indptr, origin_count
            PD, ctypes.c_size_t,           # dist_all, nnz
            P64,                           # ids_all
            PD, ctypes.c_size_t,           # d_weights, node_count
            ctypes.c_double, ctypes.c_double, ctypes.c_double,
            ctypes.c_double, ctypes.c_double,   # cutoff..growth_rate
            PD, ctypes.c_size_t,           # knn_weights, len
            ctypes.c_int64,                # knn_decay code
            PD, PD, PD, PD,                # reach, ge, gl, kn
            P64,                           # kept per origin
            ctypes.c_void_p, PD, PD,       # scratch
        ]
        _LIB = lib
    return _LIB


def _req(a, dtype, name):
    # Layout is normalized by ascontiguousarray (a copy, so caller
    # memory is never touched); the dtype/byteorder law still binds.
    if a.dtype != np.dtype(dtype):
        raise NativeRefused(f"{name} dtype {a.dtype} != {np.dtype(dtype)}")


def native_replay_fold(indptr, ids_all, dist_all, d_weights, cutoff,
                       gravity_beta, gravity_plateau,
                       gravity_logistic_midpoint, gravity_growth_rate,
                       knn_weights, knn_decay):
    """Whole-structure native replay fold (the packed trace layout).

    Returns ``(reach, ge, gl, kn, kept)`` fresh float64 arrays (the
    caller owns them; inputs are treated as immutable).  Refuses the
    domain loudly on any deviation from the traced layout law.
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
        raise NativeRefused("indptr must be a non-empty 1-D int64 array")
    if indptr[0] != 0 or indptr[-1] != ids_all.shape[0]:
        raise NativeRefused("indptr endpoints must bound ids_all")
    origin_count = indptr.shape[0] - 1
    nnz = ids_all.shape[0]
    if dist_all.shape[0] != nnz:
        raise NativeRefused("dist_all must be packed to ids_all")
    # Mirror the production fold exactly: any kind other than
    # exponential/logistic takes the flat else-branch (Accessibility
    # has no refusal there either).
    decay = _DECAY_CODES.get(knn_decay, 2)
    cutoff = float(cutoff)
    for v in (cutoff, gravity_beta, gravity_plateau,
              gravity_logistic_midpoint, gravity_growth_rate):
        fv = float(v)
        if not np.isfinite(fv):
            raise NativeRefused("fold scalars must be finite")

    lib = _lib()
    reach = np.zeros(origin_count)
    ge = np.zeros(origin_count)
    gl = np.zeros(origin_count)
    kn = np.zeros(origin_count)
    kept = np.zeros(origin_count, dtype=np.int64)
    if nnz == 0:
        # knn law for empty slices: kn == 0.0, sums == +0.0
        return reach, ge, gl, kn, kept
    pairs = np.zeros(nnz, dtype=[("d", "<f8"), ("i", "<i8")])
    scratch_d = np.zeros(nnz)
    scratch_w = np.zeros(nnz)
    data = lambda a: a.ctypes.data_as  # noqa: E731
    import ctypes
    rc = lib.una_replay_fold(
        data(indptr)(ctypes.POINTER(ctypes.c_int64)), origin_count,
        data(dist_all)(ctypes.POINTER(ctypes.c_double)), nnz,
        data(ids_all)(ctypes.POINTER(ctypes.c_int64)),
        data(d_weights)(ctypes.POINTER(ctypes.c_double)),
        d_weights.shape[0],
        cutoff, float(gravity_beta), float(gravity_plateau),
        float(gravity_logistic_midpoint), float(gravity_growth_rate),
        data(knn_weights)(ctypes.POINTER(ctypes.c_double)),
        knn_weights.shape[0], decay,
        data(reach)(ctypes.POINTER(ctypes.c_double)),
        data(ge)(ctypes.POINTER(ctypes.c_double)),
        data(gl)(ctypes.POINTER(ctypes.c_double)),
        data(kn)(ctypes.POINTER(ctypes.c_double)),
        data(kept)(ctypes.POINTER(ctypes.c_int64)),
        pairs.ctypes.data_as(ctypes.c_void_p),
        data(scratch_d)(ctypes.POINTER(ctypes.c_double)),
        data(scratch_w)(ctypes.POINTER(ctypes.c_double)))
    if rc == 2:
        raise NativeRefused("slice-order corruption (Phase B refusal)")
    if rc != 0:
        raise NativeRefused(f"native fold refused, rc={rc}")
    return reach, ge, gl, kn, kept
