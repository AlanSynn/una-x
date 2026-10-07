"""N00 binding surface: checked interfaces, ownership, cancellation,
engagement law, and opt-in wiring.

The producer's obligations (dossier 08): refused-not-coerced inputs,
explicit per-array owners, bounded scratch, cancellation at tile
barriers, and engagement counters that record REAL dispatches (never
import-time detection).  The opt-in law: importing the wrapper or the
engine engages nothing; only install() points the cohort route's build
hook at the native twin, and uninstall() restores the numba route.
"""
import ctypes
import gc
import hashlib
import inspect
import json

import numpy as np
import pytest

from conftest import build_case, ring_case


@pytest.fixture(scope="module")
def ext():
    pytest.importorskip("pybind11")
    import trace_build_native as tbn
    return tbn._load_extension()


def binding_kwargs(case):
    """An admitted case as the binding's checked arguments (reverse
    incidence built exactly as the trace module builds it)."""
    from urban_network_analysis.Engines import _trace_access_scratch as trm
    v = case["ptr"].shape[0] - 1
    rp, rv = trm._reverse_incidence(case["d"], v)
    return dict(
        o_terminal_idxs=case["o"], o_terminal_weights=case["ow"],
        adjacency_pointer=case["ptr"], adjacency_vector=case["vec"],
        adjacency_vector_weights=case["wts"],
        adjacency_flag_column=case["flag"],
        d_terminal_idxs=case["d"], d_terminal_weights=case["dw"],
        reverse_pointer=rp, reverse_vector=rv,
        cutoff=case["cutoff"], epoch_rollover=2 ** 62,
        tile_origins=64, lanes=2, budget_cand_bytes=2 ** 62,
        max_degree=int(np.max(case["ptr"][1:] - case["ptr"][:-1])))


@pytest.fixture(scope="module")
def arrays():
    return build_case(3)


def assert_refused(ext, arrays, mutate, expect_fragment):
    """Run the binding's argument checks on a FRESH copy of the valid
    arguments; the refusal must name the offending array (refused,
    never coerced)."""
    kw = binding_kwargs(arrays)
    mutate(kw)
    with pytest.raises(RuntimeError) as ei:
        ext.run_stream(**kw)
    assert expect_fragment in str(ei.value)


def test_dtype_refused_not_coerced(ext, arrays):
    assert_refused(ext, arrays,
                   lambda k: k.update(
                       adjacency_vector=arrays["vec"].astype(np.int32)),
                   "adjacency_vector: dtype must be int64")
    assert_refused(ext, arrays,
                   lambda k: k.update(
                       adjacency_vector_weights=arrays["wts"].astype(
                           np.float32)),
                   "adjacency_vector_weights: dtype must be float64")
    assert_refused(ext, arrays,
                   lambda k: k.update(
                       adjacency_flag_column=arrays["flag"].astype(np.int8)),
                   "adjacency_flag_column: dtype must be bool")


def test_ndim_refused(ext, arrays):
    assert_refused(ext, arrays,
                   lambda k: k.update(
                       o_terminal_idxs=arrays["o"][:, 0].copy()),
                   "o_terminal_idxs: ndim must be 2")


def test_noncontiguous_refused(ext, arrays):
    assert_refused(ext, arrays,
                   lambda k: k.update(
                       adjacency_vector_weights=arrays["wts"][::2]),
                   "adjacency_vector_weights: must be C-contiguous")


def test_shape_crosschecks_refused(ext, arrays):
    assert_refused(ext, arrays,
                   lambda k: k.update(
                       reverse_vector=k["reverse_vector"][:-1].copy()),
                   "reverse_vector: shape must be (2 * D,)")
    assert_refused(ext, arrays,
                   lambda k: k.update(
                       adjacency_flag_column=arrays["flag"][:-1].copy()),
                   "adjacency_flag_column: must match adjacency_vector")


def test_output_ownership_and_read_only(ext, arrays):
    """Binding level: outputs are capsule-backed (OWNDATA False — the
    extension's buffer, no copy).  Wrapper level: the published record
    arrays are marked read-only (the record's geometry is immutable
    through the Python surface)."""
    out = ext.run_stream(**binding_kwargs(arrays))
    assert not out["ids"].flags.owndata
    assert not out["dist"].flags.owndata
    assert not out["counts"].flags.owndata

    import trace_build_native as tbn
    case_args = (arrays["o"], arrays["ow"], arrays["ptr"], arrays["vec"],
                 arrays["wts"], arrays["flag"], arrays["d"], arrays["dw"],
                 arrays["dwt"], arrays["cutoff"])
    nat = tbn.native_build_trace(*case_args)
    assert not nat.destination_ids.flags.writeable
    assert not nat.distance_bits.flags.writeable
    assert not nat.indptr.flags.writeable


def test_owner_lifetime_survives_teardown(ext, arrays):
    """Capsule-owned buffers stay valid after the wrapper's locals are
    dropped: the owner is the per-array capsule, not global state."""
    out = ext.run_stream(**binding_kwargs(arrays))
    ids, dist = out["ids"], out["dist"]
    del out
    gc.collect()
    assert ids.shape[0] == dist.shape[0]
    if ids.shape[0]:
        assert int(ids[-1]) >= 0  # touch the tail of the owned buffer


# --------------------------------------------------------------------------
# Cancellation at the core ABI (ctypes over native/trace_producer.h):
# a pre-set cancel flag stops the stream at a tile barrier with status
# UNA_N00_CANCELED (3), never a truncated success.
# --------------------------------------------------------------------------
class N00Input(ctypes.Structure):
    """Mirror of struct una_n00_input (native/trace_producer.h), field
    for field -- including lanes being int32 BETWEEN tile_origins and
    budget_cand_bytes (ctypes applies the same alignment padding as
    the C compiler)."""
    _fields_ = [
        ("o_idx", ctypes.POINTER(ctypes.c_int64)),
        ("o_w", ctypes.POINTER(ctypes.c_double)),
        ("adj_ptr", ctypes.POINTER(ctypes.c_int64)),
        ("adj_vec", ctypes.POINTER(ctypes.c_int64)),
        ("adj_w", ctypes.POINTER(ctypes.c_double)),
        ("net", ctypes.POINTER(ctypes.c_uint8)),
        ("d_idx", ctypes.POINTER(ctypes.c_int64)),
        ("d_w", ctypes.POINTER(ctypes.c_double)),
        ("rev_ptr", ctypes.POINTER(ctypes.c_int64)),
        ("rev_vec", ctypes.POINTER(ctypes.c_int64)),
        ("O", ctypes.c_int64), ("N", ctypes.c_int64),
        ("D", ctypes.c_int64), ("E", ctypes.c_int64),
        ("cutoff", ctypes.c_double),
        ("epoch_rollover", ctypes.c_int64),
        ("tile_origins", ctypes.c_int64),
        ("lanes", ctypes.c_int32),
        ("budget_cand_bytes", ctypes.c_int64),
        ("max_degree", ctypes.c_int64),
    ]


class N00Output(ctypes.Structure):
    """Mirror of struct una_n00_output (native/trace_producer.h)."""
    _fields_ = [
        ("out_ids", ctypes.POINTER(ctypes.c_int64)),
        ("out_dist", ctypes.POINTER(ctypes.c_double)),
        ("nnz", ctypes.c_int64),
        ("counts", ctypes.POINTER(ctypes.c_int64)),
        ("cand_total", ctypes.c_int64),
        ("touched_total", ctypes.c_int64),
        ("tiles_done", ctypes.c_int64),
        ("budget_hit", ctypes.c_int64),
        ("scratch_peak_bytes", ctypes.c_int64),
        ("lane_counters", ctypes.POINTER(ctypes.c_int64)),
        ("lanes_written", ctypes.c_int32),
        ("status", ctypes.c_int32),
    ]


def test_core_abi_cancel(arrays):
    import trace_build_native as tbn
    from urban_network_analysis.Engines import _trace_access_scratch as trm
    so = tbn._load_build_script().ensure_built()
    lib = ctypes.CDLL(str(so))
    lib.una_n00_run_stream.argtypes = [
        ctypes.POINTER(N00Input), ctypes.POINTER(N00Output),
        ctypes.POINTER(ctypes.c_int)]
    lib.una_n00_run_stream.restype = ctypes.c_int
    lib.una_n00_output_free.argtypes = [ctypes.POINTER(N00Output)]

    v = arrays["ptr"].shape[0] - 1
    rp, rv = trm._reverse_incidence(arrays["d"], v)

    def cptr(a, t):
        return a.ctypes.data_as(ctypes.POINTER(t))

    cancel = ctypes.c_int(1)  # canceled BEFORE the first tile barrier
    inp = N00Input(
        o_idx=cptr(arrays["o"], ctypes.c_int64),
        o_w=cptr(arrays["ow"], ctypes.c_double),
        adj_ptr=cptr(arrays["ptr"], ctypes.c_int64),
        adj_vec=cptr(arrays["vec"], ctypes.c_int64),
        adj_w=cptr(arrays["wts"], ctypes.c_double),
        net=cptr(arrays["flag"], ctypes.c_uint8),
        d_idx=cptr(arrays["d"], ctypes.c_int64),
        d_w=cptr(arrays["dw"], ctypes.c_double),
        rev_ptr=cptr(rp, ctypes.c_int64),
        rev_vec=cptr(rv, ctypes.c_int64),
        O=arrays["o"].shape[0], N=v, D=arrays["d"].shape[0],
        E=arrays["vec"].shape[0], cutoff=float(arrays["cutoff"]),
        epoch_rollover=2 ** 62, tile_origins=1,
        budget_cand_bytes=2 ** 62,
        max_degree=int(np.max(arrays["ptr"][1:] - arrays["ptr"][:-1])),
        lanes=1)
    out = N00Output()
    rc = lib.una_n00_run_stream(ctypes.byref(inp), ctypes.byref(out),
                                ctypes.byref(cancel))
    assert rc == 3  # UNA_N00_CANCELED returned AND recorded
    assert out.status == 3
    assert out.budget_hit == -1  # never a truncated success
    lib.una_n00_output_free(ctypes.byref(out))  # no leak, no double free


def test_engagement_real_dispatch_law():
    """The engagement counters move ONLY on real native dispatches:
    importing the module and running the NUMBA route leave them alone;
    a native record moves dispatches/origins/tiles/candidates by the
    record's own numbers."""
    import trace_build_native as tbn
    from conftest import run_pair
    from urban_network_analysis.Engines import _trace_access_scratch as trm

    before = tbn.engagement()
    case = ring_case(seed=23, origins=8, d_count=20, v=60, extra=40)
    args = (case["o"], case["ow"], case["ptr"], case["vec"], case["wts"],
            case["flag"], case["d"], case["dw"], case["dwt"], case["cutoff"])
    trm.build_trace(*args)  # numba route: engagement untouched
    mid = tbn.engagement()
    for k in before:
        assert mid[k] == before[k], k
    nat = tbn.native_build_trace(*args)
    after = tbn.engagement()
    assert after["dispatches"] == before["dispatches"] + 1
    assert after["origins"] == before["origins"] + 8
    assert after["tiles"] == before["tiles"] + nat.counters.tiles
    assert (after["candidates"]
            == before["candidates"] + nat.counters.candidates_total)
    assert after["last_status"] == 0
    assert nat.complete


def test_opt_in_install_uninstall():
    """install() points the cohort route's hook at the native twin;
    uninstall() restores None (the numba route).  The hook is the ONLY
    production wiring and defaults to None."""
    from urban_network_analysis.batch import cohort as cohort_mod
    import trace_build_native as tbn
    assert cohort_mod._NATIVE_BUILD_HOOK is None
    try:
        tbn.install()
        assert cohort_mod._NATIVE_BUILD_HOOK is tbn.native_build_trace
    finally:
        tbn.uninstall()
    assert cohort_mod._NATIVE_BUILD_HOOK is None


def test_hook_signature_matches_call_site():
    """The cohort build site passes 10 positionals ending in cutoff
    (d_weights admission-only, last); the native twin must accept
    exactly the shape build_trace accepts."""
    import trace_build_native as tbn
    from urban_network_analysis.Engines import _trace_access_scratch as trm

    hook_params = list(
        inspect.signature(tbn.native_build_trace).parameters)
    ref_params = list(inspect.signature(trm.build_trace).parameters)
    assert hook_params[:10] == ref_params[:10]
    assert hook_params[10] == "epoch_rollover"
    assert ref_params[10] == "epoch_rollover"


def test_build_manifest_law():
    """The recorded compile flags contain no unsafe FP flag, and the
    manifest's source digests match the bytes on disk (stale-check
    law)."""
    import trace_build_native as tbn
    tbn._load_extension()
    man = json.loads((tbn.BUILD_DIR / "build_manifest.json").read_text())
    flags = " ".join(man["compile_flags"])
    for bad in ("-ffast-math", "-funsafe-math", "-ffinite-math",
                "-freciprocal", "-Ofast"):
        assert bad not in flags
    assert "-ffp-contract=off" in man["compile_flags"]
    for name, digest in man["sources"].items():
        h = hashlib.sha256((tbn.NATIVE_DIR / name).read_bytes()).hexdigest()
        assert h == digest, name
    assert man["schema"] == "una-10x-n00-build-manifest-1"
