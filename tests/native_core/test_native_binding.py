"""Binding ABI enforcement (dossier 08): refuse, never cast or reorder.

Every test here drives the RAW compiled module (not the route) to prove
the binding's own argument law: off-contract dtype/layout is REFUSED
with a loud Python error before the GIL is released; no silent cast,
byte-order fixup, reshuffle or reallocation ever happens.  The paired
route tests prove the policy layer mirrors these refusals as delegation.
"""
import numpy as np
import pytest

from urban_network_analysis.backends.native import route_scope_access
from urban_network_analysis.backends.native._loader import (
    NativeUnavailableError,
    load_core,
)

from conftest import bits, corrected_case

try:
    CORE = load_core()
except NativeUnavailableError:  # pragma: no cover - unbuilt host
    CORE = None

pytestmark = pytest.mark.skipif(CORE is None,
                                reason="native core not built on this host")


def small_case():
    """3-node canonical case (hand-checkable engagement)."""
    return dict(
        ptr=np.array([0, 1, 2, 2], dtype=np.int64),
        vec=np.array([1, 2], dtype=np.int64),
        wts=np.array([1.0, 5.0], dtype=np.float64),
        flag=np.array([True, True]),
        o=np.array([[0, 2]], dtype=np.int64),
        ow=np.array([[0.0, 9e9]], dtype=np.float64),
        d=np.array([[1, 2], [0, 1]], dtype=np.int64),
        dw=np.array([[0.25, 0.5], [0.0, 1.0]], dtype=np.float64),
        dwt=np.array([1.0, 2.0], dtype=np.float64),
        cutoff=10.0,
    )


def run_corrected(c, out=None, **over):
    out = out if out is not None else np.zeros(
        (c["o"].shape[0], c["d"].shape[0]), dtype=np.float64)
    return CORE.run_corrected(
        over.get("ptr", c["ptr"]), over.get("vec", c["vec"]),
        over.get("wts", c["wts"]), over.get("o", c["o"]),
        over.get("ow", c["ow"]), over.get("d", c["d"]),
        over.get("dw", c["dw"]), over.get("cutoff", c["cutoff"]),
        1, None, out), out


# ---------------------------------------------------------------------------
# Dtype law: refusal, never cast
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mutate", [
    "wts_f32", "ptr_i32", "vec_i32", "o_idx_f64", "ow_f32",
    "big_endian_wts", "big_endian_ptr",
])
def test_binding_refuses_offcontract_dtypes(mutate):
    c = small_case()
    over = {}
    if mutate == "wts_f32":
        over["wts"] = c["wts"].astype(np.float32)
    elif mutate == "ptr_i32":
        over["ptr"] = c["ptr"].astype(np.int32)
    elif mutate == "vec_i32":
        over["vec"] = c["vec"].astype(np.int32)
    elif mutate == "o_idx_f64":
        over["o"] = c["o"].astype(np.float64)
    elif mutate == "ow_f32":
        over["ow"] = c["ow"].astype(np.float32)
    elif mutate == "big_endian_wts":
        over["wts"] = c["wts"].astype(">f8")
    elif mutate == "big_endian_ptr":
        over["ptr"] = c["ptr"].astype(">i8")
    with pytest.raises(RuntimeError, match="dtype"):
        run_corrected(c, **over)
    # Control: the identical values in canonical dtype are accepted.
    if mutate in ("wts_f32", "big_endian_wts"):
        rep, _ = run_corrected(c, **{"wts": c["wts"].astype(np.float64)})
        assert int(rep["status"]) == 0


def test_binding_refuses_offcontract_flag_dtype():
    """The legacy push-gate column is bool or nothing: a uint8 view of
    the same bytes is refused (no reinterpretation), and a length
    mismatch is refused before any pointer is read."""
    c = small_case()
    out = np.zeros((c["o"].shape[0], c["d"].shape[0]), dtype=np.float64)
    with pytest.raises(RuntimeError, match="dtype"):
        CORE.run_legacy("a3", c["ptr"], c["vec"], c["wts"],
                        c["flag"].view(np.uint8), c["o"], c["ow"], c["d"],
                        c["dw"], c["cutoff"], 1, None, out)
    with pytest.raises(RuntimeError, match="length"):
        CORE.run_legacy("a3", c["ptr"], c["vec"], c["wts"], c["flag"][:-1],
                        c["o"], c["ow"], c["d"], c["dw"], c["cutoff"], 1,
                        None, out)


def test_binding_no_implicit_cast_happens():
    """A refused call raises BEFORE any work: the caller's buffer is
    untouched and the returned error is the binding's (Python-level),
    not a converted computation."""
    c = small_case()
    out = np.ones((c["o"].shape[0], c["d"].shape[0]), dtype=np.float64)
    with pytest.raises(RuntimeError):
        CORE.run_corrected(c["ptr"], c["vec"], c["wts"].astype(np.float32),
                           c["o"], c["ow"], c["d"], c["dw"], c["cutoff"],
                           1, None, out)
    assert (out == 1.0).all()


# ---------------------------------------------------------------------------
# Layout law: ndim, shape, contiguity, alignment, output contract
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mutate,match", [
    ("vec_2d", "1-D"),
    ("o_idx_3col", r"\(n, 2\)"),
    ("out_wrong_shape", "out shape"),
    ("out_readonly", "writeable"),
    ("out_noncontig", "out must be C-contiguous"),
    ("out_f32", "out must be float64"),
    ("ptr_empty", "at least one slot"),
])
def test_binding_refuses_bad_layout(mutate, match):
    c = small_case()
    out = np.zeros((c["o"].shape[0], c["d"].shape[0]), dtype=np.float64)
    kwargs = {}
    if mutate == "vec_2d":
        kwargs["vec"] = c["vec"].reshape(1, 2)
    elif mutate == "o_idx_3col":
        kwargs["o"] = np.zeros((1, 3), dtype=np.int64)
    elif mutate == "out_wrong_shape":
        out = np.zeros((c["o"].shape[0] + 1, c["d"].shape[0]),
                       dtype=np.float64)
    elif mutate == "out_readonly":
        out.setflags(write=False)
    elif mutate == "out_noncontig":
        big = np.zeros((4, 4), dtype=np.float64)
        out = big[:c["o"].shape[0], ::2][:, :c["d"].shape[0]]
        assert not out.flags["C_CONTIGUOUS"]
    elif mutate == "out_f32":
        out = np.zeros((c["o"].shape[0], c["d"].shape[0]), dtype=np.float32)
    elif mutate == "ptr_empty":
        kwargs["ptr"] = np.empty(0, dtype=np.int64)
    with pytest.raises(RuntimeError, match=match):
        CORE.run_corrected(kwargs.get("ptr", c["ptr"]),
                           kwargs.get("vec", c["vec"]),
                           kwargs.get("wts", c["wts"]),
                           kwargs.get("o", c["o"]),
                           kwargs.get("ow", c["ow"]),
                           c["d"], c["dw"], c["cutoff"], 1, None, out)


def test_binding_refuses_unaligned_views():
    """A 4-byte-offset float64 view is refused: alignment is part of the
    layout law (no silent realignment)."""
    c = small_case()
    raw = np.zeros(2 * 8 + 8, dtype=np.uint8)
    unaligned = raw[4:4 + 16].view(np.float64)
    unaligned[:] = c["wts"]
    assert not unaligned.flags["ALIGNED"] or unaligned.ctypes.data % 8 != 0
    with pytest.raises(RuntimeError, match="aligned"):
        CORE.run_corrected(c["ptr"], c["vec"], unaligned, c["o"], c["ow"],
                           c["d"], c["dw"], c["cutoff"], 1, None,
                           np.zeros((1, 2), dtype=np.float64))


def test_route_delegates_on_unaligned_admitted_array():
    """The policy mirror of the alignment refusal: the legacy route
    delegates instead of engaging (reason recorded, oracle result)."""
    from urban_network_analysis.kernels import scope_access as SA
    c = corrected_case("base")
    raw = np.zeros(c["wts"].size * 8 + 8, dtype=np.uint8)
    unaligned = raw[4:4 + c["wts"].size * 8].view(np.float64)
    unaligned[:] = c["wts"]
    c = dict(c, wts=unaligned)
    args = (c["o"], c["ow"], c["ptr"], c["vec"], c["wts"], c["flag"],
            c["d"], c["dw"], c["dwt"], 0.02, 5.0, 8.0, 0.3, "exponential",
            np.array([0.9, 0.5, 0.25, 0.1]), c["cutoff"])
    run = route_scope_access("una_legacy", *args)
    assert run.native_engaged is False
    assert "aligned" in run.fallback_reason
    oracle = SA.integrated_scope_access_oracle(*args)
    assert bits(run.result.reach) == bits(oracle.reach)


def test_binding_route_tag_enforced():
    c = small_case()
    out = np.zeros((c["o"].shape[0], c["d"].shape[0]), dtype=np.float64)
    with pytest.raises(RuntimeError, match="route"):
        CORE.run_legacy("bogus", c["ptr"], c["vec"], c["wts"], c["flag"],
                        c["o"], c["ow"], c["d"], c["dw"], c["cutoff"], 1,
                        None, out)


# ---------------------------------------------------------------------------
# Overflow gate (exposed unit surface; no giant arrays materialized)
# ---------------------------------------------------------------------------

def test_check_domain_size_overflow_gate():
    assert CORE.check_domain_size(5, 0) == 5
    assert CORE.check_domain_size(5, 7) == 12
    big = 2**62
    with pytest.raises(OverflowError):
        CORE.check_domain_size(big, big)
    with pytest.raises(OverflowError):
        CORE.check_domain_size(-1, 0)


# ---------------------------------------------------------------------------
# Edge shapes and the engagement fail-sentinel (hand-counted contract)
# ---------------------------------------------------------------------------

def test_empty_origin_and_destination_domains():
    c = small_case()
    # Zero origins: nothing to do, zero engagement, status OK.
    o0 = np.empty((0, 2), dtype=np.int64)
    ow0 = np.empty((0, 2), dtype=np.float64)
    out0 = np.zeros((0, c["d"].shape[0]), dtype=np.float64)
    rep, _ = run_corrected(c, out=out0, o=o0, ow=ow0)
    codes = CORE.status_codes()
    assert int(rep["status"]) == int(codes["OK"])
    assert rep["heap_pushes"] == 0 and rep["heap_pops"] == 0
    # Zero destinations: searches run, adjust writes nothing.
    d0 = np.empty((0, 2), dtype=np.int64)
    dw0 = np.empty((0, 2), dtype=np.float64)
    outd = np.zeros((c["o"].shape[0], 0), dtype=np.float64)
    rep, _ = run_corrected(c, out=outd, d=d0, dw=dw0)
    assert int(rep["status"]) == int(codes["OK"])
    assert rep["rows_scanned"] > 0


def test_engagement_counters_hand_counted():
    """Fail-sentinel: the engagement counters are the NATIVE core's
    observable work, hand-counted on the tiny graph (3 nodes, arc chain
    0->1->2): the discarded initial pop, the single start-seed push (end
    seed 9e9 fails the strict < cutoff gate), one scanned row, one
    eligible pair, the leaf degree gate suppressing the second push.
    A shim that claims engagement without doing this exact work fails
    here; combined with the absence tests (native_engaged False on
    fallback) any fake-engagement path is caught."""
    # 4 nodes: 0->2 (1.0), 1->3 (1.0).  Start node 0 (seed 0.0), end
    # node 1 (seed 9e9).  Node 2 is reached (label 1.0) but has degree 0,
    # so the leaf gate suppresses its push; node 3 is unreached and keeps
    # the sentinel 1.0 + 10.0 bit-exactly.
    ptr = np.array([0, 1, 2, 2, 2], dtype=np.int64)
    vec = np.array([2, 3], dtype=np.int64)
    wts = np.array([1.0, 1.0], dtype=np.float64)
    flag = np.array([True, True])
    o = np.array([[0, 1]], dtype=np.int64)
    ow = np.array([[0.0, 9e9]], dtype=np.float64)
    d = np.empty((0, 2), dtype=np.int64)
    dw = np.empty((0, 2), dtype=np.float64)
    out_l = np.zeros((1, 0), dtype=np.float64)
    rep = CORE.run_legacy("a3", ptr, vec, wts, flag, o, ow, d, dw,
                          10.0, 1, None, out_l)
    assert rep["heap_pushes"] == 1      # start seed only: 9e9 !< 10
    assert rep["heap_pops"] == 2        # discarded initial pop + row pop
    assert rep["rows_scanned"] == 1     # == heap_pops - 1 (the discard)
    assert rep["eligible_collected"] == 1
    assert rep["first_incomplete_origin"] == 2**63 - 1
    # Labels behind the counters (search_labels, same core, same law).
    labels = CORE.search_labels_legacy(
        o[0], ow[0], ptr, vec, wts, flag, 10.0, "a3", 0)
    assert bits(labels) == bits(
        np.array([0.0, 9e9, 1.0, 11.0], dtype=np.float64))
    # node 3: sentinel 1.0 + cutoff, one binary64 RN add.
    assert bits(labels[3:]) == bits(np.array([1.0]) + 10.0)
    assert rep["status"] == 0


def test_abi_version_and_fingerprint_reported():
    from urban_network_analysis.backends.native import EXPECTED_ABI_VERSION
    assert int(CORE.abi_version()) == EXPECTED_ABI_VERSION
    fp = str(CORE.build_fingerprint())
    assert "-ffp-contract=off" in fp and "-fno-fast-math" in fp
    assert "-march" not in fp  # baseline dispatch, recorded honestly
    isa = CORE.isa_report()
    assert "baseline" in isa  # the dispatch level is reported, not acted on


def test_cancel_flag_type_enforced_by_route():
    """A foreign cancel object never crosses into the released-GIL
    region: the route refuses it with TypeError (binding type check is
    the backstop)."""
    c = small_case()
    args = (c["o"], c["ow"], c["ptr"], c["vec"], c["wts"], c["flag"],
            c["d"], c["dw"], c["dwt"], 0.02, 5.0, 8.0, 0.3, "exponential",
            np.array([0.9, 0.5, 0.25, 0.1]), c["cutoff"])
    with pytest.raises(TypeError, match="cancel"):
        route_scope_access("una_legacy", *args, cancel=object())
    with pytest.raises((TypeError, RuntimeError)):
        route_scope_access("corrected_v1", *args, cancel="not-a-flag")
