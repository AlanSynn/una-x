"""Refusal and fallback parity for the sparse route (dossier 02 gates).

Every mutation outside the dossier-02 admitted domain must (a) refuse
BEFORE workspace allocation with the named reason, and (b) produce
bitwise-baseline outputs through the UNCHANGED original driver.  The
two shape truncations marked UB truncate arrays the original driver
indexes without bounds checks (its native behavior is out-of-bounds
reads, not exceptions); for those the gate asserted is the refusal
itself — output comparison against undefined garbage is not a check.
A -0.0 seed is NOT a refusal: the domain retains the negative-zero bit.
ENGINE FINDING (recorded, not patched — S10 F-discipline): the ORIGINAL
driver cannot COMPILE float32 origin-terminal weights (its a3 route
types heappush against the seed dtype; numba TypingError from
_large_access_scratch.py:422), so the ow_float32 case asserts exception
parity: the sparse gate refuses a1_guard, the fallback reproduces the
baseline's own crash, and both raise the same exception type.
"""
import numpy as np
import pytest

from conftest import admitted_case, bit_equal_result, driver_args

MUTATORS = {}
EXPECTED = {}  # name -> (route, refusal, compare_outputs)


def mut(name, route, refusal, compare=True):
    def deco(fn):
        MUTATORS[name] = fn
        EXPECTED[name] = (route, refusal, compare)
        return fn
    return deco


@mut("ow_float32", "exception", "a1_guard", compare=False)
def m(c):
    c["ow"] = c["ow"].astype(np.float32)


@mut("wts_float32", "fallback", "a1_guard")
def m(c):
    c["wts"] = c["wts"].astype(np.float32)


@mut("flag_uint8", "fallback", "a1_guard")
def m(c):
    c["flag"] = c["flag"].astype(np.uint8)


@mut("cutoff_bool", "fallback", "a1_guard")
def m(c):
    c["cutoff"] = True


@mut("cutoff_float32_scalar", "fallback", "a1_guard")
def m(c):
    c["cutoff"] = np.float32(c["cutoff"])


@mut("seed_nan", "fallback", "values")
def m(c):
    w = c["ow"].copy()
    w[0, 1] = np.nan
    c["ow"] = w


@mut("seed_negative", "fallback", "values")
def m(c):
    w = c["ow"].copy()
    w[0, 1] = -0.5
    c["ow"] = w


@mut("seed_inf", "fallback", "values")
def m(c):
    w = c["ow"].copy()
    w[0, 0] = np.inf
    c["ow"] = w


@mut("terminal_cost_nan", "fallback", "values")
def m(c):
    w = c["dw"].copy()
    w[0, 0] = np.nan
    c["dw"] = w


@mut("terminal_cost_negative", "fallback", "values")
def m(c):
    w = c["dw"].copy()
    w[0, 1] = -0.25
    c["dw"] = w


@mut("destination_weight_nan", "fallback", "values")
def m(c):
    w = c["dwt"].copy()
    w[0] = np.nan
    c["dwt"] = w


@mut("destination_weight_negative", "fallback", "values")
def m(c):
    w = c["dwt"].copy()
    w[0] = -1.0
    c["dwt"] = w


@mut("rounded_sentinel_cutoff", "fallback", "sentinel")
def m(c):
    c["cutoff"] = 1.7976931348623157e308


@mut("terminal_costs_narrow", "fallback", "shape", compare=False)
def m(c):
    c["dw"] = np.ascontiguousarray(c["dw"][:, :1])


@mut("destination_weights_short", "fallback", "shape", compare=False)
def m(c):
    c["dwt"] = c["dwt"][:-1].copy()


@mut("weights_noncontiguous", "fallback", "layout")
def m(c):
    big = np.zeros(c["wts"].shape[0] * 2)
    big[::2] = c["wts"]
    c["wts"] = big[::2]


@mut("weights_subclass", "fallback", "layout")
def m(c):
    class Sub(np.ndarray):
        pass
    c["wts"] = c["wts"].view(Sub)


@mut("seed_negative_zero", "sparse", "")
def m(c):
    w = c["ow"].copy()
    w[0, 0] = -0.0
    c["ow"] = w


@pytest.mark.parametrize("name", sorted(MUTATORS))
def test_gate_and_fallback(name, sp, baseline):
    case = admitted_case("base")
    MUTATORS[name](case)
    args = driver_args(case)
    route, refusal, compare = EXPECTED[name]
    if route == "exception":
        with pytest.raises(Exception) as eb:
            baseline(*args)
        with pytest.raises(Exception) as es:
            sp.sparse_scope_access(*args)
        assert type(eb.value) is type(es.value)
        return
    ref = baseline(*args)
    res = sp.sparse_scope_access(*args)
    if route == "fallback":
        assert res.route == sp.ROUTE_FALLBACK
        assert res.refusal == refusal
    else:
        assert res.route == sp.ROUTE_SPARSE
        assert res.refusal == ""
    if compare:
        assert bit_equal_result(res, ref)


def test_read_only_inputs_admitted(sp, baseline):
    """Read-only arrays are never written by the kernel: admitted."""
    case = admitted_case("base")
    for key in ("ptr", "vec", "wts", "flag", "o", "ow", "d", "dw", "dwt"):
        case[key] = case[key].copy()
        case[key].setflags(write=False)
    args = driver_args(case)
    ref = baseline(*args)
    res = sp.sparse_scope_access(*args)
    assert res.route == sp.ROUTE_SPARSE and res.refusal == ""
    assert bit_equal_result(res, ref)


def test_tail_domain_falls_back_to_a1(sp, baseline):
    """Out-of-network destination terminal -> tail refusal -> the
    original driver's a1 route (S10 binding: A1-tail coverage)."""
    case = admitted_case("base")
    d = case["d"].copy()
    d[0, 0] = case["ptr"].shape[0] - 1  # == node_count: out of range
    case["d"] = d
    args = driver_args(case)
    ref = baseline(*args)
    res = sp.sparse_scope_access(*args)
    assert res.route == sp.ROUTE_FALLBACK and res.refusal == "tail"
    assert bit_equal_result(res, ref)


def test_arena_capacity_refusal(sp, baseline):
    """H*(3V+4D+2*max_degree)*8 over the 256 MiB cap -> refuse BEFORE
    allocation; the fallback stays bitwise."""
    case = admitted_case("base")
    import numba
    lanes = int(numba.config.NUMBA_NUM_THREADS)
    n = 8388608 // lanes + 2  # strictly over 8*lanes*4*n = 256 MiB
    case["d"] = np.zeros((n, 2), dtype=np.int64)
    case["dw"] = np.full((n, 2), 0.25)
    case["dwt"] = np.full(n, 0.5)
    args = driver_args(case)
    ref = baseline(*args)
    res = sp.sparse_scope_access(*args)
    assert res.route == sp.ROUTE_FALLBACK and res.refusal == "capacity"
    assert bit_equal_result(res, ref)


def test_high_degree_within_a1_cap_still_refusable(sp, baseline):
    """max_degree far above any sane row buffer is caught by the a1
    scratch cap; parity holds through the fallback.  The case keeps the
    battery's own node count (destination terminals stay in range — a
    smaller graph would make the ORIGINAL driver's compact route read
    out of bounds, which is undefined behavior, not a comparison)."""
    from urban_network_analysis.Engines._large_access_scratch import (
        _A1_SCRATCH_CAP_ELEMS)
    import numba
    lanes = int(numba.config.NUMBA_NUM_THREADS)
    md = _A1_SCRATCH_CAP_ELEMS // lanes + 1
    case = admitted_case("base")
    node_count = case["ptr"].shape[0] - 1
    case["o"] = np.array([[0, 1]], dtype=np.int64)  # in-range terminals
    case["vec"] = np.zeros(md + 2, dtype=np.int64)  # 0->0 self-loops, 1->0
    case["wts"] = np.full(md + 2, 0.5)
    case["flag"] = np.ones(md + 2, dtype=bool)
    ptr = np.full(node_count + 1, md + 2, dtype=np.int64)
    ptr[0] = 0
    ptr[1] = md + 1
    ptr[2] = md + 2
    case["ptr"] = ptr
    args = driver_args(case)
    ref = baseline(*args)
    res = sp.sparse_scope_access(*args)
    assert res.route == sp.ROUTE_FALLBACK and res.refusal == "a1_guard"
    assert bit_equal_result(res, ref)


def test_empty_domains(sp, baseline):
    """Empty O and empty D are legal only as the baseline handles them:
    both take the sparse route here and match bitwise incl. dtypes."""
    case = admitted_case("base")

    empty_o = dict(case)
    empty_o["o"] = case["o"][:0]
    empty_o["ow"] = case["ow"][:0]
    res = sp.sparse_scope_access(*driver_args(empty_o))
    ref = baseline(*driver_args(empty_o))
    assert res.route == sp.ROUTE_SPARSE
    assert res.reach.shape == (0,) and res.reach.dtype == case["o"].dtype
    assert bit_equal_result(res, ref)

    empty_d = dict(case)
    empty_d["d"] = case["d"][:0]
    empty_d["dw"] = case["dw"][:0]
    empty_d["dwt"] = case["dwt"][:0]
    res = sp.sparse_scope_access(*driver_args(empty_d))
    ref = baseline(*driver_args(empty_d))
    assert res.route == sp.ROUTE_SPARSE
    assert bit_equal_result(res, ref)
    assert res.knn_access[0] == 0.0
