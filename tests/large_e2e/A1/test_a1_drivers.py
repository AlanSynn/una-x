"""A1 driver-level battery (proof.md section 9): T2, T3, T11, T15 and
the driver-routing half of T13.

Both engine mirrors (Accessibility + AccessibilityWElevation; PR-N5)
run integrated and od drivers on every oracle engine case; outputs are
compared byte-for-byte against the compiled B0 mirrors. Inputs are
checked unchanged; outputs checked freshly owned; refusal inputs
checked byte-equal to B0 (same fallback path).
"""
from __future__ import annotations

import numpy as np
import pytest

import fixtures
import fixtures_a1
from comparator import assert_array_bytes_equal

from harness_a1 import (INTEGRATED_OUTPUT_NAMES, assert_integrated_bytes,
                        assert_od_bytes, b0ns, cns, engine_case_kernel_arrays,
                        run_integrated, run_od)


@pytest.mark.parametrize(
    "case", fixtures.engine_cases(), ids=lambda c: c.name)
def test_t2_engine_battery_bytes(case):
    arrays = engine_case_kernel_arrays(case)
    ins = {k: np.asarray(v).copy() for k, v in arrays.items()
           if isinstance(v, np.ndarray)}

    assert_integrated_bytes(arrays, f"t2/{case.name}")
    assert_od_bytes(arrays, f"t2/{case.name}")

    # inputs unchanged through both routes
    for k, v in ins.items():
        assert np.asarray(arrays[k]).tobytes() == v.tobytes(), k


def test_t2_outputs_freshly_owned_and_repeatable():
    """Outputs alias no input, and two consecutive candidate calls give
    distinct, byte-identical arrays (no retained state)."""
    case = [c for c in fixtures.engine_cases()
            if c.name == "l0_parallel_cheap_last"][0]
    arrays = engine_case_kernel_arrays(case)
    inputs = [np.asarray(v) for v in arrays.values() if isinstance(v, np.ndarray)]

    out1 = run_integrated(cns(), arrays)
    out2 = run_integrated(cns(), arrays)
    for a, b in zip(out1, out2):
        assert a is not b
        assert_array_bytes_equal(a, b, "t2/ownership/repeat")
    for out in (*out1, *out2):
        for inp in inputs:
            assert not np.shares_memory(out, inp)


@pytest.mark.parametrize(
    "case", fixtures.engine_cases(), ids=lambda c: c.name)
def test_t15_od_path_bytes(case):
    arrays = engine_case_kernel_arrays(case)
    assert_od_bytes(arrays, f"t15/{case.name}")


# ------------------------------------------------- T3 specialization
def test_t3_int32_endpoints_kernel_refusal_byte_equal():
    """Direct int32 kernel-typed driver call: guard refuses (typing
    gate), fallback executes, bytes equal to B0 on the same inputs."""
    case = [c for c in fixtures.engine_cases() if c.name == "l1_int32_endpoints"][0]
    arrays = engine_case_kernel_arrays(case)
    int32_arrays = dict(arrays)
    int32_arrays["adjacency_pointer"] = arrays["adjacency_pointer"].astype(np.int32)
    int32_arrays["adjacency_vector"] = arrays["adjacency_vector"].astype(np.int32)
    int32_arrays["o_terminal_idxs"] = arrays["o_terminal_idxs"].astype(np.int32)
    int32_arrays["d_terminal_idxs"] = arrays["d_terminal_idxs"].astype(np.int32)

    out_b0 = run_integrated(b0ns(), int32_arrays)
    out_c = run_integrated(cns(), int32_arrays)
    for i, name in enumerate(INTEGRATED_OUTPUT_NAMES):
        assert_array_bytes_equal(out_c[i], out_b0[i], f"t3/int32/{name}")


# --------------------------------------- T11 reuse-state independence
def test_int_cutoff_driver_fresh_specialization_byte_equal():
    """PR-N2: integer cutoff takes a fresh driver specialization with
    the guard admitting (cutoff in {float64, int64}); bytes equal."""
    arrays = fixtures_a1.admitted_base()
    arrays = dict(arrays)
    arrays["cutoff"] = 12  # python int -> int64 specialization
    out_b0 = run_integrated(b0ns(), arrays)
    out_c = run_integrated(cns(), arrays)
    for i, name in enumerate(INTEGRATED_OUTPUT_NAMES):
        assert_array_bytes_equal(out_c[i], out_b0[i], f"int_cutoff/{name}")


def test_t11_repeated_calls_and_origin_permutation():
    arrays = fixtures_a1.admitted_base()
    n = arrays["o_terminal_idxs"].shape[0]

    first = run_integrated(cns(), arrays)
    for _ in range(3):
        again = run_integrated(cns(), arrays)
        for a, b in zip(first, again):
            assert_array_bytes_equal(a, b, "t11/repeat")

    # different origin ORDER: rows permute consistently, bytes per row equal
    perm = np.arange(n)[::-1].copy()
    permuted = dict(arrays)
    permuted["o_terminal_idxs"] = arrays["o_terminal_idxs"][perm]
    permuted["o_terminal_weights"] = arrays["o_terminal_weights"][perm]
    out_perm = run_integrated(cns(), permuted)
    for i, name in enumerate(INTEGRATED_OUTPUT_NAMES):
        assert_array_bytes_equal(np.asarray(out_perm[i]),
                                 np.asarray(first[i])[perm],
                                 f"t11/perm/{name}")

    # per-origin kernel rows byte-equal across orders (scratch reuse
    # history cannot leak)
    from harness_a1 import run_kernel, run_scratch_kernel
    for row in range(n):
        kcase = dict(arrays)
        kcase["o_terminal_idxs"] = arrays["o_terminal_idxs"][row:row + 1]
        kcase["o_terminal_weights"] = arrays["o_terminal_weights"][row:row + 1]
        out_b0 = run_kernel(b0ns(), kcase)
        out_c = run_scratch_kernel(cns(), kcase)
        assert_array_bytes_equal(out_c[0], out_b0[0], f"t11/row{row}/labels")
        assert_array_bytes_equal(out_c[1], out_b0[1], f"t11/row{row}/pred")


# ----------------------------- T13 driver routing: refused == B0 bytes
def _refusal_driver_cases():
    base = fixtures_a1.admitted_base()

    neg_w = dict(base)
    neg_w["adjacency_vector_weights"] = base["adjacency_vector_weights"].copy()
    neg_w["adjacency_vector_weights"][0] = -0.5

    nan_w = dict(base)
    nan_w["adjacency_vector_weights"] = base["adjacency_vector_weights"].copy()
    nan_w["adjacency_vector_weights"][0] = float("nan")

    inf_w = dict(base)
    inf_w["adjacency_vector_weights"] = base["adjacency_vector_weights"].copy()
    inf_w["adjacency_vector_weights"][0] = float("inf")

    nan_cut = dict(base)
    nan_cut["cutoff"] = float("nan")

    broken = dict(base)
    broken["adjacency_pointer"] = base["adjacency_pointer"].copy()
    broken["adjacency_pointer"][1] = broken["adjacency_pointer"][2] + 1

    oob = dict(base)
    oob["adjacency_vector"] = base["adjacency_vector"].copy()
    oob["adjacency_vector"][0] = base["adjacency_pointer"].shape[0]

    # (the OOB-terminal class is excluded here: the fallback kernel
    # writes labels[o_idx] with bounds checking off — B0's true UB —
    # so it runs only in the bounded child of test_z_ub_terminal.py.
    # int32 pointer/vector is excluded too: with int64 terminals the
    # ORIGINAL driver itself cannot type (heappush item-type mismatch)
    # on either arm — pinned as exception identity in
    # test_t13_int32_ptr_driver_typing_error_identity below.)

    return dict(negative_weight=neg_w, nan_weight=nan_w, inf_weight=inf_w,
                nan_cutoff=nan_cut, broken_pointer=broken,
                oob_neighbor=oob)


@pytest.mark.parametrize("name", sorted(_refusal_driver_cases().keys()))
def test_t13_refusal_routing_byte_equal(name):
    arrays = _refusal_driver_cases()[name]
    out_b0 = run_integrated(b0ns(), arrays)
    out_c = run_integrated(cns(), arrays)
    for i, oname in enumerate(INTEGRATED_OUTPUT_NAMES):
        assert_array_bytes_equal(out_c[i], out_b0[i], f"t13/{name}/{oname}")
    out_b0 = run_od(b0ns(), arrays)
    out_c = run_od(cns(), arrays)
    assert_array_bytes_equal(np.asarray(out_c), np.asarray(out_b0),
                             f"t13/{name}/od")


def test_t13_admitted_routing_byte_equal_on_base():
    """The valid base must also be byte-equal — this is the admitted
    route (engagement itself is pinned by the NRT child, T14)."""
    arrays = fixtures_a1.admitted_base()
    out_b0 = run_integrated(b0ns(), arrays)
    out_c = run_integrated(cns(), arrays)
    for i, oname in enumerate(INTEGRATED_OUTPUT_NAMES):
        assert_array_bytes_equal(out_c[i], out_b0[i], f"t13/admitted/{oname}")


def test_t13_int32_ptr_driver_typing_error_identity():
    """int32 pointer/vector with int64 terminals refuses at the typing
    gate — and the ORIGINAL driver body cannot type this mix either
    (heappush queue item (float64,int64) vs pushed (float64,int32)).
    The candidate must not change that: identical exception type on
    both arms, raised before any execution."""
    base = fixtures_a1.admitted_base()
    int32 = dict(base)
    int32["adjacency_pointer"] = base["adjacency_pointer"].astype(np.int32)
    int32["adjacency_vector"] = base["adjacency_vector"].astype(np.int32)
    raised = []
    for ns in (b0ns(), cns()):
        with pytest.raises(Exception) as exc:
            run_integrated(ns, int32)
        raised.append(type(exc.value).__name__)
    assert raised[0] == raised[1] == "TypingError", raised
