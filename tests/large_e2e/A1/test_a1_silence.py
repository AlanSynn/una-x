"""T17: warnings/errors silence (proof.md section 9).

No new warnings on either route after warm-up; refusal and admitted
inputs raise nothing; a kernel-level typing refusal raises the SAME
exception type on both arms.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

import fixtures_a1
from harness_a1 import b0ns, cns, run_integrated


def test_t17_no_new_warnings_on_either_route():
    arrays = fixtures_a1.admitted_base()

    refused = dict(arrays)
    refused["adjacency_vector_weights"] = \
        arrays["adjacency_vector_weights"].copy()
    refused["adjacency_vector_weights"][0] = -0.5

    run_integrated(cns(), arrays)      # warm-up: compile both routes
    run_integrated(cns(), refused)
    run_integrated(b0ns(), arrays)
    run_integrated(b0ns(), refused)

    for tag, arrs in (("admitted", arrays), ("refused", refused)):
        for arm_name, ns in (("cand", cns()), ("b0", b0ns())):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                run_integrated(ns, arrs)
            names = [type(w.message).__name__ for w in caught]
            assert not names, f"t17/{tag}/{arm_name}: {names}"


def test_t17_typing_refusal_same_exception_type():
    """A string cutoff cannot type the ORIGINAL kernel on either arm —
    the candidate must not change the exception type."""
    arrays = fixtures_a1.admitted_base()
    bad_str = dict(arrays)
    bad_str["cutoff"] = "10.0"
    raised = []
    for ns in (b0ns(), cns()):
        with pytest.raises(Exception) as exc:
            run_integrated(ns, bad_str)
        raised.append(type(exc.value).__name__)
    assert raised[0] == raised[1] == "TypingError", raised


def test_t17_value_refusal_and_admitted_raise_nothing():
    """In-domain value refusals and admitted inputs raise nothing on
    either arm. (The out-of-bounds-terminal class is NOT run in-process:
    the fallback kernel writes labels[o_idx] with numba bounds checking
    off — B0's true UB — so it is covered by the bounded-child
    failure-identity test in test_z_ub_terminal.py.)"""
    arrays = fixtures_a1.admitted_base()
    refused = dict(arrays)
    refused["adjacency_vector_weights"] = \
        arrays["adjacency_vector_weights"].copy()
    refused["adjacency_vector_weights"][0] = -0.5
    for ns in (b0ns(), cns()):
        run_integrated(ns, arrays)
        run_integrated(ns, refused)
