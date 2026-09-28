"""A3 kernel-level battery (proof.md section 7, T1 and T5).

Every comparison is against the COMPILED B0 kernel with
comparator.assert_array_bytes_equal (dtype+shape+bytes; no tolerance).
The B0 kernel returns the full nd vector; the tail-free candidate
returns V entries. The pinned claims:

* tailless[:V] is byte-identical to B0's labels[:V] for every origin;
* B0's tail [V, nd) stays at the untouched typed sentinel 1 + cutoff
  on admitted inputs (the slots A3 removes were never live);
* the kernel writes nothing outside its private vector (input digests
  unchanged across the call).
"""
from __future__ import annotations

import hashlib

import numpy as np
import pytest
from comparator import assert_array_bytes_equal

import fixtures_a3
from harness_a3 import b0ns, cns, run_b0_kernel, run_tailless


KERNEL_VARIANTS = (
    "valid", "boundary_values", "shared_terminals", "zero_destinations",
)


def _digest(arr):
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def _input_digests(case):
    return {
        name: _digest(case[name])
        for name in ("o_terminal_idxs", "o_terminal_weights",
                     "adjacency_pointer", "adjacency_vector",
                     "adjacency_vector_weights",
                     "adjacynct_vector_network_node")
    }


@pytest.mark.parametrize("variant", KERNEL_VARIANTS)
@pytest.mark.parametrize("seed", [20260925, 20260926, 20260927])
def test_tailless_matches_b0_head_and_tail_stays_sentinel(variant, seed):
    arrays = getattr(fixtures_a3, variant)(seed=seed)
    node_count = arrays["node_count"]
    d_count = arrays["d_count"]
    cutoff = arrays["cutoff"]
    for o_pos in range(arrays["o_terminal_idxs"].shape[0]):
        case = fixtures_a3.kernel_case(arrays, o_pos)
        before = _input_digests(case)

        labels_b0, pred_b0 = run_b0_kernel(b0ns(), case)
        labels_a3, pred_a3 = run_tailless(cns(), case)

        assert labels_a3.shape == (node_count,)
        assert_array_bytes_equal(
            labels_a3, labels_b0[:node_count],
            f"{variant}/{seed}/o{o_pos}/head")
        if d_count > 0:
            tail = labels_b0[node_count:]
            assert_array_bytes_equal(
                tail, np.full(d_count, 1.0 + cutoff, dtype=labels_b0.dtype),
                f"{variant}/{seed}/o{o_pos}/tail-sentinel")
        assert_array_bytes_equal(pred_a3, pred_b0,
                                 f"{variant}/{seed}/o{o_pos}/pred")

        for name, digest in _input_digests(case).items():
            assert digest == before[name], f"{variant}/{seed}/o{o_pos}/{name}"


# Note: the private vector's dtype-follows-input clause is pinned by
# the P3 census (the init expression is textually identical to A1's
# except the size term). A dtype-variation probe is NOT runnable here:
# with non-float64 origin weights the shared heap code refuses to type
# in BOTH arms (numba "heap type must be the same as item type"), i.e.
# the float64 admitted typing is the only kernel-runnable domain.
