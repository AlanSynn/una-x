"""A3 mutant battery (proof.md section 7, T2).

* no_cutoff_init must DIVERGE from the real tail-free kernel (the
  suite's power to catch init accidents);
* tail_restored must be BYTE-EQUAL on [0, V) for every origin, with
  its tail slots stuck at the sentinel — the executable form of the
  tail-deadness claim (proof.md section 2);
* guard_range_disabled must return True exactly where dropping the
  value-range scan changes the verdict, compared at the guard level
  only (it never drives a kernel).
"""
from __future__ import annotations

import numpy as np
import pytest

import fixtures_a3
import mutant_a3
from harness_a3 import cns, run_tailless


def _tailless_labels(arrays, o_pos):
    case = fixtures_a3.kernel_case(arrays, o_pos)
    labels, pred = run_tailless(cns(), case)
    return labels


def _mutant_labels(arrays, o_pos):
    max_degree = int(np.diff(arrays["adjacency_pointer"]).max())
    eligible_offset = np.empty(max_degree, dtype=np.int64)
    eligible_weight = np.empty(max_degree, dtype=np.float64)
    labels, _ = mutant_a3.no_cutoff_init(
        arrays["o_terminal_idxs"][o_pos],
        arrays["o_terminal_weights"][o_pos],
        arrays["adjacency_pointer"],
        arrays["adjacency_vector"],
        arrays["adjacency_vector_weights"],
        arrays["adjacynct_vector_network_node"],
        arrays["cutoff"],
        eligible_offset,
        eligible_weight,
    )
    return labels


def test_no_cutoff_init_mutant_diverges():
    arrays = fixtures_a3.valid()
    diverged = 0
    for o_pos in range(arrays["o_terminal_idxs"].shape[0]):
        real = _tailless_labels(arrays, o_pos)
        mutated = _mutant_labels(arrays, o_pos)
        assert real.shape == mutated.shape
        if real.tobytes() != mutated.tobytes():
            diverged += 1
    assert diverged > 0, "no-cutoff mutant failed to diverge on any origin"


def test_tail_restored_mutant_is_equivalent_on_head():
    """Restoring the destination tail must change nothing on [0, V):
    the tail slots are never read and never written (proof.md section
    2), which is exactly why removing them is semantics-preserving."""
    arrays = fixtures_a3.valid()
    node_count = arrays["node_count"]
    d_count = arrays["d_count"]
    cutoff = arrays["cutoff"]
    max_degree = int(np.diff(arrays["adjacency_pointer"]).max())
    for o_pos in range(arrays["o_terminal_idxs"].shape[0]):
        real = _tailless_labels(arrays, o_pos)
        eligible_offset = np.empty(max_degree, dtype=np.int64)
        eligible_weight = np.empty(max_degree, dtype=np.float64)
        restored, _ = mutant_a3.tail_restored(
            arrays["o_terminal_idxs"][o_pos],
            arrays["o_terminal_weights"][o_pos],
            arrays["adjacency_pointer"],
            arrays["adjacency_vector"],
            arrays["adjacency_vector_weights"],
            arrays["adjacynct_vector_network_node"],
            arrays["cutoff"],
            d_count,
            eligible_offset,
            eligible_weight,
        )
        assert restored.shape == (node_count + d_count,)
        assert np.ascontiguousarray(restored[:node_count]).tobytes() \
            == np.ascontiguousarray(real).tobytes(), \
            f"tail-restored head diverged at origin {o_pos}"
        assert_array_tail = np.full(
            d_count, 1.0 + cutoff, dtype=restored.dtype)
        assert np.ascontiguousarray(restored[node_count:]).tobytes() \
            == assert_array_tail.tobytes(), \
            f"tail-restored tail touched at origin {o_pos}"


def test_guard_range_disabled_mutant_verdicts():
    ns = cns()
    oob = fixtures_a3.oob_value()
    real_oob = ns._a3_tail_admits(
        oob["d_terminal_idxs"], oob["d_count"], oob["node_count"])
    mutant_oob = mutant_a3.guard_range_disabled(
        oob["d_terminal_idxs"], oob["d_count"], oob["node_count"])
    assert real_oob is False and mutant_oob is True

    valid = fixtures_a3.valid()
    real_ok = ns._a3_tail_admits(
        valid["d_terminal_idxs"], valid["d_count"], valid["node_count"])
    mutant_ok = mutant_a3.guard_range_disabled(
        valid["d_terminal_idxs"], valid["d_count"], valid["node_count"])
    assert real_ok is True and mutant_ok is True

    extra = fixtures_a3.extra_row()
    real_shape = ns._a3_tail_admits(
        extra["d_terminal_idxs"], extra["d_count"], extra["node_count"])
    mutant_shape = mutant_a3.guard_range_disabled(
        extra["d_terminal_idxs"], extra["d_count"], extra["node_count"])
    assert real_shape is False and mutant_shape is False  # shape scan kept
