"""A3 guard semantic probes (proof.md section 7, T3; review pin P2
posture: the firing case AND the refusal cases are exercised, not just
the condition text).

`_a3_tail_admits` is a plain njit dispatcher, so probes call it
directly. `_a1_scope_admits` is an overload-based guard: its stub runs
on any pure-Python call, so it is probed THROUGH an njit wrapper (the
only frame in which the overload resolves).
"""
from __future__ import annotations

import hashlib

import numba as nb
import numpy as np
import pytest

import fixtures_a3
from harness_a3 import cns

# Bind the overload guard ONCE as a plain global: inside nopython the
# @overload registration resolves; the probe wrapper is the only frame
# in which it exists (a pure-Python call would hit the pass stub).
_A1_ADMITS = cns()._a1_scope_admits


@nb.njit
def _a1_admits_probe(adjacency_pointer, adjacency_vector,
                     adjacency_vector_weights, adjacynct_vector_network_node,
                     o_terminal_idxs, o_terminal_weights, cutoff):
    return _A1_ADMITS(
        adjacency_pointer,
        adjacency_vector,
        adjacency_vector_weights,
        adjacynct_vector_network_node,
        o_terminal_idxs,
        o_terminal_weights,
        cutoff,
    )


def _digest(arr):
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def test_tail_admits_firing_case():
    ns = cns()
    arrays = fixtures_a3.valid()
    dti = arrays["d_terminal_idxs"]
    before = _digest(dti)
    assert ns._a3_tail_admits(dti, arrays["d_count"], arrays["node_count"]) is True
    assert _digest(dti) == before  # read-only input (T5 for the guard)


def test_tail_admits_boundary_values():
    ns = cns()
    arrays = fixtures_a3.boundary_values()
    assert ns._a3_tail_admits(
        arrays["d_terminal_idxs"], arrays["d_count"], arrays["node_count"]) is True


def test_tail_admits_zero_destinations_vacuous():
    ns = cns()
    arrays = fixtures_a3.zero_destinations()
    assert ns._a3_tail_admits(
        arrays["d_terminal_idxs"], 0, arrays["node_count"]) is True


def test_tail_admits_refusal_value_equals_node_count():
    ns = cns()
    arrays = fixtures_a3.oob_value()
    assert ns._a3_tail_admits(
        arrays["d_terminal_idxs"], arrays["d_count"], arrays["node_count"]) is False


def test_tail_admits_refusal_negative_value():
    ns = cns()
    arrays = fixtures_a3.negative_value()
    assert ns._a3_tail_admits(
        arrays["d_terminal_idxs"], arrays["d_count"], arrays["node_count"]) is False


def test_tail_admits_refusal_extra_row_shape():
    ns = cns()
    arrays = fixtures_a3.extra_row()
    assert ns._a3_tail_admits(
        arrays["d_terminal_idxs"], arrays["d_count"], arrays["node_count"]) is False


def test_tail_admits_refusal_wrong_width():
    ns = cns()
    arrays = fixtures_a3.valid()
    wide = np.zeros((arrays["d_count"], 3), dtype=np.int64)
    assert ns._a3_tail_admits(wide, arrays["d_count"], arrays["node_count"]) is False


def test_tail_admits_refusal_one_dimensional():
    ns = cns()
    arrays = fixtures_a3.valid()
    flat = arrays["d_terminal_idxs"].reshape(-1)
    assert ns._a3_tail_admits(flat, arrays["d_count"], arrays["node_count"]) is False


def test_tail_admits_refusal_d_count_mismatch():
    ns = cns()
    arrays = fixtures_a3.valid()
    assert ns._a3_tail_admits(
        arrays["d_terminal_idxs"], arrays["d_count"] - 1,
        arrays["node_count"]) is False


def test_a1_guard_still_refuses_its_domain():
    """The A1 overload guard must keep refusing the non-ordered input
    (through the njit wrapper — the resolution frame where the overload
    exists) and keep admitting the valid graph."""
    arrays_bad = fixtures_a3.a1_refusal()
    admitted, max_degree = _a1_admits_probe(
        arrays_bad["adjacency_pointer"],
        arrays_bad["adjacency_vector"],
        arrays_bad["adjacency_vector_weights"],
        arrays_bad["adjacynct_vector_network_node"],
        arrays_bad["o_terminal_idxs"],
        arrays_bad["o_terminal_weights"],
        arrays_bad["cutoff"],
    )
    assert not admitted

    arrays_ok = fixtures_a3.valid()
    admitted, max_degree = _a1_admits_probe(
        arrays_ok["adjacency_pointer"],
        arrays_ok["adjacency_vector"],
        arrays_ok["adjacency_vector_weights"],
        arrays_ok["adjacynct_vector_network_node"],
        arrays_ok["o_terminal_idxs"],
        arrays_ok["o_terminal_weights"],
        arrays_ok["cutoff"],
    )
    assert admitted
    assert int(max_degree) == int(np.diff(arrays_ok["adjacency_pointer"]).max())
