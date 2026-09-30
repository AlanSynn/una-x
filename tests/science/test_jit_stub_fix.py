"""BUG-JIT-DISABLED-STUB fix — dispatch-decision parity unit tests.

The S1 review (finding 3) flagged that the plain-Python guard's cutoff
check was broader than the numba typing gate; both gates now accept
exactly Python int/float and np.int64/np.float64 scalars.  These tests
pin that the two dispatch modes refuse the same exotic-scalar cutoffs
and admit the same ordinary ones (the full diagnostic-mode crash
regression lives in tests/large_e2e/harness/test_identity_guard.py).
"""
from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.science

from urban_network_analysis.Engines._large_access_scratch import (
    _a1_scope_admits,
)


def _good_arrays():
    pointer = np.array([0, 1, 2], dtype=np.int64)
    vector = np.array([1, 0], dtype=np.int64)
    weights = np.array([10.0, 10.0], dtype=np.float64)
    net_node = np.array([True, True], dtype=np.bool_)
    o_idxs = np.array([[0, 1]], dtype=np.int64)
    o_wts = np.array([[0.0, 0.0]], dtype=np.float64)
    return pointer, vector, weights, net_node, o_idxs, o_wts


@pytest.mark.parametrize("cutoff,admitted", [
    (250, True),                       # Python int -> nb.int64
    (250.0, True),                     # Python float -> nb.float64
    (np.int64(250), True),
    (np.float64(250.0), True),
    (True, False),                     # bool -> nb.boolean, refused compiled
    (np.bool_(True), False),
    (np.int32(250), False),            # narrower ints type differently
    (np.float32(250.0), False),
])
def test_cutoff_gate_decision_parity(cutoff, admitted):
    pointer, vector, weights, net_node, o_idxs, o_wts = _good_arrays()
    decision = _a1_scope_admits(
        pointer, vector, weights, net_node, o_idxs, o_wts, cutoff)
    assert decision[0] is admitted or decision[0] == admitted


def test_short_weights_array_refused_not_raised():
    """Review finding 4: a weights array shorter than the vector offset
    count is refused in BOTH dispatch modes (the scan is shared), so
    diagnostic mode cannot raise IndexError and compiled mode cannot
    read out of bounds."""
    pointer, vector, weights, net_node, o_idxs, o_wts = _good_arrays()
    decision = _a1_scope_admits(
        pointer, vector, weights[:1], net_node, o_idxs, o_wts, 100)
    assert decision[0] is False and decision[1] == 0


def test_malformed_inputs_still_refuse_without_raising():
    pointer, vector, weights, net_node, o_idxs, o_wts = _good_arrays()
    # wrong ndim / dtype / shape all refuse silently
    assert _a1_scope_admits(
        pointer.reshape(1, -1), vector, weights, net_node,
        o_idxs, o_wts, 100.0)[0] is False
    assert _a1_scope_admits(
        pointer, vector.astype(np.int32), weights, net_node,
        o_idxs, o_wts, 100.0)[0] is False
    assert _a1_scope_admits(
        pointer, vector, weights, net_node,
        o_idxs[:, :1], o_wts, 100.0)[0] is False
    # NaN / negative cutoff refuse via the value scan
    assert _a1_scope_admits(
        pointer, vector, weights, net_node, o_idxs, o_wts, float("nan"))[0] is False
    assert _a1_scope_admits(
        pointer, vector, weights, net_node, o_idxs, o_wts, -1.0)[0] is False
