"""K0 twin: bit-exact by construction vs the fold's np.where prologue.

The compaction is the association anchor for every downstream reduction, so
ascending order and the closed-radius boundary are contractual here."""

import numpy as np

from gpu.kernels.host_twins import k0_compact


def test_matches_numpy_where_basic():
    d = np.array([3.0, 1.0, 2.0, 0.5], dtype=np.float64)
    idx, n = k0_compact(d, 2.0)
    expected = np.where(d <= 2.0)[0]
    assert idx.dtype == np.int64
    assert n == expected.shape[0]
    assert np.array_equal(idx, expected)
    assert np.array_equal(idx, [1, 2, 3])


def test_closed_radius_boundary_is_included():
    d = np.array([1.0, 2.0, 2.0000000000000004], dtype=np.float64)
    idx, n = k0_compact(d, 2.0)
    assert n == 2 and np.array_equal(idx, [0, 1])


def test_nan_excluded_by_compare_semantics():
    d = np.array([np.nan, 1.0, np.nan, 0.0], dtype=np.float64)
    idx, n = k0_compact(d, 1e9)
    assert n == 2 and np.array_equal(idx, [1, 3])


def test_inf_excluded_below_any_finite_cutoff():
    d = np.array([np.inf, 1.0, -np.inf], dtype=np.float64)
    idx, n = k0_compact(d, 10.0)
    assert n == 2 and np.array_equal(idx, [1, 2])  # -inf <= 10 keeps


def test_empty_row_and_all_out():
    idx, n = k0_compact(np.zeros(0, dtype=np.float64), 1.0)
    assert n == 0 and idx.shape == (0,)
    idx, n = k0_compact(np.array([5.0, 6.0], dtype=np.float64), 1.0)
    assert n == 0 and idx.shape == (0,)


def test_ascending_order_preserved_on_unsorted_input():
    d = np.array([9.0, 0.1, 8.0, 0.2, 7.0, 0.3], dtype=np.float64)
    idx, n = k0_compact(d, 1.0)
    assert np.array_equal(idx, [1, 3, 5])
    assert np.all(np.diff(idx) > 0)
