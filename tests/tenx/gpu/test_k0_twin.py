"""K0 twin: bit-exact by construction vs the fold's np.where prologue.

The compaction is the association anchor for every downstream reduction, so
ascending order and the closed-radius boundary are contractual here."""

import numpy as np
import pytest

from gpu.kernels.host_twins import k0_compact, kept_prefix


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


# ---------------------------------------------------------------------------
# kept_prefix (G10R F3): the finiteness validation the docstring claimed now
# exists and is enforced — refusal, never coercion.
# ---------------------------------------------------------------------------

def test_kept_prefix_passthrough_returns_views_unchanged():
    d = np.array([1.0, 2.5, 400.0], dtype=np.float64)
    w = np.array([2.0, 0.5, 1.25], dtype=np.float64)
    d2, w2 = kept_prefix(d, w, 400.0)
    assert d2 is d and w2 is w


def test_kept_prefix_length_mismatch_named():
    with pytest.raises(ValueError, match="length mismatch"):
        kept_prefix(np.zeros(3), np.zeros(2), 1.0)


def test_kept_prefix_refuses_non_finite_distances():
    for bad in (np.nan, np.inf, -np.inf):
        d = np.array([1.0, bad], dtype=np.float64)
        with pytest.raises(ValueError, match="non-finite kept distances"):
            kept_prefix(d, np.ones(2), 400.0)


def test_kept_prefix_refuses_non_finite_weights():
    d = np.array([1.0, 2.0], dtype=np.float64)
    with pytest.raises(ValueError, match="non-finite kept weights"):
        kept_prefix(d, np.array([1.0, np.nan]), 400.0)


def test_kept_prefix_kept_plan_domain_is_stricter_than_k0():
    # k0_compact KEEPS -inf (it compares <= cutoff); the kept PLAN's
    # recorded domain is finite, so kept_prefix refuses it.  The two
    # surfaces' contracts are distinct and both are intentional.
    d = np.array([np.inf, 1.0, -np.inf], dtype=np.float64)
    _, n = k0_compact(d, 10.0)
    assert n == 2  # -inf kept by the compaction predicate
    with pytest.raises(ValueError, match="non-finite kept distances"):
        kept_prefix(d, np.ones(3), 10.0)
