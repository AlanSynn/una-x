"""K2: the exact numba tie-law argsort transcription, verified against a
REAL locally compiled numba probe (cache=False; no disk cache, no engine
import).  The tie permutation is contractual: it differs from every numpy
kind on tie-dense inputs (n00/probe_order.txt: 131/200 trials)."""

import numpy as np
import pytest

nb = pytest.importorskip("numba")
from gpu.kernels.host_twins import argsort_tie_law  # noqa: E402


@nb.njit(cache=False)
def _numba_argsort(a):
    return np.argsort(a)


def _tie_battery(trials, seed):
    rng = np.random.default_rng(seed)
    for _ in range(trials):
        n = int(rng.integers(2, 60))
        yield np.round(rng.random(n) * 3, 1)  # heavy ties


def test_tie_permutation_matches_numba_exactly():
    checked = 0
    differing_from_numpy_quicksort = 0
    for a in _tie_battery(128, seed=7):
        got = argsort_tie_law(a)
        want = _numba_argsort(a)
        assert np.array_equal(got, want), (
            f"tie-law mismatch at n={a.shape[0]}: keys={a.tolist()} "
            f"got={got.tolist()} want={want.tolist()}")
        checked += 1
        if not np.array_equal(got, np.argsort(a, kind="quicksort")):
            differing_from_numpy_quicksort += 1
    assert checked == 128
    # The battery must be non-vacuous: numba's tie law actually differs
    # from numpy quicksort on tie-dense inputs (probe_order.txt finding).
    assert differing_from_numpy_quicksort > 0


def test_larger_n_with_ties_matches_numba():
    rng = np.random.default_rng(3)
    a = np.round(rng.random(1000) * 4, 1)
    assert np.array_equal(argsort_tie_law(a), _numba_argsort(a))


def test_trivial_sizes():
    for a in (np.zeros(0, dtype=np.float64),
              np.array([2.0], dtype=np.float64)):
        assert np.array_equal(argsort_tie_law(a), np.arange(a.shape[0]))


def test_all_equal_keys_is_a_valid_permutation_matching_numba():
    # All-equal keys do NOT come out in identity order: numba's partition
    # scrambles ties deterministically.  The contract is numba equality,
    # not identity order.
    a = np.full(40, 1.5)
    got = argsort_tie_law(a)
    assert np.array_equal(got, _numba_argsort(a))
    assert np.array_equal(np.sort(got), np.arange(40))


def test_sorted_and_reverse_inputs_match_numba():
    a = np.arange(50, dtype=np.float64)
    assert np.array_equal(argsort_tie_law(a), _numba_argsort(a))
    assert np.array_equal(argsort_tie_law(a[::-1].copy()),
                          _numba_argsort(a[::-1].copy()))


def test_minus_zero_vs_zero_tie_matches_numba():
    a = np.array([0.0, -0.0, 0.0, -0.0, 0.0])
    assert np.array_equal(argsort_tie_law(a), _numba_argsort(a))


def test_typed_refusals_name_the_offense():
    with pytest.raises(ValueError, match="float64"):
        argsort_tie_law(np.array([1, 2, 3], dtype=np.int64))
    with pytest.raises(ValueError, match="C-contiguous"):
        argsort_tie_law(np.ones(6, dtype=np.float64)[::2])
