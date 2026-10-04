"""C12 characterization: single-output split folds vs the combined
fold, bitwise (campaign una_10x, dossier 05).

The split folds are mechanical extractions of the engine's own fold
source (the splice tool asserted AST-node identity against the real
bytes).  Whether the compiler keeps the same bits for an extraction is
exactly what this matrix measures, across lengths, seeds, decay
branches, tie permutations, membership boundaries, extremes, knn-weight
length classes and layouts.  ONE 1-ULP divergence rejects fine-grained
reuse — there is no tolerance anywhere in this suite.
"""
import numpy as np
import pytest

from conftest import (CASE_ORDER, bits, mk_case, run_all)


def assert_slots(case, combined, splits):
    outs, sp = run_all(case, combined, splits)
    for name, v in sp.items():
        assert bits(v) == bits(outs[name]), (name, repr(v),
                                             repr(outs[name]))
    return outs


def test_live_signature_matches_extractor_order(live_params):
    assert tuple(live_params) == CASE_ORDER


def test_engine_fold_identity_second_derivation(engine_sha, table):
    """The characterization target is the C10 table's engine bytes."""
    def walk(v):
        if isinstance(v, dict):
            for x in v.values():
                yield from walk(x)
        elif isinstance(v, list):
            for x in v:
                yield from walk(x)
        elif isinstance(v, str):
            yield v
    assert engine_sha in set(walk(table))


@pytest.mark.parametrize("n", [0, 1, 2, 3, 8, 64, 256, 257, 1024])
def test_lengths(combined, splits, n):
    assert_slots(mk_case(n=n, seed=n + 1), combined, splits)


@pytest.mark.parametrize("seed", [11, 23, 101, 977])
def test_seeds(combined, splits, seed):
    assert_slots(mk_case(seed=seed), combined, splits)


@pytest.mark.parametrize("decay", ["exponential", "logistic",
                                   "linear"])
def test_decay_branches(combined, splits, decay):
    assert_slots(mk_case(decay=decay), combined, splits)


def test_tie_permutations(combined, splits):
    g = np.random.default_rng(5)
    ties = g.choice(4, 32).astype(np.float64)
    for k in range(3):
        case = mk_case(n=32)
        case["d_distance"] = ties[g.permutation(32)]
        assert_slots(case, combined, splits)


def test_membership_boundary(combined, splits):
    case = mk_case(n=4)
    case["d_distance"] = np.array([6.0, 6.0 + 2.0 ** -40,
                                   5.999999999, 6.0])
    assert_slots(case, combined, splits)


def test_extremes(combined, splits):
    case = mk_case(n=8)
    case["d_distance"] = np.array([0.0, -0.0, 5e-324, 1e308,
                                   7.0, 8.0, 9.0, 1.0])
    assert_slots(case, combined, splits)


def test_inf_cutoff(combined, splits):
    case = mk_case(n=6)
    case["d_distance"] = np.array([np.inf, 1.0, 2.0, 3.0, 4.0, 5.0])
    case["cutoff"] = np.inf
    assert_slots(case, combined, splits)


def test_nan_weight(combined, splits):
    case = mk_case(n=6)
    case["d_weights"] = np.array([np.nan, 1.0, 0.0, -0.0,
                                  1e-300, 1e300])
    outs, sp = run_all(case, combined, splits)
    for name, v in sp.items():
        assert bits(v) == bits(outs[name]), name


def test_knn_weight_length_classes(combined, splits):
    short = mk_case(n=16)
    short["knn_weights"] = np.ones(3)
    assert_slots(short, combined, splits)
    zero = mk_case(n=4)
    zero["knn_weights"] = np.ones(0)
    outs, sp = run_all(zero, combined, splits)
    for name, v in sp.items():
        assert bits(v) == bits(outs[name]), name
    assert bits(sp["knn_access"]) == bits(0.0)  # knn == 0 branch
    long_case = mk_case(n=4)
    long_case["knn_weights"] = np.ones(100)
    assert_slots(long_case, combined, splits)


def test_layouts(combined, splits):
    f = mk_case(n=16)
    f["d_distance"] = np.asfortranarray(f["d_distance"])
    f["d_weights"] = np.asfortranarray(f["d_weights"])
    assert_slots(f, combined, splits)
    g = np.random.default_rng(9)
    big_d = g.uniform(0.0, 10.0, 32)
    big_w = g.uniform(0.5, 2.0, 32)
    s = mk_case(n=16)
    s["d_distance"] = big_d[::2]
    s["d_weights"] = big_w[::2]
    assert_slots(s, combined, splits)


def test_non_contiguous_knn_weights(combined, splits):
    case = mk_case(n=16)
    case["knn_weights"] = np.arange(32, dtype=np.float64)[::2] / 32.0
    assert_slots(case, combined, splits)


def test_decay_specializations_all_sides(combined, splits):
    """knn_decay is a reflected string: every distinct literal is its
    own specialization; the three REAL call-site branches are covered
    on both the combined and the split sides."""
    for decay in ("exponential", "logistic", "linear"):
        case = mk_case(decay=decay, n=40, seed=31)
        assert_slots(case, combined, splits)
