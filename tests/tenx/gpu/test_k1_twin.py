"""K1 twins: per-element IEEE expressions; twin == the fold's summand
expression bitwise (drift guard).  No cross-element claim is made here;
device exp() bit identity is the gate's job, not this test's."""

import numpy as np

from gpu.kernels.host_twins import (k1_gravity_exponential_coeffs,
                                    k1_gravity_logistic_coeffs,
                                    k1_knn_coeffs)

RNG = np.random.default_rng(11)


def _domain(n=37):
    dn = RNG.random(n) * 400.0
    wn = RNG.random(n) * 3.0 + 1e-6
    return dn, wn


def test_gravity_exponential_twin_matches_expression_bitwise():
    dn, wn = _domain()
    beta, plateau = 0.001, 0.0
    got = k1_gravity_exponential_coeffs(dn, wn, beta, plateau)
    want = wn / np.exp(beta * np.maximum(0, dn - plateau))
    assert got.tobytes() == want.tobytes()


def test_gravity_logistic_twin_matches_expression_bitwise():
    dn, wn = _domain()
    g, plateau, mid = 1.0, 0.0, 500.0
    got = k1_gravity_logistic_coeffs(dn, wn, g, plateau, mid)
    want = wn * (1 - 1 / (1 + np.exp(-g * (dn - plateau - mid))))
    assert got.tobytes() == want.tobytes()


def test_exp_boundary_values_match_expression_including_overflow():
    vals = np.array([0.0, -0.0, 2.0 ** -1074, 1e-308, 1e-16, 1.0, 709.0,
                     -745.0, 1e308], dtype=np.float64)
    wn = np.ones_like(vals)
    # The twin must match the expression exactly, whatever IEEE says
    # (1e308 * 1 / exp(1e308) underflows to 0.0; that IS the pinned bit).
    got = k1_gravity_exponential_coeffs(vals, wn, 1.0, 0.0)
    want = wn / np.exp(np.maximum(0, vals))
    assert got.tobytes() == want.tobytes()


def test_knn_branch_coefficients_match_fold_summands():
    dn_s, wn_s = _domain(13)
    kn = np.full(13, 0.5)
    beta, g, plateau, mid = 0.001, 1.0, 0.0, 500.0
    got = k1_knn_coeffs(dn_s, wn_s, kn, "exponential", beta, g, plateau, mid)
    want = (kn * wn_s) * np.exp(-beta * np.maximum(0, dn_s - plateau))
    assert got.tobytes() == want.tobytes()
    got = k1_knn_coeffs(dn_s, wn_s, kn, "logistic", beta, g, plateau, mid)
    want = (kn * wn_s) * (1 - 1 / (1 + np.exp(-g * (dn_s - plateau - mid))))
    assert got.tobytes() == want.tobytes()
    got = k1_knn_coeffs(dn_s, wn_s, kn, "linear", beta, g, plateau, mid)
    assert got.tobytes() == (kn * wn_s).tobytes()


def test_sign_of_zero_preserved_in_linear_branch():
    wn_s = np.array([0.0, -0.0, 1.0])
    kn = np.array([1.0, 1.0, 1.0])
    got = k1_knn_coeffs(wn_s * 0.0, wn_s, kn, "linear", 0, 0, 0, 0)
    assert got.tobytes() == (kn * wn_s).tobytes()
