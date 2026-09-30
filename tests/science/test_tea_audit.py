"""BUG-TURN-ELEV-AUDIT focused audit, Engines-metric portion (SCIENCE).

Executed characterizations for the audit areas that live in
SCIENCE-owned Engines code (dossier 02 #6).  Each confirmed
sub-finding carries its own bug ID; the remaining audit areas are
routed to their owning tasks in the S2 evidence disposition table
(turn/elevation partial-edge formulas, demand normalization and
conservation, parallel-edge attribution, stale obstacle/observer
state, per-row workflow Network_File changes).

Findings audited here:
- BUG-TEA-KNN-TIE-ORDER: legacy KNN sorts with unstable introsort
  (np.argsort default); which tied destination pairs with which
  coefficient is an undeclared implementation artifact.
- BUG-TEA-NEGATIVE-DECAY-OVERFLOW: legacy accepts negative decay
  coefficients silently; exp() overflows to inf and the accessibility
  surface becomes non-finite with no error.
- BUG-TEA-KNN-DEAD-PARAMS: the kernel's knn_gravity_beta /
  knn_gravity_logistic_midpoint / knn_gravity_growth_rate parameters
  are never read -- the KNN decay silently reuses the gravity
  parameters (latent in the integrated driver, which wires them equal).
- TEA-INV-EMPTY-ARRAYS (invariant holds, pinned): empty in-scope
  destination sets and empty knn_weights produce exactly-defined
  graceful outputs.

All fixtures terminate deterministically (no supervision needed).
"""
from __future__ import annotations

import struct

import numpy as np
import pytest

pytestmark = pytest.mark.science

# Legacy kernel imported at RUN time (see test_bug_triples import note:
# module-level engine imports pollute os.environ at collection, ahead of
# the platform_geometry facade-cleanliness test).
from urban_network_analysis.reference import errors            # noqa: E402
from urban_network_analysis.reference.accessibility import (   # noqa: E402
    corrected_reach_gravity_knn,
)


def bits(x) -> int:
    return struct.unpack("<Q", struct.pack("<d", float(x)))[0]


@pytest.fixture(scope="module")
def legacy_A():
    from urban_network_analysis.Engines import Accessibility as A
    return A


def _legacy_kernel(A, d, w, coef, *, beta=0.0, growth=0.0, knn_beta=0.0,
                   knn_growth=0.0, decay="none", plateau=0.0, midpoint=0.0):
    """Call the legacy kernel with every parameter explicit (the
    signature has 13 slots; the audit depends on which ones the body
    actually reads)."""
    return A.reach_gravity_knn_access(
        np.asarray(d, dtype=np.float64),
        np.asarray(w, dtype=np.float64),
        100.0,
        beta, plateau, midpoint,        # gravity_beta / plateau / midpoint
        plateau,                        # knn_gravity_plateau
        np.asarray(coef, dtype=np.float64),
        knn_beta,                       # knn_gravity_beta
        decay,                          # knn_decay
        midpoint,                       # knn_gravity_logistic_midpoint
        growth,                         # gravity_growth_rate
        knn_growth,                     # knn_gravity_growth_rate
    )


class TestTeaKnnTieOrder:
    """BUG-TEA-KNN-TIE-ORDER: coefficient pairing on equal distances
    follows unstable-introsort tie order."""

    def test_pinned_fixture_deviation(self, legacy_A):
        # pinned fixture: seed 20260930, first all-tie draw of the S2
        # audit sweep (n=125 zero distances, weights 1..99 draws)
        rng = np.random.default_rng(20260930)
        n = int(rng.integers(20, 200))
        w = rng.integers(1, 100, size=n).astype(np.float64)
        coef = np.arange(n, dtype=np.float64) + 1.0
        distances = np.zeros(n)

        legacy_knn = float(_legacy_kernel(legacy_A, distances, w, coef)[3])
        stable_knn = float(
            (coef * w[np.argsort(distances, kind="stable")]).sum())

        # OLD-FAILS: legacy value follows the introsort artifact, not
        # the declared ascending-destination-index tie order
        assert legacy_knn != stable_knn
        # CORRECTED-PASSES: corrected uses the stable declared order
        _, _, _, knn = corrected_reach_gravity_knn(
            distances, w, 100.0, 0.0, 0.0, 0.0, 0.0, coef, "none")
        assert bits(knn) == bits(stable_knn)
        # LEGACY-RETAINS: the artifact output pinned as recorded
        assert bits(legacy_knn) == bits(389319.0)
        assert bits(stable_knn) == bits(398846.0)


class TestTeaNegativeDecayOverflow:
    """BUG-TEA-NEGATIVE-DECAY-OVERFLOW: unvalidated decay coefficients
    silently yield non-finite accessibility."""

    def test_legacy_silently_emits_inf(self, legacy_A):
        # OLD-FAILS: beta = -800, distance 10 -> exp(8000) = inf
        out = _legacy_kernel(legacy_A, [10.0], [1.0], [1.0], beta=-800.0,
                             decay="exponential")
        assert float(out[1]) == float("inf")      # gravity_exponential
        assert float(out[3]) == float("inf")      # knn exponential

    def test_corrected_rejects_negative_decay_coefficients(self, legacy_A):
        # CORRECTED-PASSES: typed rejection before any math
        with pytest.raises(errors.ParameterDomainError):
            corrected_reach_gravity_knn(
                np.array([10.0]), np.array([1.0]), 100.0,
                -800.0, 0.0, 0.0, 0.0, np.array([1.0]), "exponential")
        with pytest.raises(errors.ParameterDomainError):
            corrected_reach_gravity_knn(
                np.array([10.0]), np.array([1.0]), 100.0,
                0.0, 0.0, 0.0, -1.0, np.array([1.0]), "logistic")
        with pytest.raises(errors.ParameterDomainError):
            corrected_reach_gravity_knn(
                np.array([10.0]), np.array([1.0]), 100.0,
                float("nan"), 0.0, 0.0, 0.0, np.array([1.0]), "none")

    def test_legacy_retains_inf_bit(self, legacy_A):
        # LEGACY-RETAINS: the pinned silent-inf behavior
        out = _legacy_kernel(legacy_A, [10.0], [1.0], [1.0], beta=-800.0,
                             decay="exponential")
        assert bits(float(out[1])) == bits(float("inf"))


class TestTeaKnnDeadParams:
    """BUG-TEA-KNN-DEAD-PARAMS: knn_gravity_* kernel parameters are
    never read; the KNN decay silently follows the gravity parameters."""

    def test_kernel_ignores_distinct_knn_decay_parameters(self, legacy_A):
        # OLD-FAILS: exponential KNN decay driven by gravity_beta=0.0
        # while knn_gravity_beta=3.0 -- if the declared knn parameter
        # were read, exp(-3*10) would decay the term to ~0
        out = _legacy_kernel(legacy_A, 
            [10.0], [2.0], [1.0], beta=0.0, knn_beta=3.0,
            decay="exponential")
        assert bits(float(out[3])) == bits(2.0)   # undecayed: gravity's 0 won

        # logistic: knn decay driven by gravity_growth_rate=0.0 while
        # knn_gravity_growth_rate=5.0
        out2 = _legacy_kernel(legacy_A, 
            [10.0], [2.0], [1.0], growth=0.0, knn_growth=5.0,
            decay="logistic")
        assert bits(float(out2[3])) == bits(1.0)  # 1 - 1/(1+exp(-0)) = 0.5...

    def test_corrected_single_declared_parameter_set(self, legacy_A):
        # CORRECTED-PASSES: corrected decay responds to its declared
        # coefficients -- exponential with beta=3 decays exp(-3*10)~0
        _, g_exp, _, knn = corrected_reach_gravity_knn(
            np.array([10.0]), np.array([2.0]), 100.0,
            3.0, 0.0, 0.0, 0.0, np.array([1.0]), "exponential")
        assert 0.0 < g_exp < 1e-10
        assert 0.0 < knn < 1e-10

    def test_legacy_retains_dead_param_behavior(self, legacy_A):
        # LEGACY-RETAINS: with wired-equal parameters (the integrated
        # driver's configuration) behavior is unchanged and pinned
        out = _legacy_kernel(legacy_A, [10.0], [2.0], [1.0], beta=0.0, knn_beta=0.0,
                             decay="exponential")
        assert bits(float(out[3])) == bits(2.0)


class TestTeaInvEmptyArrays:
    """TEA-INV-EMPTY-ARRAYS (audit invariant, holds): graceful exact
    outputs for empty in-scope sets and empty knn weights."""

    def test_empty_destinations_all_zeros_legacy_and_corrected(self, legacy_A):
        legacy = _legacy_kernel(legacy_A, [], [], [1.0], beta=0.3, decay="exponential")
        assert [float(x) for x in legacy] == [0.0, 0.0, 0.0, 0.0]
        corrected = corrected_reach_gravity_knn(
            np.array([50.0]), np.array([1.0]), 10.0,
            0.3, 1.0, 2.0, 1.0, np.array([1.0]), "exponential")
        assert bits(corrected[0]) == bits(0.0)
        assert bits(corrected[3]) == bits(0.0)

    def test_empty_knn_weights_zero_knn_only(self, legacy_A):
        legacy = _legacy_kernel(legacy_A, [5.0], [2.0], [], beta=0.3, decay="exponential")
        # reach/gravity computed; knn exactly 0.0
        assert float(legacy[0]) == 2.0
        assert float(legacy[3]) == 0.0
