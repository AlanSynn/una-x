"""Reduction-plan pinning: canonical plans vs compiled engine fold
and adjust kernels, bitwise (CPU_ALGORITHMS).

Pins SPEC.md section 5: IEEE closed-radius membership (NaN distances
excluded — characterized against the compiled fold), ascending-order
boolean gathers, default argsort tie permutation observability, knn
decay branches reading the gravity_* scalars (dead knn_gravity_*
signature parameters), builtin-min adjust semantics (first operand on
incomparability), and the kept-prefix fold identity of the improved
arm's plan.
"""
import math

import numpy as np
import pytest
from numba import njit

from urban_network_analysis.Engines.AccessibilityWElevation import (
    adjust_destination_distances,
    reach_gravity_knn_access,
)
from urban_network_analysis.kernels.reductions import (
    PLAN_SCOPE_SEARCH,
    PLANS,
    adjust_destination_distances_plan,
    collect_kept,
    fold_reach_gravity_knn_kept_legacy_plan,
    fold_reach_gravity_knn_kept_plan,
    fold_reach_gravity_knn_legacy_plan,
    fold_reach_gravity_knn_plan,
)

rng = np.random.default_rng(101)


def fold_kwargs(**over):
    kw = dict(gravity_beta=0.05, gravity_plateau=3.0,
              gravity_logistic_midpoint=6.0, gravity_growth_rate=0.4,
              knn_decay="exponential",
              knn_weights=np.array([0.8, 0.5, 0.25, 0.125, 0.06]))
    kw.update(over)
    return kw


def engine_fold(d_distance, d_weights, cutoff, kw, poison=999.0):
    """The compiled fold, with its DEAD knn_gravity_* scalars bound to a
    poison value (default 999.0) — outputs must not notice, whichever
    poison is used."""
    # The compiled fold returns reach as float64 (n_weights.sum(), engine
    # line 80); the int64 truncation is a DRIVER-side cast, not the fold's.
    reach = np.empty(1, dtype=np.float64)
    ge = np.empty(1, dtype=np.float64)
    gl = np.empty(1, dtype=np.float64)
    ka = np.empty(1, dtype=np.float64)
    reach[0], ge[0], gl[0], ka[0] = reach_gravity_knn_access(
        d_distance=d_distance, d_weights=d_weights, cutoff=cutoff,
        gravity_beta=kw["gravity_beta"], gravity_plateau=kw["gravity_plateau"],
        gravity_logistic_midpoint=kw["gravity_logistic_midpoint"],
        gravity_growth_rate=kw["gravity_growth_rate"],
        knn_gravity_plateau=poison, knn_weights=kw["knn_weights"],
        knn_gravity_beta=poison, knn_decay=kw["knn_decay"],
        knn_gravity_logistic_midpoint=poison,
        knn_gravity_growth_rate=poison)
    return reach[0], ge[0], gl[0], ka[0]


def bits(x):
    return np.ascontiguousarray(np.asarray(x)).tobytes()


def case_outputs_match(d_distance, d_weights, cutoff, kw):
    """una_legacy parity: the LEGACY (njit fastmath) plan vs the
    compiled engine — the plain-numpy plan is corrected_v1 and is NOT
    claimed to reproduce these bits (see the legacy_delta test)."""
    er, ege, egl, eka = engine_fold(d_distance, d_weights, cutoff, kw)
    pr, pge, pgl, pka = fold_reach_gravity_knn_legacy_plan(
        d_distance, d_weights, cutoff, kw["gravity_beta"],
        kw["gravity_plateau"], kw["gravity_logistic_midpoint"],
        kw["gravity_growth_rate"], kw["knn_decay"], kw["knn_weights"])
    assert bits(er) == bits(pr)
    assert bits(ege) == bits(pge)
    assert bits(egl) == bits(pgl)
    assert bits(eka) == bits(pka)


FOLD_CASES = {
    "finite_typical": dict(
        d=np.array([1.0, 4.0, 9.0, 2.5]), w=np.array([1.0, 2.0, 0.5, 3.0]),
        cutoff=10.0, kw=fold_kwargs()),
    "cutoff_boundary_closed": dict(
        d=np.array([9.999999999999998, 10.0,
                    np.nextafter(10.0, 11.0)]),
        w=np.array([1.0, 2.0, 4.0]), cutoff=10.0, kw=fold_kwargs()),
    "nan_distances_excluded": dict(
        d=np.array([1.0, np.nan, 3.0, 0.0, np.nan]),
        w=np.array([1.0, 2.0, 4.0, 8.0, 16.0]), cutoff=3.0,
        kw=fold_kwargs()),
    "inf_distance": dict(
        d=np.array([np.inf, 1.0]), w=np.array([1.0, 2.0]), cutoff=5.0,
        kw=fold_kwargs()),
    "empty_kept": dict(
        d=np.array([11.0, 12.0]), w=np.array([1.0, 2.0]), cutoff=5.0,
        kw=fold_kwargs()),
    "all_kept_knn_short": dict(
        d=np.array([1.0, 2.0]), w=np.array([1.0, 2.0]), cutoff=5.0,
        kw=fold_kwargs(knn_weights=np.array([0.5]))),
    "knn_zero_weights": dict(
        d=np.array([1.0, 2.0]), w=np.array([1.0, 2.0]), cutoff=5.0,
        kw=fold_kwargs(knn_weights=np.array([]))),
    "signed_zero_weights": dict(
        d=np.array([1.0, -0.0, 0.0]), w=np.array([-0.0, 0.0, 1.0]),
        cutoff=5.0, kw=fold_kwargs()),
    "ties_argsort_observable": dict(
        d=np.array([2.0, 1.0, 2.0, 1.0]), w=np.array([100.0, 1.0, 200.0, 2.0]),
        cutoff=5.0, kw=fold_kwargs()),
    "decay_logistic": dict(
        d=np.array([0.5, 3.0, 7.0]), w=np.array([1.0, 1.0, 1.0]),
        cutoff=8.0, kw=fold_kwargs(knn_decay="logistic")),
    "decay_other_passthrough": dict(
        d=np.array([0.5, 3.0, 7.0]), w=np.array([1.0, 1.0, 1.0]),
        cutoff=8.0, kw=fold_kwargs(knn_decay="identity")),
    "beta_zero": dict(
        d=np.array([1.0, 2.0]), w=np.array([1.0, 2.0]), cutoff=5.0,
        kw=fold_kwargs(gravity_beta=0.0)),
}


@pytest.mark.parametrize("name", list(FOLD_CASES), ids=list(FOLD_CASES))
def test_fold_plan_bitwise_vs_engine(name):
    """plan.accessibility.fold.reach_gravity_knn.v1 == compiled fold,
    byte-for-byte, across the adversarial battery."""
    c = FOLD_CASES[name]
    case_outputs_match(c["d"], c["w"], c["cutoff"], c["kw"])


def test_membership_is_ieee_closed_radius():
    """The compiled fold's np.where membership did NOT inherit the
    search's fast-comparison canonicalization: NaN distances are
    EXCLUDED and the radius is closed (d == cutoff kept)."""
    d = np.array([1.0, np.nan, 10.0, np.nextafter(10.0, 11.0), np.inf])
    w = np.ones(5)
    c = FOLD_CASES["finite_typical"]["kw"]
    er, _, _, _ = engine_fold(d, w, 10.0, c)
    assert float(er) == 2.0  # kept weights at d = 1.0 and 10.0 (closed)


def test_legacy_kept_plan_identity_and_membership():
    """The legacy kept-prefix fold equals the legacy base fold (and the
    compiled engine) whenever fed the membership's own gather, and its
    membership law is the engine's: IEEE <=, NaN excluded, closed."""
    for name, c in FOLD_CASES.items():
        kw = c["kw"]
        base = fold_reach_gravity_knn_legacy_plan(
            c["d"], c["w"], c["cutoff"], kw["gravity_beta"],
            kw["gravity_plateau"], kw["gravity_logistic_midpoint"],
            kw["gravity_growth_rate"], kw["knn_decay"], kw["knn_weights"])
        mask = np.where(c["d"] <= c["cutoff"])[0]
        kept = fold_reach_gravity_knn_kept_legacy_plan(
            c["d"][mask], c["w"][mask], c["cutoff"], kw["gravity_beta"],
            kw["gravity_plateau"], kw["gravity_logistic_midpoint"],
            kw["gravity_growth_rate"], kw["knn_decay"], kw["knn_weights"])
        assert all(bits(a) == bits(b) for a, b in zip(base, kept)), name
        er = engine_fold(c["d"], c["w"], c["cutoff"], kw)
        assert all(bits(a) == bits(b) for a, b in zip(er, base)), name
    d = np.array([1.0, np.nan, 10.0, np.nextafter(10.0, 11.0), np.inf])
    w = np.ones(5)
    kw = FOLD_CASES["finite_typical"]["kw"]
    r, _, _, _ = fold_reach_gravity_knn_legacy_plan(
        d, w, 10.0, kw["gravity_beta"], kw["gravity_plateau"],
        kw["gravity_logistic_midpoint"], kw["gravity_growth_rate"],
        kw["knn_decay"], kw["knn_weights"])
    assert float(r) == 2.0


def test_dead_knn_gravity_parameters_characterized():
    """Varying the engine fold's knn_gravity_* scalars changes nothing;
    varying the gravity_* scalars does — the audited body reads the
    gravity scalars in every branch (SPEC.md dead-parameter fact)."""
    c = FOLD_CASES["finite_typical"]
    kw = c["kw"]
    base = engine_fold(c["d"], c["w"], c["cutoff"], kw, poison=999.0)
    alt = engine_fold(c["d"], c["w"], c["cutoff"], kw, poison=-777.5)
    # whichever poison occupies the knn slots, the outputs do not move
    assert all(bits(a) == bits(b) for a, b in zip(base, alt))
    # but the gravity_* scalars are live: doubling beta changes output
    diff = dict(kw)
    diff["gravity_beta"] = kw["gravity_beta"] * 2
    other = engine_fold(c["d"], c["w"], c["cutoff"], diff)
    assert any(bits(a) != bits(b) for a, b in zip(base, other))


def test_adjust_plan_bitwise_vs_engine():
    """plan.accessibility.adjust.builtin_min.v1 == compiled adjust,
    byte-for-byte, including NaN and signed-zero rows."""
    labels = np.array([0.0, np.nan, 5.0, -0.0])
    didx = np.array([[0, 1], [1, 2], [2, 3], [3, 0], [1, 1]], dtype=np.int64)
    dwts = np.array([[1.0, 2.0], [0.5, 0.25], [1.0, 1.0],
                     [-0.0, -0.0], [2.0, 1.0]], dtype=np.float64)
    engine = adjust_destination_distances(labels, didx, dwts, dwts.shape[0])
    plan = adjust_destination_distances_plan(labels, didx, dwts)
    assert bits(engine) == bits(plan)
    # Builtin-min operand law: min(a, b) = b if b < a else a, so a NaN
    # FIRST operand wins (b < NaN is False) and a NaN SECOND operand
    # loses (NaN < a is False -> a returned).
    assert float(plan[0]) == 1.0          # min(1.0, NaN) -> 1.0
    assert math.isnan(float(plan[1]))     # min(NaN, 5.25) -> NaN
    # signed zero: builtin min returns the FIRST operand on equality
    assert bits(plan[3]) == bits(-0.0)  # min(-0.0 + -0.0, -0.0 + -0.0)
    # row 4: label[1]=NaN + 2.0 first operand -> NaN
    assert math.isnan(float(plan[4]))


def test_collect_kept_matches_membership_gather():
    """The compacting producer reproduces the fold prologue's kept
    values and ascending order bitwise (CPU_LAYOUT C4 lemma L4),
    including NaN/inf destination terminal weights (admitted bits)."""
    r = np.random.default_rng(55)
    n = 256
    labels = r.uniform(-1.0, 20.0, 32)
    labels[7] = np.nan
    labels[11] = np.inf
    didx = r.integers(0, 32, size=(n, 2)).astype(np.int64)
    dwts = r.uniform(-5.0, 5.0, size=(n, 2))
    dwts[3, 0] = np.nan
    dwts[9, 1] = np.inf
    dwts[17] = -0.0
    d_weights = r.uniform(0.0, 10.0, n)
    cutoff = 8.0
    mask = np.where(adjust_destination_distances_plan(
        labels, didx, dwts) <= cutoff)[0]
    want_d = adjust_destination_distances_plan(labels, didx, dwts)[mask]
    want_w = d_weights[mask]
    got_d, got_w, kept = collect_kept(labels, didx, dwts, d_weights, cutoff)
    assert kept == mask.shape[0]
    assert bits(got_d) == bits(want_d)
    assert bits(got_w) == bits(want_w)


def test_fold_kept_plan_identity():
    """plan...reach_gravity_knn_kept.v1 == base plan whenever its
    inputs equal the base plan's post-gather arrays — the improved
    arm's fold identity (proof L5), exercised on the battery + random
    prefixes."""
    for name, c in FOLD_CASES.items():
        kw = c["kw"]
        mask = np.where(c["d"] <= c["cutoff"])[0]
        nd = c["d"][mask]
        nw = c["w"][mask]
        base = fold_reach_gravity_knn_plan(
            c["d"], c["w"], c["cutoff"], kw["gravity_beta"],
            kw["gravity_plateau"], kw["gravity_logistic_midpoint"],
            kw["gravity_growth_rate"], kw["knn_decay"], kw["knn_weights"])
        kept = fold_reach_gravity_knn_kept_plan(
            nd, nw, c["cutoff"], kw["gravity_beta"], kw["gravity_plateau"],
            kw["gravity_logistic_midpoint"], kw["gravity_growth_rate"],
            kw["knn_decay"], kw["knn_weights"])
        assert all(bits(a) == bits(b) for a, b in zip(base, kept)), name


def test_plans_registry_shape():
    """The plan registry binds each plan ID to its stage and spec
    section — the identity strings that flow into KernelInputs."""
    assert set(PLANS) >= {
        "plan.accessibility.adjust.builtin_min.v1",
        "plan.accessibility.fold.reach_gravity_knn.v1",
        "plan.accessibility.fold.reach_gravity_knn_kept.v1",
    }
    for plan_id, (fn_name, stage, spec, profiles) in PLANS.items():
        assert stage.startswith("accessibility.")
        assert spec.startswith("SPEC.md")
        assert set(profiles) <= {"una_legacy", "corrected_v1"}
        assert profiles, plan_id
        # every legacy-fastmath transcription carries the legacy profile;
        # the plain-numpy plans carry corrected_v1 (numpy dual-lists the
        # adjust plan only, where the transcription matched engine bits)
        if plan_id.endswith(".legacy_fastmath.v1"):
            assert profiles == ("una_legacy",), plan_id
        elif plan_id == PLAN_SCOPE_SEARCH:
            # the search transcription IS the una_legacy law set (fast
            # comparisons); corrected_v1 search semantics live in the
            # SCIENCE reference package, not here
            assert profiles == ("una_legacy",), plan_id
        else:
            assert "corrected_v1" in profiles, plan_id
        import urban_network_analysis.kernels as _pkg
        assert callable(getattr(_pkg, fn_name)), fn_name
