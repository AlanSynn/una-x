"""P03M: the empirical per-field dependency matrix (campaign una_10x
post_p00, designs/04 "Admission is a dependency proof, not the name of
a parameter" + "Test field-level independence by changing ONE actual
dependency at a time").

Every row perturbs exactly ONE real argument of the LIVE compiled
replay on a REAL admitted C11 case and records which of the four fold
outputs change bitwise.  The PREDICTION column is derived from the
fold source itself (``AccessibilityWElevation.reach_gravity_knn_access``):

* reach reads d_distance <= cutoff membership + d_weights only
  (AWE:75-80);
* gravity_exponential reads membership + d_weights + gravity_beta +
  gravity_plateau (AWE:83);
* gravity_logistic reads membership + d_weights + gravity_growth_rate
  + gravity_plateau + gravity_logistic_midpoint — NOT beta (AWE:85,
  the E6 fact);
* knn_access reads membership + knn_weights shape/crop + argsort
  permutation, then branches on knn_decay: "exponential" reads
  gravity_beta + gravity_plateau (AWE:104); "logistic" reads
  gravity_growth_rate + gravity_plateau + gravity_logistic_midpoint
  (AWE:106); every other value reads no decay scalar (AWE:108).

A row whose empirical outcome contradicts its prediction FAILS the
suite — the matrix is a proof obligation, not a survey.
"""
import numpy as np
import pytest

from conftest import (BUNDLE, METRIC_PARAMS, RESULT_FIELDS, admitted_case,
                      changed_fields, fields_bits, replay_binding,
                      replay_with)

BASELINE_BUNDLE = BUNDLE


def _per_branch_perturbations():
    """One row per (dependency, knn_decay branch) with the prediction
    derived from the fold source lines quoted in the module docstring."""
    rows = []
    decay_branch_knn = {
        "exponential": {"gravity_beta", "gravity_plateau"},
        "logistic": {"gravity_growth_rate", "gravity_plateau",
                     "gravity_logistic_midpoint"},
        "other": set(),
    }
    scalar_effects = {
        "gravity_beta": {"gravity_exponential"},
        "gravity_plateau": {"gravity_exponential", "gravity_logistic"},
        "gravity_logistic_midpoint": {"gravity_logistic"},
        "gravity_growth_rate": {"gravity_logistic"},
    }
    for decay, knn_reads in decay_branch_knn.items():
        for scalar, base_effects in scalar_effects.items():
            # knn_access changes iff the branch's expression reads this
            # scalar (AWE:104/106/108)
            predicted = sorted(set(base_effects)
                               | ({"knn_access"} if scalar in knn_reads
                                  else set()))
            bundle = dict(gravity_beta=BASELINE_BUNDLE[0],
                          gravity_plateau=BASELINE_BUNDLE[1],
                          gravity_logistic_midpoint=BASELINE_BUNDLE[2],
                          gravity_growth_rate=BASELINE_BUNDLE[3])
            perturbed = dict(bundle)
            perturbed[scalar] = {
                "gravity_beta": 2.5,
                "gravity_plateau": 0.5,
                "gravity_logistic_midpoint": 1.25,
                "gravity_growth_rate": 0.4,
            }[scalar]
            rows.append((f"{scalar}|{decay}", decay, bundle, perturbed,
                         tuple(predicted)))
    return rows


def _run_bundle(tr, case, decay, bundle):
    rec = tck_build(tr, case)
    res = replay_with(tr, rec, case, knn_decay=decay,
                      bundle=(bundle["gravity_beta"],
                              bundle["gravity_plateau"],
                              bundle["gravity_logistic_midpoint"],
                              bundle["gravity_growth_rate"]))
    return rec, fields_bits(res)


def tck_build(trm, case):
    from conftest import tck
    return tck.build(trm, case)


@pytest.mark.parametrize("row", _per_branch_perturbations(),
                         ids=lambda r: r[0])
def test_scalar_dependency_per_branch(tr, row):
    _, decay, bundle, perturbed, predicted = row
    case = admitted_case("base")
    _, base_bits = _run_bundle(tr, case, decay, bundle)
    _, pert_bits = _run_bundle(tr, case, decay, perturbed)
    changed = changed_fields(base_bits, pert_bits)
    assert sorted(changed) == list(predicted), (row[0], changed, predicted)


def test_d_weights_value_dependency(tr):
    """All four fields consume the destination metric weights.  The
    perturbation shifts EVERY weight by +2.0 because the reach output
    is materialized through the int64 cast (``reach_dtype``): a sub-
    quantum bump can be absorbed by the truncation and MASK a real
    dependency (P03C conftest documents the same cast behavior)."""
    case = admitted_case("base")
    rec = tck_build(tr, case)
    base = fields_bits(replay_with(tr, rec, case))
    dwt2 = np.asarray(case["dwt"], np.float64) + 2.0
    pert = fields_bits(replay_with(tr, rec, case, d_weights=dwt2))
    assert sorted(changed_fields(base, pert)) == sorted(RESULT_FIELDS)


def test_knn_weights_value_dependency(tr):
    """Only knn_access consumes the KNN coefficient vector's VALUES."""
    case = admitted_case("base")
    rec = tck_build(tr, case)
    base = fields_bits(replay_with(tr, rec, case))
    kw2 = METRIC_PARAMS[4] * 2.0
    pert = fields_bits(replay_with(tr, rec, case, knn_weights=kw2))
    assert changed_fields(base, pert) == ["knn_access"]


@pytest.mark.parametrize("decay", ["exponential", "logistic", "other"])
def test_knn_decay_branch_dependency(tr, decay):
    """Switching ONLY the branch selector changes knn_access alone; the
    other three fields are branch-invariant."""
    case = admitted_case("base")
    rec = tck_build(tr, case)
    base = fields_bits(replay_with(tr, rec, case,
                                   knn_decay="exponential"))
    pert = fields_bits(replay_with(tr, rec, case, knn_decay=decay))
    changed = changed_fields(base, pert)
    expected = [] if decay == "exponential" else ["knn_access"]
    assert changed == expected


def test_cutoff_membership_dependency_cross_trace(tr):
    """Cutoff changes membership (AWE:75), so all four fields depend on
    it — proven across TWO traces (the replay refuses a cutoff that
    mismatches the trace identity, so the perturbed binding builds its
    own trace over the same arrays).  The perturbed cutoff is derived
    from the retained-distance distribution itself: a fixed small
    decrement can retain every candidate (no membership change at
    all), so we cut at the midpoint of the retained distances, which
    provably drops the far candidates."""
    case = admitted_case("base")
    rec_a = tck_build(tr, case)
    res_a = replay_with(tr, rec_a, case)
    db = np.asarray(rec_a.distance_bits)
    cutoff_b = float((db.min() + db.max()) / 2.0)
    case_b = dict(case)
    case_b["cutoff"] = cutoff_b
    rec_b = tck_build(tr, case_b)
    res_b = replay_with(tr, rec_b, case_b)
    assert res_b.route == "trace_replay"
    assert int(rec_b.indptr[-1]) < int(rec_a.indptr[-1])   # membership shrank
    assert sorted(changed_fields(fields_bits(res_a),
                                 fields_bits(res_b))) == \
        sorted(RESULT_FIELDS)


def test_cutoff_dtype_only_refuses_not_reuses(tr):
    """An int cutoff over the same arrays is a DIFFERENT trace identity
    (typed cutoff in the C11 identity): replay must refuse the
    mismatched pair — never silently reuse across the dtype."""
    case = admitted_case("base")
    rec = tck_build(tr, case)
    res = replay_with(tr, rec, case, cutoff=25)
    assert res.route == "trace_replay_refused"
    assert res.refusal == "cutoff_mismatch_with_trace_identity"


# --------------------------------------------------------------------------
# dtype-only / byte-order-only / layout-only REPLAY rows (empirical):
# the identity distinguishes them (identity suite); here we record what
# the fold's bits do — the row SUCCEEDS on whatever the empirical
# outcome is, but the outcome is pinned (a silent change is a defect).
# --------------------------------------------------------------------------
def test_d_weights_dtype_only_replay_empirical(tr):
    """EMPIRICAL, pinned: float32 d_weights replay bitwise-DIFFERS in
    exactly the three float-accumulated fields.  reach is ABSORBED by
    its int64 output cast at this workload scale (the float32 sum
    truncates to the same integer — P03C's conftest documents the cast
    behavior), so its bits coincide; the identity still distinguishes
    the dtype (identity suite), which is the conservative direction:
    no reuse can cross the dtype boundary."""
    case = admitted_case("base")
    rec = tck_build(tr, case)
    base = fields_bits(replay_with(tr, rec, case))
    d32 = np.asarray(case["dwt"], np.float64).astype(np.float32)
    res = replay_with(tr, rec, case, d_weights=d32)
    assert res.route == "trace_replay"
    empir = changed_fields(base, fields_bits(res))
    assert empir == ["gravity_exponential", "gravity_logistic",
                     "knn_access"], f"unpinned empirical outcome: {empir}"


def test_d_weights_byteorder_only_refuses_replay(tr):
    """EMPIRICAL DISCOVERY, pinned: a big-endian d_weights array is not
    a bitwise-identical replay — numba's dispatcher REFUSES the
    non-native byte order outright (``TypingError: Unsupported array
    dtype: >f8``).  In the cohort route this exception lands in the
    sparse gate and the row falls back to the engine's own path
    unchanged.  MetricID v2 independently distinguishes the byte order
    (identity suite); the replay layer's behavior is the refusal
    pinned here."""
    import numba
    case = admitted_case("base")
    rec = tck_build(tr, case)
    dle = np.asarray(case["dwt"], np.float64)
    dbe = dle.astype(dle.dtype.newbyteorder(">"))
    with pytest.raises(numba.core.errors.TypingError):
        replay_with(tr, rec, case, d_weights=dbe)


def test_zero_sign_bit_identity_and_replay_bits(tr):
    """EMPIRICAL 0/-0 record, pinned.  A PURE sign-bit difference (a
    retained destination's weight +0.0 vs -0.0, all other values
    equal) replays to BITWISE IDENTICAL outputs on this case: IEEE-754
    sum semantics absorb the sign of a zero term added to nonzero
    partials (x + (-0.0) == x for x != 0).  MetricID v2 nonetheless
    distinguishes the two inputs (identity suite:
    test_scalar_negative_zero_and_ulp_distinct and the array-bit
    closure), so no reuse can cross the sign boundary — the identity
    is strictly more discriminating than one case's output bits, which
    is the safe direction.  (Replacing a POSITIVE weight with -0.0 is
    a value change, not a 0/-0 probe; all four fields move and the
    value-dependency row covers it.)"""
    case = admitted_case("base")
    rec = tck_build(tr, case)
    retained = np.unique(rec.destination_ids)
    d_plus = np.asarray(case["dwt"], np.float64).copy()
    d_plus[retained[0]] = 0.0
    d_minus = d_plus.copy()
    d_minus[retained[0]] = -0.0
    base = fields_bits(replay_with(tr, rec, case, d_weights=d_plus))
    pert = fields_bits(replay_with(tr, rec, case, d_weights=d_minus))
    assert changed_fields(base, pert) == []
