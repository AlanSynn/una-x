"""P03C L1: scalar/dtype specialization edges on the compiled path
(campaign una_10x post_p00, designs/03 "Arithmetic and compiler
obligations"): cutoff scalar typing and bit identity (-0.0 is not
+0.0, nextafter is a different trace identity), raw scalar forwarding,
weight layout acceptance, and output dtype law.
"""
import numpy as np
import pytest

from conftest import (BUNDLE, METRIC_PARAMS, admitted_case,
                      first_difference, replay_legacy, replay_live,
                      retamper, synth_record, synth_weights)

FIELDS = ("reach", "gravity_exponential", "gravity_logistic",
          "knn_access")


def _ring_case_for(trm):
    """A real producer record with several non-trivial slices."""
    slices = [[0, 3, 4], [1], [], [2, 5], [0, 6], [7], [1, 2, 3]]
    dists = np.arange(sum(len(s) for s in slices),
                      dtype=np.float64) * 0.5
    rec = synth_record(trm, slices, dists, dest_count=8)
    return rec, dict(dwt=synth_weights(8), cutoff=rec.cutoff_value)


@pytest.mark.parametrize("bad_cutoff", [
    25,                    # int64 typing vs the record's float64
    np.float32(25.0),      # float32 typing
    np.nextafter(25.0, np.inf),   # 1 ULP up
    np.nextafter(25.0, 0.0),      # 1 ULP down
])
def test_cutoff_identity_gates_identical_both_arms(tr, legacy,
                                                   bad_cutoff):
    """The typed+bit cutoff gate fires with the same refusal on the
    compiled path as on the frozen bytes (C11-NOTE-3 law re-pinned)."""
    rec, case = _ring_case_for(tr)
    old = replay_legacy(rec, case, cutoff=bad_cutoff)
    new = replay_live(tr, rec, case, cutoff=bad_cutoff)
    assert old.route == "trace_replay_refused"
    assert new.route == "trace_replay_refused"
    assert new.refusal == old.refusal \
        == "cutoff_mismatch_with_trace_identity"
    assert new.folds_done == old.folds_done == 0


def test_cutoff_negative_zero_is_not_positive_zero(tr, legacy):
    """The trace was built with cutoff +0.0; replaying with -0.0 is a
    bit-identity refusal on both arms (IEEE-754 -0.0 != +0.0)."""
    case0 = dict(dwt=synth_weights(8), cutoff=0.0)
    slices = [[0, 3], [1], []]
    dists = np.array([0.0, 0.5, 0.25], dtype=np.float64)
    rec = synth_record(tr, slices, dists, dest_count=8,
                       cutoff=0.0, trace_id="p03c-negzero")
    old = replay_legacy(rec, case0, cutoff=-0.0)
    new = replay_live(tr, rec, case0, cutoff=-0.0)
    assert new.refusal == old.refusal \
        == "cutoff_mismatch_with_trace_identity"
    assert new.route == old.route == "trace_replay_refused"
    # and the +0.0 replay succeeds bitwise on both arms
    old0 = replay_legacy(rec, case0, cutoff=0.0)
    new0 = replay_live(tr, rec, case0, cutoff=0.0)
    assert new0.route == "trace_replay"
    assert first_difference(new0, old0) is None


def test_raw_scalar_forwarding_specializations(tr):
    """Passing the cutoff/gravity scalars as numpy scalars (raw
    forwarding into the compiled driver) specializes identically to
    the frozen bytes — no hidden coercion at the driver boundary."""
    from conftest import legacy_replay_trace
    rec, case = _ring_case_for(tr)
    variants = [
        dict(cutoff=np.float64(rec.cutoff_value)),
        dict(gravity_beta=np.float64(BUNDLE[0]),
             gravity_plateau=np.float64(BUNDLE[1])),
    ]
    for kw in variants:
        args = (rec, case["dwt"],
                kw.get("gravity_beta", BUNDLE[0]),
                kw.get("gravity_plateau", BUNDLE[1]),
                BUNDLE[2], BUNDLE[3], "exponential",
                METRIC_PARAMS[4], kw.get("cutoff", case["cutoff"]))
        old = legacy_replay_trace(*args)
        new = tr.replay_trace(*args)
        assert new.route == "trace_replay", kw
        assert first_difference(new, old) is None, kw


def test_dest_weights_refusals_identical_both_arms(tr, legacy):
    """Layout refusals on the compiled path are byte-identical in
    reason to the frozen bytes."""
    rec, case = _ring_case_for(tr)
    for bad in (case["dwt"].reshape(1, -1), list(case["dwt"]),
                case["dwt"][:-1]):
        old = replay_legacy(rec, case, d_weights=bad)
        new = replay_live(tr, rec, case, d_weights=bad)
        assert old.route == "trace_replay_refused"
        assert new.route == "trace_replay_refused"
        assert new.refusal == old.refusal
        assert new.folds_done == old.folds_done == 0


def test_strided_weights_accepted_bitwise(tr, legacy):
    """A non-contiguous (stride-2) float64 weight vector passes the
    gates; the gather inside the compiled driver produces the same
    fresh writable copy bits as the frozen bytes' np.array(gather)."""
    rec, case = _ring_case_for(tr)
    wide = np.zeros(16, np.float64)
    wide[::2] = case["dwt"]
    strided = wide[::2]
    assert strided.shape[0] == rec.dest_count
    assert not strided.flags.c_contiguous
    old = replay_legacy(rec, case, d_weights=strided)
    new = replay_live(tr, rec, case, d_weights=strided)
    assert new.route == "trace_replay"
    assert first_difference(new, old) is None
    # the strided gather is genuinely the same VALUES as the compact
    # vector (no layout-dependent value drift)
    compact = replay_live(tr, rec, case)
    assert first_difference(new, compact) is None


def test_output_dtype_law(tr):
    """Compiled replay outputs carry exactly the record's declared
    dtypes (reach int64 from the record, metrics float64)."""
    rec, case = _ring_case_for(tr)
    new = replay_live(tr, rec, case)
    assert new.reach.dtype == np.dtype(rec.reach_dtype)
    for f in ("gravity_exponential", "gravity_logistic", "knn_access"):
        assert getattr(new, f).dtype == np.dtype(rec.metric_dtype), f


def test_admitted_base_case_still_three_way(tr, baseline, legacy):
    """Anchor: the real producer's base case stays three-way bitwise
    through the specializations file too (guards against the synthetic
    fixtures drifting from the real trace shape)."""
    from conftest import bits, driver_args, result_equal, tck
    case = admitted_case("base")
    rec = tck.build(tr, case)
    old = replay_legacy(rec, case)
    new = replay_live(tr, rec, case)
    ref = baseline(*driver_args(case))
    assert new.route == "trace_replay"
    assert first_difference(new, old) is None
    assert result_equal(new, ref)
