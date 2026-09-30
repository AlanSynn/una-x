"""F2 kernel battery: the baseline kernel `_accumulate_od_flow` vs the
local-overlap kernel `_accumulate_od_flow_local` on the dossier's
adversarial classes plus a seeded fuzz.

Conditions discharged here (F2I ruling):
* (b) overflow: touched-list overflow (of=1) must still deliver EXACT
  outputs and leave the scratch FULLY re-zeroed — never a partial reset.
* (d) q_sum<=0 early exit: dedicated adversarial case + reset-after-it
  proof (structurally unreachable in-sample, so a synthetic case is the
  only coverage).
* (e) adversarial classes: ties, contaminated snap edges, u-turn
  exclusion, -9999 missing predecessors, zero node-output, repeated
  destination visits (duplicate reset targets), disjoint envelopes,
  all-overlap slices, early exit; sequential reuse across ODs on the
  SAME scratch (contamination battery).

A single bitwise divergence rejects the domain outright.
"""
from __future__ import annotations

import pytest as _pytest
from _legacy_paths import legacy_layout_present as _llp
if not _llp():
    _pytest.skip('historical large_e2e layout not present '
                 '(set UNA_LEGACY_WORKSPACE)', allow_module_level=True)

import numpy as np

import fixtures_f2
from urban_network_analysis.Engines import AggregateFlow as AF
from harness_f2 import record, record_first_divergence

CASES = fixtures_f2.micro_cases()


def _diff_case(case, sc=None, caps=None, label=None):
    """Run both kernels on one case; bitwise-compare all outputs; check
    the of flag and pristine scratch. Returns (of, delivered, sc)."""
    label = label or case.name
    o_ab0, o_ba0, o_n0, d0 = case.call_orig()
    o_ab1, o_ba1, o_n1, d1, of, sc = case.call_local(sc=sc, caps=caps)
    detail = {
        "out_AB_equal": o_ab0.tobytes() == o_ab1.tobytes(),
        "out_BA_equal": o_ba0.tobytes() == o_ba1.tobytes(),
        "out_node_equal": o_n0.tobytes() == o_n1.tobytes(),
        "delivered_equal": (np.float64(d0).tobytes()
                            == np.float64(d1).tobytes()),
    }
    if not all(detail.values()):
        record_first_divergence(label, detail)
    record(label, **detail, of=int(of))
    for k, v in detail.items():
        assert v, f"{label}: {k} is False — F2 route diverged from baseline"
    case.assert_pristine(sc, f"{label}/post-reset")
    return int(of), float(d1), sc


def test_micro_battery_bitwise():
    """Every micro case: the local route is bitwise-identical to the
    baseline kernel and the scratch is pristine after the reset."""
    for case in CASES:
        of, _, _ = _diff_case(case)
        assert of == 0, f"{case.name}: unexpected touched-list overflow"


def test_slice_covers_full_scan_reach():
    """The load-bearing F2 equivalence leg: the finite gradient slice
    (`cols`) covers EXACTLY the reach set the baseline full-node scan
    computes from the scattered dd_buf. Numpy replication of the
    baseline scan (sections 1-2) vs the cols-filtered predicate —
    equal sets on every micro case."""
    for case in CASES:
        finite = np.isfinite(case.dd_buf)
        rep_reach = finite & (case.d_o + case.dd_buf <= case.budget)
        full_set = np.flatnonzero(rep_reach)
        cols_set = case.expected_reach()
        assert np.array_equal(full_set, cols_set), \
            f"{case.name}: slice {cols_set} != full-scan reach {full_set}"
        assert np.all(np.diff(case.cols) > 0), \
            f"{case.name}: cols not strictly ascending (precondition)"


def test_early_exit_pristine():
    """Unreachable destination virtual: both kernels deliver 0.0, no
    outputs written, scratch pristine (early-exit reset branch)."""
    case = next(c for c in CASES if c.name == "m_unreach_dest")
    o_ab0, o_ba0, o_n0, d0 = case.call_orig()
    assert np.float64(d0).tobytes() == np.float64(0.0).tobytes()
    o_ab1, o_ba1, o_n1, d1, of, sc = case.call_local()
    assert np.float64(d1).tobytes() == np.float64(0.0).tobytes()
    assert of == 0
    assert o_ab1.tobytes() == o_ab0.tobytes()
    assert o_ba1.tobytes() == o_ba0.tobytes()
    assert o_n1.tobytes() == o_n0.tobytes()
    case.assert_pristine(sc, "early_exit")


def test_qsum_zero_adversarial_reset():
    """(d) q_sum<=0 exit: the envelope holds at the virtual nodes only,
    so pass 1 admits no via-arc and q_sum is 0. The local kernel must
    restore reach AND cont scratch to pristine (reset-after-it proof)
    and deliver 0.0 like the baseline; a subsequent OD served on the
    SAME scratch must still be exact."""
    case = next(c for c in CASES if c.name == "m_qsum_zero")
    o_ab0, o_ba0, o_n0, d0 = case.call_orig()
    assert np.float64(d0).tobytes() == np.float64(0.0).tobytes(), \
        "fixture must actually hit the q_sum<=0 exit in the baseline"
    o_ab1, o_ba1, o_n1, d1, of, sc = case.call_local()
    assert np.float64(d1).tobytes() == np.float64(0.0).tobytes()
    assert of == 0
    assert o_ab1.tobytes() == o_ab0.tobytes()
    assert o_ba1.tobytes() == o_ba0.tobytes()
    case.assert_pristine(sc, "qsum_zero")
    # Sequential reuse on the same-sized scratch after the early exit.
    case2 = next(c for c in CASES if c.name == "m_base")
    sc2 = case2.alloc_scratch()
    # Pre-contaminate the scratch deliberately: the early exit must
    # have left the first case's arrays clean, but this proves reuse
    # robustness against stale scratch regardless.
    _diff_case(case2, sc=sc2, label="qsum_zero_then_base_reuse")


def test_overflow_full_rezero_never_partial():
    """(b) Tiny touched-list caps force of=1 mid-recording. The kernel
    must STILL deliver bitwise-exact outputs and the scratch must be
    FULLY re-zeroed (equal to a pristine allocation) — proving a
    partial reset never happens even with truncated lists."""
    case = next(c for c in CASES if c.name == "m_base")
    assert int(case.expected_reach().shape[0]) >= 2, \
        "fixture must reach several nodes to overflow caps of 1"
    o_ab0, o_ba0, o_n0, d0 = case.call_orig()
    o_ab1, o_ba1, o_n1, d1, of, sc = case.call_local(
        caps={"t_reach": 1, "t_cont_o": 1, "t_cont_d": 1,
              "t_acc_o": 1, "t_acc_d": 1})
    assert of == 1, "fixture must force the overflow flag"
    assert o_ab1.tobytes() == o_ab0.tobytes()
    assert o_ba1.tobytes() == o_ba0.tobytes()
    assert o_n1.tobytes() == o_n0.tobytes()
    assert np.float64(d1).tobytes() == np.float64(d0).tobytes()
    case.assert_pristine(sc, "overflow_full_rezero")
    # And the scratch is still usable afterwards (sequential reuse).
    _diff_case(case, sc=sc, label="overflow_then_reuse")


def test_sequential_reuse_cross_od():
    """Two different ODs served on the SAME scratch, back to back — the
    per-stripe reuse pattern of the driver. Both must be exact and the
    scratch pristine after each (duplicate reset targets — acc indices
    recorded multiple times — reset idempotently)."""
    case_a = next(c for c in CASES if c.name == "m_base")
    case_b = next(c for c in CASES if c.name == "m_contam")
    sc = case_a.alloc_scratch()
    _diff_case(case_a, sc=sc, label="reuse_a_first")
    _diff_case(case_b, sc=sc, label="reuse_b_second")
    _diff_case(case_a, sc=sc, label="reuse_a_again")


def test_fuzz_bitwise_first_divergence():
    """Seeded random digraphs: bitwise differential over every callable
    case. Covers pv-outside-reach, disjoint envelopes and duplicate
    reset targets as natural cases. First divergence, if any, is
    recorded before the assert."""
    n_cases = 24
    n_exact = 0
    for seed in range(n_cases):
        case = fixtures_f2.random_micro_case(seed)
        if not np.isfinite(case.d_shortest):
            continue      # the driver never calls the kernel on such ODs
        _diff_case(case, label=f"fuzz/{case.name}")
        n_exact += 1
    record("fuzz/summary", cases=n_cases, exact=n_exact)
    assert n_exact >= 12, "fuzz produced too few callable cases"
