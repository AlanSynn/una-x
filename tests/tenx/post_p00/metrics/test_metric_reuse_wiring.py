"""P03M-B: the production metric-reuse executor laws (campaign una_10x
post_p00, designs/04 implementation order 2-5, TEST_MATRIX MR01).

Every test drives the REAL production entry point
``_metric_reuse_scratch.row_replay`` over a REAL admitted C11 trace and
compares every served row BITWISE against a direct ``replay_trace`` of
the same binding (no tolerance, no allclose).  The C12 planner pins
planning laws; these pins cover the EXECUTOR: gates-before-hit, fresh
copies, admitted-domain gating, typed agreement, collapse, eviction,
canonical registration and the cohort transport (counters, kill
switch).
"""
import dataclasses

import numpy as np
import pytest

from conftest import (BUNDLE, METRIC_PARAMS, RESULT_FIELDS, admitted_case,
                      fields_bits, replay_binding, replay_with)

ALL_OUTS = RESULT_FIELDS

# The REAL Settings signature of the frozen cohort rows: int
# search_radius / int plateau / int midpoint (Settings types), float
# beta/growth.
COHORT_BUNDLE = (2.5, 0, 500, 0.8)   # int plateau + int midpoint


def tck_build(tr, case):
    from conftest import tck
    return tck.build(tr, case)


def _bits_tuple(outs):
    return {f: np.asarray(o).tobytes()
            for f, o in zip(ALL_OUTS, outs)}


def _bits_res(res):
    return fields_bits(res)


def _row(store, tr, rec, case, counters, **kw):
    """One production row_replay call with the exact binding."""
    from urban_network_analysis.Engines import _metric_identity as midm
    from urban_network_analysis.Engines import _metric_reuse_scratch as mre
    binding = replay_binding(case, **kw)
    midv = midm.compute_metric_id(rec, *binding)
    return mre.row_replay(store, rec, midv, *binding, counters)


def _register(store, rec, outs_tuple, case, **kw):
    """Full-fold canonical registration (the cohort's post-replay
    act): private copies keyed by the row's typed MetricID."""
    from urban_network_analysis.Engines import _metric_identity as midm
    from urban_network_analysis.Engines import _metric_reuse_scratch as mre
    binding = replay_binding(case, **kw)
    midv = midm.compute_metric_id(rec, *binding)
    store.register(midv, rec.trace_id, mre.binding_job(*binding),
                   dict(zip(mre.ALL_OUTS, outs_tuple)))


def _bundle_with(scalar, value):
    b = {"gravity_beta": BUNDLE[0], "gravity_plateau": BUNDLE[1],
         "gravity_logistic_midpoint": BUNDLE[2],
         "gravity_growth_rate": BUNDLE[3]}
    b[scalar] = value
    return (b["gravity_beta"], b["gravity_plateau"],
            b["gravity_logistic_midpoint"], b["gravity_growth_rate"])


def _expected_split(decay, scalar):
    """Expected (copied, folded) outputs when `scalar` changes, read
    from the committed dependency map (the C12 planner's own source)."""
    from urban_network_analysis.Engines import _metric_reuse_scratch as mre
    deps = mre.output_dependencies(decay)
    copied, folded = [], []
    for out in ALL_OUTS:
        if scalar in deps[out]:
            folded.append(out)
        else:
            copied.append(out)
    return copied, folded


# --------------------------------------------------------------------------
# Bundle mode: MetricID v2 hit -> four fresh copies, zero folds
# --------------------------------------------------------------------------
def test_bundle_hit_bitwise_and_fresh(tr):
    from urban_network_analysis.Engines import _metric_reuse_scratch as mre
    case = admitted_case("base")
    rec = tck_build(tr, case)
    store = mre.MetricReuseStore()
    counters = {}
    # Row 1: empty store -> collapse -> None (the full path serves it).
    assert _row(store, tr, rec, case, counters) is None
    assert counters["reuse_collapse_full_folds"] == 1
    ref = replay_with(tr, rec, case)
    _register(store, rec, (ref.reach, ref.gravity_exponential,
                           ref.gravity_logistic, ref.knn_access), case)
    # Row 2: identical binding -> bundle, bitwise-equal fresh arrays.
    served = _row(store, tr, rec, case, counters)
    assert served is not None
    outs, mode = served
    assert mode == mre.REUSE_MODE_BUNDLE
    assert counters["reuse_bundle_hits"] == 1
    assert _bits_res(ref) == _bits_tuple(outs)
    # Freshness: mutating the served arrays cannot corrupt the store.
    for o in outs:
        o[:] = -1
    served3 = _row(store, tr, rec, case, counters)
    assert _bits_res(ref) == _bits_tuple(served3[0])
    assert counters["reuse_bundle_hits"] == 2


def test_bundle_handouts_are_independent_objects(tr):
    from urban_network_analysis.Engines import _metric_reuse_scratch as mre
    case = admitted_case("base")
    rec = tck_build(tr, case)
    store = mre.MetricReuseStore()
    ref = replay_with(tr, rec, case)
    _register(store, rec, (ref.reach, ref.gravity_exponential,
                           ref.gravity_logistic, ref.knn_access), case)
    a, _ = _row(store, tr, rec, case, {})
    b, _ = _row(store, tr, rec, case, {})
    for x, y in zip(a, b):
        assert x is not y
    for x in a:
        x[:] = 123.0
    c, _ = _row(store, tr, rec, case, {})
    for x, z in zip(a, c):
        assert np.asarray(x).tobytes() != np.asarray(z).tobytes()
    for y, z in zip(b, c):
        assert np.asarray(y).tobytes() == np.asarray(z).tobytes()


# --------------------------------------------------------------------------
# Split mode: the frozen-cohort shape (logistic branch, beta varies)
# --------------------------------------------------------------------------
def test_split_frozen_cohort_shape_logistic(tr):
    """THE frozen-cohort shape: all rows share the trace and differ in
    gravity_beta only, knn_decay='logistic' — reach/gravity_logistic/
    knn_access are beta-invariant (E6), gravity_exponential folds.
    3 copies + 1 split fold, bitwise-equal to the row's own full
    replay."""
    from urban_network_analysis.Engines import _metric_reuse_scratch as mre
    case = admitted_case("base")
    rec = tck_build(tr, case)
    store = mre.MetricReuseStore()
    counters = {}
    ref0 = replay_with(tr, rec, case, knn_decay="logistic")
    _register(store, rec, (ref0.reach, ref0.gravity_exponential,
                           ref0.gravity_logistic, ref0.knn_access),
              case, knn_decay="logistic")
    served = _row(store, tr, rec, case, counters, knn_decay="logistic",
                  bundle=_bundle_with("gravity_beta", 2.5))
    assert served is not None
    outs, mode = served
    assert mode == mre.REUSE_MODE_SPLIT
    assert counters["reuse_split_copies"] == 3
    assert counters["reuse_split_folds"] == 1
    ref1 = replay_with(tr, rec, case, knn_decay="logistic",
                       bundle=_bundle_with("gravity_beta", 2.5))
    assert _bits_res(ref1) == _bits_tuple(outs)


def test_int_cohort_signature_split_engages(tr):
    """The REAL Settings signature (int search_radius/plateau/midpoint)
    is inside the admitted domain: split engages and stays bitwise."""
    from urban_network_analysis.Engines import _metric_reuse_scratch as mre
    case = dict(admitted_case("base"))
    case["cutoff"] = 400
    rec = tck_build(tr, case)
    store = mre.MetricReuseStore()
    counters = {}
    ref0 = replay_with(tr, rec, case)
    _register(store, rec, (ref0.reach, ref0.gravity_exponential,
                           ref0.gravity_logistic, ref0.knn_access), case)
    served = _row(store, tr, rec, case, counters, bundle=COHORT_BUNDLE)
    assert served is not None
    outs, mode = served
    assert mode == mre.REUSE_MODE_SPLIT
    ref1 = replay_with(tr, rec, case, bundle=COHORT_BUNDLE)
    assert _bits_res(ref1) == _bits_tuple(outs)


def test_split_row_registration_makes_it_canonical(tr):
    """A split-served row becomes canonical (the cohort registers it):
    a later row with the SAME binding bundle-hits on its identity."""
    from urban_network_analysis.Engines import _metric_reuse_scratch as mre
    case = admitted_case("base")
    rec = tck_build(tr, case)
    store = mre.MetricReuseStore()
    counters = {}
    served2 = _row(store, tr, rec, case, counters,
                   bundle=_bundle_with("gravity_beta", 2.5))
    assert served2 is None              # nothing canonical yet
    ref_full = replay_with(tr, rec, case,
                           bundle=_bundle_with("gravity_beta", 2.5))
    # The cohort's post-full-replay registration of the beta row:
    _register(store, rec, (ref_full.reach, ref_full.gravity_exponential,
                           ref_full.gravity_logistic, ref_full.knn_access),
              case, bundle=_bundle_with("gravity_beta", 2.5))
    served = _row(store, tr, rec, case, counters,
                  bundle=_bundle_with("gravity_beta", 2.5))
    outs, mode = served
    assert mode == mre.REUSE_MODE_BUNDLE
    assert _bits_res(ref_full) == _bits_tuple(outs)


# --------------------------------------------------------------------------
# MR01 at the wiring level: one dependency at a time, every branch
# --------------------------------------------------------------------------
@pytest.mark.parametrize("decay", ["exponential", "logistic", "other"])
@pytest.mark.parametrize("scalar,value", [
    ("gravity_beta", 2.5),
    ("gravity_plateau", 0.5),
    ("gravity_logistic_midpoint", 1.25),
    ("gravity_growth_rate", 0.4),
])
def test_mr01_one_dependency_at_a_time(tr, decay, scalar, value):
    from urban_network_analysis.Engines import _metric_reuse_scratch as mre
    case = admitted_case("base")
    rec = tck_build(tr, case)
    store = mre.MetricReuseStore()
    counters = {}
    ref0 = replay_with(tr, rec, case, knn_decay=decay)
    _register(store, rec, (ref0.reach, ref0.gravity_exponential,
                           ref0.gravity_logistic, ref0.knn_access),
              case, knn_decay=decay)
    pert = _bundle_with(scalar, value)
    served = _row(store, tr, rec, case, counters, knn_decay=decay,
                  bundle=pert)
    assert served is not None
    outs, mode = served
    assert mode == mre.REUSE_MODE_SPLIT
    copied, folded = _expected_split(decay, scalar)
    assert counters["reuse_split_copies"] == len(copied)
    assert counters["reuse_split_folds"] == len(folded)
    ref1 = replay_with(tr, rec, case, knn_decay=decay, bundle=pert)
    assert _bits_res(ref1) == _bits_tuple(outs)


def test_mr01_branch_change_knn_folds_rest_copy(tr):
    from urban_network_analysis.Engines import _metric_reuse_scratch as mre
    case = admitted_case("base")
    rec = tck_build(tr, case)
    store = mre.MetricReuseStore()
    counters = {}
    ref0 = replay_with(tr, rec, case, knn_decay="exponential")
    _register(store, rec, (ref0.reach, ref0.gravity_exponential,
                           ref0.gravity_logistic, ref0.knn_access),
              case, knn_decay="exponential")
    served = _row(store, tr, rec, case, counters, knn_decay="logistic")
    outs, mode = served
    assert mode == mre.REUSE_MODE_SPLIT
    assert counters["reuse_split_copies"] == 3
    assert counters["reuse_split_folds"] == 1
    ref1 = replay_with(tr, rec, case, knn_decay="logistic")
    assert _bits_res(ref1) == _bits_tuple(outs)


def test_mr01_weights_change_collapses_to_full_fold(tr):
    from urban_network_analysis.Engines import _metric_reuse_scratch as mre
    case = admitted_case("base")
    rec = tck_build(tr, case)
    store = mre.MetricReuseStore()
    counters = {}
    ref0 = replay_with(tr, rec, case)
    _register(store, rec, (ref0.reach, ref0.gravity_exponential,
                           ref0.gravity_logistic, ref0.knn_access), case)
    dwt2 = np.asarray(case["dwt"], np.float64) + 2.0
    assert _row(store, tr, rec, case, counters, d_weights=dwt2) is None
    assert counters["reuse_collapse_full_folds"] == 1


def test_mr01_knn_weights_change_folds_knn_only(tr):
    from urban_network_analysis.Engines import _metric_reuse_scratch as mre
    case = admitted_case("base")
    rec = tck_build(tr, case)
    store = mre.MetricReuseStore()
    counters = {}
    ref0 = replay_with(tr, rec, case)
    _register(store, rec, (ref0.reach, ref0.gravity_exponential,
                           ref0.gravity_logistic, ref0.knn_access), case)
    kw2 = METRIC_PARAMS[4] * 2.0
    served = _row(store, tr, rec, case, counters, knn_weights=kw2)
    outs, mode = served
    assert counters["reuse_split_copies"] == 3
    assert counters["reuse_split_folds"] == 1
    ref1 = replay_with(tr, rec, case, knn_weights=kw2)
    assert _bits_res(ref1) == _bits_tuple(outs)


# --------------------------------------------------------------------------
# Typed agreement, admitted domain, gates-before-hit
# --------------------------------------------------------------------------
def test_f32_weights_stay_outside_the_split_domain(tr):
    """float32 d_weights is outside the admitted domain (split_domain
    refuses before any sourcing) — the row falls to the full fold."""
    from urban_network_analysis.Engines import _metric_reuse_scratch as mre
    case = admitted_case("base")
    rec = tck_build(tr, case)
    store = mre.MetricReuseStore()
    counters = {}
    ref0 = replay_with(tr, rec, case)
    _register(store, rec, (ref0.reach, ref0.gravity_exponential,
                           ref0.gravity_logistic, ref0.knn_access), case)
    d32 = np.asarray(case["dwt"], np.float64).astype(np.float32)
    assert _row(store, tr, rec, case, counters, d_weights=d32) is None
    assert counters["reuse_split_domain_refused"] == 1
    assert counters.get("reuse_bundle_hits", 0) == 0


def test_int_cutoff_mismatch_refuses_before_any_hit(tr):
    """int cutoff over a float-cutoff trace: the binding gate refuses
    (typed trace identity) BEFORE the bundle lookup — no hit crosses a
    refused pair."""
    from urban_network_analysis.Engines import _metric_reuse_scratch as mre
    case = admitted_case("base")
    rec = tck_build(tr, case)
    store = mre.MetricReuseStore()
    counters = {}
    ref0 = replay_with(tr, rec, case)
    _register(store, rec, (ref0.reach, ref0.gravity_exponential,
                           ref0.gravity_logistic, ref0.knn_access), case)
    served = _row(store, tr, rec, case, counters, cutoff=25)
    assert served is None
    assert counters["reuse_refusals"] == 1
    assert counters.get("reuse_bundle_hits", 0) == 0


def test_corrupt_trace_refuses_before_bundle_hit(tr):
    """A bit-flipped packed distance (digest NOT updated) breaks the
    content digest: the gates run BEFORE the bundle lookup, so a
    registered identity still never serves a corrupt trace."""
    from urban_network_analysis.Engines import _metric_reuse_scratch as mre
    case = admitted_case("base")
    rec = tck_build(tr, case)
    store = mre.MetricReuseStore()
    counters = {}
    ref0 = replay_with(tr, rec, case)
    _register(store, rec, (ref0.reach, ref0.gravity_exponential,
                           ref0.gravity_logistic, ref0.knn_access), case)
    dist = np.asarray(rec.distance_bits).copy()
    dist[0] = np.nextafter(dist[0], np.float64(1e300))
    forged = dataclasses.replace(rec, distance_bits=dist)  # old digest
    served = _row(store, tr, forged, case, counters)
    assert served is None
    assert counters["reuse_refusals"] == 1
    assert counters.get("reuse_bundle_hits", 0) == 0


def test_np_float64_beta_stays_outside_split_but_bundles(tr):
    """The domain gate blocks only SPLIT mode; an out-of-domain row
    with an identical typed binding still serves as a bundle (the
    canonical entry was produced by the full fold on that exact
    binding)."""
    from urban_network_analysis.Engines import _metric_reuse_scratch as mre
    case = admitted_case("base")
    rec = tck_build(tr, case)
    store = mre.MetricReuseStore()
    counters = {}
    ref0 = replay_with(tr, rec, case)
    _register(store, rec, (ref0.reach, ref0.gravity_exponential,
                           ref0.gravity_logistic, ref0.knn_access), case)
    npb = (np.float64(BUNDLE[0]), BUNDLE[1], BUNDLE[2], BUNDLE[3])
    served = _row(store, tr, rec, case, counters, bundle=npb)
    assert served is not None
    outs, mode = served
    assert mode == mre.REUSE_MODE_BUNDLE
    assert _bits_res(ref0) == _bits_tuple(outs)


# --------------------------------------------------------------------------
# Store laws: bounds, eviction, immutability, LRU, registration
# --------------------------------------------------------------------------
def test_store_bounds_eviction_lru():
    from urban_network_analysis.Engines import _metric_reuse_scratch as mre
    store = mre.MetricReuseStore(max_entries=2)
    job = mre.binding_job(np.zeros(3), 1.0, 0.0, 1.0, 0.5, "logistic",
                          np.ones(1), 400)
    outs = {f: np.zeros(4) for f in mre.ALL_OUTS}
    store.register("a", "t", job, outs)
    store.register("b", "t", job, outs)
    assert len(store) == 2
    store.register("c", "t", job, outs)
    assert len(store) == 2 and store.evictions == 1
    assert store.lookup("a") is None          # oldest evicted
    assert store.lookup("b") is not None
    assert store.lookup("c") is not None
    store.lookup("b")                          # LRU touch protects b
    store.register("d", "t", job, outs)
    assert store.lookup("b") is not None
    assert store.lookup("c") is None
    entry = store.lookup("d")
    for arr in entry["outs"].values():
        assert not arr.flags.writeable


def test_register_repeat_is_noop():
    from urban_network_analysis.Engines import _metric_reuse_scratch as mre
    store = mre.MetricReuseStore()
    job = mre.binding_job(np.zeros(3), 1.0, 0.0, 1.0, 0.5, "logistic",
                          np.ones(1), 400)
    outs = {f: np.zeros(4) for f in mre.ALL_OUTS}
    store.register("a", "t", job, outs)
    outs2 = {f: np.ones(4) for f in mre.ALL_OUTS}
    store.register("a", "t", job, outs2)   # first-arrival canonical stays
    entry = store.lookup("a")
    assert entry["outs"]["reach"].tobytes() == np.zeros(4).tobytes()


def test_register_owns_its_binding_lease_arrays(tr):
    """P03R F1: the stored binding lease holds PRIVATE copies of the
    array-valued fields — a caller mutating a previously-passed array
    in place after registration cannot make a later split agreement
    check serve the old canonical bits against NEW binding bytes."""
    from urban_network_analysis.Engines import _metric_identity as midm
    from urban_network_analysis.Engines import _metric_reuse_scratch as mre
    case = admitted_case("base")
    rec = tck_build(tr, case)
    store = mre.MetricReuseStore()
    counters = {}
    dwt = np.array(case["dwt"], copy=True)
    binding0 = replay_binding(case, d_weights=dwt)
    mid0 = midm.compute_metric_id(rec, *binding0)
    expected_lease_bits = np.asarray(dwt).tobytes()
    ref0 = replay_with(tr, rec, case)
    _register(store, rec, (ref0.reach, ref0.gravity_exponential,
                           ref0.gravity_logistic, ref0.knn_access),
              case, d_weights=dwt)
    entry = store.lookup(mid0)
    assert entry is not None
    assert entry["job"]["d_weights"] is not dwt      # private lease copy
    assert entry["job"]["d_weights"].tobytes() == expected_lease_bits
    # Caller mutates the array IN PLACE after registration.
    dwt[0] += 1.5
    served = _row(store, tr, rec, case, counters, d_weights=dwt)
    assert served is None                    # no stale serve: collapse
    assert counters["reuse_collapse_full_folds"] == 1
    # The stored lease still records the registration-time bytes.
    assert entry["job"]["d_weights"].tobytes() == expected_lease_bits


def test_lease_copy_is_load_bearing(tr):
    """Negative control for P03R F1: with the pre-fix SHALLOW lease
    (dict(job), aliased arrays) the same scenario stale-serves — the
    aliased lease 'agrees' with the mutated live array and copies
    canonical outputs computed from the OLD bytes.  Proves the
    private-copy law in register() is what prevents the defect."""
    from urban_network_analysis.Engines import _metric_identity as midm
    from urban_network_analysis.Engines import _metric_reuse_scratch as mre

    class _ShallowStore(mre.MetricReuseStore):
        # The pre-P03R-fix register (P03M-B 1523ad9 bytes).
        def register(self, metric_id, trace_id, job, outs):
            if metric_id in self._entries:
                return
            stored = {}
            for name in mre.ALL_OUTS:
                arr = np.array(outs[name], copy=True)
                arr.flags.writeable = False
                stored[name] = arr
            self._entries[metric_id] = {"trace_id": trace_id,
                                        "job": dict(job), "outs": stored}
            self._order.append(metric_id)

    case = admitted_case("base")
    rec = tck_build(tr, case)
    store = _ShallowStore()
    dwt = np.array(case["dwt"], copy=True)
    ref0 = replay_with(tr, rec, case)
    binding0 = replay_binding(case, d_weights=dwt)
    mid0 = midm.compute_metric_id(rec, *binding0)
    store.register(mid0, rec.trace_id, mre.binding_job(*binding0),
                   dict(zip(mre.ALL_OUTS,
                            (ref0.reach, ref0.gravity_exponential,
                             ref0.gravity_logistic, ref0.knn_access))))
    dwt[0] += 1.5                              # caller mutates in place
    counters = {}
    served = _row(store, tr, rec, case, counters, d_weights=dwt)
    assert served is not None                  # the stale serve happens
    outs, mode = served
    assert mode == mre.REUSE_MODE_SPLIT
    assert counters["reuse_split_copies"] == 4  # every dep "agreed" (alias)
    assert counters.get("reuse_split_folds", 0) == 0
    ref1 = replay_with(tr, rec, case, d_weights=dwt)
    assert fields_bits(ref1) != _bits_tuple(outs)   # served bits WRONG


# --------------------------------------------------------------------------
# Cohort transport: counters exposed, kill switch
# --------------------------------------------------------------------------
def test_cohort_session_counters_and_kill_switch(monkeypatch, tmp_path):
    from urban_network_analysis.batch.cohort import CohortSession
    monkeypatch.delenv("UNA_METRIC_REUSE", raising=False)
    s = CohortSession(str(tmp_path / "root"), 1 << 26)
    assert s.reuse_enabled is True and s._reuse is not None
    d = s.delta(s.mark())
    for key in ("reuse_bundle_hits", "reuse_split_rows",
                "reuse_split_copies", "reuse_split_folds",
                "reuse_collapse_full_folds", "reuse_split_domain_refused",
                "reuse_refusals", "reuse_registrations"):
        assert key in d and d[key] == 0
    monkeypatch.setenv("UNA_METRIC_REUSE", "0")
    s2 = CohortSession(str(tmp_path / "root2"), 1 << 26)
    assert s2.reuse_enabled is False and s2._reuse is None


def test_cohort_session_metric_store_is_process_pinned(monkeypatch,
                                                       tmp_path):
    """P04I lifecycle fix: the batch worker constructs a fresh
    CohortSession for EVERY row, so the metric-reuse store must be
    pinned at PROCESS level per cohort root (same lifecycle law as the
    trace store).  Two sessions on one root share ONE store object and
    therefore each other's registrations; a different root gets a
    different store; the kill switch still yields None."""
    from urban_network_analysis.batch import cohort as cohort_mod
    from urban_network_analysis.batch.cohort import CohortSession
    monkeypatch.delenv("UNA_METRIC_REUSE", raising=False)
    root_a = str(tmp_path / "root_a")
    root_b = str(tmp_path / "root_b")
    s1 = CohortSession(root_a, 1 << 26)
    s2 = CohortSession(root_a, 1 << 26)   # next row, same worker process
    s3 = CohortSession(root_b, 1 << 26)   # different cohort root
    assert s1._reuse is s2._reuse
    assert s1._reuse is not s3._reuse
    # Shared state is live: a registration in s1's store is visible
    # from s2's (same object), and evictions are counted in one place.
    s1._reuse.register(
        "mid-pinning-probe", "pinning-probe-trace",
        {"gravity_beta": 0.1, "gravity_plateau": 0,
         "gravity_logistic_midpoint": 500, "gravity_growth_rate": 0.8,
         "knn_decay": "logistic", "knn_weights": np.array([1.0]),
         "cutoff": 400},
        {name: np.zeros((2,), dtype=np.float64) for name in ALL_OUTS})
    assert len(s2._reuse.entries_for_trace("pinning-probe-trace")) == 1
    assert s2.delta(s2.mark())["reuse_split_rows"] == 0
    assert s1._reuse.evictions == s2._reuse.evictions
    # Kill switch still bypasses the shared cache entirely.
    monkeypatch.setenv("UNA_METRIC_REUSE", "0")
    s4 = CohortSession(root_a, 1 << 26)
    assert s4.reuse_enabled is False and s4._reuse is None
