"""C30 cohort route: public-API sharing with bitwise parity (dossier 06).

Every test drives the installed public API.  The route may only save
work — the engine's own numbers must survive bitwise, the serial route
must never notice the cohort machinery, and produce-once must hold per
trace identity across processes.
"""
from __future__ import annotations

import os

import numpy as np
import pytest

from conftest import tree_bytes

pytestmark = pytest.mark.tenx_cohort

COHORT_DIRNAME = ".una-batch-cohort"
METRICS = ("reach", "gravity_exponential", "gravity_logistic",
           "knn_access")


def _composite(i, s):
    s.batch_composite_output = True
    s.batch_composite_result_column = "knn_access"


def test_parallel_cohort_bitwise_matches_serial(
        make_batch, tmp_path, cohort_summary):
    """Four distinct-radius jobs through the cohort route: every exported
    artifact and every composite capture is bitwise-identical to the
    serial route, each distinct identity built its own trace, and the
    rehydrated parent state is the PLAIN engine (the wrapper is
    runtime-only, never observable state)."""
    from urban_network_analysis.batch.cohort import CohortGraphEngine

    ser = make_batch(4, out_folder=tmp_path / "ser", extra=_composite)
    ser.RunBatch("accessibility")

    par = make_batch(4, out_folder=tmp_path / "par", extra=_composite)
    par.RunBatch("accessibility", parallel=True, workers=2)

    cs = cohort_summary(par)
    assert cs is not None, "cohort route did not report"
    assert cs["rows"] == 4
    assert cs["refusals"] == {}, cs["refusals"]
    assert cs["replays"] == 4
    assert cs["folds_done"] > 0
    assert cs["builds"] == 4          # one build per distinct identity
    assert cs["distinct_trace_identities"] == 4

    # row artifacts bitwise-identical (hidden runtime entries excluded)
    assert tree_bytes(tmp_path / "ser", skip_hidden=True) == \
        tree_bytes(tmp_path / "par", skip_hidden=True)

    # composite captures bitwise-identical, same column order
    cols = [c for c in par.composite_result.columns
            if c.startswith("knn_access_")]
    ser_cols = [c for c in ser.composite_result.columns
                if c.startswith("knn_access_")]
    assert cols == ser_cols
    for c in cols:
        assert np.array_equal(par.composite_result[c].to_numpy(),
                              ser.composite_result[c].to_numpy())

    # per-row identity attribution: every row's stats carries its own
    # trace id, and the four distinct radii produced four distinct ids
    ids = {stats["trace_id"] for stats in cs["per_row"].values()}
    assert len(ids) == 4
    assert all(stats["result_key"] for stats in cs["per_row"].values())

    # the parent's rehydrated engine is the plain view, not the wrapper
    assert not isinstance(par.accessibility.graph_engine,
                          CohortGraphEngine)


def test_shared_identity_builds_once_and_warm_repeat_serves(
        make_batch, tmp_path, cohort_summary):
    """Four DISTINCT parameter jobs (different gravity_beta) sharing ONE
    trace identity (same inputs, same cutoff): exactly one build serves
    the whole cohort; a second RunBatch over the same output folder is
    warm — the persisted cohort store serves every row, nothing builds
    again, and the artifacts are unchanged."""
    def extra(i, s):
        s.search_radius = 120          # one identity for the whole cohort
        s.gravity_beta = 0.010 + 0.001 * i   # distinct jobs, shared trace
        _composite(i, s)

    out = tmp_path / "out"
    p1 = make_batch(4, out_folder=out, extra=extra)
    p1.RunBatch("accessibility", parallel=True, workers=2)
    cs1 = cohort_summary(p1)
    assert cs1["rows"] == 4
    assert cs1["builds"] == 1, cs1     # produce-once per identity
    assert cs1["distinct_trace_identities"] == 1
    # P04I: the process-pinned P03M metric-reuse arm engages here —
    # per worker process the first row full-folds (collapse, canonical
    # register) and every later distinct-beta row split-serves (reach
    # copied, gravity refolded).  Every row is accounted, none refused.
    modes1 = [stats["reuse_mode"] for stats in cs1["per_row"].values()]
    assert len(modes1) == 4
    assert cs1["replays"] == modes1.count(None) >= 1
    assert {m for m in modes1 if m} == {"metric_reuse_split"}
    # every non-owner row was served from the store or the residency
    assert cs1["faultins"] + cs1["resident_hits"] == 3
    assert cs1["refusals"] == {}, cs1["refusals"]
    # one identity -> all rows report the same trace id (identity closure:
    # any disagreement would have forked keys and forced extra builds)
    ids = {stats["trace_id"] for stats in cs1["per_row"].values()}
    assert ids and len(ids) == 1
    # the cohort store root persists beside the row outputs (warm resumes)
    assert os.path.isdir(os.path.join(str(out), COHORT_DIRNAME))

    cold_tree = tree_bytes(out, skip_hidden=True)
    p2 = make_batch(4, out_folder=out, extra=extra)
    p2.RunBatch("accessibility", parallel=True, workers=2)
    cs2 = cohort_summary(p2)
    assert cs2["builds"] == 0, cs2     # nothing is built again
    # The metric store is process-local (designs/04: no persistence) —
    # fresh workers collapse their first row again, then split-serve;
    # the TRACE layer stays warm (faultins only, zero builds).
    modes2 = [stats["reuse_mode"] for stats in cs2["per_row"].values()]
    assert cs2["replays"] == modes2.count(None) >= 1
    assert {m for m in modes2 if m} == {"metric_reuse_split"}
    assert cs2["faultins"] + cs2["resident_hits"] == 4
    assert cs2["refusals"] == {}, cs2["refusals"]
    # warm artifacts unchanged, bitwise
    assert tree_bytes(out, skip_hidden=True) == cold_tree


def test_serial_and_single_job_never_touch_the_cohort_route(
        make_batch, tmp_path, cohort_summary):
    """The serial route and a plain single-job RunAccessibility must not
    install the wrapper, report a cohort block, or create the cohort
    store root: the route exists only inside the parallel worker path."""
    from urban_network_analysis.batch.cohort import CohortGraphEngine

    p = make_batch(3, out_folder=tmp_path / "out")
    p.RunBatch("accessibility")
    assert cohort_summary(p) is None
    assert not os.path.exists(os.path.join(str(tmp_path / "out"),
                                           COHORT_DIRNAME))

    q = make_batch(1, out_folder=tmp_path / "single")
    q.RunAccessibility()
    assert cohort_summary(q) is None
    assert not os.path.exists(os.path.join(str(tmp_path / "single"),
                                           COHORT_DIRNAME))
    assert not isinstance(q.accessibility.graph_engine,
                          CohortGraphEngine)


def test_refusal_falls_back_to_inner_bitwise(
        make_batch, tmp_path, monkeypatch):
    """A refused trace build must degrade to the inner engine's own
    o_access and produce the SAME bits: the cohort route can only save
    work, never change a number.  Counted, not silent."""
    from urban_network_analysis.batch.cohort import (CohortGraphEngine,
                                                     CohortSession)
    from urban_network_analysis.Engines import _trace_access_scratch as trm

    # control: the plain engine's own four arrays for this job
    ctrl = make_batch(1, out_folder=tmp_path / "ctrl")
    ctrl.RunAccessibility()
    ref = {k: np.array(getattr(ctrl.accessibility, k)) for k in METRICS}

    incomplete = trm.TraceRecord(
        trace_id="poisoned", schema_version=trm.TRACE_SCHEMA_VERSION,
        origin_count=0, dest_count=0, cutoff_value=0.0,
        cutoff_dtype="float64", reach_dtype="float64",
        metric_dtype="float64", indptr=np.zeros(1, dtype=np.int64),
        destination_ids=np.zeros(0, dtype=np.int64),
        distance_bits=np.zeros(0), complete=False,
        route="unit_poison", refusal="poisoned-by-test",
        counters=trm.TraceCounters(*((0,) * 12)))
    monkeypatch.setattr(trm, "build_trace", lambda *a, **k: incomplete)

    out = tmp_path / "out"
    p = make_batch(1, out_folder=out)
    p._cohort_session = CohortSession(
        os.path.join(str(out), COHORT_DIRNAME), 1 << 26)
    p.RunAccessibility()

    # the integrator dispatch installed the wrapper, and the arrays the
    # run left are the FALLBACK's — bitwise-equal to the control
    assert isinstance(p.accessibility.graph_engine, CohortGraphEngine)
    for k in METRICS:
        assert np.array_equal(np.array(getattr(p.accessibility, k)),
                              ref[k]), k

    session = p._cohort_session
    c = session.counters
    assert c["rows_attempted"] == 1
    assert c["builds"] == 0 and c["replays"] == 0
    assert sum(c["refusals"].values()) == 1
    reason = next(iter(c["refusals"]))
    assert reason.startswith("trace_build:"), reason
