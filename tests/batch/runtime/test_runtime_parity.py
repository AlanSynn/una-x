"""Serial/parallel parity: final state and artifacts match bitwise across
worker configurations (dossier 04 tests; dossier 05 acceptance).

The serial route is the reference.  For W in (1, 2, 4) the parallel route
must produce byte-identical artifacts and bitwise-identical observable
state — settings/projects (identity preserved), topology arrays, engine
arrays, result flags, report rows — with the same outputs at the same
paths.
"""
from __future__ import annotations

import numpy as np
import pytest

from conftest import tree_bytes, values_equal

from urban_network_analysis.batch.report import BatchExecutionReport

pytestmark = pytest.mark.batch_runtime

_ENGINE_ATTRS = ("reach", "gravity_exponential", "gravity_logistic",
                 "knn_access")
_TOPO_ATTRS = ("node_points", "weights", "start_nodes", "end_nodes")


def _assert_same_state(serial, par):
    # settings identity + content: the serial loop leaves self.settings
    # bound to the LAST project object, mutated in place.  output_folder
    # is excluded from the content diff: each route wrote to its own tree
    # (compared byte-for-byte by the callers via tree_bytes).
    assert par.settings is par.projects[-1]
    assert serial.settings is serial.projects[-1]
    ds, dp = vars(serial.settings), vars(par.settings)
    differing = {k for k in set(ds) | set(dp)
                 if k != "output_folder"
                 and not values_equal(ds.get(k), dp.get(k))}
    assert not differing, f"final settings differ: {sorted(differing)}"
    # every project object keeps its identity and its serial mutations
    for a, b in zip(serial.projects, par.projects):
        assert type(a) is type(b)
        da, db = vars(a), vars(b)
        diff = {k for k in set(da) | set(db)
                if k != "output_folder"
                and not values_equal(da.get(k), db.get(k))}
        assert not diff, f"project row differs: {sorted(diff)}"
    # observable flags
    assert (par.has_flow_results, par.has_centrality_results) == \
           (serial.has_flow_results, serial.has_centrality_results)
    # a FRESH parallel run rehydrates from the last worker's bundle — the
    # state-only rerun is a resume-path fallback and must never fire here
    if par.batch_report is not None:
        assert not any("state-only rerun" in n
                       for n in par.batch_report.notes_runtime)
    # engine arrays bitwise
    for attr in _ENGINE_ATTRS:
        a = getattr(serial.accessibility, attr, None)
        b = getattr(par.accessibility, attr, None)
        if a is None or b is None:
            assert a is None and b is None, attr
            continue
        assert np.asarray(a).dtype == np.asarray(b).dtype, attr
        assert np.array_equal(np.asarray(a), np.asarray(b)), \
            f"engine.{attr} differs from serial"
    # topology arrays bitwise
    for attr in _TOPO_ATTRS:
        assert np.array_equal(getattr(serial.topology.network, attr),
                              getattr(par.topology.network, attr)), attr


@pytest.mark.parametrize("workers", [1, 2, 4])
def test_parallel_matches_serial_bitwise(make_batch, tmp_path, workers):
    serial = make_batch(3, out_folder=tmp_path / "ser")
    serial.RunBatch("accessibility")

    par = make_batch(3, out_folder=tmp_path / "par")
    par.RunBatch("accessibility", parallel=True, workers=workers)

    _assert_same_state(serial, par)
    assert tree_bytes(tmp_path / "ser") == tree_bytes(tmp_path / "par")


def test_parallel_matches_serial_bitwise_multithreaded_workers(
        make_batch, tmp_path):
    """Parity must also hold when each worker owns >1 thread (dossier 05:
    equal bits across admitted worker configurations)."""
    from dataclasses import replace

    serial = make_batch(3, out_folder=tmp_path / "ser")
    serial.RunBatch("accessibility")

    par = make_batch(3, out_folder=tmp_path / "par")
    par.RunBatch(
        "accessibility", parallel=True, workers=2,
        execution=replace(par.execution, threads_per_worker=2))

    _assert_same_state(serial, par)
    assert tree_bytes(tmp_path / "ser") == tree_bytes(tmp_path / "par")
    report = par.batch_report
    assert "2 spawn worker(s) x 2 thread(s)" in \
        " ".join(report.notes_runtime)


def test_parallel_report_shape(make_batch, tmp_path):
    par = make_batch(2, out_folder=tmp_path / "out")
    par.RunBatch("accessibility", parallel=True, workers=2)
    report = par.batch_report
    assert isinstance(report, BatchExecutionReport)
    assert report.parallel_requested is True
    assert report.workers_requested == 2
    assert report.workers_effective == 2
    assert "row_workers" in report.pools
    assert report.commit_order == (0, 1)
    assert report.fold_order == (0, 1)
    assert report.worker_admissible == (0, 1)
    assert report.cancelled is False
    assert [r.index for r in report.rows] == [0, 1]
    assert all(r.status == "ran" and r.phase == "COMMITTED"
               for r in report.rows)
    assert all(r.worker_pid is not None and r.worker_pid != __import__("os").getpid()
               for r in report.rows)
    # output manifest lists the committed artifacts per row
    for r in report.rows:
        assert r.output_files, "manifest row empty"
        for path in r.output_files:
            import os
            assert os.path.exists(path)
    assert report.output_manifest == {str(r.index): r.output_files
                                      for r in report.rows}


def test_all_worker_rows_run_in_worker_processes(make_batch, tmp_path):
    """A worker-admissible row must run in a spawn worker, not silently in
    the parent (a benchmark-only pool cannot satisfy dossier 05)."""
    import os
    par = make_batch(2, out_folder=tmp_path / "out")
    par.RunBatch("accessibility", parallel=True, workers=2)
    parent_pid = os.getpid()
    pids = {r.worker_pid for r in par.batch_report.rows}
    assert parent_pid not in pids
    assert len(pids) >= 1


def test_flow_analysis_parity(make_batch, tmp_path):
    serial = make_batch(2, out_folder=tmp_path / "ser")
    serial.RunBatch("flow")
    par = make_batch(2, out_folder=tmp_path / "par")
    par.RunBatch("flow", parallel=True, workers=2)

    _assert_same_state(serial, par)
    assert tree_bytes(tmp_path / "ser") == tree_bytes(tmp_path / "par")
    assert np.array_equal(np.asarray(serial.flow.edge_flow),
                          np.asarray(par.flow.edge_flow))
