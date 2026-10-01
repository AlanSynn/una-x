"""Staged-ABI compatibility and import hygiene for the runtime (dossier 05).

* ``RunBatch(parallel=True)`` is refused inside worker processes (guarded
  entry point) — no pool recursion.
* ``urban_network_analysis.batch`` stays plan-pure at import time (numpy
  via the Settings dataclass is the reviewed BATCH_PLAN floor — nothing
  heavier), and the spawn TARGET is the stdlib-only ``una_batch_worker_main``
  shim: thread budgets apply before its first heavy import.
* The runtime report remains an ``isinstance`` of the EXECUTION-stage
  ``BatchReport``/``RowOutcome`` contracts (field set is a superset).
"""
from __future__ import annotations

import dataclasses
import os
import subprocess
import sys

import pytest

pytestmark = pytest.mark.batch_runtime

_ALAN = sys.executable
_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))


def test_nested_parallel_batch_refused(make_batch, tmp_path, monkeypatch):
    p = make_batch(2, out_folder=tmp_path / "out")
    monkeypatch.setenv("_UNA_BATCH_IN_WORKER", "1")
    with pytest.raises(RuntimeError, match="nested parallel"):
        p.RunBatch("accessibility", parallel=True, workers=2)
    assert p.batch_report is None


def test_batch_package_import_stays_plan_pure():
    """Importing the batch package must not pull the analysis stack —
    the planner is usable from any environment and the runtime is
    imported lazily by RunBatch only.  numpy itself is ALLOWED and
    expected: the reviewed BATCH_PLAN contract pins ``Settings`` as
    stdlib+numpy ("nothing heavier") — plan.py imports the Settings
    dataclass for row typing."""
    code = (
        "import sys;"
        "import urban_network_analysis.batch as b;"
        "heavy = ('pandas', 'geopandas', 'shapely', 'numba', 'sklearn');"
        "loaded = [m for m in heavy if m in sys.modules];"
        "assert not loaded, f'analysis-stack modules loaded: {loaded}';"
        "assert not any(m == 'urban_network_analysis.UNA' or"
        "               m.startswith('urban_network_analysis.UNA.')"
        "               for m in sys.modules), 'UNA pulled in';"
        "assert 'urban_network_analysis.batch.runtime' not in sys.modules,"
        "       'runtime eagerly imported'"
    )
    env = dict(os.environ, PYTHONPATH=os.path.join(_REPO, "src"),
               PYTHONHASHSEED="0")
    res = subprocess.run([_ALAN, "-c", code], env=env, cwd=_REPO,
                         capture_output=True, text=True, timeout=120)
    assert res.returncode == 0, res.stderr


def test_spawn_target_import_stays_light():
    """The spawn worker TARGET (the stdlib-only shim the pool actually
    references) must import without numpy or any heavy module: spawn
    imports the target's module before any of our code runs, and the
    thread budgets must be applicable BEFORE numpy initializes its BLAS
    thread pool from the environment.  The shim then hands off to
    ``urban_network_analysis.batch.worker.pool_main`` inside the budget."""
    code = (
        "import sys;"
        "import una_batch_worker_main as m;"
        "heavy = [mod for mod in ('numpy', 'pandas', 'geopandas', 'numba',"
        "         'sklearn', 'urban_network_analysis.batch.worker')"
        "         if mod in sys.modules];"
        "assert not heavy, f'heavy modules loaded: {heavy}'"
    )
    env = dict(os.environ, PYTHONPATH=os.path.join(_REPO, "src"),
               PYTHONHASHSEED="0")
    res = subprocess.run([_ALAN, "-c", code], env=env, cwd=_REPO,
                         capture_output=True, text=True, timeout=120)
    assert res.returncode == 0, res.stderr


def test_report_and_outcome_extend_execution_abis(make_batch, tmp_path):
    from urban_network_analysis.Execution import BatchReport, RowOutcome
    from urban_network_analysis.batch.report import (BatchExecutionReport,
                                                     RowExecutionOutcome)

    base_row_fields = {f.name for f in dataclasses.fields(RowOutcome)}
    base_rep_fields = {f.name for f in dataclasses.fields(BatchReport)}
    row_fields = {f.name for f in dataclasses.fields(RowExecutionOutcome)}
    rep_fields = {f.name for f in dataclasses.fields(BatchExecutionReport)}
    assert base_row_fields <= row_fields
    assert base_rep_fields <= rep_fields

    par = make_batch(2, out_folder=tmp_path / "out")
    par.RunBatch("accessibility", parallel=True, workers=2)
    report = par.batch_report
    assert isinstance(report, BatchReport)
    assert isinstance(report, BatchExecutionReport)
    for r in report.rows:
        assert isinstance(r, RowOutcome)
        assert isinstance(r, RowExecutionOutcome)
