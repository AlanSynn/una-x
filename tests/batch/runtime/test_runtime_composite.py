"""Composite-output parity for the parallel runtime (dossier 04).

The composite fold is float order-sensitive: capture columns must fold in
``plan.fold_order`` (caller order), never completion order, and the
composite files must match the serial route byte for byte.
"""
from __future__ import annotations

import numpy as np
import pytest

from conftest import tree_bytes

pytestmark = pytest.mark.batch_runtime


def _composite_extra(i, s):
    s.batch_composite_output = True


def test_composite_parity_across_workers(make_batch, tmp_path):
    serial = make_batch(3, out_folder=tmp_path / "ser",
                        extra=_composite_extra)
    serial.RunBatch("accessibility")

    par = make_batch(3, out_folder=tmp_path / "par",
                     extra=_composite_extra)
    par.RunBatch("accessibility", parallel=True, workers=2)

    # one group (all rows share the origins layer), same grouping name
    assert len(par.composite_results) == len(serial.composite_results) == 1
    assert par.composite_results[0][0] == serial.composite_results[0][0]

    s_gdf, p_gdf = serial.composite_result, par.composite_result
    assert list(p_gdf.columns) == list(s_gdf.columns)
    caps = [c for c in p_gdf.columns if c.startswith("knn_access_")]
    # exactly one column per row, in caller order — the fold never follows
    # completion order (a completion-order fold would permute these)
    assert caps == ["knn_access_row0", "knn_access_row1", "knn_access_row2"]
    for c in caps + ["composite_sum"]:
        assert np.array_equal(np.asarray(p_gdf[c]), np.asarray(s_gdf[c]))
    assert tree_bytes(tmp_path / "ser") == tree_bytes(tmp_path / "par")


def test_composite_respects_per_row_opt_out(make_batch, tmp_path):
    """A row with batch_composite_output=False contributes no column,
    identically on both routes."""
    def extra(i, s):
        # set EXPLICITLY every row: make_batch reuses one live Settings
        # object across rows, so a conditional assignment would leak the
        # previous row's value into this row's project snapshot
        s.batch_composite_output = (i != 1)

    serial = make_batch(3, out_folder=tmp_path / "ser", extra=extra)
    serial.RunBatch("accessibility")
    par = make_batch(3, out_folder=tmp_path / "par", extra=extra)
    par.RunBatch("accessibility", parallel=True, workers=2)

    for gdf in (serial.composite_result, par.composite_result):
        caps = [c for c in gdf.columns if c.startswith("knn_access_")]
        assert caps == ["knn_access_row0", "knn_access_row2"]
    assert tree_bytes(tmp_path / "ser") == tree_bytes(tmp_path / "par")


def test_composite_fold_is_caller_order_under_completion_noise(
        make_batch, tmp_path):
    """The composite is float order-sensitive: with the completion order
    forced OPPOSITE to caller order (early rows sleep longest), the folded
    columns and the float sum must still match the serial route exactly —
    this is the deterministic selector for the fold-by-completion-order
    mutant (review MINOR-6)."""
    def composite_only(i, s):
        s.batch_composite_output = True

    def extra(i, s):
        # the sleep specs ride the parallel route only: the serial route
        # runs rows on the live instance, whose Validation() rejects the
        # test-transport field
        sleeps = [1.2, 0.8, 0.4, 0.0]
        composite_only(i, s)
        s.inject_specs = ([{"phase": "pre_run", "action": "sleep",
                            "arg": sleeps[i]}] if sleeps[i] else [])

    serial = make_batch(4, out_folder=tmp_path / "ser", extra=composite_only)
    serial.RunBatch("accessibility")

    par = make_batch(4, out_folder=tmp_path / "par", extra=extra)
    par.RunBatch("accessibility", parallel=True, workers=4)

    assert par.batch_report.commit_order == (0, 1, 2, 3)
    assert par.batch_report.fold_order == (0, 1, 2, 3)
    caps = [c for c in par.composite_result.columns
            if c.startswith("knn_access_")]
    ser_caps = [c for c in serial.composite_result.columns
                if c.startswith("knn_access_")]
    assert caps == ser_caps == ["knn_access_row0", "knn_access_row1",
                                "knn_access_row2", "knn_access_row3"]
    for c in caps + ["composite_sum"]:
        assert np.array_equal(np.asarray(par.composite_result[c]),
                              np.asarray(serial.composite_result[c]))
    assert tree_bytes(tmp_path / "ser") == tree_bytes(tmp_path / "par")
