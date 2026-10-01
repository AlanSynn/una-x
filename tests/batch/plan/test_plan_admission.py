"""Admission: only proven-independent rows go to workers; anything
unproven is serial WITH a reason; independent rows are never rejected
wholesale because other rows in the batch are not independent."""
from __future__ import annotations

import pytest

from urban_network_analysis.Settings import Settings
from urban_network_analysis.batch import (BATCH_TRANSITIONS, ROW_TRANSITIONS,
                                          plan_batch)


class CustomSettings(Settings):
    """A user subclass — unshared behavior cannot be proven row-local."""


@pytest.mark.batch_plan
def test_independent_rows_admitted_despite_serialized_neighbors(make_settings,
                                                                tmp_path):
    """A hazardous pair does not drag independent rows down with it: rows 2
    and 3 are proven independent and remain worker-admissible."""
    out = tmp_path / "out"
    rows = [
        make_settings("a", stem="same", folder=out),
        make_settings("b", stem="same", folder=out,
                      inputs=("nb.geojson", "ob.geojson", "db.geojson")),
        make_settings("c", stem="c", folder=tmp_path / "out_c",
                      inputs=("nc.geojson", "oc.geojson", "dc.geojson")),
        make_settings("d", stem="d", folder=tmp_path / "out_d",
                      inputs=("nd.geojson", "od.geojson", "dd.geojson")),
    ]
    plan = plan_batch(rows)
    assert plan.rows[0].admission == "serial"
    assert plan.rows[1].admission == "serial"
    assert plan.rows[2].admission == "worker"
    assert plan.rows[3].admission == "worker"
    assert plan.worker_admissible == (2, 3)
    assert plan.serialized[0] == (0, plan.rows[0].admission_reasons)
    assert any("output_conflict with row 1" in r
               for r in plan.rows[0].admission_reasons)


@pytest.mark.batch_plan
def test_custom_settings_subclass_runs_serial_with_reason(make_settings):
    rows = [make_settings("plain"), CustomSettings(name="fancy")]
    rows[1].data_folder = rows[0].data_folder
    rows[1].network_file, rows[1].origins_file, rows[1].destinations_file = (
        "n2.geojson", "o2.geojson", "d2.geojson")
    rows[1].output_folder = str(rows[0].output_folder) + "_x"
    plan = plan_batch(rows)
    assert plan.rows[0].admission == "worker"
    assert plan.rows[1].admission == "serial"
    reason = " ".join(plan.rows[1].admission_reasons)
    assert "not the base Settings type" in reason
    assert "CustomSettings" in reason


@pytest.mark.batch_plan
def test_serial_note_and_timestamp_note(make_settings, tmp_path):
    out = tmp_path / "out"
    rows = [make_settings("a", stem="same", folder=out),
            make_settings("b", stem="same", folder=out,
                          inputs=("nb.geojson", "ob.geojson", "db.geojson"))]
    plan = plan_batch(rows)
    joined = " | ".join(plan.notes)
    assert "serial route" in joined

    # a worker batch with default wStamp=True carries the clock note
    plan2 = plan_batch([make_settings("w", folder=tmp_path / "ow",
                                      wstamp=True)])
    assert any("timestamped subfolder" in n for n in plan2.notes)


@pytest.mark.batch_plan
def test_commit_and_fold_order_are_caller_order(make_settings):
    rows = [make_settings(f"r{i}") for i in range(4)]
    rows[1].network_file = ""
    rows[1].origins_file = ""
    rows[1].destinations_file = ""
    plan = plan_batch(rows)
    assert plan.commit_order == (0, 2, 3)
    assert plan.fold_order == plan.commit_order


@pytest.mark.batch_plan
def test_composite_output_note_when_a_row_asks_for_it(make_settings):
    """Review NOTE-5: composite artifacts are coordinator-written AFTER the
    last commit; the plan says so explicitly when any row requests one."""
    rows = [make_settings("a", folder="/tmp/oa"),
            make_settings("b", folder="/tmp/ob",
                          inputs=("nb.geojson", "ob.geojson", "db.geojson"))]
    assert plan_batch(rows).notes == ()   # no composite -> no note
    rows[0].batch_composite_output = True
    plan = plan_batch(rows)
    assert any("finalize_composite" in n and "fold_order" in n
               for n in plan.notes)
    # the note does not change admission
    assert all(r.admission == "worker" for r in plan.rows)


@pytest.mark.batch_plan
def test_transition_spec_is_contractual():
    """The ordered transition names are part of the plan's public contract
    (the coordinator reproduces exactly these, in exactly this order)."""
    assert ROW_TRANSITIONS == (
        "bind_settings",
        "resolve_output_folder",
        "validate_or_skip",
        "substitute_output_name",
        "run_analysis",
        "capture_composite",
        "record_outcome",
    )
    assert BATCH_TRANSITIONS == (
        "admit_execution",
        "load_pairing",
        "init_compositor",
        "run_rows",
        "finalize_composite",
        "assemble_report",
    )
