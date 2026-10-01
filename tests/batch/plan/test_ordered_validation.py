"""Ordered validation outcomes (dossier 04 step 3): problems are held in
caller order — nothing about row 8 preempts row 2."""
from __future__ import annotations

import pytest

from urban_network_analysis.batch import plan_batch


@pytest.mark.batch_plan
def test_outcomes_are_in_caller_order_including_skips(make_settings):
    rows = [make_settings("ok0"),
            make_settings("broken", inputs=("", "o.geojson", "d.geojson")),
            make_settings("ok2"),
            make_settings("broken2", inputs=("", "", "")),
            make_settings("ok4")]
    plan = plan_batch(rows)
    assert [r.row.index for r in plan.rows] == [0, 1, 2, 3, 4]
    assert [r.validation.status for r in plan.rows] == [
        "ok", "skipped", "ok", "skipped", "ok"]
    # the missing lists match the serial loop's computation exactly,
    # including their order
    assert plan.rows[1].validation.missing == ("network_file",)
    assert plan.rows[3].validation.missing == (
        "network_file", "origins_file", "destinations_file")
    assert "missing required fields" in plan.rows[1].validation.detail


@pytest.mark.batch_plan
def test_a_late_problem_never_prevents_earlier_planning(make_settings):
    """The planner inspects every row; a problem on the last row cannot
    stop rows 0..n-1 from being fully planned (the serial rule: an error
    surfaces at ITS position, not at inspection time)."""
    rows = [make_settings(f"ok{i}") for i in range(4)]
    rows.append(make_settings("broken", inputs=("", "", "")))
    plan = plan_batch(rows)                     # must not raise
    assert len(plan.rows) == 5
    assert plan.rows[4].validation.status == "skipped"
    assert all(plan.rows[i].validation.status == "ok" for i in range(4))
    assert all(plan.rows[i].admission == "worker" for i in range(4))
    assert plan.commit_order == (0, 1, 2, 3)


@pytest.mark.batch_plan
def test_empty_batch_raises_like_the_serial_call():
    with pytest.raises(RuntimeError, match="No settings to run"):
        plan_batch([])


@pytest.mark.batch_plan
def test_single_row_batch_plans_trivially(make_settings):
    plan = plan_batch([make_settings("solo")])
    assert plan.commit_order == (0,)
    assert plan.rows[0].admission == "worker"
    assert plan.rows[0].hazards == ()
    assert plan.notes == ()


@pytest.mark.batch_plan
def test_truthy_non_string_required_field_is_an_ordered_error(make_settings):
    """Review MINOR-2: the serial loop fails at this row ((value or '').
    strip() -> AttributeError); the plan holds that as an ordered error —
    it must NOT raise at plan time."""
    rows = [make_settings("ok0"), make_settings("bad"),
            make_settings("ok2")]
    rows[1].network_file = 12345          # truthy, not a string
    plan = plan_batch(rows)               # must not raise
    v1 = plan.rows[1].validation
    assert v1.status == "error"
    assert "network_file" in v1.detail and "int" in v1.detail
    assert plan.rows[1].admission == "error"
    # serial semantics: falsy values are MISSING (the `or ''` idiom)
    rows[1].network_file = 0              # falsy non-string: missing
    plan = plan_batch(rows)
    assert plan.rows[1].validation.status == "skipped"
    assert "network_file" in plan.rows[1].validation.missing


@pytest.mark.batch_plan
def test_nothing_commits_at_or_after_the_first_error(make_settings):
    """The serial call aborts at the earliest row-position failure: the
    valid prefix commits, later rows keep their recorded outcomes but are
    not in commit_order."""
    rows = [make_settings("ok0"), make_settings("bad"),
            make_settings("ok2")]
    rows[1].network_file = object()       # truthy non-string -> error
    plan = plan_batch(rows)
    assert plan.rows[0].validation.status == "ok"
    assert plan.rows[1].validation.status == "error"
    assert plan.rows[2].validation.status == "ok"      # still planned
    assert plan.commit_order == (0,)                   # truncated
    assert plan.rows[2].admission == "worker"


@pytest.mark.batch_plan
def test_non_string_data_folder_is_an_ordered_error_not_a_plan_crash(
        make_settings, tmp_path):
    """A truthy non-string data_folder makes the serial loop fail in-row
    (os.path.join TypeError); the plan records the ordered error instead of
    raising at plan time — including through the default-output branch,
    which the serial loop evaluates BEFORE the skip gate."""
    bad = make_settings("badfolder")
    bad.data_folder = 5
    plan = plan_batch([bad])              # defaulting branch -> error
    assert plan.rows[0].validation.status == "error"
    assert "data_folder" in plan.rows[0].validation.detail

    explicit = make_settings("badfolder2", folder=tmp_path / "out")
    explicit.data_folder = 5              # explicit folder: fails later,
    plan = plan_batch([explicit])         #  at the loaders -> still held
    assert plan.rows[0].validation.status == "error"
    assert plan.rows[0].admission == "error"


@pytest.mark.batch_plan
def test_unreadable_attribute_is_an_ordered_error():
    class Raising:
        name = "r"

        @property
        def network_file(self):
            raise RuntimeError("boom")

        origins_file = "o.geojson"
        destinations_file = "d.geojson"
        data_folder = "/tmp"

    plan = plan_batch([Raising()])        # must not raise at plan time
    assert plan.rows[0].validation.status == "error"
    assert "unreadable" in plan.rows[0].validation.detail


@pytest.mark.batch_plan
def test_plan_is_immutable(make_settings):
    import dataclasses
    plan = plan_batch([make_settings("a"), make_settings("b")])
    with pytest.raises(dataclasses.FrozenInstanceError):
        plan.rows[0].validation.status = "skipped"
    with pytest.raises(dataclasses.FrozenInstanceError):
        plan.commit_order = ()
