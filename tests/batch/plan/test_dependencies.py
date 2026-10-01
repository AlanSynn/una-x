"""Dependency hazard derivation: output conflicts, read/write overlap,
object aliasing — dossier 04 step 2."""
from __future__ import annotations

import pytest

from urban_network_analysis.Settings import Settings
from urban_network_analysis.batch import plan_batch


@pytest.mark.batch_plan
def test_disjoint_rows_have_no_hazards(make_settings, tmp_path):
    rows = [make_settings("a", folder=tmp_path / "out_a"),
            make_settings("b", folder=tmp_path / "out_b",
                          inputs=("n2.geojson", "o2.geojson", "d2.geojson"))]
    plan = plan_batch(rows)
    assert all(r.hazards == () for r in plan.rows)
    assert [r.admission for r in plan.rows] == ["worker", "worker"]
    assert plan.worker_admissible == (0, 1)


@pytest.mark.batch_plan
def test_same_folder_same_stem_conflicts(make_settings, tmp_path):
    out = tmp_path / "out"
    rows = [make_settings("a", stem="city", folder=out),
            make_settings("b", stem="city", folder=out,
                          inputs=("n2.geojson", "o2.geojson", "d2.geojson"))]
    plan = plan_batch(rows)
    kinds = {h.kind for r in plan.rows for h in r.hazards}
    assert kinds == {"output_conflict"}
    assert plan.rows[0].admission == "serial"
    assert plan.rows[1].admission == "serial"
    assert "output_conflict with row 1" in plan.rows[0].admission_reasons[0]


@pytest.mark.batch_plan
def test_same_folder_distinct_stems_independent(make_settings, tmp_path):
    """Different stems in one folder write different files — admitted."""
    out = tmp_path / "out"
    rows = [make_settings("a", stem="cityA", folder=out),
            make_settings("b", stem="cityB", folder=out,
                          inputs=("n2.geojson", "o2.geojson", "d2.geojson"))]
    plan = plan_batch(rows)
    assert all(r.admission == "worker" for r in plan.rows)


@pytest.mark.batch_plan
def test_prefix_stems_conflict(make_settings, tmp_path):
    """Row stems 'city' vs 'city_routes': every basename is <stem> or
    <stem>_<suffix> (Engines/Base.py), so one stem prefixing the other is a
    possible collision — serialized (conservative)."""
    out = tmp_path / "out"
    rows = [make_settings("a", stem="city", folder=out),
            make_settings("b", stem="city_routes", folder=out,
                          inputs=("n2.geojson", "o2.geojson", "d2.geojson"))]
    plan = plan_batch(rows)
    assert any(h.kind == "output_conflict" for h in plan.rows[0].hazards)
    assert all(r.admission == "serial" for r in plan.rows)


@pytest.mark.batch_plan
def test_later_row_reads_earlier_rows_output(make_settings, tmp_path):
    """Row 1's input is row 0's published artifact: read-write hazard, and
    the serial commit order is the only safe schedule."""
    data0 = tmp_path / "d0"
    rows = [
        make_settings("producer", data_folder=data0,
                      inputs=("net.geojson", "o.geojson", "d.geojson"),
                      stem="flow_out", folder=data0 / "Results"),
        make_settings("consumer", data_folder=data0 / "Results",
                      inputs=("flow_out.csv", "o.geojson", "d.geojson"),
                      stem="downstream", folder=tmp_path / "out_c"),
    ]
    plan = plan_batch(rows)
    rw = [h for h in plan.rows[1].hazards if h.kind == "read_write"]
    assert rw and rw[0].with_row == 0
    assert plan.rows[0].admission == "serial"
    assert plan.rows[1].admission == "serial"


@pytest.mark.batch_plan
def test_earlier_row_reads_later_rows_output_is_also_a_hazard(make_settings,
                                                              tmp_path):
    """The writer comes SECOND: overlapping out of caller order would clobber
    the file the first row already read.  Direction of the data flow does
    not matter — any read/write overlap is a hazard."""
    shared = tmp_path / "shared"
    rows = [
        make_settings("reader", data_folder=shared,
                      inputs=("net.geojson", "o.geojson", "d.geojson"),
                      stem="reader_out", folder=tmp_path / "out_r"),
        make_settings("writer", data_folder=shared,
                      inputs=("n2.geojson", "o2.geojson", "d2.geojson"),
                      stem="net.geojson",        # writes shared/net.geojson*
                      folder=shared),
    ]
    plan = plan_batch(rows)
    rw = [h for h in plan.rows[0].hazards if h.kind == "read_write"]
    assert rw and rw[0].with_row == 1
    assert plan.rows[0].admission == "serial"
    assert plan.rows[1].admission == "serial"


@pytest.mark.batch_plan
def test_aliased_settings_object_is_a_hazard(make_settings):
    """The same Settings object twice: the serial loop's second row observes
    the first row's mutations (output_folder/naming are written onto it)."""
    s = make_settings("shared")
    plan = plan_batch([s, s])
    kinds = {h.kind for r in plan.rows for h in r.hazards}
    assert kinds == {"aliased_settings"}
    assert plan.rows[0].admission == "serial"
    assert plan.rows[1].admission == "serial"
    assert plan.rows[0].row.aliased_with == (1,)
    assert plan.rows[1].row.aliased_with == (0,)
    assert "aliased_settings with row 1" in plan.rows[0].admission_reasons[0]


@pytest.mark.batch_plan
def test_distinct_objects_with_equal_fields_are_not_aliased(make_settings):
    rows = [make_settings("a"), make_settings("b")]
    rows[1].network_file = rows[0].network_file       # same input path,
    rows[1].origins_file = rows[0].origins_file       # different objects:
    rows[1].destinations_file = rows[0].destinations_file
    plan = plan_batch(rows)
    # shared READS are fine; only writes/aliasing hazard
    assert all(r.admission == "worker" for r in plan.rows)


@pytest.mark.batch_plan
def test_skipped_rows_are_no_hazard_source(make_settings, tmp_path):
    """A row that will be skipped runs no transitions and writes nothing:
    a later row sharing its folder/stem is still provably independent."""
    out = tmp_path / "out"
    rows = [make_settings("broken", inputs=("", "", ""), stem="city",
                          folder=out),
            make_settings("live", stem="city", folder=out,
                          inputs=("n2.geojson", "o2.geojson", "d2.geojson"))]
    plan = plan_batch(rows)
    assert plan.rows[0].admission == "skipped"
    assert plan.rows[1].admission == "worker"
    assert plan.commit_order == (1,)


@pytest.mark.batch_plan
def test_obstacle_file_participates_as_input(make_settings, tmp_path):
    shared = tmp_path / "shared"
    rows = [
        make_settings("writer", data_folder=shared,
                      inputs=("n2.geojson", "o2.geojson", "d2.geojson"),
                      stem="obstacles", folder=shared),
        make_settings("reader", data_folder=shared,
                      inputs=("net.geojson", "o.geojson", "d.geojson"),
                      stem="reader_out", folder=tmp_path / "out_r"),
    ]
    rows[1].obstacle_points_file = "obstacles.geojson"   # the 4th input
    plan = plan_batch(rows)
    assert len(plan.rows[1].row.input_paths) == 4
    rw = [h for h in plan.rows[1].hazards if h.kind == "read_write"]
    assert rw and rw[0].with_row == 0


@pytest.mark.batch_plan
def test_case_insensitive_stems_are_not_independent(make_settings, tmp_path):
    """Dossier 04: 'a different spelling is not independent.'  On a
    case-sensitive filesystem 'City' and 'city' are different files, but
    the planner compares case-insensitively — the conservative direction
    (over-serializing can never corrupt)."""
    out = tmp_path / "out"
    rows = [make_settings("a", stem="City", folder=out),
            make_settings("b", stem="city", folder=out,
                          inputs=("n2.geojson", "o2.geojson", "d2.geojson"))]
    plan = plan_batch(rows)
    assert any(h.kind == "output_conflict" for h in plan.rows[0].hazards)
    assert all(r.admission == "serial" for r in plan.rows)


@pytest.mark.batch_plan
def test_observer_file_participates_as_input(make_settings, tmp_path):
    """Review MAJOR-1: RunFlow loads settings.observer_points_file per row
    (UNA.RunBatch -> AddObservers, resolved like the other layers), so it
    is part of the read set — a flow row consuming an earlier row's
    published observer file is a read-write hazard, not two workers."""
    shared = tmp_path / "shared"
    rows = [
        make_settings("writer", data_folder=shared,
                      inputs=("n2.geojson", "o2.geojson", "d2.geojson"),
                      stem="obs", folder=shared),
        make_settings("consumer", data_folder=shared,
                      inputs=("net.geojson", "o.geojson", "d.geojson"),
                      stem="downstream", folder=tmp_path / "out_c"),
    ]
    rows[1].observer_points_file = "obs.feather"   # writer's output
    plan = plan_batch(rows)
    assert len(plan.rows[1].row.input_paths) == 4
    rw = [h for h in plan.rows[1].hazards if h.kind == "read_write"]
    assert rw and rw[0].with_row == 0
    assert plan.rows[0].admission == "serial"
    assert plan.rows[1].admission == "serial"
    assert plan.worker_admissible == ()


@pytest.mark.batch_plan
@pytest.mark.parametrize("stem", ["../evil", "sub/evil"])
def test_separator_stems_are_not_contained(make_settings, tmp_path, stem):
    """Review MINOR-1: a stem with a path separator writes OUTSIDE the
    modeled folder ('../evil' escapes it, 'sub/evil' descends beneath it).
    The write-set model cannot contain such a row, so NO row is provably
    independent of it — every executing row is serial with the reason,
    even where no literal path overlap is detectable."""
    rows = [
        make_settings("escaper", stem=stem, folder=tmp_path / "out_a"),
        make_settings("innocent", stem="innocent",
                      folder=tmp_path / "out_b",
                      inputs=("nb.geojson", "ob.geojson", "db.geojson")),
    ]
    plan = plan_batch(rows)
    assert plan.worker_admissible == ()
    assert all(r.admission == "serial" for r in plan.rows)
    for r in plan.rows:
        assert any("path separator" in reason
                   for reason in r.admission_reasons)


@pytest.mark.batch_plan
def test_non_settings_object_is_unprovable_serial():
    class Odd:
        pass

    odd = Odd()
    odd.name = "odd"
    odd.network_file = "n.geojson"
    odd.origins_file = "o.geojson"
    odd.destinations_file = "d.geojson"
    odd.data_folder = "/tmp"
    odd.output_folder = ""
    odd.output_file_name = "odd_out"
    plan = plan_batch([odd])
    assert plan.rows[0].admission == "serial"
    assert any("Settings" in r for r in plan.rows[0].admission_reasons)
