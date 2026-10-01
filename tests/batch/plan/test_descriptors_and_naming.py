"""Row descriptors: the serial loop's naming decisions, captured at their
original logical boundaries — recorded, never applied to live Settings."""
from __future__ import annotations

import os

import pytest

from urban_network_analysis.batch import plan_batch


@pytest.mark.batch_plan
def test_decisions_captured_not_applied(make_settings):
    """The serial loop mutates s.output_folder and s.output_file_name; the
    planner records the same decisions WITHOUT touching the live objects."""
    s = make_settings("alpha")           # no output_folder, stem "Results"
    s.output_file_name = "Results"       # default; triggers substitution
    plan = plan_batch([s], script_output_folder="/tmp/script_out")
    d = plan.rows[0].row
    assert d.output_folder == "/tmp/script_out"
    assert d.output_folder_defaulted is True
    assert d.output_stem == "alpha"      # "Results" -> settings.name decision
    assert d.output_name_substituted is True
    # live objects untouched:
    assert s.output_folder is None           # Settings default
    assert s.output_file_name == "Results"


@pytest.mark.batch_plan
def test_output_folder_precedence_row_over_script_over_default(make_settings,
                                                                tmp_path):
    s_row = make_settings("r", folder=tmp_path / "own")
    plan = plan_batch([s_row], script_output_folder="/tmp/script_out")
    assert plan.rows[0].row.output_folder == str(tmp_path / "own")
    assert plan.rows[0].row.output_folder_defaulted is False

    s_default = make_settings("d")       # no row value, no script value
    plan = plan_batch([s_default])
    d = plan.rows[0].row
    assert d.output_folder == os.path.join(s_default.data_folder, "Results")
    assert d.output_folder_defaulted is True


@pytest.mark.batch_plan
def test_input_paths_canonicalized_through_data_folder(make_settings):
    s = make_settings("a")               # relative inputs + data_folder
    plan = plan_batch([s])
    d = plan.rows[0].row
    assert len(d.input_paths) == 3       # obstacle file unset -> 3 inputs
    for field in ("network_file", "origins_file", "destinations_file"):
        expected = os.path.realpath(
            os.path.normpath(os.path.join(s.data_folder,
                                          getattr(s, field))))
        assert expected in d.input_paths
    assert all(os.path.isabs(p) for p in d.input_paths)


@pytest.mark.batch_plan
def test_symlinked_input_resolves_to_the_same_artifact(tmp_path, make_settings):
    """A different spelling via symlink is NOT independent: the reader of
    the link and the writer into the real location hazard on each other."""
    real_dir = tmp_path / "real"
    real_dir.mkdir()
    real = real_dir / "city.csv"
    real.write_text("x,y\n1,2\n")
    link = tmp_path / "links"
    link.mkdir()
    link_net = link / "city_link.csv"
    os.symlink(real, link_net)

    reader = make_settings("reader",
                           data_folder=link,
                           inputs=("city_link.csv", "o.geojson", "d.geojson"),
                           stem="reader_out",
                           folder=tmp_path / "out_reader")
    writer = make_settings("writer",
                           data_folder=real_dir,
                           inputs=("city.csv", "o.geojson", "d.geojson"),
                           stem="city",          # writes real/city*.csv
                           folder=real_dir)      # output folder == input dir
    plan = plan_batch([reader, writer])
    rw = [h for h in plan.rows[0].hazards if h.kind == "read_write"]
    assert rw and rw[0].with_row == 1     # reader hazards on the writer
    assert plan.rows[0].admission == "serial"
    assert plan.rows[1].admission == "serial"


@pytest.mark.batch_plan
def test_semantic_profile_recorded_per_row(make_settings):
    import dataclasses
    rows = [make_settings("legacy"), make_settings("corrected")]
    rows[1].execution = dataclasses.replace(rows[1].execution,
                                            semantic_profile="corrected_v1")
    plan = plan_batch(rows)
    assert plan.rows[0].row.semantic_profile == "una_legacy"
    assert plan.rows[1].row.semantic_profile == "corrected_v1"


@pytest.mark.batch_plan
def test_skipped_rows_keep_descriptors_and_order(make_settings):
    rows = [make_settings("ok0"),
            make_settings("broken", inputs=("", "o.geojson", "d.geojson")),
            make_settings("ok2")]
    plan = plan_batch(rows)
    assert [r.row.index for r in plan.rows] == [0, 1, 2]
    # the substitution decision fires only AFTER the skip gate: a skipped
    # row keeps its raw naming (review NOTE-1) and reads nothing
    assert plan.rows[1].row.output_stem == "Results"
    assert plan.rows[1].row.output_name_substituted is False
    assert plan.rows[1].row.input_paths == ()
    assert plan.rows[1].admission == "skipped"
    # the skip does not disturb the executed rows' caller-order positions
    assert plan.commit_order == (0, 2)
