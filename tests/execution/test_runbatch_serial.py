"""RunBatch serial behavior: existing calls preserved, new plumbing truthful.

Runs the real accessibility engine on the committed 3x3 smoke grid — the
same workload class HARNESS froze — so the report reflects actual rows, not
mocks.  Scope: serial route only (parallel is a later DAG task and raises).
"""
from __future__ import annotations

import pytest

pytestmark = pytest.mark.execution

from urban_network_analysis import (
    ExecutionOptions,
    CacheOptions,
)
from urban_network_analysis.Settings import Settings  # direct class path
# NOTE: urban_network_analysis.UNA is imported inside the `project` fixture,
# NOT at module level: a collection-time import pulls the numba stack and
# Topology.py's USE_PYGEOS env write into the WHOLE-REPO collection phase,
# where it collides with tests/large_e2e conftest's NUMBA_NUM_THREADS
# setdefault (science cutoff tests then fail with "Cannot set
# NUMBA_NUM_THREADS") and the platform_geometry env probe.  Keep heavy
# imports call-time in suites that live in the shared whole-repo run.
# (EXECUTION review F1, 2026-10-01.)


PAIRING_HEADER = (
    "name,data_folder,network_file,origins_file,destinations_file,"
    "search_radius,output_folder,output_wStamp\n")


def _pairing(tmp_path, fixture, extra_rows=()) -> str:
    lines = [PAIRING_HEADER]
    for name, radius in (("r1", 100), ("r2", 200)):
        lines.append(f"{name},{fixture},network.geojson,origins.geojson,"
                     f"destinations.geojson,{radius},{tmp_path / 'out'},"
                     f"False\n")
    lines.extend(extra_rows)
    csv = tmp_path / "pairing.csv"
    csv.write_text("".join(lines))
    return str(csv)


@pytest.fixture()
def project(smoke_fixture):
    from urban_network_analysis import UNA  # call-time: see import note above
    p = UNA(verbosity=0)
    p.settings.data_folder = str(smoke_fixture)
    return p


def test_serial_run_row_outcomes_and_report(project, tmp_path, smoke_fixture):
    out = tmp_path / "out"
    csv = _pairing(tmp_path, smoke_fixture)

    result = project.RunBatch("accessibility", csv)

    assert result is None  # existing contract: RunBatch returns None
    report = project.batch_report
    assert report is not None
    assert report.parallel_requested is False
    assert report.workers_requested is None
    assert report.requested == ExecutionOptions()
    assert report.effective.backend == "reference"  # auto -> trusted reference
    assert report.effective.fallback_reason
    assert report.effective.options == ExecutionOptions()

    assert [r.status for r in report.rows] == ["ran", "ran"]
    assert [r.index for r in report.rows] == [0, 1]
    assert [r.name for r in report.rows] == ["r1", "r2"]
    assert all(r.backend == "reference" for r in report.rows)
    assert all(r.semantic_profile == "una_legacy" for r in report.rows)
    assert report.rows[0].detail is None
    assert report.notes == ()

    # existing behavior preserved: each ran row produced its outputs
    assert (out / "r1.geojson").is_file()
    assert (out / "r1.feather").is_file()
    assert (out / "r2.geojson").is_file()
    assert (out / "r2.feather").is_file()


def test_rows_missing_required_fields_are_skipped_in_the_report(
        project, tmp_path, smoke_fixture):
    """The project-list route can express an emptied field directly (pairing
    CSVs cannot: ApplyRow keeps the field default for an empty cell)."""
    s = Settings()
    s.data_folder = str(smoke_fixture)
    s.network_file = "network.geojson"
    s.origins_file = ""  # explicitly emptied -> missing required field
    s.destinations_file = "destinations.geojson"
    s.output_folder = str(tmp_path / "out")
    s.name = "r_skip"
    s.output_wStamp = False
    project.settings = s
    project.SaveSettingsToProject()

    s2 = Settings()
    s2.data_folder = str(smoke_fixture)
    s2.network_file = "network.geojson"
    s2.origins_file = "origins.geojson"
    s2.destinations_file = "destinations.geojson"
    s2.search_radius = 100
    s2.output_folder = str(tmp_path / "out")
    s2.name = "r_run"
    s2.output_wStamp = False
    project.settings = s2
    project.SaveSettingsToProject()

    project.RunBatch("accessibility")  # no pairing file -> self.projects

    report = project.batch_report
    assert [r.status for r in report.rows] == ["skipped", "ran"]
    assert report.rows[0].name == "r_skip"
    assert "origins_file" in report.rows[0].detail
    assert (tmp_path / "out" / "r_run.geojson").is_file()
    assert not (tmp_path / "out" / "r_skip.geojson").exists()


def test_serial_run_with_explicit_reference_execution(project, tmp_path, smoke_fixture):
    out = tmp_path / "out"
    csv = _pairing(tmp_path, smoke_fixture)
    execution = ExecutionOptions(
        backend="reference",
        cache=CacheOptions(mode="memory", verification="always"))
    project.RunBatch("accessibility", csv, execution=execution)

    report = project.batch_report
    assert report.requested == execution
    assert report.effective.backend == "reference"
    assert report.effective.fallback_reason is None  # honoured verbatim
    assert report.effective.options == execution


def test_instance_execution_is_the_call_level_request(project, tmp_path,
                                                      smoke_fixture):
    """The instance/call-level execution drives ADMISSION and the report's
    request; row profiles are the rows' own identity (pairing rows default
    to una_legacy — deliberately different from the call-level profile)."""
    csv = _pairing(tmp_path, smoke_fixture)
    project.execution = ExecutionOptions(semantic_profile="corrected_v1",
                                         backend="reference")
    project.RunBatch("accessibility", csv)
    report = project.batch_report
    assert report.requested == project.execution
    assert report.requested.semantic_profile == "corrected_v1"
    # rows restored from the CSV carry their own (default) profile
    assert all(r.semantic_profile == "una_legacy" for r in report.rows)
    assert all(r.backend == "reference" for r in report.rows)


def test_workers_on_serial_route_is_recorded_not_used(project, tmp_path, smoke_fixture):
    csv = _pairing(tmp_path, smoke_fixture)
    project.RunBatch("accessibility", csv, workers=3)
    report = project.batch_report
    assert report.workers_requested == 3
    assert report.parallel_requested is False
    assert any("workers=3" in n and "serial" in n for n in report.notes)


def test_existing_positional_call_shapes_still_bind(project, tmp_path, smoke_fixture):
    """Pre-EXECUTION call sites pass (analysis, pairing_file) — both call
    forms must keep binding identically."""
    csv = _pairing(tmp_path, smoke_fixture)
    project.RunBatch("accessibility", csv)
    assert project.batch_report is not None and \
        len(project.batch_report.rows) == 2

    project.RunBatch(analysis="accessibility", pairing_file=csv)
    assert project.batch_report is not None and \
        len(project.batch_report.rows) == 2


def test_report_is_cleared_at_admission_of_a_new_call(project, tmp_path, smoke_fixture):
    csv = _pairing(tmp_path, smoke_fixture)
    project.RunBatch("accessibility", csv)
    first = project.batch_report
    assert first is not None

    from urban_network_analysis import BackendNotAvailableError
    with pytest.raises(BackendNotAvailableError):
        project.RunBatch("accessibility", csv,
                         execution=ExecutionOptions(backend="gpu"))
    # a rejected call does not touch instance state: the last ADMITTED
    # call's report remains (admission failures precede the stale-clear).
    assert project.batch_report is first


def test_per_row_profile_from_pairing_csv_flows_to_outcomes(
        project, tmp_path, smoke_fixture):
    """The four dotted execution keys serialize per Settings ROW: a pairing
    row carrying execution.semantic_profile is that row's reported identity,
    while the call-level request stays whatever the caller asked for."""
    header = PAIRING_HEADER.strip() + (
        ",execution.semantic_profile,execution.backend,"
        "execution.cache.mode,execution.cache.directory\n")
    csv = tmp_path / "pairing.csv"
    csv.write_text(
        header +
        f"r1,{smoke_fixture},network.geojson,origins.geojson,"
        f"destinations.geojson,100,{tmp_path / 'out'},False,"
        "corrected_v1,reference,memory,.una-cache\n")
    project.RunBatch("accessibility", str(csv))

    report = project.batch_report
    assert report.rows[0].semantic_profile == "corrected_v1"
    assert report.requested == ExecutionOptions()  # call level: untouched
    assert report.effective.backend == "reference"
