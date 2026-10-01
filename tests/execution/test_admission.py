"""Typed admission policy: forced routes raise, auto records, reference is
honoured — and RunBatch enforces all of it BEFORE scientific execution.
"""
from __future__ import annotations

import pytest

pytestmark = pytest.mark.execution

import urban_network_analysis as una_pkg
from urban_network_analysis import (
    BackendNotAvailableError,
    CacheOptions,
    CapabilityError,
    ExecutionNotAdmittedError,
    ExecutionOptions,
)
from urban_network_analysis.Execution import admit_execution
from urban_network_analysis.Settings import Settings  # direct class path


# ---- lazy export surface -------------------------------------------------

def test_lazy_exports_resolve_to_the_same_class_objects():
    assert una_pkg.ExecutionOptions is ExecutionOptions
    assert una_pkg.CacheOptions is CacheOptions
    assert una_pkg.CapabilityError is CapabilityError
    assert una_pkg.BackendNotAvailableError is BackendNotAvailableError
    assert una_pkg.ExecutionNotAdmittedError is ExecutionNotAdmittedError
    assert "ExecutionOptions" in una_pkg.__dir__()


def test_error_hierarchy():
    assert issubclass(BackendNotAvailableError, CapabilityError)
    assert issubclass(ExecutionNotAdmittedError, CapabilityError)
    assert issubclass(CapabilityError, RuntimeError)


# ---- admit_execution ------------------------------------------------------

def test_reference_is_honoured_verbatim():
    o = ExecutionOptions(backend="reference")
    eff = admit_execution(o)
    assert eff.backend == "reference"
    assert eff.fallback_reason is None
    assert eff.options is o


def test_native_and_gpu_raise_without_qualified_capability():
    for backend in ("native", "gpu"):
        with pytest.raises(BackendNotAvailableError) as ei:
            admit_execution(ExecutionOptions(backend=backend))
        assert backend in str(ei.value)


def test_auto_falls_back_to_reference_with_recorded_reason():
    eff = admit_execution(ExecutionOptions(backend="auto"))
    assert eff.backend == "reference"
    assert eff.fallback_reason
    assert "reference" in eff.fallback_reason
    assert eff.options.backend == "auto"  # request preserved, resolution recorded


def test_expert_routes_never_fall_back():
    # the error must be the typed admission failure, not a fallback wrapper
    with pytest.raises(CapabilityError):
        admit_execution(ExecutionOptions(backend="native"))


# ---- RunBatch admission ordering -----------------------------------------
# Admission happens BEFORE any row runs: a rejected request must leave no
# outputs and no batch_report.  (Real serial runs are covered in
# test_runbatch_serial.py; here UNA instances are never run to completion.)

@pytest.fixture()
def project(tmp_path, smoke_fixture):
    from urban_network_analysis import UNA
    p = UNA(verbosity=0)
    p.settings.data_folder = str(smoke_fixture)
    p.settings.network_file = "network.geojson"
    p.settings.origins_file = "origins.geojson"
    p.settings.destinations_file = "destinations.geojson"
    p.settings.output_folder = str(tmp_path / "out")
    return p


def _pairing_csv(tmp_path):
    csv = tmp_path / "pairing.csv"
    csv.write_text(
        "name,network_file,origins_file,destinations_file,search_radius\n"
        "r1,network.geojson,origins.geojson,destinations.geojson,100\n")
    return str(csv)


def test_runbatch_parallel_true_is_admitted_and_delegates(project, tmp_path,
                                                          monkeypatch):
    """BATCH_EXEC staged delivery closed: parallel=True no longer raises
    the typed unadmitted-mode error — it delegates to the parallel batch
    runtime after admission.  Observed via a sentinel coordinator (this
    suite never runs rows to completion); the runtime's own behavior
    tests live in tests/batch/runtime/."""
    import urban_network_analysis.batch.runtime as batch_runtime

    seen = {}

    def _sentinel(una, analysis, **kwargs):
        seen['analysis'] = analysis
        seen['parallel_requested'] = kwargs['effective'] is not None
        assert una is project

    monkeypatch.setattr(batch_runtime, "run_batch_parallel", _sentinel)
    project.RunBatch("accessibility", _pairing_csv(tmp_path), parallel=True)
    assert seen['analysis'] == "accessibility"
    assert project.batch_report is None      # sentinel didn't set one
    assert not (tmp_path / "out").exists() or not any((tmp_path / "out").iterdir())


def test_runbatch_forced_native_raises_before_any_work(project, tmp_path):
    with pytest.raises(BackendNotAvailableError):
        project.RunBatch(
            "accessibility", _pairing_csv(tmp_path),
            execution=ExecutionOptions(backend="native"))
    assert project.batch_report is None
    out = tmp_path / "out"
    assert not out.exists() or not any(out.iterdir())


def test_runbatch_forced_gpu_raises_before_any_work(project, tmp_path):
    with pytest.raises(BackendNotAvailableError):
        project.RunBatch("accessibility", _pairing_csv(tmp_path),
                         execution=ExecutionOptions(backend="gpu"))
    assert project.batch_report is None


def test_runbatch_rejects_non_execution_options(project, tmp_path):
    with pytest.raises(TypeError):
        project.RunBatch("accessibility", _pairing_csv(tmp_path),
                         execution={"backend": "reference"})


def test_runbatch_rejects_workers_below_one(project, tmp_path):
    with pytest.raises(ValueError):
        project.RunBatch("accessibility", _pairing_csv(tmp_path), workers=0)


def test_runbatch_rejects_non_integer_workers(project, tmp_path):
    """Review F8: a non-int workers value must hit the documented
    ValueError path, not an incidental TypeError from the comparison."""
    for bad in ("4", 1.5, True):
        with pytest.raises(ValueError):
            project.RunBatch("accessibility", _pairing_csv(tmp_path),
                             workers=bad)


def test_runbatch_unknown_analysis_raises_value_error(project):
    # unknown analysis is the pre-existing ValueError (unchanged contract)
    with pytest.raises(ValueError):
        project.RunBatch("centrality")


def test_row_forcing_native_raises_before_any_work(project, tmp_path,
                                                   smoke_fixture):
    """A Settings ROW can carry execution too (the four dotted keys are
    per-row Settings state).  A row forcing native/gpu must fail the call
    BEFORE any row runs — never silently run on the call-level route."""
    s = Settings()
    s.data_folder = str(smoke_fixture)
    s.network_file = "network.geojson"
    s.origins_file = "origins.geojson"
    s.destinations_file = "destinations.geojson"
    s.output_folder = str(tmp_path / "out")
    s.name = "row_native"
    s.output_wStamp = False
    s.execution = ExecutionOptions(backend="native",
                                   semantic_profile="corrected_v1")
    project.settings = s
    project.SaveSettingsToProject()

    with pytest.raises(BackendNotAvailableError) as ei:
        project.RunBatch("accessibility")
    assert "row_native" in str(ei.value)
    assert project.batch_report is None
    out = tmp_path / "out"
    assert not out.exists() or not any(out.iterdir())
