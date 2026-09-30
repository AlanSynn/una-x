"""Positive control: one real job through the installed public API.

Executes the frozen smoke manifest against the installed una_legacy wheel:
identity -> one timed public call -> separate traced engagement leg ->
bitwise same-kernel binding -> obligations -> complete record -> append-only
custody.  This is HARNESS's "installed identity and correct dispatch
verified" leg.
"""
from __future__ import annotations

import json

import pytest

from benchmarks.platform.identity import resolve_identity, assert_installed
from benchmarks.platform.records import validate_record, write_session
from benchmarks.platform.runner import run_one_real_job
from benchmarks.platform.validation import validate_output_obligations

pytestmark = pytest.mark.harness_heavy


def test_installed_identity_is_truthful(legacy_python, legacy_wheel_verified,
                                        frozen_constants):
    info = resolve_identity(legacy_python)
    assert_installed(info, "urban_network_analysis",
                     expected_wheel_tree_sha256=
                     frozen_constants["wheel_tree_sha256"],
                     forbidden_roots=[frozen_constants["src_root"]])
    assert info["distribution_version"] == "2.6.0"
    assert "site-packages" in info["module_file"]
    assert not info["module_file"].startswith(frozen_constants["src_root"])


def test_one_real_job_end_to_end(legacy_python, legacy_wheel_verified,
                                 smoke_manifest_frozen, run_out, smoke_root,
                                 frozen_constants):
    from benchmarks.platform.manifest import load_manifest
    manifest = load_manifest(smoke_root / "manifest.json", verify_inputs=True)

    record = run_one_real_job(
        manifest, repo=frozen_constants['repo'], python_exe=legacy_python,
        out_dir=run_out, timeout_s=540.0)

    # correct dispatch: the requested analysis is what actually executed
    assert record["dispatch_called"] == [
        "urban_network_analysis.UNA.RunAccessibility"]
    assert record["effective_backend"] == "reference"
    assert record["requested_backend"] == "reference"
    assert record["export_synced"] is True

    # validation isolation: obligations (schema/rows/features) then record
    obligations = smoke_manifest_frozen["output_obligations"]
    out_manifest = validate_output_obligations(run_out / "leg_timed",
                                               obligations)
    assert out_manifest["validated"]["Results.geojson"]["features"] == 2
    assert out_manifest["validated"]["Results.feather"]["rows"] == 2
    record["output_manifest"] = out_manifest
    record["reference_source_sha256"] = "not-computed-in-this-test"
    validate_record(record)

    # simultaneous process-tree memory was sampled, not parent-only
    assert record["memory"]["peak_is_simultaneous_tree_sum"] is True
    assert record["memory"]["samples_taken"] > 0
    assert record["memory"]["peak_simultaneous_rss_bytes"] > 0

    # separate boundaries: import and compute are individually recorded
    assert 0 < record["timings"]["child_import_s"] < \
        record["timings"]["child_wall_s"]
    assert 0 < record["timings"]["child_compute_s"]

    # append-only custody: same run ID can never be rewritten
    session = write_session(run_out / "sessions", record["run_id"],
                            [record], commands=["pytest test_one_real_job"])
    assert session.is_file()
    with pytest.raises(Exception, match="append-only custody"):
        write_session(run_out / "sessions", record["run_id"], [record])


def test_baseline_determinism_timed_vs_engagement(legacy_python,
                                                  legacy_wheel_verified,
                                                  smoke_manifest_frozen,
                                                  run_out, frozen_constants,
                                                  smoke_root):
    """Bitwise binding between the timed kernel and the traced kernel."""
    from benchmarks.platform.manifest import load_manifest
    manifest = load_manifest(smoke_root / "manifest.json", verify_inputs=True)
    run_one_real_job(manifest, repo=frozen_constants['repo'],
                     python_exe=legacy_python, out_dir=run_out,
                     timeout_s=540.0)
    # run_one_real_job already compared leg_timed vs leg_engagement bitwise
    # (raises BitwiseMismatch on any difference); assert artifacts exist:
    timed = sorted(p.name for p in (run_out / "leg_timed").iterdir()
                   if p.is_file())
    eng = sorted(p.name for p in (run_out / "leg_engagement").iterdir()
                 if p.is_file())
    assert timed == eng == ["Results.feather", "Results.geojson"]
