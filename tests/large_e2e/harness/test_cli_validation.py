"""CLI validation negative tests: every failure must happen BEFORE work.

Each case runs the real run.py subprocess and asserts exit code 2, a
"rejected_before_work" session record, and that no worker processes,
outputs or warmups were produced.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

import stubkit
from stubkit import read_session, run_harness


def assert_rejected_before_work(completed, out_dir: Path, reason: str | None = None):
    assert completed.returncode == 2, (
        f"expected exit 2, got {completed.returncode}\n"
        f"stdout={completed.stdout}\nstderr={completed.stderr}")
    session = read_session(out_dir)
    assert session["status"] == "rejected_before_work"
    assert session["qualification_valid"] is False
    assert session["installed_qualified"] is False
    if reason is not None:
        assert session["failure"]["reason"] == reason
    # No work happened: no outputs, no warmup, no raw job records.
    assert not (out_dir / "warmup").exists()
    assert not (out_dir / "outputs").exists()
    raw = out_dir / "raw"
    if raw.exists():
        assert not list(raw.glob("job_*.json"))
    return session


def test_missing_required_arguments_fail(tmp_path):
    """A missing frozen parameter must fail before any job execution."""
    import subprocess
    completed = subprocess.run(
        [sys.executable, str(stubkit.RUN_PY), "--arm", "x"],
        capture_output=True, text=True, cwd=str(tmp_path), timeout=60)
    assert completed.returncode == 2
    assert "--manifest" in completed.stderr


def test_nonexistent_manifest_rejected(tmp_path, stub_identity, run_dir):
    manifest = tmp_path / "missing.json"
    completed = run_harness(run_dir, manifest, stub_identity)
    assert_rejected_before_work(completed, run_dir, "manifest_invalid")


def test_zero_cpu_budget_rejected(tmp_path, stub_identity, run_dir):
    manifest = stubkit.make_manifest(tmp_path / "w.json", tmp_path / "inputs")
    completed = run_harness(run_dir, manifest, stub_identity, cpu_budget=0)
    assert completed.returncode == 2  # argparse: must be a positive integer


def test_negative_worker_count_rejected(tmp_path, stub_identity, run_dir):
    manifest = stubkit.make_manifest(tmp_path / "w.json", tmp_path / "inputs")
    completed = run_harness(run_dir, manifest, stub_identity, workers=-3)
    assert completed.returncode == 2


def test_non_integral_jobs_rejected(tmp_path, stub_identity, run_dir):
    manifest = stubkit.make_manifest(tmp_path / "w.json", tmp_path / "inputs")
    completed = run_harness(run_dir, manifest, stub_identity, jobs=2.5)
    assert completed.returncode == 2


def test_zero_memory_budget_rejected(tmp_path, stub_identity, run_dir):
    manifest = stubkit.make_manifest(tmp_path / "w.json", tmp_path / "inputs")
    completed = run_harness(run_dir, manifest, stub_identity, memory_budget_mib=0)
    assert completed.returncode == 2


def test_zero_timeout_rejected(tmp_path, stub_identity, run_dir):
    manifest = stubkit.make_manifest(tmp_path / "w.json", tmp_path / "inputs")
    completed = run_harness(run_dir, manifest, stub_identity, timeout_s=0)
    assert completed.returncode == 2


def test_nonempty_out_dir_rejected(tmp_path, stub_identity, run_dir):
    manifest = stubkit.make_manifest(tmp_path / "w.json", tmp_path / "inputs")
    (run_dir / "previous_run_leftover.txt").write_text("stale")
    completed = run_harness(run_dir, manifest, stub_identity)
    session = assert_rejected_before_work(completed, run_dir, "path_collision")
    assert "must be a new (or empty) unique directory" in \
        session["failure"]["detail"]["error"]


def test_out_colliding_with_cache_root_rejected(tmp_path, stub_identity):
    manifest = stubkit.make_manifest(tmp_path / "w.json", tmp_path / "inputs")
    out = tmp_path / "same_dir"
    completed = run_harness(out, manifest, stub_identity,
                            cache_root=out)
    assert_rejected_before_work(completed, out, "path_collision")


def test_out_nested_in_cache_root_rejected(tmp_path, stub_identity):
    manifest = stubkit.make_manifest(tmp_path / "w.json", tmp_path / "inputs")
    cache_root = tmp_path / "cache"
    out = cache_root / "nested_out"
    out.parent.mkdir(parents=True, exist_ok=True)
    completed = run_harness(out, manifest, stub_identity, cache_root=cache_root)
    assert_rejected_before_work(completed, out, "path_collision")


def test_out_colliding_with_input_dir_rejected(tmp_path, stub_identity):
    data_dir = tmp_path / "inputs"
    manifest = stubkit.make_manifest(tmp_path / "w.json", data_dir)
    completed = run_harness(data_dir / "out", manifest, stub_identity)
    assert_rejected_before_work(completed, data_dir / "out", "path_collision")


def test_unknown_analysis_rejected_not_defaulted(tmp_path, stub_identity, run_dir):
    manifest = stubkit.make_manifest(tmp_path / "w.json", tmp_path / "inputs")
    data = json.loads(manifest.read_text())
    data["analysis"] = "centrality"
    manifest.write_text(json.dumps(data))
    completed = run_harness(run_dir, manifest, stub_identity)
    session = assert_rejected_before_work(completed, run_dir, "manifest_invalid")
    assert "centrality" in session["failure"]["detail"]["error"]


def test_malformed_manifest_unknown_top_key_rejected(tmp_path, stub_identity, run_dir):
    manifest = stubkit.make_manifest(tmp_path / "w.json", tmp_path / "inputs")
    data = json.loads(manifest.read_text())
    data["surprise_key"] = True
    manifest.write_text(json.dumps(data))
    completed = run_harness(run_dir, manifest, stub_identity)
    assert_rejected_before_work(completed, run_dir, "manifest_invalid")


def test_malformed_manifest_bad_schema_version_rejected(tmp_path, stub_identity, run_dir):
    manifest = stubkit.make_manifest(tmp_path / "w.json", tmp_path / "inputs")
    data = json.loads(manifest.read_text())
    data["schema_version"] = 99
    manifest.write_text(json.dumps(data))
    completed = run_harness(run_dir, manifest, stub_identity)
    assert_rejected_before_work(completed, run_dir, "manifest_invalid")


def test_manifest_settings_output_folder_rejected(tmp_path, stub_identity, run_dir):
    manifest = stubkit.make_manifest(tmp_path / "w.json", tmp_path / "inputs")
    data = json.loads(manifest.read_text())
    data["settings"]["output_folder"] = "/somewhere/evil"
    manifest.write_text(json.dumps(data))
    completed = run_harness(run_dir, manifest, stub_identity)
    assert_rejected_before_work(completed, run_dir, "manifest_invalid")


def test_manifest_settings_unknown_key_rejected(tmp_path, stub_identity, run_dir):
    manifest = stubkit.make_manifest(tmp_path / "w.json", tmp_path / "inputs")
    data = json.loads(manifest.read_text())
    data["settings"]["not_a_real_setting"] = 1
    manifest.write_text(json.dumps(data))
    completed = run_harness(run_dir, manifest, stub_identity)
    session = assert_rejected_before_work(completed, run_dir, "manifest_invalid")
    assert "not_a_real_setting" in session["failure"]["detail"]["error"]


def test_w_beyond_cpu_budget_rejected(tmp_path, stub_identity, run_dir):
    manifest = stubkit.make_manifest(tmp_path / "w.json", tmp_path / "inputs")
    completed = run_harness(run_dir, manifest, stub_identity,
                            workers=4, numba_threads=2, cpu_budget=4)
    session = assert_rejected_before_work(completed, run_dir, "admission_refused")
    assert "W*H_numba" in session["failure"]["detail"]["error"]


def test_numba1_with_larger_flow_executor_mismatch_detected(tmp_path, run_dir,
                                                            stub_identity):
    """NUMBA_NUM_THREADS=1 beside a larger flow executor must be caught."""
    manifest = stubkit.make_manifest(tmp_path / "w.json", tmp_path / "inputs",
                                     analysis="flow")
    completed = run_harness(run_dir, manifest, stub_identity, workers=1,
                            numba_threads=1, flow_stripes=4, cpu_budget=2)
    session = assert_rejected_before_work(completed, run_dir, "admission_refused")
    assert "flow admission" in session["failure"]["detail"]["error"]


def test_flow_w_times_h_plus_k_must_fit_budget(tmp_path, stub_identity, run_dir):
    manifest = stubkit.make_manifest(tmp_path / "w.json", tmp_path / "inputs",
                                     analysis="flow")
    completed = run_harness(run_dir, manifest, stub_identity, workers=2,
                            numba_threads=1, flow_stripes=4, cpu_budget=8)
    session = assert_rejected_before_work(completed, run_dir, "admission_refused")
    assert "W*(H_numba+K)" in session["failure"]["detail"]["error"]


def test_memory_budget_above_available_rejected(tmp_path, stub_identity, run_dir):
    import psutil
    available_mib = psutil.virtual_memory().available // (1024 * 1024)
    manifest = stubkit.make_manifest(tmp_path / "w.json", tmp_path / "inputs")
    completed = run_harness(run_dir, manifest, stub_identity,
                            memory_budget_mib=int(available_mib * 2))
    session = assert_rejected_before_work(completed, run_dir, "admission_refused")
    assert "memory budget" in session["failure"]["detail"]["error"]


def test_pythonpath_rejected_in_installed_mode(tmp_path, stub_identity, run_dir):
    manifest = stubkit.make_manifest(tmp_path / "w.json", tmp_path / "inputs")
    completed = run_harness(run_dir, manifest, stub_identity,
                            extra_env={"PYTHONPATH": "/some/source/tree"})
    session = assert_rejected_before_work(completed, run_dir,
                                          "environment_policy")
    assert "PYTHONPATH" in session["failure"]["detail"]["error"]


def test_numba_disable_jit_rejected_in_installed_mode(tmp_path, stub_identity,
                                                      run_dir):
    manifest = stubkit.make_manifest(tmp_path / "w.json", tmp_path / "inputs")
    completed = run_harness(run_dir, manifest, stub_identity,
                            extra_env={"NUMBA_DISABLE_JIT": "1"})
    session = assert_rejected_before_work(completed, run_dir,
                                          "environment_policy")
    assert "NUMBA_DISABLE_JIT" in session["failure"]["detail"]["error"]


def test_unknown_test_fault_rejected(tmp_path, stub_identity, run_dir):
    manifest = stubkit.make_manifest(tmp_path / "w.json", tmp_path / "inputs")
    completed = run_harness(run_dir, manifest, stub_identity,
                            test_fault="not_a_fault")
    assert_rejected_before_work(completed, run_dir, "unknown_test_fault")


def test_diagnostic_mode_requires_diagnostic_identity(tmp_path, stub_identity,
                                                      run_dir):
    manifest = stubkit.make_manifest(tmp_path / "w.json", tmp_path / "inputs")
    completed = run_harness(run_dir, manifest, stub_identity,
                            diagnostic_source_root=tmp_path)
    assert_rejected_before_work(completed, run_dir, "identity_invalid")


def test_cached_filename_with_changed_bytes_rejected(tmp_path, stub_identity,
                                                     run_dir):
    """Input rehash: same filenames, mutated bytes -> fail before work."""
    data_dir = tmp_path / "inputs"
    manifest = stubkit.make_manifest(tmp_path / "w.json", data_dir)

    # First run succeeds and caches nothing hostile: inputs stay in place.
    ok = run_harness(run_dir, manifest, stub_identity)
    assert ok.returncode == 0, ok.stderr

    # Second run on a MUTATED input (same filename, changed bytes).
    run_dir2 = tmp_path / "run_out_2"
    run_dir2.mkdir()
    (data_dir / "origins.bin").write_bytes(b"tampered payload\n")
    completed = run_harness(run_dir2, manifest, stub_identity)
    session = assert_rejected_before_work(completed, run_dir2,
                                          "input_rehash_mismatch")
    detail = session["failure"]["detail"]["error"]
    assert "origins.bin" in detail
    # No worker work happened after the rehash failure.
    assert not (run_dir2 / "warmup").exists()
