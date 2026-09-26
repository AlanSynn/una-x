"""Identity guard tests (HARNESS.md behavior 2 + negative tests).

The wheel guard must fail for sibling 'site-packages-fake' prefixes,
editable .pth imports, repository shadowing, changed module hashes and
version drift — using path-RESOLUTION checks, not string prefixes.  Every
case here drives the real run.py or a real subprocess against a fake venv
layout; nothing is mocked at the guard level.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

import stubkit
from stubkit import read_session, run_harness

HARNESS_DIR = stubkit.HARNESS_DIR


# ----------------------------------------------------------------------
# Guard primitives under a real subprocess (path-resolution semantics)
# ----------------------------------------------------------------------
def run_guard_check(code: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-c", code],
                          capture_output=True, text=True, timeout=60)


GUARD_IMPORT = (
    "import sys\n"
    "sys.path.insert(0, {harness!r})\n"
    "from pathlib import Path\n"
    "from harness.identity import check_under_site_packages, "
    "guard_repository_shadow, scan_pth_files, scan_editable_install, "
    "require_dist_info\n"
    "from harness.spec import ValidationError\n"
).format(harness=str(HARNESS_DIR))


def test_site_packages_fake_sibling_prefix_rejected(tmp_path):
    """'/x/site-packages-fake/pkg' is NOT under '/x/site-packages'.

    This is exactly the string-prefix trap the historical runner had; the
    resolution-based guard must refuse it even though the path string
    contains 'site-packages'.
    """
    fake = tmp_path / "site-packages-fake" / "urban_network_analysis"
    declared = tmp_path / "site-packages"
    declared.mkdir(parents=True)
    fake.mkdir(parents=True)
    completed = run_guard_check(
        GUARD_IMPORT +
        f"try:\n"
        f"    check_under_site_packages(Path({str(fake)!r}), [Path({str(declared)!r})])\n"
        f"except ValidationError:\n"
        f"    print('GUARD_REJECTED')\n"
        f"else:\n"
        f"    print('GUARD_ACCEPTED_BAD')\n")
    assert completed.returncode == 0, completed.stderr
    assert "GUARD_REJECTED" in completed.stdout


def test_site_packages_real_containment_accepted(tmp_path):
    sp = tmp_path / "site-packages"
    pkg = sp / "urban_network_analysis"
    pkg.mkdir(parents=True)
    completed = run_guard_check(
        GUARD_IMPORT +
        f"matched = check_under_site_packages(Path({str(pkg)!r}), "
        f"[Path({str(sp)!r})]); print('GUARD_OK', matched)")
    assert "GUARD_OK" in completed.stdout


def test_repository_import_shadow_rejected(tmp_path):
    pkg = tmp_path / "urban_network_analysis"
    completed = run_guard_check(
        GUARD_IMPORT +
        f"try:\n"
        f"    guard_repository_shadow(Path({str(pkg)!r}), "
        f"Path({str(tmp_path)!r}), [])\n"
        f"except ValidationError:\n"
        f"    print('GUARD_REJECTED')\n"
        f"else:\n"
        f"    print('GUARD_ACCEPTED_BAD')\n")
    assert "GUARD_REJECTED" in completed.stdout


def test_sys_path_source_entry_rejected(tmp_path):
    src_entry = tmp_path / "src"
    src_entry.mkdir()
    completed = run_guard_check(
        GUARD_IMPORT +
        f"try:\n"
        f"    guard_repository_shadow(Path({str(tmp_path / 'elsewhere')!r}), "
        f"Path({str(tmp_path)!r}), [{str(src_entry)!r}])\n"
        f"except ValidationError:\n"
        f"    print('GUARD_REJECTED')\n"
        f"else:\n"
        f"    print('GUARD_ACCEPTED_BAD')\n")
    assert "GUARD_REJECTED" in completed.stdout


# ----------------------------------------------------------------------
# End-to-end installed-mode rejections via run.py subprocesses
# ----------------------------------------------------------------------
def test_installed_run_with_stub_venv_passes(tmp_path, stub_venv,
                                             stub_identity, run_dir):
    """Positive control: a properly 'installed' stub arm passes the guard."""
    manifest = stubkit.make_manifest(tmp_path / "w.json", tmp_path / "inputs")
    completed = run_harness(run_dir, manifest, stub_identity,
                            cache_root=tmp_path / "cache")
    assert completed.returncode == 0, completed.stderr
    session = read_session(run_dir)
    assert session["status"] == "valid"
    assert session["measurement_class"] == "installed"
    assert session["qualification_valid"] is True
    assert session["identity"]["identity_kind"] == "installed_wheel"
    job = json.loads((run_dir / "raw" / "job_00000.json").read_text())
    ready = json.loads((run_dir / "raw" / "worker_0_ready.json").read_text())
    assert ready["identity"]["package_root"] == str(stub_venv["package_root"])
    # Per-job provenance ties the record to the identity WITHOUT trusting
    # the ready record alone.
    assert job["identity_sha256"] == ready["identity"]["identity_sha256"]
    assert job["identity_fingerprint"] == ready["identity"]["identity_fingerprint"]
    assert job["identity_kind"] == "installed_wheel"
    assert job["writer"]["writer_peak_active"] >= 1
    assert job["writer"]["writer_limit"] == 1


def _installed_run_rejected(tmp_path, stub_venv, mutate, expected_reason):
    """Copy the stub venv, mutate it, regenerate identity, expect refusal."""
    venv_root = tmp_path / "venv_mutated"
    layout = stubkit.build_stub_venv(venv_root)
    mutate(layout)
    identity_path = tmp_path / "identity_mutated.json"
    stubkit.generate_identity(
        stubkit.CAMPAIGN_PYTHON, arm="stub", kind="installed_wheel",
        out_path=identity_path, site_packages=[layout["site_packages"]])
    out = tmp_path / "run_out"
    out.mkdir()
    manifest = stubkit.make_manifest(tmp_path / "w.json", tmp_path / "inputs")
    completed = run_harness(out, manifest, identity_path,
                            cache_root=tmp_path / "cache")
    assert completed.returncode != 0
    session = read_session(out)
    assert session["status"] != "valid"
    if session["status"] == "invalid":
        assert session["failure"]["reason"] == expected_reason
    return session, completed


def test_editable_pth_import_rejected(tmp_path, stub_venv, stub_identity):
    """A .pth executable hook (editable finder) in site-packages fails the guard."""
    manifest = stubkit.make_manifest(tmp_path / "w.json", tmp_path / "inputs")
    pth = stub_venv["site_packages"] / "__editable___urban_network_analysis_finder.pth"
    pth.write_text("import __editable___urban_network_analysis_finder; "
                   "__editable___urban_network_analysis_finder.install()\n")
    try:
        out = tmp_path / "run_out"
        out.mkdir()
        completed = run_harness(out, manifest, stub_identity,
                                cache_root=tmp_path / "cache")
        assert completed.returncode != 0
        session = read_session(out)
        assert session["status"] == "invalid"
        assert session["failure"]["reason"] == "worker_startup_failed"
        assert ".pth" in str(session["failure"]["detail"])
    finally:
        pth.unlink()


def test_path_pth_into_source_tree_rejected(tmp_path, stub_venv, stub_identity):
    """A plain-path .pth pointing at a source tree must fail the guard."""
    manifest = stubkit.make_manifest(tmp_path / "w.json", tmp_path / "inputs")
    pth = stub_venv["site_packages"] / "zz_shadow.pth"
    pth.write_text("/some/repository/src\n")
    try:
        out = tmp_path / "run_out"
        out.mkdir()
        completed = run_harness(out, manifest, stub_identity,
                                cache_root=tmp_path / "cache")
        assert completed.returncode != 0
        session = read_session(out)
        assert session["status"] == "invalid"
        assert ".pth" in str(session["failure"]["detail"])
    finally:
        pth.unlink()


def test_changed_module_source_hash_invalidates_warm_identity(tmp_path,
                                                              stub_venv,
                                                              stub_identity):
    """Change a cached module's bytes after identity generation -> refusal."""
    out = tmp_path / "run_out"
    out.mkdir()
    manifest = stubkit.make_manifest(tmp_path / "w.json", tmp_path / "inputs")
    target = stub_venv["package_root"] / "UNA.py"
    original = target.read_bytes()
    try:
        target.write_bytes(original + b"\n# tampered\n")
        completed = run_harness(out, manifest, stub_identity,
                                cache_root=tmp_path / "cache")
        assert completed.returncode != 0
        session = read_session(out)
        assert session["status"] == "invalid"
        assert session["failure"]["reason"] == "worker_startup_failed"
        assert "module hashes" in str(session["failure"]["detail"])
    finally:
        target.write_bytes(original)


def test_version_drift_rejected_with_consistent_identity(tmp_path, stub_venv):
    """Same package, different declared version -> identity mismatch."""
    identity_path = tmp_path / "identity_v2.json"
    data = stubkit.generate_identity(
        stubkit.CAMPAIGN_PYTHON, arm="stub", kind="installed_wheel",
        out_path=identity_path, site_packages=[stub_venv["site_packages"]])
    data["package_version"] = "1.2.3-different"
    data = stubkit.recompute_identity_sha(data)
    identity_path.write_text(json.dumps(data))

    out = tmp_path / "run_out"
    out.mkdir()
    manifest = stubkit.make_manifest(tmp_path / "w.json", tmp_path / "inputs")
    completed = run_harness(out, manifest, identity_path,
                            cache_root=tmp_path / "cache")
    assert completed.returncode != 0
    session = read_session(out)
    assert session["status"] == "invalid"
    assert "version mismatch" in str(session["failure"]["detail"])


def test_tampered_identity_integrity_check_rejected(tmp_path, stub_venv,
                                                    stub_identity):
    """Hand-edited identity (no re-sign) must fail the integrity check."""
    data = json.loads(stub_identity.read_text())
    data["package_version"] = "9.9.9-evil"
    tampered = tmp_path / "identity_tampered.json"
    tampered.write_text(json.dumps(data))
    out = tmp_path / "run_out"
    out.mkdir()
    manifest = stubkit.make_manifest(tmp_path / "w.json", tmp_path / "inputs")
    completed = run_harness(out, manifest, tampered,
                            cache_root=tmp_path / "cache")
    assert completed.returncode == 2  # rejected BEFORE work
    session = read_session(out)
    assert session["failure"]["reason"] == "identity_invalid"


def test_package_only_in_fake_sibling_cannot_be_reached(tmp_path):
    """Package physically relocated into a 'site-packages-fake' sibling AFTER
    identity generation; the identity still declares the real site-packages.
    The worker imports only from DECLARED locations, so the import cannot
    resolve — the string-prefix similarity of the sibling name buys nothing.
    (The prefix itself is refused at guard level in
    test_site_packages_fake_sibling_prefix_rejected.)"""
    venv_root = tmp_path / "venv"
    layout = stubkit.build_stub_venv(venv_root)
    identity_path = tmp_path / "identity.json"
    stubkit.generate_identity(
        stubkit.CAMPAIGN_PYTHON, arm="stub", kind="installed_wheel",
        out_path=identity_path,
        site_packages=[layout["site_packages"]])

    fake_sp = venv_root / "lib" / "python3.11" / "site-packages-fake"
    fake_sp.mkdir(parents=True)
    # Move the package + dist-info OUT of the declared site-packages.
    shutil.move(str(layout["package_root"]),
                str(fake_sp / "urban_network_analysis"))
    shutil.move(str(layout["dist_info"]), str(fake_sp / layout["dist_info"].name))

    out = tmp_path / "run_out"
    out.mkdir()
    manifest = stubkit.make_manifest(tmp_path / "w.json", tmp_path / "inputs")
    completed = run_harness(out, manifest, identity_path,
                            cache_root=tmp_path / "cache")
    assert completed.returncode != 0, completed.stdout
    session = read_session(out)
    assert session["status"] == "invalid"
    assert session["qualification_valid"] is False
    # The worker refused at import time; nothing ran.
    assert session["failure"]["reason"] == "worker_startup_failed"


def test_diagnostic_mode_runs_source_and_labels_records(tmp_path):
    """Diagnostic mode: distinctly named, imports the REAL source tree, runs
    one real accessibility analysis on a tiny fixture, and structurally
    cannot emit installed-qualified records.  NUMBA_DISABLE_JIT is legal
    ONLY here (installed mode rejects it before work)."""
    data_dir = tmp_path / "inputs_real"
    input_files = stubkit.make_real_tiny_fixture(data_dir)
    manifest = stubkit.make_manifest(
        tmp_path / "w_real.json", data_dir, analysis="accessibility",
        settings_overrides={
            "network_file": "network.feather",
            "origins_file": "origins.feather",
            "destinations_file": "destinations.feather",
            "search_radius": 250,
            "output_wStamp": False,
            "progressbar": False,
        },
        input_files=input_files)
    identity_path = tmp_path / "identity_diag.json"
    stubkit.generate_identity(
        stubkit.CAMPAIGN_PYTHON, arm="diag", kind="diagnostic_source",
        out_path=identity_path,
        source_root=stubkit.REPO_ROOT / "src",
        extra_sys_path=[stubkit.REPO_ROOT / "src"])
    out = tmp_path / "run_out"
    out.mkdir()
    completed = run_harness(out, manifest, identity_path,
                            diagnostic_source_root=stubkit.REPO_ROOT / "src",
                            cache_root=tmp_path / "cache",
                            extra_env={"NUMBA_DISABLE_JIT": "1"},
                            timeout=600.0)
    assert completed.returncode == 0, completed.stderr[-4000:]
    session = read_session(out)
    assert session["status"] == "valid"
    assert session["measurement_class"] == "diagnostic_source"
    assert session["qualification_valid"] is False
    assert session["installed_qualified"] is False
    assert any("diagnostic" in lim for lim in session["limitations"])
    job = json.loads((out / "raw" / "job_00000.json").read_text())
    ready = json.loads((out / "raw" / "worker_0_ready.json").read_text())
    assert job["measurement_class"] == "diagnostic_source"
    assert ready["identity"]["identity_kind"] == "diagnostic_source"
    assert stubkit.REPO_ROOT / "src" in \
        Path(ready["identity"]["package_root"]).parents
    assert job["identity_sha256"] == ready["identity"]["identity_sha256"]
    # The real public method really ran and was timed.
    assert job["application_ns"] > 0
    assert job["method_called"] == "RunAccessibility"
