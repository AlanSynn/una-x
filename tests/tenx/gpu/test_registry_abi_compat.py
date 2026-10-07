"""Structural ABI compatibility with the retained src backend contracts.

contracts.py is stdlib+numpy only ("importing this module is cheap and
safe everywhere" — its own docstring), so it loads standalone by path
WITHOUT importing the urban_network_analysis package.  The test pins that
gpu.backends.gpu.capability_entry speaks the exact frozen field sets of
KernelCapabilities and CapabilityRegistryEntry."""

import importlib.util
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
CONTRACTS = (REPO_ROOT / "src" / "urban_network_analysis" / "backends"
             / "contracts.py")


def _load_contracts():
    pytest.importorskip("numpy")
    spec = importlib.util.spec_from_file_location(
        "una_contracts_under_test", CONTRACTS)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_capability_entry_speaks_kernel_capabilities_fields():
    contracts = _load_contracts()
    import dataclasses
    fields = {f.name for f in dataclasses.fields(
        contracts.KernelCapabilities)}
    import gpu.backends.gpu as gpub
    entry = gpub.capability_entry(binary_fingerprint="unavailable "
                                  "(no nvcc on this host)",
                                  cell_id="c")
    assert set(entry["capabilities"].keys()) == fields
    assert entry["capabilities"]["backend"] == "gpu"
    assert entry["capabilities"]["qualified"] is False
    assert entry["capabilities"]["evidence_ref"] is None


def test_capability_entry_speaks_registry_entry_fields():
    contracts = _load_contracts()
    import dataclasses
    fields = {f.name for f in dataclasses.fields(
        contracts.CapabilityRegistryEntry)}
    import gpu.backends.gpu as gpub
    entry = gpub.capability_entry(
        binary_fingerprint="unavailable (no nvcc on this host)",
        cell_id="c", memory_estimate_bytes=1024,
        crossover="unmeasured (no device; G11 deliverable)")
    assert set(entry.keys()) == fields
    assert entry["semantic_math_version"] == gpub.SEMANTIC_MATH_VERSION


def test_execution_context_route_vocabulary_includes_gpu():
    contracts = _load_contracts()
    ctx = contracts.ExecutionContext(
        semantic_profile="una_legacy",
        logical_reduction_plan="plan.accessibility.fold.reach_gravity_knn"
                               ".v1.legacy_fastmath",
        backend="gpu", cpu_budget=None, threads_per_worker=1,
        memory_limit_bytes=64 * 1024 * 1024,
        workspace_limit_bytes=16 * 1024 * 1024, queue_depth=1,
        writer_concurrency=1, timeout_s=None, owner="gpu-engineer")
    assert ctx.backend == "gpu"


def test_gpu_surface_is_import_light():
    """Importing the whole gpu/ surface must not pull numba, the engine,
    or any accelerator stack (import-light law; probes stay lazy)."""
    import os
    import subprocess
    import sys
    env = dict(os.environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    code = (
        "import sys; "
        "import gpu, gpu.backends.gpu, gpu.runtime.availability, "
        "gpu.runtime.buffers, gpu.runtime.region, gpu.runtime.errors, "
        "gpu.kernels.host_twins; "
        "banned = ('numba', 'urban_network_analysis', 'torch', 'cupy', "
        "'cuda', 'pycuda'); "
        "leaked = [m for m in banned if m in sys.modules]; "
        "assert not leaked, f'gpu import leaked {leaked}'; "
        "print('hygiene-ok')"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True,
                          text=True, cwd=str(REPO_ROOT), env=env,
                          timeout=120)
    assert proc.returncode == 0, proc.stderr
    assert "hygiene-ok" in proc.stdout
