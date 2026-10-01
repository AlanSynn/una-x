"""Dossier-10 kernel ABI: construction IS the enforcement.

KernelInputs structurally requires stage / semantic_profile /
logical_reduction_plan / owner on every invocation — "profile, model,
partition and ownership are sent to every kernel" is a type-level fact,
not a convention.  The fingerprint pins numerical identity; the restart
contract forbids lying about partial work.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.execution

from urban_network_analysis.backends import (
    ArrayDescriptor,
    ArrayResult,
    CapabilityRegistryEntry,
    ExecutionContext,
    KernelCapabilities,
    KernelInputs,
    KernelOutputs,
    kernel_inputs_fingerprint,
)


def _desc(name="edges", content="payload", **over) -> ArrayDescriptor:
    import hashlib
    kw = dict(dtype="float64", shape=(2, 2), layout="dense",
              content_hash=hashlib.sha256(content.encode()).hexdigest(),
              owner="topology")
    kw.update(over)
    return ArrayDescriptor(name=name, **kw)


def _inputs(**over) -> KernelInputs:
    kw = dict(stage="accessibility.reach",
              semantic_profile="una_legacy",
              logical_reduction_plan="canonical",
              arrays=(_desc(),),
              owner="UNA.RunAccessibility")
    kw.update(over)
    return KernelInputs(**kw)


# ---- ArrayDescriptor -------------------------------------------------------

def test_array_descriptor_rejects_bad_metadata():
    with pytest.raises(ValueError):
        ArrayDescriptor(name="", dtype="float64", shape=(1,),
                        content_hash="ab", owner="topology")
    with pytest.raises(ValueError):
        ArrayDescriptor(name="a", dtype="", shape=(1,),
                        content_hash="ab", owner="topology")
    with pytest.raises(ValueError):
        ArrayDescriptor(name="a", dtype="float64", shape=(1,),
                        content_hash="ab", owner="")
    with pytest.raises(ValueError):
        ArrayDescriptor(name="a", dtype="float64", shape=(1,), layout="sparse",
                        content_hash="ab", owner="topology")
    with pytest.raises(ValueError):
        ArrayDescriptor(name="a", dtype="float64", shape=(-1,),
                        content_hash="ab", owner="topology")


def test_array_descriptor_content_hash_must_be_sha256_hex():
    """Review F3: content_hash is the identity a cache key or parity check
    consumes — it must be a real 64-char sha256 hex string, never invented."""
    import hashlib
    good = hashlib.sha256(b"x").hexdigest()
    assert ArrayDescriptor(name="a", dtype="float64", shape=(1,),
                           content_hash=good,
                           owner="t").content_hash == good
    for bad in ("", "ab", "z" * 64, "A" * 64,  # non-hex / wrong length
                hashlib.sha256(b"x").hexdigest() + "0"):
        with pytest.raises(ValueError):
            ArrayDescriptor(name="a", dtype="float64", shape=(1,),
                            content_hash=bad, owner="t")


# ---- KernelInputs: profile/partition/ownership are structural --------------

@pytest.mark.parametrize("missing", [
    {"stage": ""},
    {"stage": "   "},
    {"semantic_profile": ""},
    {"semantic_profile": None},
    {"logical_reduction_plan": ""},
    {"logical_reduction_plan": None},
    {"owner": ""},
    {"owner": None},
])
def test_kernel_inputs_cannot_exist_without_profile_partition_owner(missing):
    kw = dict(stage="accessibility.reach",
              semantic_profile="una_legacy",
              logical_reduction_plan="canonical",
              arrays=(), owner="UNA.RunAccessibility")
    kw.update(missing)
    with pytest.raises(ValueError):
        KernelInputs(**kw)


def test_kernel_inputs_rejects_duplicate_array_names():
    with pytest.raises(ValueError, match="duplicate"):
        KernelInputs(stage="s", semantic_profile="una_legacy",
                     logical_reduction_plan="canonical",
                     arrays=(_desc("edges"), _desc("edges")),
                     owner="o")


def test_kernel_inputs_is_frozen():
    ki = _inputs()
    with pytest.raises(Exception):
        ki.stage = "other"


# ---- fingerprint ------------------------------------------------------------

def test_fingerprint_is_stable_for_identical_requests():
    assert kernel_inputs_fingerprint(_inputs()) == \
        kernel_inputs_fingerprint(_inputs())


def test_fingerprint_follows_numerical_identity():
    base = kernel_inputs_fingerprint(_inputs())
    assert kernel_inputs_fingerprint(
        _inputs(semantic_profile="corrected_v1")) != base
    assert kernel_inputs_fingerprint(
        _inputs(logical_reduction_plan="stripe64")) != base
    assert kernel_inputs_fingerprint(_inputs(stage="flow.assign")) != base
    assert kernel_inputs_fingerprint(
        _inputs(arrays=(_desc(content="other"),))) != base
    assert kernel_inputs_fingerprint(
        _inputs(arrays=(_desc(layout="csr"),))) != base


def test_fingerprint_excludes_owner():
    """Two owners sending numerically identical work share a fingerprint —
    ownership is tracked per invocation, not in the numerical identity."""
    a = _inputs(owner="UNA.RunAccessibility")
    b = _inputs(owner="compat.madina.accessibility")
    assert a.owner != b.owner
    assert kernel_inputs_fingerprint(a) == kernel_inputs_fingerprint(b)


# ---- KernelOutputs: the restart contract -----------------------------------

def test_complete_outputs_need_no_restart_state():
    ko = KernelOutputs(
        arrays=(ArrayResult(name="reach",
                            array=np.array([1.0, 2.0])),),
        status="complete")
    assert ko.next_logical_state is None


def test_incomplete_outputs_must_name_the_restart_point():
    with pytest.raises(ValueError, match="next_logical_state"):
        KernelOutputs(arrays=(), status="incomplete")
    ok = KernelOutputs(arrays=(), status="incomplete",
                       next_logical_state="stripe:7")
    assert ok.next_logical_state == "stripe:7"


def test_kernel_outputs_reject_unknown_status_and_duplicate_names():
    with pytest.raises(ValueError):
        KernelOutputs(arrays=(), status="partial")
    with pytest.raises(ValueError, match="duplicate"):
        KernelOutputs(
            arrays=(ArrayResult(name="a", array=np.zeros(1)),
                    ArrayResult(name="a", array=np.zeros(1))),
            status="complete")


# ---- capabilities / registry / context -------------------------------------

def _caps(**over) -> KernelCapabilities:
    kw = dict(backend="native", binary_fingerprint="gcc13-abc123",
              supported_profiles=("una_legacy",), stage="accessibility.reach",
              isa_or_device_requirements=("avx2",))
    kw.update(over)
    return KernelCapabilities(**kw)


def test_capabilities_validate_backend_and_fingerprint():
    with pytest.raises(ValueError):
        _caps(backend="tpu")
    with pytest.raises(ValueError):
        _caps(binary_fingerprint="")
    with pytest.raises(ValueError):
        _caps(stage="")
    assert _caps().qualified is False  # unqualified by default


def test_capabilities_reject_empty_tuple_members():
    """Review F3: empty-string members inside supported_profiles /
    isa_or_device_requirements would silently widen a capability claim."""
    with pytest.raises(ValueError):
        _caps(supported_profiles=("una_legacy", ""))
    with pytest.raises(ValueError):
        _caps(supported_profiles=("",))
    with pytest.raises(ValueError):
        _caps(isa_or_device_requirements=("avx2", "  "))


def test_execution_context_backend_is_the_post_admission_route():
    """Review F3: the context carries the route POST-admission — 'auto' is
    resolved away by admit_execution, so only concrete routes are valid."""
    base = dict(semantic_profile="una_legacy",
                logical_reduction_plan="canonical", backend="reference",
                cpu_budget=None, threads_per_worker=1,
                memory_limit_bytes=None, workspace_limit_bytes=None,
                queue_depth=64, writer_concurrency=1, timeout_s=None,
                owner="UNA.RunAccessibility")
    for route in ("reference", "native", "gpu"):
        assert ExecutionContext(**{**base, "backend": route}).backend == route
    for bad in ("auto", "totally-bogus", ""):
        with pytest.raises(ValueError):
            ExecutionContext(**{**base, "backend": bad})


def test_registry_entry_requires_semantic_math_version():
    with pytest.raises(ValueError):
        CapabilityRegistryEntry(capabilities=_caps(),
                                semantic_math_version="")
    entry = CapabilityRegistryEntry(capabilities=_caps(),
                                    semantic_math_version="una-legacy-1")
    assert entry.crossover is None  # no calibrated crossover yet


def test_execution_context_requires_profile_partition_owner():
    base = dict(semantic_profile="una_legacy",
                logical_reduction_plan="canonical", backend="reference",
                cpu_budget=None, threads_per_worker=1,
                memory_limit_bytes=None, workspace_limit_bytes=None,
                queue_depth=64, writer_concurrency=1, timeout_s=None,
                owner="UNA.RunAccessibility")
    for empty in ("semantic_profile", "logical_reduction_plan", "owner"):
        kw = dict(base)
        kw[empty] = ""
        with pytest.raises(ValueError):
            ExecutionContext(**kw)
    ctx = ExecutionContext(**base)
    assert ctx.cancellation is None  # runtime-only handle, lands with FAULTS


def test_backends_package_never_pulls_the_analysis_stack():
    # PYTHONPATH pins the subprocess to THIS tree (repo src), not any
    # installed copy — an installed wheel must never silently stand in.
    repo_src = str(Path(__file__).resolve().parents[2] / "src")
    env = {**os.environ, "PYTHONPATH": repo_src}
    code = (
        "import sys; import urban_network_analysis.backends as b; "
        "pulled = [m for m in ('numba', 'sklearn', 'geopandas', 'pandas') "
        "if m in sys.modules]; assert not pulled, pulled; "
        "assert b.__file__.startswith(sys.argv[1]); "
        "assert hasattr(b, 'KernelInputs'); print('CLEAN')"
    )
    r = subprocess.run([sys.executable, "-c", code, repo_src],
                       capture_output=True, text=True, env=env)
    assert "CLEAN" in r.stdout, r.stderr[-400:]
