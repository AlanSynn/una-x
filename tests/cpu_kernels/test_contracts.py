"""ABI integration (CPU_ALGORITHMS): the kernels package's frozen plan
IDs, stage IDs and semantic profiles flow into the backend-neutral
contracts delivered by EXECUTION — ``KernelInputs`` accepts them, the
fingerprint covers every numerical dependency and deliberately excludes
ownership, and ``KernelOutputs``/``KernelCapabilities``/
``ExecutionContext`` behave per their typed contracts."""
import hashlib

import numpy as np
import pytest

from urban_network_analysis.backends.contracts import (
    ArrayDescriptor,
    ArrayResult,
    CapabilityRegistryEntry,
    ExecutionContext,
    KernelCapabilities,
    KernelInputs,
    KernelOutputs,
    kernel_inputs_fingerprint,
)
from urban_network_analysis.kernels import (
    PLAN_ADJUST,
    PLAN_FOLD,
    PLAN_SCOPE_SEARCH,
    PLANS,
    PROFILE_CORRECTED,
    PROFILE_LEGACY,
    STAGE_ADJUST,
    STAGE_FOLD,
    STAGE_SCOPE_SEARCH,
)


def descriptor(name, arr, layout="dense", owner="topology"):
    return ArrayDescriptor(
        name=name, dtype=str(arr.dtype), shape=arr.shape, layout=layout,
        content_hash=hashlib.sha256(
            np.ascontiguousarray(arr)).hexdigest().lower(), owner=owner)


def kernel_inputs(plan, stage, owner="UNA.RunAccessibility"):
    ptr = np.array([0, 2, 3], dtype=np.int64)
    vec = np.array([1, 0, 0], dtype=np.int64)
    wts = np.array([1.5, 2.5, 0.5], dtype=np.float64)
    return KernelInputs(
        stage=stage, semantic_profile=PROFILE_LEGACY,
        logical_reduction_plan=plan,
        arrays=(descriptor("adjacency_pointer", ptr, layout="csr"),
                descriptor("adjacency_vector", vec, layout="csr"),
                descriptor("adjacency_vector_weights", wts, layout="csr")),
        owner=owner)


def test_plan_ids_and_stages_flow_into_kernel_inputs():
    """Every registered plan ID builds a valid KernelInputs under its
    registered stage — the identity strings are contract-compatible."""
    pairs = {
        PLAN_SCOPE_SEARCH: STAGE_SCOPE_SEARCH,
        PLAN_ADJUST: STAGE_ADJUST,
        PLAN_FOLD: STAGE_FOLD,
    }
    for plan, stage in pairs.items():
        assert plan in PLANS
        inputs = kernel_inputs(plan, stage)
        assert inputs.logical_reduction_plan == plan
        assert inputs.semantic_profile == PROFILE_LEGACY
        assert kernel_inputs_fingerprint(inputs) == \
            kernel_inputs_fingerprint(kernel_inputs(plan, stage))


def test_fingerprint_covers_every_numerical_dependency():
    """Stage, profile, plan, dtype/shape/layout and content hash all
    enter the fingerprint; the owner does not (shared-cache design)."""
    base = kernel_inputs(PLAN_FOLD, STAGE_FOLD)
    ref = kernel_inputs_fingerprint(base)
    assert ref == kernel_inputs_fingerprint(
        kernel_inputs(PLAN_FOLD, STAGE_FOLD, owner="some.other.owner"))

    def variant(**over):
        kw = dict(stage=base.stage,
                  semantic_profile=base.semantic_profile,
                  logical_reduction_plan=base.logical_reduction_plan,
                  arrays=base.arrays, owner=base.owner)
        kw.update(over)
        return kernel_inputs_fingerprint(KernelInputs(**kw))

    assert variant(stage=STAGE_SCOPE_SEARCH) != ref
    assert variant(semantic_profile=PROFILE_CORRECTED) != ref
    assert variant(logical_reduction_plan=PLAN_FOLD + ".x") != ref
    # content change (same shape/dtype)
    wts = np.array([1.5, 2.5, 0.5000001], dtype=np.float64)
    changed = list(base.arrays)
    changed[2] = descriptor("adjacency_vector_weights", wts, layout="csr")
    assert variant(arrays=tuple(changed)) != ref


def test_kernel_outputs_complete_and_incomplete():
    reach = np.arange(3, dtype=np.int64)
    done = KernelOutputs(
        arrays=(ArrayResult(name="reach", array=reach),),
        status="complete")
    assert done.next_logical_state is None
    with pytest.raises(ValueError):
        KernelOutputs(arrays=(), status="incomplete")
    partial = KernelOutputs(arrays=(), status="incomplete",
                            next_logical_state="origin-block-4")
    assert partial.next_logical_state == "origin-block-4"
    with pytest.raises(ValueError):
        KernelOutputs(arrays=(), status="partial")
    with pytest.raises(ValueError):
        KernelOutputs(
            arrays=(ArrayResult(name="reach", array=reach),
                    ArrayResult(name="reach", array=reach)),
            status="complete")


def test_capabilities_and_registry_entry_typed_gates():
    with pytest.raises(ValueError):
        KernelCapabilities(backend="reference",
                           binary_fingerprint="fp",
                           supported_profiles=(PROFILE_LEGACY,),
                           stage=STAGE_SCOPE_SEARCH,
                           isa_or_device_requirements=("avx2",))
    caps = KernelCapabilities(
        backend="native", binary_fingerprint="gcc13-o3-march=native",
        supported_profiles=(PROFILE_LEGACY, PROFILE_CORRECTED),
        stage=STAGE_SCOPE_SEARCH,
        isa_or_device_requirements=("x86-64-v3",),
        known_refusals=("float32-terminal-weights",), qualified=False)
    entry = CapabilityRegistryEntry(
        capabilities=caps,
        semantic_math_version="una_legacy/cpu-algorithms-freeze",
        workload_features={"domain": "W3_medium"},
        memory_estimate_bytes=4096,
        crossover="see BACKEND_POLICY calibrated crossover")
    assert entry.capabilities.qualified is False
    assert entry.crossover is not None


def test_execution_context_post_admission_backends():
    kwargs = dict(
        semantic_profile=PROFILE_LEGACY,
        logical_reduction_plan=PLAN_SCOPE_SEARCH,
        backend="reference", cpu_budget=None, threads_per_worker=1,
        memory_limit_bytes=None, workspace_limit_bytes=None,
        queue_depth=1, writer_concurrency=1, timeout_s=None,
        owner="UNA.RunAccessibility")
    ctx = ExecutionContext(**kwargs)
    assert ctx.backend == "reference"
    with pytest.raises(ValueError):
        ExecutionContext(**{**kwargs, "backend": "auto"})
