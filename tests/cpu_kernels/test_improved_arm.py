"""Improved-arm conformance (CPU_ALGORITHMS): the compacted-producer
arm is bitwise-equal to the reference oracle AND to the compiled
engines, on every admitted case, with honest refusal routing — and its
attribution constants record the CPU_LAYOUT measurements verbatim."""
import numpy as np
import pytest

from urban_network_analysis.Engines import _large_access_scratch as LAS
from urban_network_analysis.Engines.AccessibilityWElevation import (
    integrated_scope_access,
)
from urban_network_analysis.kernels import (
    ATTRIBUTION,
    ATTRIBUTION_EVIDENCE,
    IMPROVED_ARM_ID,
    integrated_scope_access_oracle,
    scope_access_compacted,
)

from conftest import CASE_PARAMS, bits, build_case


@pytest.mark.parametrize("name", CASE_PARAMS)
def test_compacted_arm_bitwise(name):
    """Arm outputs == reference oracle outputs == engine outputs,
    byte-for-byte; route attribution reports 'compacted' on admitted
    domains and the delegated legacy route otherwise."""
    from conftest import case_kwargs
    c = build_case(3 + CASE_PARAMS.index(name), **case_kwargs(name))
    fold = dict(gravity_beta=0.02, gravity_plateau=5.0,
                gravity_logistic_midpoint=8.0, gravity_growth_rate=0.3,
                knn_decay="exponential",
                knn_weights=np.array([0.9, 0.5, 0.25, 0.1]))
    engine = integrated_scope_access(
        c["o"], c["ow"], c["ptr"], c["vec"], c["wts"], c["flag"],
        c["d"], c["dw"], c["dwt"],
        fold["gravity_beta"], fold["gravity_plateau"],
        fold["gravity_logistic_midpoint"], fold["gravity_growth_rate"],
        fold["knn_decay"], fold["knn_weights"], c["cutoff"])
    reference = integrated_scope_access_oracle(
        c["o"], c["ow"], c["ptr"], c["vec"], c["wts"], c["flag"],
        c["d"], c["dw"], c["dwt"],
        fold["gravity_beta"], fold["gravity_plateau"],
        fold["gravity_logistic_midpoint"], fold["gravity_growth_rate"],
        fold["knn_decay"], fold["knn_weights"], c["cutoff"])
    arm = scope_access_compacted(
        c["o"], c["ow"], c["ptr"], c["vec"], c["wts"], c["flag"],
        c["d"], c["dw"], c["dwt"],
        fold["gravity_beta"], fold["gravity_plateau"],
        fold["gravity_logistic_midpoint"], fold["gravity_growth_rate"],
        fold["knn_decay"], fold["knn_weights"], c["cutoff"])

    admitted, _ = LAS._a1_scope_admits(
        c["ptr"], c["vec"], c["wts"], c["flag"], c["o"], c["ow"],
        c["cutoff"])
    assert arm.route == ("compacted" if admitted
                         else "delegated:" + reference.route)
    for arm_arr, ref_arr, eng_arr in (
            (arm.reach, reference.reach, engine[0]),
            (arm.gravity_exponential, reference.gravity_exponential, engine[1]),
            (arm.gravity_logistic, reference.gravity_logistic, engine[2]),
            (arm.knn_access, reference.knn_access, engine[3])):
        assert bits(arm_arr) == bits(ref_arr) == bits(eng_arr)


def test_compacted_arm_empty_battery():
    """Zero origins and zero destinations run through the compacted
    route with empty (correctly typed) outputs."""
    ptr = np.array([0], dtype=np.int64)
    o = np.empty((0, 2), dtype=np.int64)
    ow = np.empty((0, 2), dtype=np.float64)
    d = np.empty((0, 2), dtype=np.int64)
    dw = np.empty((0, 2), dtype=np.float64)
    dwt = np.empty(0, dtype=np.float64)
    arm = scope_access_compacted(
        o, ow, ptr, np.empty(0, dtype=np.int64),
        np.empty(0, dtype=np.float64), np.empty(0, dtype=np.bool_),
        d, dw, dwt, 0.02, 5.0, 8.0, 0.3, "exponential",
        np.array([0.5]), 5.0)
    assert arm.route == "compacted"
    assert arm.reach.shape == (0,) and arm.reach.dtype == np.int64
    assert arm.knn_access.shape == (0,)


def test_compacted_admits_refuses_giant_destination_counts(monkeypatch):
    """The scratch cap (threads * (16d + 16) bytes under 256 MiB) is
    enforced by the mirror exactly as the C module's guard did —
    probed through an inflated thread bound instead of allocating
    256 MiB of destination rows."""
    from urban_network_analysis.kernels import scope_access as SA
    from urban_network_analysis.kernels import compacted as CP
    c = build_case(9)
    assert CP.compacted_admits(
        c["ptr"], c["vec"], c["wts"], c["flag"], c["o"], c["ow"],
        c["d"], c["dw"], c["dwt"], c["cutoff"]) is True
    monkeypatch.setattr(SA, "_default_threads_bound", lambda: 10 ** 9)
    assert CP.compacted_admits(
        c["ptr"], c["vec"], c["wts"], c["flag"], c["o"], c["ow"],
        c["d"], c["dw"], c["dwt"], c["cutoff"]) is False


def test_attribution_constants_are_the_freeze_record():
    """The attribution table records the CPU_LAYOUT paired medians
    verbatim and states the wall-neutral disposition — the numbers a
    NATIVE_CORE/GPU estimate must cite, not re-derive."""
    assert IMPROVED_ARM_ID == "scope_access_compacted_v1"
    assert ATTRIBUTION["W3_medium_proxy_4t"]["speedup_median"] == 0.9895
    assert ATTRIBUTION["W3_medium_proxy_1t"]["speedup_median"] == 1.0029
    assert ATTRIBUTION["W1_small_genuine_4t"]["speedup_median"] == 0.8512
    assert ATTRIBUTION["outputs"].startswith("bitwise-equal")
    assert "BENCHMARKS.md" in ATTRIBUTION["gate"]
    assert "cpul-20261002T2305Z" in ATTRIBUTION_EVIDENCE
