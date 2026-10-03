"""Frozen kernel contracts for the accessibility scope-access family
(task CPU_ALGORITHMS, campaign una-platform-2026-09).

The freeze (SPEC.md, this directory) consists of:
- the characterized una_legacy scope-search schedule with its
  fast-math comparison and heap order laws (`scope_access`, executed
  numba-free and pinned bitwise against the compiled engines);
- the canonical reduction plans (`reductions`) whose ID strings flow
  into ``backends.contracts.KernelInputs.logical_reduction_plan``;
- the improved-CPU comparison arm (`compacted`): CPU_LAYOUT candidate
  C4, bitwise-equal, wall-neutral by measurement, attribution recorded
  in-module — never a default route;
- the conformance suite (`tests/cpu_kernels`) that pins all of it
  against the compiled engines per platform.

Consumed, not redefined: the backend-neutral ABI
(``urban_network_analysis.backends.contracts``, task EXECUTION) and
the corrected_v1 independent reference
(``urban_network_analysis.reference.*``, task SCIENCE).  Semantic
profiles are ``una_legacy`` (this package) and ``corrected_v1``
(reference package); cross-profile comparison is an error.
"""
from .compacted import (
    ATTRIBUTION,
    ATTRIBUTION_EVIDENCE,
    IMPROVED_ARM_ID,
    compacted_admits,
    scope_access_compacted,
)
from .reductions import (
    PLAN_ADJUST,
    PLAN_FOLD,
    PLAN_FOLD_KEPT,
    PLAN_FOLD_KEPT_LEGACY,
    PLAN_FOLD_LEGACY,
    PLAN_SCOPE_SEARCH,
    PLANS,
    STAGE_ADJUST,
    STAGE_FOLD,
    STAGE_SCOPE_SEARCH,
    adjust_destination_distances_plan,
    collect_kept,
    fold_reach_gravity_knn_kept_legacy_plan,
    fold_reach_gravity_knn_kept_plan,
    fold_reach_gravity_knn_legacy_plan,
    fold_reach_gravity_knn_plan,
)
from .scope_access import (
    ROUTE_A1,
    ROUTE_A3,
    ROUTE_COMPACT,
    ScopeAccessResult,
    fast_lt,
    fast_le,
    integrated_scope_access_oracle,
    scope_route_admits,
    scope_search_schedule,
    tail_admits,
)

PROFILE_LEGACY = "una_legacy"
PROFILE_CORRECTED = "corrected_v1"

__all__ = [
    "PROFILE_LEGACY", "PROFILE_CORRECTED",
    "STAGE_SCOPE_SEARCH", "STAGE_ADJUST", "STAGE_FOLD",
    "PLAN_SCOPE_SEARCH", "PLAN_ADJUST", "PLAN_FOLD", "PLAN_FOLD_KEPT",
    "PLANS",
    "ROUTE_A3", "ROUTE_A1", "ROUTE_COMPACT",
    "ScopeAccessResult",
    "fast_lt", "fast_le",
    "scope_route_admits", "tail_admits",
    "scope_search_schedule",
    "integrated_scope_access_oracle",
    "adjust_destination_distances_plan",
    "collect_kept",
    "fold_reach_gravity_knn_plan",
    "fold_reach_gravity_knn_kept_plan",
    "fold_reach_gravity_knn_legacy_plan",
    "fold_reach_gravity_knn_kept_legacy_plan",
    "IMPROVED_ARM_ID", "ATTRIBUTION", "ATTRIBUTION_EVIDENCE",
    "compacted_admits", "scope_access_compacted",
]
