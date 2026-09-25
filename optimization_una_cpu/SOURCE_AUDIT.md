# Source audit

Repository target: `AlanSynn/una-x`.
Upstream audited baseline: `City-Form-Lab/urban_network_analysis@c15ebda6981397f46eed5c2d55229f71e57d44fb`.

This fork was seeded from that upstream source for the package runtime, build metadata, setup environment, and examples. Large tutorial datasets were not copied into the fork; obtain genuine proxy fixtures from the pinned upstream dataset or another immutable manifest and record hashes.

Observed hot-path fact: AccessibilityWElevation constructs CSR by scanning full `start_nodes` and `end_nodes` arrays for every node in both count and fill passes. RunAccessibility routes all non-turn accessibility through AccessibilityWElevation, including symmetric no-elevation runs. The same construction pattern exists in Accessibility.py.

Prior evidence is preserved under `optimization_una_cpu/prior/una_optimization/`. It is historical synthetic/component evidence only. It does not prove the complete fork, installed wheel, tutorial GIS pipeline, Feather export, actual laptop configuration, or user production workload.

Before source changes, record:
- current main SHA and merge base,
- git status and untracked files,
- Python/dependency versions,
- CPU logical/physical cores and admitted CPU quota,
- total/available RAM and swap,
- platform/ISA,
- Numba threading layer and effective thread count,
- filesystem/output path,
- any CI checks visible for the exact SHA.

Reconcile source drift explicitly. Do not blind-apply the patch to a different implementation.