# D10: Backend-neutral contracts and calibrated automatic selection

Public code requests semantic profile, resource limits and optional expert backend; it never
needs to import CUDA/Numba/pybind types. Define KernelInputs/Outputs/Capabilities/ExecutionContext
with dtype/layout, logical reduction plan, stage identity, owner, cancellation and memory budget.
Every route implements the same profile-specific observable contract and reports work engagement.

`backend='reference'` forces trusted CPU. `native` and `gpu` are explicit substantive routes:
unavailable or unsupported requests raise a typed capability error before execution. `auto`
may choose reference/native/GPU per qualified fused region, recording reason and effective route.
Do not silently broaden a domain because random tests passed. Once admitted work starts,
numerical mismatch/internal bugs fail loudly; they are not hidden by normal fallback.

Registry entries bind semantic+math version, backend binary/compiler fingerprint, device/ISA,
workload features, memory estimate, validation evidence and calibrated end-to-end crossover.
Absent/stale/malformed entries choose trusted reference under auto. Experts may exercise correct
unqualified domains explicitly but their results are not default qualification. Empty capability
registries cannot satisfy the required native/GPU deliverables.

Features: graph/arcs/origins/destinations, local frontier estimates, dtype/profile, turn state size,
route model, requested outputs, cache/residency state, batch length, memory budget and hardware.
Start with conservative piecewise thresholds from measured rows. Do not add ML scheduling until
simple rules demonstrably fail. Cache numerical identity need not equal backend identity when
bitwise-certified outputs coincide, but profile/math differences always segregate keys.

Optimize regions: compare A_CPU+B_CPU against A_CPU+B_GPU boundary costs and A_GPU+B_GPU residency.
Use equal resource policies and complete output availability. A faster leaf surrounded by slow
packing is rejected. Recalculate thresholds after algorithm/layout changes invalidate calibration.

Tests: missing device/library, stale qualification, wrong ISA, changed profile/dtype, memory refusal,
small below crossover, cache hit vs miss, multi-GPU request when only one available, injected kernel
failure and cancellation. Negative controls hardwire GPU-available -> GPU or reuse wrong-profile
cache and must fail. Publish a capabilities table and evidence-backed dispatch matrix.
