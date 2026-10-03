# 08: specialize the reduced computation, preserve platform capabilities

Native and GPU tasks are retained platform requirements, not automatically claimed fulfilled by
this CPU-focused performance result. Reuse ongoing reviewed NATIVE_CORE/NATIVE_SIMD/GPU work and
existing backends/contracts.py. One integrator owns interface changes. Start from the sparse
query/trace/replay plan, not a slower Python oracle or old global-scan layout.

## Native
Substantive AOT C++17/pybind11 (retained program starting choice), checked int64/f64 interfaces,
immutable input buffers, explicit owners, bounded scratch, precise exception mapping before GIL
release. Match original per-output heap/snapshot/ordered fold plan. Parallelize origins or
independent metric jobs, never reassociate a reduction. Characterize SIMD with compiler reports/
assembly, including gathers, lane divergence, alignment and FMA; no `-ffast-math` as a workaround.
Benchmarks: current backend algorithm A, reduced algorithm B in current backend, same B native C.
Cold build/install/startup and conversion costs are included in their declared boundaries.
Optional native install cannot break CPU Python fallback or existing supported platforms.

## GPU
Record actual installed hardware and FP64/profile capabilities. Do not assume the previous M1
or the author's container is the target. Reuse validated GPU backend if present; choose one
hardware-specific route at a time (CUDA-class on NVIDIA; appropriate actual device API elsewhere).
A vendor API lacking required arithmetic is an unsupported exact domain, not license for FP32.
At least one substantive real-device region plus installed engagement and numerical validation is
required for GPU support. Capability detection or copying through a GPU is not that region.

Candidate residency region: upload immutable prepared graph/index/trace once, execute many
independent origin queries or metric replays, retain ordered per-output accumulator state, download
only required results/checkpoints. Host computes delicate coefficients/transcendentals initially
unless device math is independently bit-validated. Legacy exact matching is per characterized
profile/environment; corrected reference authority remains separate. Do not change a reduction
because its CPU compiler used a different vector tree. No floating atomics or completion-order sums.

Model FULL service: prepare+conversion+H2D+launch+kernel+D2H+sync+remaining, with cold compile/setup J.
Measure device/global/shared/register traffic only when counters exist; otherwise use explicitly
labeled payload models. Fuse only where it removes traffic without spilling/occupancy damage.
Bound queues and tiles, no dense O*D*jobs tensor, explicit ownership of double buffers, cancellation
and publication barriers. Synchronize before timing ends or leases are released.

CPU vs GPU selection uses validated measured crossover including neighbors. For identical fixed
resource policy both arms have the same allowed machine/device budget; record hardware additions
separately and never label a different machine equal-resource. Missing hardware blocks qualification,
not independent native/CPU/cache/batch work. Provide buildable implementation and a real-hardware
validation command, but state unrun gates. No paid external compute without authorization.
