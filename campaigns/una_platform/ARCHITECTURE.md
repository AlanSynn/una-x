# Architecture and ownership boundaries

Public UNA / Madina facade -> validated immutable JobSpec -> semantic model/profile ->
content-addressed prepared graph -> backend-neutral kernel contract -> reference/native/GPU
execution -> deterministic result assembly -> ordered publication and live-state rehydration.

Separate these identities:
- semantic profile/model and complete numerical inputs;
- numerical artifact contents and dtype/layout/order;
- backend schedule/device/compiled artifact capabilities;
- export/provenance identity and publication transaction.
A warm cache is not permission to skip validation or re-use changed objects.

Core immutable contracts: KernelInputs, KernelOutputs, KernelCapabilities, ExecutionContext.
Use flat typed buffers with explicit shape, stride, dtype, index origin, version and ownership.
A capability describes supported models/profiles/dtypes, device features, memory bounds,
algorithm version and qualification digest. A result includes completed status, output bytes,
backend actually executed, task ranges, fallback reason and work counters. Never return success
for partially filled arrays. Backend internals are private; public users request semantics.

Retain existing NumPy/Numba as a trusted provider. Native C++ and GPU implementations share
semantic specifications, not necessarily schedules. Do not expose CUDA streams or a vendor
array type in public result schemas. Geometry input/output remains GeoPandas/Shapely unless
an explicitly additive API requests another representation.

A prepared graph is immutable and separate from mutable UNA/Topology/Settings. Results must
not alias writable cached buffers. Per-job mutable state and per-logical-stripe accumulators
have a unique owner. Read-only host/device graph residency may be shared only through leased
handles with lifetime beyond all queued work. No process shares a live mutable Zonal object.

One resource coordinator admits processes, native threads, Numba pools, flow pools, writers,
cache producers and device contexts. Do not create a GPU context per CPU worker. Use one
bounded device service or one explicit device owner. Numerical stripes do not change with
physical scheduling. Checkpoints contain no opaque device pointers or executable pickle.

Architecture should remain small: no generic distributed platform, ML autotuner or vendor
plugin marketplace. Implement only abstractions needed for these actual engines. Keep test
oracles and historical packet imports outside the installed runtime closure.
