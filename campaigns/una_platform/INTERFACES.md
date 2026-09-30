# Public interfaces to implement, not current functionality claims

Implement these concrete control points after CONTRACT freezes field types/defaults:

```python
from urban_network_analysis import UNA, ExecutionOptions, CacheOptions
p = UNA()
p.execution = ExecutionOptions(
    semantic_profile="corrected_v1", backend="auto",
    cache=CacheOptions(mode="disk", directory=".una-cache", max_bytes=2_000_000_000),
)
p.RunAccessibility()             # existing no-argument public method still works
p.RunBatch("flow", parallel=True, workers=4, execution=p.execution)
# Existing return remains None; read p.batch_report for effective execution and row outcomes.
```

The example's memory size is a configuration example, not a recommended/measured laptop budget.
Omitted p.execution uses the default UNA legacy profile. Zonal.execution provides the equivalent
optional control for the Madina facade; omitted control uses madina_legacy. Existing Madina
function positional parameters/defaults remain accepted and visible. Avoid replacing meaningful
signatures with anonymous *args/**kwargs dispatch. Optional additive keywords must not change
legacy parameter interpretation. CONTRACT settles one authoritative serialized representation;
saving/loading/resetting a project must preserve explicitly selected numerical profile/options.

ExecutionOptions is immutable and validated before job admission. Suggested fields:
semantic_profile, backend, device, cpu_budget, threads_per_worker, logical_reduction_plan,
memory_limit_bytes, workspace_limit_bytes, queue_depth, writer_concurrency, timeout_s,
cache, checkpoint. Device/context objects and cancellation handles are runtime-only ownership
references, not blindly serialized into Settings/CSV. CPU budget and field defaults are explicit.

CacheOptions: off/memory/disk, directory, memory/disk byte quota, verification policy and schema
version. Never let expert verification options disable numerical identity validation silently.
Expose inspect/clear/stats without deleting in-flight or unrelated user files.

KernelInputs: immutable ordered CSR/geometry/OD descriptors with dtypes/layout/owner/profile and
content hashes. KernelOutputs: typed arrays plus complete/incomplete status and next logical
state for restart. KernelCapabilities: backend binary, supported profiles/types, stage/domain,
ISA/device requirements, known refusals and qualification state. ExecutionContext: budget,
logical schedule, physical resources, cancellation, scratch/transfer ownership and provenance.

BatchReport includes requested/effective workers and every independent pool, row dependency and
fallback reasons, ordered committed prefix, per-row model/profile/backend, failures/cancellation,
cache engagement, peak resources, output manifest and checkpoint generation. Timings are not
numerical cache identity. Public reports must not expose credentials or private environment data.

Forced backend requests raise a typed capability/admission error before scientific execution
when unavailable. Auto falls back to a trusted admitted CPU implementation with a reason. Internal
numerical mismatch is never hidden as ordinary fallback. Explicit GPU kernel capacity retry
restarts an uncommitted logical owner, not a partially applied accumulation.

When GPU/native hardware is unavailable, implement source/host/error/contract work and mark
unrun device/binary cells blocked. Continue CPU/API/cache/batch packaging and evidence. Do not
present a stub or simulator-only route as feature completion. Final manifest records exactly
which profiles/stages/devices are supported, not the vague phrase 'GPU supported everywhere'.
