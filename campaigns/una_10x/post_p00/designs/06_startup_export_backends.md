# Secondary designs: startup, export and heterogeneous execution

These are decision-ready alternatives, not additional measured speedups or mandatory rewrites.
Apply only after P02/P03 exposes their residual service. Retained platform native/GPU obligations
still require substantive implementations and real qualification; this amendment does not waive them.

## Worker-local prepared cache vs one producer
Lowest-risk milestone: each existing spawn worker decodes/builds an identity once, retaining only
immutable prepared payload. W workers still pay W first preparations, but K rows no longer pay K.
Measure this matched arm before adding cross-process publication. Selected target is one producer
per identity plus bounded per-worker reconstruction. Do not multiply their speedups.

## Lightweight bootstrap
Current public calls create and close a pool each time. First reduce import closure, not move it
outside the timer. Record exactly which imports are required by the admitted prepared worker,
then lazily import optional flow/turn/visualization stacks only when selected. Global environment
and native pool limits must be applied BEFORE heavy imports. Retain spawn entrypoint guards,
module path metadata and unsupported-route behavior. Public API/class imports must still work.

A faster ready signal that precedes heavy imports does not eliminate their time; report ready and
ready-for-compute separately. Keep JIT cache loading and first specialization in the correct cold/
warm regime. Do not fork an already initialized GDAL/Numba process to avoid spawn overhead.

## Persistent execution context (separate API/measurement review)
If startup remains dominant, an explicit owned context could retain workers across public calls.
This changes lifecycle and needs an additive, separately reviewed API: close/cancel, thread policy,
input verification, version changes, memory bounds and no stale state. Existing one-shot calls keep
compatibility. Compare initial call + all subsequent calls + shutdown against equal-policy baseline;
report FIRST_COHORT and RESIDENT_DISTINCT separately. It cannot retroactively change PRIMARY or hide
first-use costs. Repeating existing RunBatch calls alone does not retain its current pool.

## Single process with compiled independent-origin threads
This can share prepared geometry without cross-process reconstruction, but is not the first
implementation. Admission requires all needed kernels release the GIL and distinct writer/data
ownership. Preserve worker option/report semantics or introduce an explicitly reviewed schedule,
not secretly replace process contracts. Never share one GDAL dataset across threads. Compare
Python control, native pool and serializer costs against the selected process path.

## Export optimization ladder
1. Keep current exporters and required 64 files. Share immutable prepared geometry/schema only;
   allocate fresh result columns and retain feature order, float text, dtypes and CRS metadata.
2. If encoding dominates after P02, separate immutable schema/geometry encoding from varying
   columns only through a proven byte-identical producer. GeoJSON key order/float formatting and
   Feather metadata/compression/library behavior are part of the artifact contract. Do not adopt
   a different writer just because values parse approximately the same.
3. Bounded overlap with metrics and ordered commits is allowed only under explicit ownership.
   Filesystem throughput, encoding CPU, digest and fsync are separate costs. Output-byte counts
   are not physical disk traffic or evidence that I/O is negligible.
Avoid hardlinks, output thinning, delayed flush beyond timer and dropping duplicate-looking
fields. A required unchanged field is reusable computation, not an omitted required artifact.

## Native/GPU regions
The immediate P00 bottleneck is upstream repeated preparation. Port the reduced-work replay or
sparse search after algorithm admission, not the same unnecessary GIS scans. Preserve typed
legacy fold bits or keep trusted CPU math-sensitive coefficients/folds in a heterogeneous region.
Measure prepare/upload/launch/kernel/download/conversion/sync/remaining service and device/live
memory. No hardware -> hardware qualification unavailable, not simulated success. Strongest
matched CPU configuration remains comparator; do not multiply a kernel ratio with P02/P03 gains.

## Flow extension
Keep existing F2 local-overlap consumer. Candidate sparse-output bounded searches upstream of it
must preserve predecessor/tie/update order. Gradient caching needs graph/profile/cost/radius and
query identity. A pre-summed unit-flow matrix scaled by new weights is not bitwise equivalent to
ordered updates. Flow remains a separately frozen family; its gains cannot qualify OBS_COHORT.
