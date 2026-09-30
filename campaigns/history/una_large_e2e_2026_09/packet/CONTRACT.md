# Semantic, numerical, state, artifact and DX contract

## Definitions
Keep mathematical equivalence, floating-point execution equivalence, API/artifact equivalence, scientific equivalence and equal-resource performance claims separate. The compatibility oracle is the actual B0 implementation under a frozen compiler/library/ISA/threading profile, not ideal Dijkstra or a mathematical transport model. No tolerance-based relaxation is authorized.

## Public surfaces
Inventory public exports from __init__, UNA, Settings, Topology and engine classes; capture signatures/defaults, import/package/version behavior and supported file formats. Keep RunAccessibility, RunFlow, RunODM, RunBatch and their state mutations. Keep Settings validation/serialization, readiness flags, graph array attributes, per-origin/per-edge/per-node outputs and ordering. Do not remove or shrink public scope/ODM arrays. New private routines may specialize only internal callers whose complete observation graph is documented.

Input contract includes CRS, geometric precision, feature order, custom weights/IDs, snapping tie behavior, elevation and obstacle corrections, observers, and invalid-input warning/exception types, messages and order. No coordinate reprojection, graph simplification, edge deduplication, reindexing or changed cutoff is an optimization. Keep all mandated outputs and file-writing order.

## Numerical frozen details
Preserve float64/int64/int32/bool roles, casts, overflow/rounding, signed zero, nonfinite masks, sentinel initialization, terminal assignment order, heap tuple ordering, neighbor incidence order, eligibility snapshot, degree-based queue decisions, destination filtering sequence, existing argsort algorithm and tie inputs, exp/log implementations and reduction order. Preserve current Numba flags; source fastmath is not permission to relax further. Compare actual compiled behavior, not only .py_func.

A1 does not fix duplicate-label overwrites, stale queue entries, self-loop terminal overwrites or degree-one quirks. A2 does not assume labels are globally optimal or finalized. F2 does not assume every predecessor is in the overlap. F1 cannot change logical stripe membership. Preserve current integer-typed reach even with fractional destination weights.

Exact finite values and signed zeros require byte equality after dtype/shape checks. NaN payload equality is required for pure copy paths; for arithmetic establish baseline-vs-baseline determinism and retain at least masks plus any stronger stable baseline property. Do not replace a failing bit comparison with allclose. Nondeterminism discovered in baseline blocks the affected exact claim until independently characterized, not silently normalized.

## Flow partition
Let K be the baseline logical stripe count and O origin count. Stripe s owns origins s,s+K,... in that order, destinations in original order, and private AB/BA/node partial arrays. Final sum visits s=0..K-1. K, original per-stripe operations and final order are numerical dependencies. H physical threads may be less than K only with an explicit schedule proof that executes whole fixed stripes serially on available workers. No work stealing of individual origins across arithmetic stripes and no atomics/tree reduction. This scheduling extension is not required for F1; do not implement it unless S01 identifies necessity and review approves a separate bounded schedule.

## State and errors
Observe every relevant public-call boundary, Settings before/after, gravity-cap derivation/writeback before flow, topology replacement, engine readiness, result arrays, schema/geometry and required log messages. Timing fields and performance-rate text are inherently variable; compare them by a preregistered structural rule, never delete non-timing warnings. Shared mutable UNA objects are not independent jobs.

Fast admission has complete inputs and an explicit fallback. Guards are charged to timing and cannot perform unsafe casts/indexing before baseline validation. Unsupported types, shapes, weights, platform/math profiles or capacity execute the unchanged original path. Do not catch arbitrary exceptions after optimized mutation and replay baseline. Private scratch cleanup on early return/error must prevent contamination of future calls.

Inherited NaN-weight queue growth is documented, not fixed here. Test such inputs only in isolated memory/time-bounded children. An expected kill is hazard characterization, not successful numerical validation. Do not run uncontrolled nonfinite cases in the shared benchmark process.

## Artifacts and durability
Compare file sets, data/geometry order, dtypes, indices, column names, CRS, metadata, values and writer settings for CSV/GeoJSON/Feather/ODM as applicable. Byte comparison first; a semantic comparator is permitted only for baseline-established nondeterministic metadata fields named before candidate execution. Never drop IDs/CRS/row order to get equality.

Keep baseline durability: public return follows synchronous writers. Do not claim fsync, atomic publication or crash recovery where baseline lacks it, and do not add stronger durability to only one timing arm. Benchmark verification time is separate and explicit.

## DX and scope
Package/import name, Python support, ordinary install workflow and public defaults stay unchanged. No new mandatory native compiler/runtime, runtime network download, backend selector, private data upload or credential change. Runtime imports cannot depend on campaigns/, benchmark adapters or source-tree loaders. Build baseline and candidate wheels in matching clean environments. Check all imported module paths and native versions, not just the top-level version string.
