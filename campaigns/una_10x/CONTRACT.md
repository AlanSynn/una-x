# Frozen semantic, numerical and operational contract

## Independent equivalence classes
Track mathematical, floating-point execution, API/artifact, scientific/domain and equal-resource
performance equivalence separately. A proof in one class does not establish another.

## Public surface
Keep existing imports, supported calls, method signatures and default behavior; preserve current
Execution/Cache/Batch options. New internal routing/trace machinery stays private. Any genuinely
necessary additive execution option must be reviewed before coding, default to compatibility,
serialize explicitly, and leave old positional calls unchanged. No automatic backend selected
only because a device exists. Existing Madina adapters and their public mutations remain supported.

Geometry/CRS/units, node and edge identities, duplicates, self-loops, terminal ordering, turns,
elevation, obstacle direction and partial-edge corrections, destination weights, path model,
search cutoff, origin order, and every requested artifact remain part of the workload. No pruning
by Euclidean geometry unless a separate exact stored-cost bound is proved; not part of S10.

## Three numerical authorities
una_legacy: current pinned compiled engine on a pinned implementation/math environment.
madina_legacy: pinned Madina valid behavior under audited dependencies.
corrected_v1: independently reviewed corrected reference as actually specified and implemented.
Preserve the distinction. Do not compare two profiles for a speed ratio or transfer caches across
profiles. Unknown corrected-plan conflicts block that route until resolved; they do not license
changing legacy bits. Nonfinite/negative legacy domains may deliberately refuse new routes.

Require same output dtypes/shapes, finite bytes, signed zeros, categorical results and required
nonfinite masks/payloads. Copied payloads preserve bytes. No blanket NaN exemption, rounding before
comparison, allclose acceptance, or FP32 substitution. Preserve original compiled fold/heap/tie
behavior or prove direct equivalence. A new compilation can change reductions with identical
source expressions. Record ISA, compiler, math library, Numba/LLVM, fastmath, contraction, rounding
and denormal mode. Do not enable blanket fast-math on a new backend.

## Sparse optimization contract
Only admit the actual A3 finite/nonnegative domain verified in dossier 02. Check metadata and
allocation bounds before reading arrays or launching unsafe code. Compute the original typed
sentinel and explicitly test it exceeds the cutoff. Validation checks must not be compiled under
unsafe nonfinite assumptions. Refuse unknown layouts/types rather than casting silently.
The private epoch projection must match each dense label read. Seeds are assigned in order,
even at the same node or outside expansion cutoff. Heap prologue, seed strict < gate, row <= gate,
snapshot phase, assignment phase, node degree and flags all stay unchanged. No stale-entry skip,
settled mask, queue de-duplication or scalar-immediate relaxation as an incidental optimization.
Touched membership includes every seed and write. Retained destinations return to original ID
order BEFORE the same filter/fold. Do not replace default argsort tie order with a new tie policy.

## Reuse contract
Only identical complete numerical inputs justify reuse. Decode or prepare once into immutable,
owned content, not mutable DataFrame identity. Distinct provenance may share numerical payload,
but current validation, metadata, warnings, state and output publication must still execute.
No source filename/mtime-only key. Include profile and arithmetic implementation identity.
Cutoff is an exact key: a larger-radius trace is NOT assumed valid for a smaller radius.
Changing destination weight must not invalidate routing only when it truly does not affect graph,
connectors, masks or model; prove each projection, never assume it from the name.

## Observable state and recovery
RunBatch must match the existing ordered commit state/prefix at every boundary, including
Settings mutations, gravity-cap resolution, skipped rows, output names, composite assembly,
last-engine state, exception type/message and cancellation. Speculation cannot expose later
artifacts or replace the failure prefix. Cohort results are not published as a group unless
that is already the declared contract. Partial results must be marked incomplete, never cached
or checkpointed as complete. Retain required staging/hash/durability checks in timed execution.

A result cache hit must not alias another job's mutable arrays. Internal read-only views require
leases, explicit ownership and immutable source identity. Memory pressure evicts caches, not
scientific workload. Do not catch arbitrary runtime exceptions and silently retry the original
route after partially mutating outputs. Pre-execution capability refusal may use the trusted
path; error/cancellation after admission follows existing explicit transactional boundaries.
