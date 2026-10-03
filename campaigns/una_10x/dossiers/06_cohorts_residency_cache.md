# 06: public RunBatch cohort execution and bounded immutable residency

Reuse the CURRENT batch/plan.py, runtime.py, worker.py, checkpoint.py and report.py contracts.
No new external benchmark-only scheduler that leaves public RunBatch slow. Baseline comparisons
must exercise the existing public implementation and strongest legal caches/concurrency.

## Plan
Evaluate row binding, missing-field skips, substitutions and statically resolvable Settings
transitions in the same original order. Build dependency edges for any row that reads an earlier
result/state, gravity-cap mutation, topology revision, input-file mutation, output conflict or
composite state. Unknown dependencies form a serial barrier, not a guessed-independent cohort.
Group ONLY independent numerical substeps with identical GraphID/QueryID/DestID. A cohort is not
necessarily contiguous rows; publication still follows row order. Runtime-resolved dependencies
wait for the prerequisite commit before deriving keys. Serialize state transitions, not all compute.

## Execute
One owner prepares immutable bytes and produces each missing trace once. Work descriptors contain
content identities, validated paths/lease handles and small metric parameters, not live mutable
UNA/GeoDataFrames. CPU workers can reuse read-only snapshots while owning private search/metric
scratch and outputs. Share only where serialization cost and lifetime are measured. For separate
processes use explicit shared-memory/mmap/read-only-buffer leases; no implicit fork inheritance.
Cache disabling uses the original route without expensive identity/serialization work.

## Commit
For each original row r: wait for its dependencies/result, verify staged artifact checksums and
state delta, apply EXACT serial Settings/project mutations, publish required files at original
paths/policy, perform original composite updates, record checkpoint completion, release leases.
Do not collapse all K rows into one final state update. A failed earlier row cancels later
speculation; no later public files leak. Preserve skipped-row mutations and last-engine identity
or observable state according to the existing tested contract. Any full state rehydration remains
inside timing. Do not remove it solely because benchmark code reads only final metrics.

## Caches and recovery
Extend content-addressed storage rather than duplicate it. Resident cache stores immutable decoded
arrays only with bounded byte accounting and ownership. Public getters/results remain fresh.
Numerical identity is separate from export provenance. Single-flight owner death, timeouts,
corruption, partial trace, schema/profile/code mismatch, and eviction mid-read are tested. Never
execute pickle from disk cache. Atomic commit completion follows current durability guarantee;
no hash-only validity or mtime-only stale keys. All data-affecting public mutations force correct
invalidation, even when callers edit exposed arrays; snapshot/hash or an enforceable revision
mechanism is required. Untracked user mutation cannot be assumed absent.

Resume verifies input/profile/code/trace/metric identities, not merely 'row done'. Reuse numerical
payload while republishing metadata only when the publication policy allows it. Exactly-once
public effects cannot be inferred from at-least-once worker execution. Commit marker/manifest and
files must recover to the existing consistent prefix after injected crashes.

Performance controls: count unique searches vs requested rows, source reads, trace bytes, metric
folds, cache decodes, IPC bytes and public commits. Retain K outputs and all required validation.
Limit queued trace tiles/results and writers; avoid head-of-line accumulation exceeding RAM.
Cohort cache miss, warm distinct and exact-repeat regimes reported separately. Initial trace and
resident construction cost is INCLUDED in FIRST_COHORT. Re-measure startup and one-off latency.
