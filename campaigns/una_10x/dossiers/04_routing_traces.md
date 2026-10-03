# 04: sparse ordered routing trace and complete numerical identity

Goal: compute routing once for distinct metric jobs with identical prepared search inputs.
Do not confuse graph equivalence over real weights with execution equivalence over ordered bits.

## Layered identity (canonically typed, schema versioned)
GraphID = profile + ordered CSR pointers/neighbors/cost bits/flags + prepared topology/turn/
obstacle/elevation semantics + compiler/search plan identity + relevant numerical environment.
QueryID = GraphID + exact ordered origin seed IDs and seed cost bits + exact typed cutoff +
search/sentinel/expansion policy. Same geometric point is not enough; different prepared bits miss.
DestID = ordered destination terminal IDs and cost bits + extraction/min/membership plan identity.
TraceID = QueryID + DestID + trace schema. Initial route uses full origin vector identity.
A later per-origin memo may separate QueryIDs only after public duplicates/ownership are proved.
MetricID = TraceID + selected destination weight bits in ORIGINAL order + ALL actual metric
parameters/branch/plan/dtype/compiler identity. ArtifactID = names/formats/CRS/output metadata/
publication policy; it is separate and never bypasses current validation, state or exports.

Cutoff equality is exact. Do not share a trace produced at larger R with smaller R merely by
filtering: b, seed gates, queue chronology and legacy labels depend on R. Do not reverse a query
or contract degree-two paths without a rounded-operation proof. Changed graph flags/edge ordering,
turns/obstacles/costs/profile/code cause misses even when many distances look unchanged.

## Representation
Packed origin-major sparse trace: indptr[O+1], destination_ids[nnz] int64, distance_bits[nnz]
float64, immutable metadata. Each slice is strictly ascending original destination IDs. Store
actual evaluated retained distance bits, not rounded/compressed surrogates. No dense O*D or
K*O*D intermediates. If original downstream behavior observes unretained values or full ODM,
that consumer uses its original required representation; sparse trace doesn't replace it.

Use one bounded producer tile at a time. Measure/index trace byte counts, temporary packing,
queues and leased memory. Exceeding trace budget selects recompute/streaming, not truncated trace
success. Partially computed trace carries incomplete status and cannot be published as a cache
hit. Checked integer capacities and signed indices; remove the prototype's typed-list narrowing
warning by a proved representation, not warning suppression.

API-internal concept (adapt names to existing backends/contracts, do not create a parallel API):
  prepare_snapshot -> immutable GraphHandle
  route(GraphHandle, query, dest) -> TraceHandle(capabilities, owner, bytes, complete)
  replay(TraceHandle, MetricPlan, weights) -> fresh ResultBundle
  release(handle) after all dependent tasks/writers/commits complete.
A handle is not a durable cache identity and a pointer address is not a hash.

## Correctness gate
Compare each trace slice to actual full reference distances filtered in original index order.
Compare per-job replay to an independently executed full job; vary beta/plateau/midpoint/growth/
KNN branch/coefficients/destination weights individually. Force different results for at least
two cohort rows and validate all K jobs. Exact-repeat control uses current full-result cache.
No invalid input becomes successful because an earlier trace hit avoids its validation.
