# 07: flow sparse-output producer and ordered update reuse

Conditional measured performance track; inherited flow correctness support remains mandatory.
Read current AggregateFlow before changing: local overlap scratch, fixed ordering and budgeted
chunks are already implemented. The new opportunity is bounded traversal -> sparse result without
allocating a dense V' result for every source, and repeated identical routing across parameter jobs.

## Sparse producer
Measure destination-gradient dense initialization/output/finite mask/pack cost vs traversal and
flow loading. Instrument outside final timings. Native candidate returns sorted global node IDs,
exact distances and predecessors only for the original finite entries. Graph includes virtual
nodes. Reproduce the pinned SciPy directed graph, sorted/duplicate adjacency handling, queue/tie,
cutoff and predecessor semantics; equal distances alone do NOT establish equivalent tree flow.
First compare every sparse expansion re-materialized to full original dist/pred arrays. When
library tie behavior cannot be duplicated, that legacy domain retains SciPy; corrected profile
may use a separately reviewed canonical tree but cannot compare its outputs/speed as legacy.

Don't compare all-destinations-from-one-source vs many-origin searches as exact interchange without
proof. Reverse accumulation and shortcuts change rounding. Do not assume degree-two contraction
preserves RN sums. CSR deduplication can change arc attribution even if distances match.

## Ordered update tape
Only admit repeated OD routing inputs identical in graph/cost/snap/order/radius/detour/profile and
contamination rules. Store sufficient ordered route predicates/arc IDs/coefficients to replay the
actual parameter-dependent operations. Do not cache a pre-summed unit-flow matrix then scale:
RN(RN(a+w*x)+w*y) generally differs from RN(a+w*RN(x+y)). Keep original q_sum/scale/order where it
is observed, independent stripe accumulation and slot fold. Tapes can be larger than recompute;
bound by ROI and RAM. Cheap values may be recomputed with identical operations instead of stored.

Run mass-conservation/scientific checks separately from bitwise implementation parity. Turn-aware
state/line graph and K-alternative path semantics are separate profiles, not auto-covered by the
node-graph optimization. All-route enumeration that is inherently huge needs honest progress,
resource-limit failure and complete-output semantics; do not impose a hidden path count cap.

Reject if stage coverage cannot plausibly reach material application gains under MODEL.md, if
predecessor/update bits differ, if trace bytes dominate, or if packaging complexity is unjustified.
Retain source/tests/counterexample/reason rather than cycling through new frameworks indefinitely.
