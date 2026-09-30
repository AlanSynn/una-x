# D07: CPU algorithm/layout frontier after existing work

Current source ALREADY contains ordered CSR, A1 two-phase search scratch, A3 tailless scopes,
F1 GIL changes, F2 overlap scratch and F3 chunk budgeting. Read their implementations and
closeout, then profile current-main installed execution. Do not relabel old wins as new gains.

## Authorized candidates, admitted by stage evidence
- Local destination enumeration from touched terminal reverse indexes. Prove actual typed
  sentinel b > cutoff and nonnegative terminal costs; include seeded nodes even when never
  expanded; preserve original destination order after deduplication and original metric fold.
- Reuse owned per-origin/per-stripe scratch with touched resets across searches. Restore every
  read-default field on every return/exception; no stale epoch wrap. Public OD scope outputs
  remain full-shape where required even if internal accessibility scratch is sparse.
- Exact redundant guard/graph preparation reuse under owned immutable cached snapshots.
- Fuse producer/consumer of cheap values and avoid repeated full-array passes; do not recache
  cheap values that cost more DRAM traffic than recomputation.
- Sparse flow candidate enumeration using stored finite-gradient indices while retaining node,
  arc, sort-tie, predecessor traversal and accumulation order. Predecessors outside overlap
  still require defined scratch values and touched-write cleanup.
- Stream ALL-path enumeration/geometry materialization where supported without changing returned
  route set/order. Exponential output size remains work; finite-K is a different model.

## Proof format
State old and new state variables, input dependencies, write/read sets, complete observable
projection, and induction across each ordered step. For any skipped region, establish exact
stored lower/upper bounds, including rounding and infinity behavior. Scalar min vs np.minimum
and reordered NaN comparisons are not automatically equivalent. Corrected algorithms may fix
semantics only under a registered bug specification, not as an unlabelled fast path.

## Measurement
Count expanded states, heap pushes/pops, inspected arcs, destinations scanned vs retained,
overlap size, compiler allocation calls and bytes by lifetime. Inspect generated LLVM/assembly
before claiming removal or SIMD. Benchmark leaf, adapter, setup+adapter, full public job and
public batch. Keep no more than three alternatives per active bottleneck before reprofile.

## Tests
Duplicate neighbors with simultaneous eligibility, self-loops, equal distances, zero costs,
seeded but unexpanded endpoints, huge/nextafter radius, signed zero, nonfinite refusal,
empty destinations, KNN ties and cutoff membership. Reused scratch: adversarial sequence of
large/small/empty/error searches on same worker. Mutating a cached topology invalidates reuse.
Accept only exact outputs/state and complete-region improvement under equal resources.
