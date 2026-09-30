# F2: consume sparse gradients locally; reset all private writes

## Objective and boundaries
In AggregateFlow._accumulate_od_flow, replace repeated full V' reach scans and zeroed private arrays when the actual OD envelope is small. Begin with the node-graph path only. Turn-aware path stays unchanged unless independently admitted with its own state graph/tests. Preserve all full public outputs and dense buffer semantics seen by the original arithmetic. Do NOT rewrite SciPy Dijkstra, switch flow models, compact public graph IDs, or approximate the envelope.

## Admission
Measured per-OD full-node scan/init coverage on observed large graph, with distributions of sparse destination entries, reachable OD count and overlap size. Inspect original compiled nonfinite comparisons: d_o and d_d contain +inf for unreachable states even for finite valid edge costs. Fastmath assumptions cannot be hand-waved. The optimized domain must reproduce the compiled baseline's rejection of absent destination entries for finite budget; fail the candidate if this cannot be characterized.

## Minimal staged implementation
1. Keep per-thread dd_buf/pd_buf scatter exactly as driver already does. Destination g_nodes slices originate from np.where(finite[row]) and are ascending global node IDs. Verify sorted uniqueness; unsupported construction falls back.
2. Instead of scanning all node IDs, inspect g_nodes[d] in that order using the UNCHANGED reach predicate on stored d_o,d_d,budget. Outside that slice d_d is the original +inf sentinel; prove original predicate is false there for the admitted profile. Set reach[v] and construct reach_nodes in ascending GLOBAL node order exactly as baseline.
3. Allocate reach/cont_o/cont_d/acc_o/acc_d once per physical active stripe, initialized to the original False/+0.0 values. Preserve dense indexing initially, avoiding new ID mapping or changed sort inputs.
4. Run the ORIGINAL ordering, contamination, two via-arc passes, q_sum/scale, two tree passes, node updates and delivered calculation. Same arrays in same order reach np.argsort; keep sort kind/ties and every floating operation.
5. Record a reset list for EVERY scratch index written, not merely overlap nodes. A predecessor pv can be outside reach; acc_o[pv]+=f or acc_d[pv]+=f still writes and must be reset. Duplicate reset IDs are safe but potentially costly; deduplicate using integer-only logic when justified.
6. One cleanup epilogue resets all recorded scratch values to exact original False/+0.0. It runs after success, no-reach return, q_sum<=0 return and failures before workspace reuse. The Python wrapper must not leave contaminated scratch in a surviving worker after an exception.

## Read/write closure checklist
reach reads: source/target membership, origin/destination acceptance, arc filters.
cont reads: predecessor values and arc-filter values; default False outside writes.
acc reads/writes: via seeds, tree propagation for v AND pv, node-flow combination, destination delivered scalar. Output AB/BA/node arrays are NOT reset.
orders: origin/destination sort input is exactly baseline reach_nodes; no stable-sort substitution.
_find_arc: original adjacency scan and edge selection remain unchanged, including parallel-edge behavior.
dd/pd: driver scatter/reset unchanged and executed in original destination order; early exceptions cannot result in reuse with stale finite entries.

For any unenumerated read/write, stop and update the proof before implementation. Do not assert 'predecessors must be inside the envelope' based only on ideal positive-cost shortest paths; original traversal/budget/numerical implementation is the contract.

## Proof
Induct over OD calls within each fixed stripe. Initially private scratch equals original fresh arrays. Sparse enumeration creates the identical ordered reach list under the verified predicate. Original operations then receive identical inputs, including default values at off-envelope predecessors, so all public updates and delivered scalar match. Complete reset restores initial scratch for the next OD, closing the induction. Concurrency is safe only when physical owners never share scratch and stripe order/partials remain fixed.

## Adversarial tests
OD with no overlap, absent origin/destination in reach, q_sum zero, equal distances/ties, immediate U-turn exclusions, contaminated origin/destination snap edges, predecessor outside reach, missing predecessor -9999, zero node-output length, repeated same destination with changing origin, disjoint successive envelopes, duplicate reset targets, all-overlap dense case, and injected early error then continued worker. Compare every intermediate array/list/q_sum/scale and output update at each OD, actual compiled results plus explanatory trace. Nonfinite invalid costs use bounded fallback probes; +inf unreachable sentinels are normal test cases.

## Cost and memory
Avoid O(V') zeroing per OD but retain O(V') scratch per active owner; do not call it sparse memory merely because touched iteration is sparse. Account fixed logical stripe outputs separately from physical scratch. New touched-list construction, cleanup, guards and sorting are charged. For dense overlap original may be faster: retain original route when measured crossover and memory justify it, with deterministic admitted selection and exact tests on both sides.

Stop if proof fails, no net complete-job gain, excessive touched metadata, aliasing, or memory regression. Do not combine with decay memoization, compact-ID graph, predecessor redesign or destination-parallel reduction in this campaign.
