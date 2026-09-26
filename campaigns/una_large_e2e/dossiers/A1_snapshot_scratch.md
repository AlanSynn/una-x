# A1: snapshot-preserving search scratch

## Objective and source
Remove repeated per-heap-pop array allocation/gather overhead in compact_vector_node_view_scope (Accessibility.py and AccessibilityWElevation.py) WITHOUT replacing its state transition with conventional Dijkstra. This is a private optimized internal route; the original function/path remains the reference and unsupported-domain fallback. Do not touch _ordered_csr.

## Admission
H05 must show surviving allocation/gather overhead inside search with material complete-job coverage. Inspect compiled allocations or a suitable allocation/call-count probe; NumPy source syntax alone is not proof of a heap allocation in machine code. At entry admit ordinary owned/nonconcurrently-mutated typed arrays with valid CSR pointers and endpoint indices, float64 weights/labels, finite nonnegative costs and finite nonnegative cutoff. Refuse unsafe sum-overflow domains or show exact original overflow behavior. Charge guard/graph metadata computation to full job; do it once per graph when identical inputs permit, not every pop. Wrong dtype/shape, invalid indices, nonfinite/negative weights, unsupported math/ISA profile or excessive scratch capacity use original behavior. Do not issue new warnings before original validation.

## Exact baseline transition
For popped tuple (w,u), let S0 be the entire label vector BEFORE this row is updated. For each adjacency offset i, baseline first forms c_i=RN(edge_weight_i+w), then p_i=(c_i<=R and c_i<S0[neighbor_i]). It computes ALL p_i before any assignment. It then visits true offsets i in increasing row order, assigns S[neighbor_i]=c_i, and conditionally pushes exactly as the original network-node/degree checks specify. Duplicate destinations can therefore receive a later, larger value. Preserve this behavior.

## Proposed algorithm
Keep scope allocation length V+D, dtype, sentinel construction, both terminal assignments and heap seed/pop/push sequence IDENTICAL in the first implementation. Preallocate two private buffers sized to the graph's maximum adjacency degree: eligible_offset:int64 and eligible_weight:float64. Compute maximum degree once from unchanged pointer data and bound multiplied active-search scratch under RESOURCES. Every independent origin/search owns its buffers; no module-global or shared Numba array.

```text
while queue:
    w,u = original_heappop(queue)
    n_eligible = 0
    for i in original row order:
        c = original_typed_add(weight[i],w)
        # no label writes in this phase
        if original_comparisons(c,R,S[neighbor[i]]):
            eligible_offset[n_eligible] = i
            eligible_weight[n_eligible] = c
            n_eligible += 1
    for j in range(n_eligible):
        i = eligible_offset[j]
        c = eligible_weight[j]
        v = original_neighbor_at(i)
        S[v] = c
        if original_is_network(i):
            if original_degree(v)>1:
                original_heappush(queue,(c,v))
return original_shapes_and_values
```

Do not recompute eligibility during phase two. Do not sort eligible neighbors by weight. Do not skip stale queue entries. Do not mark a node settled, filter duplicate incidences, add <= versus < fixes, change terminal overwrite order, or enqueue degree-one nodes that baseline does not enqueue.

## Proof required before code promotion
Induct on popped heap events. Base: identical label initialization and heap. Phase-one inputs are unchanged S0 and identical row order; identical typed adds/comparisons yield identical eligible offsets AND weights. Phase two performs identical assignments/pushes in identical order, including duplicate overwrites. Thus labels and heap contents after the row match, establishing the next induction step. This is an execution argument, not shortest-path optimality.

Compiler obligation: same scalar expression does not itself prove machine behavior when vector array expressions and fastmath interact. Compare generated add/compare instructions/LLVM where practical; test actual compiled baseline versus candidate on cutoff-adjacent, signed-zero, overflow-refused and duplicate cases. Keep no unsafe extra fastmath/FMA/precision change. A mismatch blocks promotion even if mathematical proof looks right.

## Test matrix and observables
Two parallel edges to the same neighbor, cheaper-first and cheaper-last; equal weights; row with self-loop; origin terminals equal with different costs; cutoff exactly equal/below/above nextafter; unreachable terminal; degree-one neighbor; stale popped tuple; all-refused candidates; row of maximum degree; empty CSR accepted by baseline. Preserve final public return lengths, empty predecessor result, graph attributes and aliases. Input arrays remain unchanged. Assert no scratch sharing across origin prange iterations or separate jobs.

Trace each pop, snapshot labels, eligible pairs, each write/push and resulting queue in Python explanatory tests, and compare actual compiled final/intermediate snapshots on small cases. Repeated-call and multi-thread tests must be bit-identical under fixed numerical profile. Add negative tests that intentionally replace phase one/two by immediate update and confirm the duplicate fixture fails.

## Benchmark and stop
Measure leaf, private adapter, full search/centrality and installed public job. Count live scratch and allocations; warm-up/compilation/guards are included in appropriate boundaries. Reject if compiler already removed the target work, if scratch is excessive for high-degree networks, or if full-job gain is within noise. Do not keep a larger buffer for every global node when only maximum degree is needed. One design and one performance revision maximum.
