# A2: local destination enumeration with unchanged metric sequence

## Objective
Avoid unconditional O(D) distance adjustment/filtering per origin when bounded searches touch only a small neighborhood. Operates ONLY in private accessibility integration, not public scope/OD_Matrix. First version retains original V+D scope initialization; A3 is separate. It can build on accepted A1, or a new private tracked wrapper if A1 is rejected. Do not change metric arithmetic.

## Admission and complete dependencies
Require H05 measured destination-scan coverage on observed large D with low touched fraction. Validate terminal indices in [0,V), finite nonnegative destination terminal costs and finite cutoff. Graph costs/seed inputs must be in the tracked-search admitted domain. Reverse index depends on BOTH destination terminal arrays and their order; build once per graph/job from current arrays, never reuse across reloads/Settings without identity proof. No cross-run cache.

Track every seed label assignment and every subsequent label assignment, including assignments to degree-one nodes never pushed and to nodes whose seed exceeds cutoff. 'Popped', 'settled' or final distance<=R is not a substitute. Appending duplicate touched IDs then integer-sorting/deduplicating is initially safer than a new mutable epoch system. Measure its cost. If dynamic list growth/memory defeats benefit, reject rather than smuggling A3 state changes into this task.

## Sentinel proof using actual stored values
Baseline scope has length V+D. When D>0 and all routing indices are <V, scope[V] is an untouched destination-tail slot holding the ACTUAL baseline typed initialization value b. Read that stored value after the identical search. Only admit local pruning when b is finite and b>R. For D=0 use the unchanged empty-destination behavior. If b<=R (e.g. large cutoff rounds R+1 back to R) use dense destination adjustment.

For a destination whose terminals were never assigned, both labels equal b. With finite nonnegative terminal costs and unchanged rounded addition, each sum is >=b>R, so the baseline minimum cannot pass the cutoff. This proof does NOT assume ideal shortest paths, monotone final labels or a continuous b=R+1. It excludes NaN/negative terminal costs and any unsupported rounding profile. Seed assignments must be tracked even if their stored value equals b.

## Proposed dataflow
Build integer reverse incidence lists mapping each network terminal to ORIGINAL destination IDs. Include both terminals (duplicates allowed at construction). Preserve full destinations data externally.

```text
scope,touched_assignments = identical_private_search(origin)
if D==0 or sentinel/typed/capacity preconditions fail:
    original_dense_adjustment_and_metrics(scope)
else:
    U = sorted_unique_integer(touched_assignments)
    candidate_ids = sorted_unique_integer(concatenate(reverse_index[u] for u in U))
    for d in candidate_ids in ascending ORIGINAL ID order:
        dist[d] = original_min(original_add(scope[start[d]],start_cost[d]),
                               original_add(scope[end[d]],end_cost[d]))
    call the ORIGINAL compiled reach_gravity_knn_access on
        candidate_distances and d_weights[candidate_ids], same scalar parameters
```

The helper caller signature is also part of admission: preserve dtype and array-layout specialization of d_distance/d_weights and record both dispatchers' compiled signatures. A gather that changes the original A/F/C layout specialization must fall back unless independently validated. The original metric helper filters by cutoff. Show that its retained n_distances/n_weights arrays are identical in length, order, dtype, layout and bit pattern to those produced from the full D-vector. Then its existing reductions/argsort see the SAME sequence. Do not replace sum, exp/log, unstable argsort, KNN tie policy or exponential/logistic formulas. Do not use a Python set iteration order. Integer sorted-unique is permitted; floating reordering is not.

## State/ownership and fallback
Scope and original destination arrays remain public-compatible; only private transient candidate arrays shrink. No stale reverse index after AddDestinations/AddNetwork/reload. Precondition checks cannot index malformed inputs before baseline validation. If candidate enumeration would allocate more than the dense route, choose dense before mutation and charge the decision overhead. No concurrent writes to shared reverse index.

## Tests
Compare all D dense adjusted values against computed candidate values at included IDs, and prove excluded baseline distances fail cutoff. Compare the final filtered arrays BEFORE metrics, all four outputs, output dtype/shape and artifact bytes. Cases: both terminals unassigned, only one assigned, duplicate terminals, repeated destination weights, equal distances with unequal weights/knn coefficients, terminal assigned but never queued, seed exactly b, radius around 2^53 and maxfloat causing b<=R/nonfinite, D=0, no reachable destination, all reachable destinations, negative/nonfinite terminal costs refused, reordered/reloaded destination data and two successive origins.

## Cost and stop
New work is reverse-index build plus touched/candidate enumeration/sorting/gather; it is not free. Avoid per-origin O(V) clearing for the new tracking structure, but do not add an unproven generation counter. Report candidate fraction and actual D scan avoided. Reject dense/nearly-global regimes with no net gain. Preserve dense original public ODM behavior even if a streamed ODM would be faster. One design/revision budget under DECISION.
