# 03: terminal indexing, exact local enumeration and crossover

Input from dossier 02: full logical labels represented by a private epoch view plus touched nodes.
Build reverse incidence CSR terminal_node -> original destination_index from BOTH terminal
columns. Keep both entries when a destination has the same terminal twice; candidate deduplication
handles it. Checked int64 counts/prefix sums; allocate <=2D IDs. Source graph/IDs are not renumbered.

Proof of omitted evaluations: for a destination with neither terminal touched, both logical
labels equal the actual typed b. With finite nonnegative terminal costs and b>R, each original
rounded sum is >=b>R, so the original builtin min cannot pass the original membership <=R.
This excludes only unobserved computations, not requested destination records. Do not use
Euclidean distance, a spatial crop, top-k shortcuts or an approximate radius as this proof.

Pseudocode:
  ids=[]
  for v in touched_nodes:
    for d in reverse_index[v]:
      if destination_mark[d] != epoch: mark; ids.append(d)
  sort ids by ascending ORIGINAL destination index
  for d in ids:
    a = original_add(read(t0[d]),tw0[d])
    b2 = original_add(read(t1[d]),tw1[d])
    dist = original_builtin_min(a,b2)
    append(dist, original_weight[d])
  invoke SAME original filter/fold on this sequence

The fold's retained values/weights must be byte-identical and in the same order as the global
scan. Sorting by discovery order or nearest distance before the fold is wrong. Different input
length may affect compiler specialization, so direct actual-fold equivalence is required even
when filtered values are the same. If necessary feed a characterized kept-fold that is separately
proven for this host/profile. Default argsort tie behavior is part of the contract.

For destination-weight cohorts, trace stores original IDs and distances, NOT reusable copied
weights. Each job gathers its own weights in trace order and applies its own metric. Only reuse
weights when their numerical identity matches. Empty kept lists still produce baseline dtypes,
zeros and exceptions. Seed nodes not expanded are still touched and included.

Dispatch: do not promote unconditional sparse mode; historical tiny/dense regressions exist.
At P00 estimate retained fraction, V,D,O, degree and index/arena cost using the baseline-only
sampled descriptors. If sampling searches, reuse their exact computed results and charge them,
or record duplicate sampling cost. A deterministic threshold table is sufficient; no ML predictor.
Freeze thresholds on selection cells, test withheld dense/tiny/other-morphology cells. Guard
failure and performance fallback are distinguishable in evidence, both retain original semantics.
If runtime density is learned late, switching ONLY destination enumeration to a full original
scan from the same logical view is allowed after proof; no restart/reorder of a search after
public mutation. Budget-aware refusal occurs before workspace allocation and backend launch.

No new public API needed. Entry integration uses the existing Execution/backend contracts, with
route/capability counters in diagnostics outside hot loops where possible. Counters must prove
actual work reduction (nodes initialized, candidates/kept, folds, fallback), not just function entry.
