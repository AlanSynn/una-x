# F3: byte-budgeted destination-gradient chunks

## Objective
Bound transient dense memory in AggregateFlow._precompute_dest_gradients (and separately reviewed turns equivalent if selected), preserving the exact set/order of destination-source queries and sparse results. This is a memory-layout/lifetime experiment, not a new shortest-path algorithm.

## Baseline live set
Current chunk is approximately floor(1e8/V'), clamped to at least one. A chunk allocates float64 dist, predecessor array (inspect returned dtype), full finite mask, source indices, sparse gathered copies appended to lists, SciPy/native workspace and Python overhead. At loop assignment, previous dist/preds/mask may remain live while next results allocate. At final np.concatenate, sparse parts coexist with final arrays. Existing 0.8 GB comment counts distance payload only; with i32 predecessors plus bool mask the dense payload is about 1.3 GB at 1e8 elements. This is a DERIVED payload estimate, not measured RSS or exact allocator peak.

## Admission and warnings
Changed chunk count can change warning/exception multiplicity. Admit only validated CSR and normal finite nonnegative cost profiles for which the original SciPy calls are warning-free; unsupported negative/nonfinite costs, invalid sources/indices, unusual sparse subclasses or warning-producing conversions keep original chunking and error behavior. Count warnings in B0/candidate tests. Do not classify warnings as harmless timing metadata.

## Proposed implementation
Extract a PRIVATE pure chunk-selection helper taking V', remaining sources, returned dtype sizes and a conservative workspace cap. The cap is internal policy derived from approved resource profile or a fixed validated safe default, not a new required public Settings field. Do not query changing free RAM every row and call it deterministic numerical identity; numerical behavior must be independent of chosen chunk, but record the actual schedule.

Let fixed_live include graph, retained sparse parts, source/result arrays and reserved native/runtime margin. Let per_source include V'*(8+pred_itemsize+mask_itemsize) plus actual row temporaries/source-index cost and a measured safety margin. Choose c<=remaining such that fixed_live+c*per_source fits workspace budget. If even c=1 cannot fit, the experiment has no admitted memory-safe fast path; preserve original behavior or report benchmark capacity unavailable. Never drop destinations or return partial gradients.

For source slices in original ascending order:
- call the SAME scipy_dijkstra, SAME CSR, directed flag, source indices, limit and return_predecessors=True;
- preserve shape handling and per-row np.where(finite) order;
- preserve original gathered dtype conversions and independent copies;
- append rows in exact destination order;
- delete all dense arrays, masks and lingering views before the next Dijkstra call;
- concatenate with identical indptr/nodes/dist/pred order and dtype.

Initially keep mask formation unchanged. Per-row finite masks are a second subchange only if lifetime measurements justify it and exactness is independently checked. No min_only, batched-nearest-source substitution, sparse Dijkstra rewrite, concurrent chunks, changed limit, compression or new file cache.

## Equivalence obligation
The operation per source must be the same native search regardless of how sources are grouped into calls. Inspect the pinned SciPy implementation or official semantics and test actual predecessor tie behavior, not distances alone. If native vectorized batching chooses different ties or arithmetic under changed chunk size, reject that domain. Pure copies/deallocation cannot alter already-copied sparse arrays; verify no appended part aliases deleted dense memory.

## Tests
Chunk sizes 1,2,3,entire-source-set and uneven final tail; zero destinations; one destination; disconnected states/+inf; tied shortest paths; duplicated source nodes if baseline allows; custom/elevation/obstacle costs; sorted input/parallel arcs from actual flow CSR; empty finite rows. Compare all indptr/nodes/dist/pred arrays bit-for-bit, forward/reverse CSR unchanged and full flow/cap/artifacts. Force tiny cap and refusal at single-source capacity without running OOM in main process.

Measure simultaneous process-tree/native memory and explicit per-stage payload counts, not tracemalloc alone. Verify old chunk reference release via allocation/lifetime instrumentation outside final timing. Compare complete RunFlow and total gradients, cold and warm. A lower-memory but slower implementation can be capacity_only under DECISION; never label it throughput-qualified without full-path win.

## Stop conditions
If sparse parts/final outputs dominate persistent memory, chunk reduction cannot solve the capacity limit. Record the remaining lower bound rather than adding out-of-scope spill/checkpoint formats. If B0 fails at a large input, report censored baseline and unavailable speedup. Keep exactness evidence at a common feasible cell. At most three preselected workspace caps (policy choices 64/128/256 MiB) within this task's bounded screening budget; do not run an unbounded sweep or change caps after final freeze.
