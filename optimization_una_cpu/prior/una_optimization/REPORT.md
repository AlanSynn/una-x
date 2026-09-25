# UNA throughput optimization: order-preserving CSR construction

**Target:** City-Form-Lab/urban_network_analysis  
**Pinned upstream commit:** `c15ebda6981397f46eed5c2d55229f71e57d44fb`  
**Scope:** guarded source patch, exactness tests, and CPU component-pipeline experiments.  
**Disposition:** locally validated optimization candidate; **not a completed production or release optimization**.

## 1. Result and evidence boundary

A large repeated traversal was removed from the two non-turn accessibility CSR builders. The replacement groups directed edge incidences in exactly the existing order, without performing floating-point arithmetic on edge weights. The original construction remains the fallback. No public API, Settings field, routing engine, reduction, file writer, or scientific workload was changed.

The final warm **synthetic component batch** had a **5.3394× throughput ratio, derived from measured values**, against the best original-code worker/thread configuration in the tested grid. This is **not** a measured speedup for `UNA.RunAccessibility()`, `RunBatch()`, the complete GIS workflow, or a production dataset.

The measured component starts with already-constructed topology arrays saved in NPZ, then executes typed directional-weight preparation, CSR construction, extracted upstream Numba accessibility kernels, and extracted upstream CSV/GeoJSON export. Real network-file parsing, validation, topology construction, point snapping, obstacle construction, complete public-API state transitions, and Feather export are outside that measured boundary.

GitHub source reads succeeded, but direct cloning/downloading into the execution environment failed. Source contexts and computational functions were transcribed from those reads; a complete checkout was not available. The package was **not imported or run as a complete package**. `verify_repository.py` supplies a strict source/AST check against a real checkout, and `benchmarks/run_repository.py` supplies an actual public-API benchmark; **neither gate was run here**. These limitations prevent promoting the local measurements to a production-performance or complete-integration claim.

### Evidence labels

Every experimental figure below is labeled **measured**, **derived from measured values**, **hypothetical sensitivity analysis**, or **unavailable**. Configuration values describe the actual experiment. Symbolic work/storage formulas are algorithmic derivations, not measurements of hardware traffic. Source observations are attributed to the pinned source inventory at the end.

## 2. Contract established for the changed dependency closure

The exact contract for every feature in this project has not been exhaustively established. The following is the audited boundary; features outside it remain unchanged and require their existing release checks.

| Surface | Observed behavior and preservation requirement |
|---|---|
| Public program | `UNA`, `Settings`, `RunAccessibility`, `RunFlow`, `RunODM`, `RunBatch`, Settings/project serialization. The patch changes no signatures or dispatch. No console-script entry point appears in the inspected `pyproject.toml`. [S1,S5,S8] |
| Current non-turn dispatch | `RunAccessibility` uses `AccessibilityWElevation` even with elevation disabled, to retain obstacle support. Therefore the elevation builder is on the ordinary accessibility path. Both it and the separately accessible plain builder are patched. Some design-document dispatch descriptions are stale; executable source takes precedence. [S1,S2,S5] |
| Graph representation | Pointer and neighbor arrays are native `int64`; prepared directional weights are `float64`; network-incidence flags are Boolean. Each row contains forward incidences in input-edge order, then reverse incidences in input-edge order. Parallel edges and both self-loop incidences survive. Returned graph arrays own their data. [S1,S2] |
| Numeric preparation | Float64 elevation subtraction, positive-part calculation, coefficient multiplication, addition, obstacle corrections, and terminal-weight calculations retain their original expressions and order. There is no new math-library implementation. [S2] |
| Kernel behavior | Existing Numba kernels use `fastmath=True`; outer origin loops already use `prange`. This patch neither enables stricter semantics nor broadens fast-math. Exact output comparison is for the same compiler/dependency/hardware environment, not a new cross-platform bitwise guarantee. [S1,S2] |
| Observable quirks | Integrated reach is allocated with the origin-index dtype, so weighted fractional reach undergoes integer assignment. The legacy Dijkstra neighbor mask is evaluated before its row updates; duplicate neighbors can make update order observable. Both behaviors remain intact. [S1,S2] |
| Results and errors | Original array fields, full/pairwise OD interfaces, missing-topology errors, validation, warnings, and metric storage flags remain unchanged. Metric flags are not used to delete computation without an error/state proof. [S1,S2,S5] |
| Artifacts | Accessibility export order and fields, geometry in accessibility CSV, CRS, delimiter, KNN suffix, enabled format flags, and folder/timestamp policy remain unchanged. The generic export helper has different CSV/GeoJSON behavior and was deliberately not substituted. Timestamp values naturally depend on wall time; only the policy is preserved. [S6] |
| Mutability and chronology | Topology is reloaded on each public run. Batch rows mutate current Settings/state and capture results sequentially before composite publication. No cross-run cache or in-place topology reuse was added; the original batch loop was not parallelized. [S5,S7] |
| Durability and recovery | Existing ordinary synchronous file-writing calls remain; the patch adds no fsync, transactional publication, checkpoint, or recovery promise. No durable cache is introduced. Numba's existing disk-cache policy remains, but source changes may invalidate old cache entries. [S1,S2,S6] |

### Distinct equivalence claims

**Mathematical:** the same directed incidence multiset and weights represent the same graph. That alone is insufficient.

**Floating-point execution:** more strongly, pointer, neighbor, weight, and flag arrays are identical in shape, dtype, order, and bits on the admitted representation. Downstream numerical code is unchanged. Thus identical inputs reach identical numerical operations; no floating reduction proof is replaced by a real-number argument. This is conditional on the unchanged runtime implementation, and the complete-checkout AST gate remains outstanding.

**API/artifact:** no source changes outside private construction/import/assignment plumbing; extraction-level CSV/GeoJSON artifacts match byte-for-byte in the local gates. Feather, full public chronology, and complete repository integration remain unverified.

**Scientific/domain:** radius, origins, destinations, directional costs, obstacles, engine choice, detour policy, and requested outputs are not reduced or approximated. Aggregate flow and K-alternative routing are not interchangeable simply because their output schemas match. [S9]

**Performance:** the measured ratio is under equal CPU/RAM limits, but only for the defined synthetic component workload, tested worker configurations, and warm lifecycle.

## 3. Resource and workload record

**Measured environment:** Intel Xeon Platinum 8370C, Linux x86-64, Python 3.13.5. CPU affinity exposes five logical CPUs, but the controlling cgroup has `cpu.max = 400000 100000`: **four CPU equivalents**. `memory.max = 4294967296`: **4 GiB**. Affinity count was not mistaken for usable sustained compute capacity. GPU device nodes were not observed; accelerator capacity, model, interconnect, power mode, and performance are **unavailable**.

**Measured dependency versions:** NumPy 2.3.5, Numba 0.65.1, llvmlite 0.47.0, pandas 2.2.3, GeoPandas 1.1.2, Shapely 2.1.2, pyogrio 0.12.1, Fiona 1.10.1, psutil 7.2.2, pytest 9.0.2. Numba threading layer: OpenMP. PyArrow: **unavailable**. Exact record: `results/environment_final.json`.

Execution policy: `NUMBA_NUM_THREADS=4`, `OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=1`; each worker explicitly selects its active Numba thread count. Only one benchmark experiment ran at a time. No independent implementation/review agents were available; benchmark processes are not independent reviewers.

The deterministic fixture is a shuffled, synthetic grid with **4,096 nodes, 8,064 edges, 512 origins, and 2,048 destinations**, seed `20260922`, radius `350`, elevation coefficient `0.3`. All four accessibility metrics are computed. File formats in this explicitly defined benchmark are CSV and GeoJSON, with timestamp folders disabled equally. That is not a change to package defaults or a claim to have benchmarked Feather. The NPZ is **335,446 bytes measured**; output files total **190,312 bytes measured** per job. No full OD matrix is requested in this timed workload; full OD is checked separately for equivalence.

Production dataset, workload distribution, target throughput, production startup budget, and user hardware constraints beyond this sandbox are **unavailable**. Therefore no target-SLA or global-optimum claim is made.

## 4. Current-stage execution and cost model

The complete logical dependency chain is:

```text
Settings validation -> network/origin/destination I/O -> topology and snapping
    -> elevation/obstacle/partial-edge preparation -> ordered CSR construction
    -> independent origin searches -> destination-distance adjustment
    -> per-origin metric reductions/KNN -> synchronous export
    -> ordered batch capture -> composite export
```

The two untouched flow engines and turn-aware line graph have separate traversal structures. AggregateFlow already computes reverse destination gradients once per Centrality call and reuses them across origins; that reuse is not a new optimization opportunity claimed here. [S9]

For the complete path use:

\[
T\ge\max\left(W/C,\ S,\ Q_{\rm memory}/B_{\rm memory},\ Q_{\rm IO}/B_{\rm IO}\right).
\]

Here actual DRAM bytes, effective memory bandwidth, cold storage bandwidth, hardware FLOP counters, and complete application span are **unavailable**. File sizes and OS I/O counters are not substitutes for those quantities. The recorded warm `read_bytes_delta` is zero in the final component samples: page-cache reads must not be presented as cold-storage throughput.

For independent jobs:

\[
T_{\rm batch}\approx\sum_j\left\lceil K_j/N_j\right\rceil t_j(N_j,H_j)
 +T_{\rm serial}+T_{\rm overhead}.
\]

Worker count and kernel threads consume the same quota. The experiment tunes them jointly rather than multiplying an assumed worker speedup by a kernel speedup. No stage-specific asynchronous pipeline was introduced.

### Work removed

With `N` nodes and `E` edges, the original builder makes two full endpoint comparisons per node in the count phase and repeats them in the fill phase: **`4NE` endpoint comparisons, algorithmically derived**. Its endpoint arrays represent **`32NE` logical input bytes visited** for native int64 endpoints. Those are repeated *logical visits*, not measured DRAM transactions: small endpoint arrays may remain in cache. Mask sums, Boolean masks, masked gathers, Python list growth, and NumPy scalar boxing add work and allocation.

The candidate replaces that `O(NE)` scan structure with a histogram, prefix sums, stable grouping, and gathers. Conservative complexity is **`O(N + E log E)`**, with **`O(N+E)` explicit storage**. No claim depends on a particular internal integer-sort implementation.

The untouched kernel still initializes a scope of length `N+D` per active origin, performs radius-bounded heap traversal, scans destinations, creates filtered metric arrays, and sorts reachable distances for KNN. Its work includes `O(O(N+D))` initialization/scanning, the actual visited-arc/heap work, and per-origin reachable-distance sorting. A radius bound does not remove the full initialization/destination scan. Scratch, heap size, search divergence, and sorting costs vary by workload; they were not replaced by a dense-matrix assumption.

### Final single-worker stage observations

**All entries are medians derived from measured values**, three paired runs per variant, one worker and one active Numba thread. See `results/final_components.csv`. Independent stage medians need not sum to the median total.

| Measured component stage | Original seconds | Candidate seconds |
|---|---:|---:|
| NPZ load | 0.001556 | 0.001435 |
| Typed weights and point geometry | 0.000295 | 0.000319 |
| Ordered CSR | 0.264188 | 0.001161 |
| Unchanged accessibility metrics | 0.018433 | 0.019357 |
| CSV + GeoJSON export | 0.017426 | 0.019247 |
| Complete defined component | **0.301943** | **0.041608** |
| Real GIS preparation/public dispatch | **unavailable** | **unavailable** |

Ratios **derived from measured values**: CSR **227.50×**; complete component latency **7.2568×**. Neither is a complete application speedup. The CSR fraction falls from **87.50%** to **2.79%**, derived from measured values; the remaining costs are metrics/export, not a need for another faster CSR backend.

For a stage fraction `f`, stage speedup `s`, and added normalized overhead `delta`, use

\[
S_{total}=1/(1-f+f/s+\delta).
\]

**Hypothetical sensitivity analysis, anchored to measured fractions:** removing the remaining CSR stage entirely would improve this candidate component by at most about **1.029×**, before any new overhead. A production total has an unknown `f`; these component fractions cannot be transplanted into it. For a requested speedup `S*`, the required stage speedup is `f/(1/S* - 1 + f - delta)` only when the denominator is positive. No numeric production target exists to evaluate that gate.

## 5. Accepted transformation and proof

### A1: stable incidence grouping

Let edges be indexed in input order. Form the ordered sequence

\[
L=[(s_e,t_e,w_{AB,e})]_{e=0}^{E-1}\ \Vert\
  [(t_e,s_e,w_{BA,e})]_{e=0}^{E-1}.
\]

For node `v`, the original builder emits exactly the subsequence of `L` whose first component equals `v`: all forward occurrences first, each in edge order, then all reverse occurrences in edge order. A **stable** sort by that first component retains precisely this subsequence order within each group. The degree histogram gives its length; prefix sums place the same rows at the same pointer offsets. Parallel edges remain distinct. A self-loop occurs once in each half, matching the original two masked selections. Isolated and empty rows retain repeated pointer offsets.

Only source-node indices are sorted. Neighbor IDs and already-computed directional float64 weights are gathered with the resulting permutation. There is no addition, rounding, reweighting, reduced precision, FMA, or transcendental evaluation in the replacement. Prefix sums are integer operations bounded by the admitted doubled edge count. The original masks/lists ultimately create the same typed sequence; bit-pattern tests specifically cover signed zero, nonfinite values, subnormals, and NaN payloads in the tested NumPy environment.

The three final `np.array(...)` conversions are skipped only when the helper already returned new, owned, writable, contiguous arrays of the required dtype. This removes an unnecessary copy without aliasing the topology input. On fallback the exact old constructors execute. The existing graph's `topology` and `logger` references and all origin/destination fields remain unchanged.

### Admitted domain and fallback

The helper accepts a 64-bit `intp` platform; an ordinary Python integer node count in representable bounds; exact one-dimensional NumPy ndarrays with native int64 endpoints and native float64 directional weights; equal lengths; bounded doubled edge count; and endpoints within the logical node domain. Strided/read-only source arrays are permitted because only new arrays are written.

Other dtypes, ndarray subclasses, unsupported shapes, byte order, endpoint domains, platforms, or counts return `None` and execute the original code. Only `MemoryError` from fast construction is caught, so its scratch frame can be released before the caller attempts the original path. Unrelated exceptions are not swallowed. This is not protection from an OS/cgroup OOM kill, and it does not promise unchanged out-of-memory success/failure thresholds.

There is no absorbing-state, finite-state projection, approximate bound, partial evaluation of transcendental functions, or recurrence reassociation in the accepted change. Those classes of optimization were considered, but none was needed to remove the measured repeated traversal. No execution block changes the logical network, radius, boundary, or chronology.

## 6. Candidate register and rejected experiments

| Candidate | Evidence and estimated coverage | Decision |
|---|---|---|
| A1: stable order-preserving CSR | Original CSR occupies 87.50% of the measured single-worker component, **derived**. Arrays/artifacts match local tests; source patch retains fallback. | **Selected local candidate**; full repository gates outstanding. |
| Independent jobs, jointly tuned W/H | Full defined component batches measured for 1×4, 2×2, 4×1 under the same quota. | **Selected benchmark scheduling policy**; not a change to `UNA.RunBatch`. |
| Numba linear counting/scatter | Warm CSR faster in small experiments; first call compilation plus execution 1.906793 s **measured**. That first call includes compiler initialization, not purely marginal warmed-application compile cost. | Not promoted: no measured end-to-end gain sufficient to repay another compiled path after A1. |
| Immediate scalar Dijkstra relaxation | Concrete duplicate-neighbor counterexample, described below. | **Rejected: changes observable execution.** |
| Unstable sorting / graph deduplication | Can reorder ties or coalesce parallel incidences; analysis, not timing evidence. | Rejected for this contract. |
| Skip unexported metrics | Errors, direct return values, state, and warning observability not fully proved dead. | Deferred; no implementation or savings claim. |
| Shrink search scope, avoid full destination scan | Potential redundant scratch/scanning, but complete valid-domain and state proofs not performed. | Deferred; performance **unavailable**. |
| Cross-run topology cache | Numerical identity would need content, CRS, precision, weight columns, obstacles, Settings and mutation/invalidation rules separate from export identity. | Deferred; no cache identity or recovery contract weakened. |
| Parallelize current RunBatch | Settings mutations, topology reloads, logs, result capture and composite order are shared/chronological. | Rejected as a blind transformation; independent job wrapper only. |
| GPU or new numeric backend | No accelerator, crossover, full residency-region timing, math parity, or packaging evidence. | Not admitted. |
| Replace K-alternatives with aggregate flow | Similar result schema does not establish scientific equivalence. | Rejected as workload/model substitution. |

### Actual rejection witness

For a row with two parallel incidences to the same neighbor, candidate distances are `1.0` then `2.0`, and the old label exceeds both. The original vectorized mask observes the old label for **both** incidences, then stores both candidates in order. An immediate scalar test sees the updated label on the second incidence and rejects it. The tested small graph finishes with neighbor label **2.0 in the original versus 1.0 in the proposed scalar update, measured**. Raw witness: `results/experiments_summary.json`. This demonstrates why an apparently standard Dijkstra rewrite is not an exact performance patch here.

The Numba scatter experiment changes both algorithm and execution system; it is not evidence attributing speed to a compiler alone. No same-algorithm backend speed claim is made. No full experiment was repeated merely to add an attractive framework comparison.

## 7. Memory-lifetime inventory

Let `O` denote origins and `D` destinations. The following are **algorithmic storage derivations**, not measured total resident memory.

| Lifetime | Explicit storage and observations |
|---|---|
| Persistent graph output | CSR pointer `8(N+1)`, neighbors `16E`, weights `16E`, flags `2E`: **`8(N+1)+34E` bytes**. This is not the entire Topology/GIS object. |
| Persistent analysis inputs | Origin terminal IDs/weights `32O`; destination terminal IDs/weights `32D`, plus destination weights `8D`. Geometry, source dataframes and native objects are additional. |
| CSR histogram | Sources `16E`, counts `8N`, pointer `8(N+1)` coexist. Counts are released before sorting. |
| CSR grouping | Sources `16E`, permutation `16E`, pointer, plus native stable-sort workspace, whose exact peak is **unavailable**. Sources are released after grouping. |
| CSR weight gather | Permutation `16E`, neighbor result `16E`, temporary weights `16E`, gathered weights `16E`, pointer coexist: **`64E+8(N+1)` explicit bytes**. The source weight arrays were already persistent inputs. |
| Search/metrics | Per active origin: scope `8(N+D)`, destination distances `8D`, input-dependent heap, masks, filtered arrays and KNN sort buffers. Actual compiler/native temporary lifetime was not exhaustively measured. Result arrays: `32O`. |
| Optional dense OD | `8OD` output bytes only when requested; not present in the timed accessibility workload. |
| Export | Result GeoDataFrame and GIS/native writer buffers after construction/search. They must not be added to already-dead CSR scratch as though concurrent. |
| Job queue/runtime | Bounded small argument descriptors and results; no full topology arrays sent through multiprocessing IPC. Each process loads its own fixture. Interpreter, Numba/OpenMP, allocator and GIS runtime are additional. |

Use

\[
M_{live}=M_{persistent}+\max_s M_{workspace,s}+M_{queues}+M_{native}.
\]

No mmap or device-unified-memory assumption is used. Native sorting/writer allocations and heap fragmentation remain unquantified. The explicit buffers explain the work reduction; empirical memory observations supplement rather than replace that model.

**Measured allocation-instrumented passes**, separate from speed tests (`results/final_memory.json`):

| Fixture | Original tracked peak bytes | Candidate tracked peak bytes | Identical CSR payload bytes |
|---|---:|---:|---:|
| N=4,096; E=8,064 | 1,521,563 | 561,672 | 306,952 |
| N=16,384; E=32,512 | 6,164,539 | 2,213,675 | 1,236,488 |

These are `tracemalloc`-tracked Python/NumPy allocations, **not** whole-process RSS or an accounting of all native memory. At final batch scale the maximum sampled cgroup memory was **1,079.06 MiB original, 787.54 MiB candidate, derived from measured samples**. Sampling occurred about every 10 ms and may miss instantaneous peaks; cgroup memory includes more than private working arrays. Both observed samples are below the 4 GiB budget, but larger datasets are not thereby certified.

## 8. CPU/GPU and heterogeneous decision

CPU is retained for the selected change because the existing CPU implementation now removes the dominant measured service cost with no device boundary or new runtime. This is **not** a measurement that GPUs are slower for UNA generally.

| Stage | Parallelism, ordering, intensity and working set | Reuse/divergence, movement and numerical constraint | Classification |
|---|---|---|---|
| GIS validation/topology/snapping | Many features, but library/stateful setup and irregular geometry; arithmetic intensity and peak native working set **unavailable**. | Host GIS objects feed multiple later stages; offload would require conversion and ownership design. CRS/errors/geometry semantics observable. | **CPU preferred provisionally**; GPU geometry evidence insufficient. |
| Directional costs / CSR | Costs are edge-independent; row incidence order must survive grouping. Sorting/gathering is movement-heavy, CPU cache/DRAM intensity not measured. Explicit storage above. | Reuse per origin is large, so constructing once matters. Existing weights copied exactly. An isolated GPU builder must return CSR to CPU search. | **CPU preferred for selected scope**; whole-region offload remains a candidate. |
| Bounded Dijkstra | Independent origins, but each priority queue and row-update chronology ordered; random gathers and divergent path lengths. Scope/heap per active origin. | Read-only CSR reused across origins. GPU residency could help only with a sufficiently large search/reduction region. Duplicate-neighbor behavior, ties, dtype and CPU fast-math behavior must survive. | **GPU candidate, unvalidated**; CPU retained. |
| Destination adjustment / metrics | Independent origins; each metric has existing reductions, sort/tie order and exponentials. Per-origin destination scans can dominate large-D work. | CSR/search output should stay resident through metrics; returning only scalar metric arrays is preferable. CPU/device exp, FMA, denormal and reduction implementations need characterization. | **Heterogeneous/GPU candidate, unvalidated**. |
| Turn-aware / flow engines | Arc-state searches, reverse gradient tables, OD/tree loading and optional path geometry; stage costs not profiled here. | Aggregate destination-gradient reuse already exists. Ordered accumulations and path-level semantics cannot be inferred from output schema. | **Insufficient evidence**. |
| Export / composite | Host object/format creation and ordered batch capture. Required output amount fixed by format flags. | Download required results once; retain existing write order, errors and policy. Composite reduction chronology observable. | **CPU preferred provisionally**. |

All classifications beyond the measured CSR region are hypotheses for investigation, not measured backend winners. No GPU numerical profile, relaxed precision, tree reduction, fast-math extension, or device-specific dependency was added.

### IO-aware accounting for an isolated CSR offload

| Metric | Selected CPU path | GPU candidate |
|---|---|---|
| Work | Histogram, stable grouping, prefix, gathers; no weight arithmetic | Equivalent ordered grouping required; implementation **unavailable** |
| CPU DRAM / device-global bytes | Actual transfers **unavailable**; explicit live arrays inventoried | **unavailable** |
| Host → device bytes | 0 by construction | **Hypothetical:** `32E` for native endpoint and AB/BA arrays |
| Device → host bytes | 0 by construction | **Hypothetical:** `8(N+1)+34E` for complete CSR, unless search also remains resident |
| Intermediate materialization | Sources/permutation/temporary neighbor and weight arrays; no dense OD tensor | **unavailable** |
| New GPU launches / synchronization | None | **unavailable** |
| Peak live memory | Explicit inventory plus measured tracked allocations above | Device/runtime/safety margin **unavailable** |
| New first-use compilation | No new compiled kernel in the patch | **unavailable** |
| Warm end-to-end CSR stage | 0.001161 s median, **derived from measured values** | **unavailable** |

A CPU-generated Boolean flag array could reduce hypothetical download bytes; persistent device search could eliminate the complete CSR download. Neither was implemented or timed. Geometry inputs are not densified merely to fit a GPU API.

Admission would require

\[
T_{prepare}+T_{H2D}+T_{launch}+T_{kernel}+T_{D2H}+T_{sync}
 +T_{conversion}+T_{remaining}<T_{CPU}.
\]

For repeated jobs, setup `J` must be separated from recurring transfer/kernel/synchronization costs. Compare fused CSR/search/metrics regions as well as isolated kernels; device residency can change the answer. Actual bandwidth/intensity/occupancy/register pressure/lane efficiency and calibrated crossover are **unavailable**. There is no reason to select a backend solely because a device is present, and no accelerator path is admitted without a real complete-pipeline win against a tuned CPU baseline.

## 9. Selected configuration and final benchmark

### Exploratory equal-budget selection

**Medians and throughput derived from measured values**, two batches of eight jobs for each pair, from `results/batch_W*_H*.csv`:

| Workers × active Numba threads | Original batch seconds | Candidate batch seconds | Candidate jobs/second |
|---|---:|---:|---:|
| 1 × 4 | 2.435067 | 0.308299 | 25.949 |
| 2 × 2 | 1.321173 | 0.190888 | 41.909 |
| 4 × 1 | 1.121441 | 0.190906 | 41.905 |

Candidate 2×2 and 4×1 are indistinguishable at this sample resolution. The selected candidate uses **two processes, two active Numba threads per process, queue depth two, at most two simultaneous writers**. It reduces replicated runtime memory/startup relative to four processes. No new block/tile size was introduced; original origin scheduling and the full logical domain remain intact. The best original configuration in this tested grid is four processes, one Numba thread each. This is not proof of a global scheduling optimum.

### Frozen final comparison

The final helper and integration-generator hashes are in `results/final_candidate_manifest.json`. The final run verified these hashes before and after benchmarking. Early exploratory runs preceded an added 64-bit-platform admission guard; the final comparison uses the frozen guard. No algorithmic change was made after final measurement.

**Final defined component-batch statistics, derived from measured values:** eight independent jobs per batch, six batches per variant, two fresh-pool rounds with reversed variant order. Input and artifact hashes matched across all jobs and variants.

| Metric | Best tested original | Selected candidate |
|---|---:|---:|
| Workers × active kernel threads | 4 × 1 | 2 × 2 |
| Median eight-job batch time | 1.172292 s | 0.219555 s |
| Observed batch range | 1.016355–1.663360 s | 0.169245–0.321068 s |
| Throughput from median | 6.8242 jobs/s | 36.4374 jobs/s |
| Throughput ratio | reference | **5.3394×** |
| Maximum sampled cgroup memory | 1,079.06 MiB | 787.54 MiB |
| Median pool startup + common warm-up | 2.548948 s | 1.799008 s |

Pool startup/warm-up is recorded separately and is excluded from warm batch time. Initializers warm the shared extracted numerical kernel and GIS writer using a tiny/common preparation policy in both variants. This is not a cold-cache or first-use comparison. The final batch timer includes task dispatch, job-local verification hashing, result collection, and completion; the single-component stage timer stops before hashing. Hence the two ratios describe different complete boundaries and must not be multiplied.

Warm filesystem caches are part of the conditions. Repeated jobs use the same fixture and distinct output directories; every job still reloads inputs, recomputes all metrics, and rewrites requested artifacts. There is no result cache or hidden work reuse. Six batches per variant are a small, noisy sample, not a confidence-qualified service-level guarantee.

### Cold lifecycle diagnostic

**Measured**, one fresh-process, empty-Numba-cache diagnostic per variant, one worker/one active thread:

| Metric | Original | Candidate |
|---|---:|---:|
| Entire process wall time | 12.65 s | 12.90 s |
| Component including first kernel use | 10.685981 s | 10.827599 s |
| Kernel stage including compilation | 10.340826 s | 10.755504 s |
| Process peak RSS | 299,028 KiB | 299,272 KiB |

**No cold-start improvement is demonstrated.** Unchanged Numba compilation dominates, and one sample cannot resolve small differences. Empty Numba cache is not cold filesystem/page cache. A short-lived process and a warm worker service are different workloads; the warm throughput result is not applied to the former.

## 10. Verification tiers and remaining gates

**Measured local outcome:** `61 passed, 1 skipped` in `results/tests_final.txt`. The initial failed Feather attempts are retained in `results/tests_initial.txt` rather than represented as passes.

| Tier | Executed evidence | Status |
|---|---|---|
| L0 | 910 exhaustive tiny ordered graph cases; self-loops, duplicates, isolated/empty graphs; special IEEE values and 4,000 seeded arbitrary bit patterns; ownership and admitted-domain checks | Passed locally |
| L1 | Original extracted CSR versus candidate; fallback error behavior; complete per-origin scopes/predecessors; unchanged Dijkstra Python-line state/heap trace; dense OD and all four metrics across decay/cutoff cases | Passed locally |
| Integration contexts | Generated plain/elevation builder excerpts, fast/fallback assignments, topology/logger references; contextual patch application with shifted offsets | Passed locally; **not full-repository integration** |
| Component artifact check | Original extracted accessibility writer, raw CSV/GeoJSON bytes, schemas/log messages | Passed locally |
| Feather | PyArrow dependency missing | **Skipped / unavailable** |
| L2 genuine application | Actual pinned package, real GIS inputs, full chronology, obstacle/turn dispatch, project/batch/composite artifacts | **Not run / unavailable** |
| L3 selection | Synthetic CSR sizes and defined component worker/thread experiments | Run; synthetic surrogate only |
| L4 production | Final immutable implementation on actual user dataset/hardware/target | **Not run / unavailable** |
| Full matrix/release CI | All supported Python/NumPy/Numba/platform versions, independent reviewer, packaging/release checks | **Not run** |
| GPU | Transfer, compilation, synchronization, math parity, capacity and end-to-end gates | **Not run** |

Exact artifact equality here is not only a final scalar comparison: CSR bits, complete search labels, heap/state chronology, metric arrays and file bytes are covered in the local extraction-based tests. However, these cannot replace full-package tests where Settings/Topology or native GIS libraries exercise other paths.

## 11. Reproduction and safe integration

`README.md` contains a minimal entry point. The following commands are the local experimental procedures; run from this bundle in an environment with the recorded dependencies. They write/overwrite the local `results` directory, so preserve this evidence bundle before rerunning.

```bash
export PYTHONPATH=.
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export NUMBA_NUM_THREADS=4

python -m pytest -q tests
python -m benchmarks.run_components --phase selection --threads 1 2 4 --repeats 3
python -m benchmarks.run_batch --workers 1 --threads 4 --jobs 8 --repeats 2
python -m benchmarks.run_batch --workers 2 --threads 2 --jobs 8 --repeats 2
python -m benchmarks.run_batch --workers 4 --threads 1 --jobs 8 --repeats 2
python -m benchmarks.run_experiments
python -m benchmarks.run_components --phase final --threads 1 --repeats 3
python -m benchmarks.run_final_batch
python -m benchmarks.run_memory
python -m benchmarks.summarize
```

The quota is an environmental condition, not created by these commands. Do not run the matrix concurrently or infer four-core equivalence on an unconstrained host from environment variables alone. The supplied final batch script selects the recorded configurations; it is not a general adaptive scheduler.

Cold diagnostic, with a distinct empty cache for each variant:

```bash
for variant in baseline candidate; do
  cache_dir=$(mktemp -d)
  NUMBA_CACHE_DIR="$cache_dir" /usr/bin/time \
    -f 'process_wall_s=%e\nuser_s=%U\nsystem_s=%S\nmax_rss_kib=%M' \
    -o "results/cold_${variant}_process.txt" \
    python -m benchmarks.run_components --phase cold --variant "$variant" --threads 1
  rm -rf "$cache_dir"
done
```

### Applying against actual upstream source

Use a clean, inactive worktree at the pinned commit. The default installer checks the complete Git blob hashes of both files before generating a patch; it refuses mismatched source and an existing helper. This is stronger than relying on a context-only patch. No package source was remotely modified here.

```bash
# Run from this bundle. /path/to/una must be an actual clean pinned checkout.
python verify_repository.py /path/to/una
python apply_optimization.py /path/to/una > verified.patch
git -C /path/to/una apply --check "$PWD/verified.patch"
git -C /path/to/una apply "$PWD/verified.patch"
python verify_repository.py /path/to/una --check-patched
```

`patches/ordered_csr.patch` is also supplied for review. Its context application was checked against the fetched source excerpts with shifted offsets, **not** a complete downloaded checkout. Prefer the blob-checking generator for integration. Optional `--write` checks both source blobs before replacing any files, but replacements are atomic per file, **not a transaction across files**. Do not apply in a live running installation; retain Git recovery and review.

The actual application gate must run separately in clean baseline and patched environments, using identical real Settings and inputs. The runner preserves all scientific and output-format flags, changing only the explicitly recorded output location/timestamp-folder policy for comparison:

```bash
python benchmarks/run_repository.py --repo /path/to/baseline \
  --settings /path/to/real_settings.json --output results/real_baseline \
  --analysis accessibility --threads 1 --runs 3
python benchmarks/run_repository.py --repo /path/to/patched \
  --settings /path/to/real_settings.json --output results/real_candidate \
  --analysis accessibility --threads 1 --runs 3
```

Those commands are **provided, not reported as executed**. Compare result arrays, artifacts, public state, warnings and Settings chronology, then tune actual independent-job concurrency against the strongest equal-policy baseline. The runner records full API time, import time, process peak RSS and artifact/array hashes; it is not itself the complete release suite or a substitute for stateful tests.

## 12. Completion assessment

The work removes a demonstrated redundant traversal and provides an explicit exactness argument, fallback, local tests, raw timing/memory data, scheduling selection, and rejected-experiment evidence. It meets those **local candidate** deliverables.

It does **not** meet the requested final production completion criteria: no genuine full-application benchmark, actual target workload/SLA demonstration, complete source-extraction verification, full artifact/dependency matrix, accelerator evaluation, or independent final review is available. Unsupported representations remain on the unchanged reference path, but the entire repository's operational contract has not been certified. Do not mark those checks passed or publish the component ratio as a UNA application speedup.

The next required gate is the pinned-checkout verification and real GIS/public-API run—not another microkernel optimization. The observed remaining CSR fraction provides no justification for continued backend complexity on this fixture.

## Source inventory

All source observations refer to commit `c15ebda6981397f46eed5c2d55229f71e57d44fb`; documentation is descriptive, executable source defines the changed behavior.

- **S1:** `src/urban_network_analysis/Engines/Accessibility.py`, blob `246fe439591379038c5e73d0388f0720726213e9`: numerical kernels and `_build_compact_graph_from_topology`.
- **S2:** `src/urban_network_analysis/Engines/AccessibilityWElevation.py`, blob `69061684380ffa9d890b75804bbcd3b4642eae90`: matching kernels, typed costs/corrections and CSR builder.
- **S5:** `src/urban_network_analysis/UNA.py`, blob `f803a536277c178f512e08243ac12ce18bf3de46`: `RunAccessibility`, `RunBatch` and public orchestration inspected in the first source window.
- **S6:** `src/urban_network_analysis/Engines/Base.py`, blob `17442082028d47a78c29806fb21ffa35071b9280`: `resolve_output_folder`, export utilities, `ExportAccessibilityResults`, beginning of ODM export.
- **S7:** `Topology.py` beginning and `DESIGN.md`: network load/validation, graph/layer mutability, design intent. Complete topology/Settings/flow source was not exhaustively audited.
- **S8:** `pyproject.toml`, blob `55c672756de24ecaa565df9ffd444d7587dc0fb4`: package dependencies, build/version configuration.
- **S9:** `AggregateFlow.py` introductory design and imports, blob `0114ce8f318a3247b62e3d468e32d54e692737a2`, plus `README.md`: destination-gradient reuse and distinction from path enumeration. Not a measured flow profile.

Source browser: `https://github.com/City-Form-Lab/urban_network_analysis/tree/c15ebda6981397f46eed5c2d55229f71e57d44fb`.

## Evidence map

| Deliverable | File(s) |
|---|---|
| Selected implementation / immutable identity | `implementation/_ordered_csr.py`, `results/final_candidate_manifest.json` |
| Integration patch / guarded generator | `patches/ordered_csr.patch`, `apply_optimization.py` |
| Contract, proof, costs, heterogeneous classification, memory lifetimes, registers | This report |
| Raw small-scale experiments and rejection witness | `results/small_csr_benchmarks.csv`, `results/experiments_summary.json` |
| Raw final warm batch and per-job measurements | `results/final_batch.csv`, `results/final_batch_jobs.csv` |
| Raw final component stage measurements | `results/final_components.csv` |
| Memory and exact environment | `results/final_memory.json`, `results/environment_final.json` |
| Cold start / compiler boundary | `results/cold_*.json`, `results/cold_*_process.txt` |
| Test results and contextual patch check | `results/tests_final.txt`, `results/tests_initial.txt`, `results/context_patch_check.txt` |
| Derived statistics | `results/summary.json` |
| Deferred actual-repository gates | `verify_repository.py`, `benchmarks/run_repository.py` |