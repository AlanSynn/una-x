# UNA-X: two structural routes toward 10x application throughput

Reference inspected: AlanSynn/una-x commit
`395cdc5f683894b6f2ba460f7dbcefee99981ba3`.
This is a research prototype and an evidence packet, **not a production patch,
not an installed-package qualification, and not a measured UNA end-to-end win**.
No repository was modified or pushed.

## What was actually executed

A source-derived prepared-array reference implements the current admitted A3
search schedule and the original metric expressions. It was compared against:

1. Epoch-tagged private labels plus a terminal-to-destination reverse index.
2. Shared routing/kept-distance traces across distinct gravity-beta jobs.
3. The same trace sharing plus exact reuse of metric fields whose complete
   numerical inputs do not change.

GitHub source reads succeeded through the connector, but the container could
not fetch an executable checkout (DNS failure). Therefore the local baseline
is a transcription of selected current-source computational behavior, NOT the
installed current UNA package. `verify_checkout.py` supplies the next direct
kernel comparison against a pinned complete checkout. **It was not run here.**

The prepared-array timer starts with CSR and point-terminal arrays already in
memory. It includes candidate guards, reverse-index construction, per-lane
workspace allocation, every origin search, metrics, and fully available result
arrays. The cohort timer also includes building the trace from scratch. It
excludes geometry/file loading, topology/CSR creation, public API wrappers,
cache-store I/O/validation, RunBatch state transfer/publication, output files,
process startup, and JIT. No disk-cache or GPU speedup was measured.

Both primary arms use one process and four Numba threads, a four-core cgroup
CPU quota and 4 GiB memory limit. `environment.json` records exact versions.
The host reports AMD EPYC 9V74; Python 3.13.5, NumPy 2.3.5, Numba 0.65.1,
llvmlite 0.47.0. These are not laptop/M1 measurements. Phase peak memory,
physical DRAM bytes, cold compile cost, and real-application stage shares are
unavailable. Post-run RSS is recorded but is not a phase peak.

## Sparse local-query experiment

Each cell has five paired repetitions; arm order reverses on alternate reps.
Every result array was compared byte-for-byte outside the timed boundary.
Numbers in this table are **derived from measured medians**.

| Synthetic workload | V | D | O | Cutoff in synthetic cost units | A3-derived reference, 4T | Sparse prototype, 4T | Ratio |
|---|---:|---:|---:|---:|---:|---:|---:|
| Tiny | 100 | 26 | 14 | 4 | 0.025960 ms | 0.072078 ms | 0.360x |
| Search-dominated control | 8,464 | 2,560 | 512 | 20 | 14.959747 ms | 17.852626 ms | 0.838x |
| Local, medium global domain | 65,536 | 65,536 | 512 | 4 | 23.910867 ms | 3.113592 ms | 7.680x |
| Local, large global domain | 262,144 | 262,144 | 512 | 4 | 109.923225 ms | 7.423693 ms | 14.807x |
| Same domain, many origins | 262,144 | 262,144 | 8,192 | 4 | 1,715.870517 ms | 35.737006 ms | 48.014x |

All geometries are synthetic grids with edge costs in [0.8, 1.2). None of these
is an observed-city or production workload. Full prepared arrays are used in
both arms; destinations are not removed from the requested workload. The fast
implementation proves that omitted *evaluations* cannot contribute on its
admitted domain. The small/search-dominated regressions must not be hidden:
this requires workload-aware dispatch and a trusted original route.

Raw observations: `raw/local_*.json`, `raw/tiny_*.json`,
`raw/search_dominated_*.json`. One-thread controls are also retained; the headline
uses the four-thread comparison, not the much larger one-thread ratio.

## Distinct-parameter cohort experiment

Prepared graph: V=8,464, D=2,560, O=512, cutoff=20, four threads. Beta values are
all distinct and retained in each raw record. Every exponential-gravity result
is computed for its requested beta. The first and last exponential outputs
are asserted different; this is not a repeated-final-answer benchmark.

Numbers are **derived from measured medians** (five paired repetitions each).

| Jobs | Change | Matched repeat arm | Candidate | Ratio |
|---|---|---:|---:|---:|
| 8 | Routing trace reuse | 0.134449459 s | 0.037409766 s | 3.594x |
| 32 | Routing trace reuse | 0.524428298 s | 0.084615158 s | 6.198x |
| 8 | Trace reuse + unchanged metric fields | 0.130594480 s | 0.023795361 s | 5.488x |
| 32 | Trace reuse + unchanged metric fields | 0.602054310 s | 0.038160851 s | 15.777x |

These are separate matched experiments; do not compare their baseline medians
as if they were simultaneous or multiply their ratios. In the fixed logistic
KNN setting, changing exponential beta does not change reach, logistic gravity,
or KNN. Those fields are evaluated once and copied into fresh, non-aliased
result arrays for each job. The exponential expression is recomputed.

The ge-only compiled expression matched the source-derived full fold in 170
small finite random cases, and every cohort result matched its independently
computed parameter result. This is not proof that a newly compiled reduction
matches every legacy compiler/math-library profile. The actual project's pinned
reduction plans and direct differential tests must decide admission.

A Numba signed-index warning from the typed-list replay is retained in
`reuse_log.txt`; indices in these fixtures are nonnegative and below 512. A
production implementation needs explicit checked index types and bounds.

## Exactness argument for the sparse query

Let b be the actual float64 result of the original initialization 1+cutoff.
Require finite nonnegative prepared costs/seeds/terminal weights and b>cutoff.
Reject the fast path when the sentinel inequality fails; it can fail for large
floating-point cutoffs. Do not infer b from real-number arithmetic.

The private epoch view is L[v]=stored[v] when stamp[v] equals this origin's epoch,
and L[v]=b otherwise. At the beginning it is identical to the dense reference.
The two seed assignments occur in the original order, even when they name the
same vertex. Each popped row computes all eligible edges before any label
updates. Thus its comparisons read the same snapshot, produce the same eligible
sequence, and replay the same updates/heap pushes. Induction gives identical
logical labels and heap transitions in the admitted arithmetic implementation.
The prototype does not claim NaN/invalid-input legacy behavior.

Every seed and label write is registered as touched. If neither terminal of a
destination is touched, both its labels equal b. Nonnegative terminal costs
imply each original rounded sum is at least b, so neither can enter the cutoff.
Only destinations linked to touched nodes need to be evaluated. Deduplicate IDs
and sort them by their ORIGINAL destination index. Compute the original two
sums and builtin min, then run the original metric expressions. The retained
values/weights have exactly the original order, including tied KNN distances.
No graph cropping, reverse search, shortcut weight, or reduction reassociation
is involved.

Private storage is reused only within a call. Epochs are origin_index+1, and
fresh stamp arrays prevent cross-call overflow. A persistent production arena
needs an explicit wrap/reset policy. Approximate lifetime accounting is
24*V+16*D+16*max_degree bytes per lane, plus local candidate/metric temporaries;
the reverse index is 8*(V+1)+16*D bytes plus construction scratch. These are
**derived array payload formulas, not measured RSS or DRAM traffic**.

Local validation: 140 component cases, 540 full final label-vector comparisons
across origins, two fast-domain refusal checks, plus 170 metric-expression cases.
Tests include multiedges, self-loops, repeated endpoints, isolated nodes, empty
destination sets, zero costs, signed zero and cutoff-adjacent cases. They do
not cover full public chronology, all error behavior, cross-platform bit parity,
compiled per-pop traces, installed wheels, batch cancellation, or output files.
Those remain required before integration.

## The application-level gate

For original application fraction f in a changed region, region speed s and
new overhead delta normalized to the old application wall:

    total_speedup = 1 / (1-f + f/s + delta)

With the measured component ratio 48.0138296 and zero added external overhead,
10x requires f >= 0.9191433. This is **derived from measured component values**,
not a measurement of f. If unchanged per-job service is U, the directly useful
inequality is:

    (U + 1.715870517) / (U + 0.035737006) >= 10
    iff U <= 0.150944495 seconds.

Thus the prototype is not sufficient if a real job spends seconds rebuilding
GIS state or serializing outputs. At hypothetical f=0.95, zero new overhead,
the modeled total speedup is 14.3295x. All these what-if application projections
are explicitly hypothetical until public-path timings exist.

For K jobs sharing routing, let f be the reusable share INCLUDING only work
whose full dependencies actually match. Ignoring double-counting:

    cohort_speedup = 1 / (f/K + 1-f + delta)

This is a model, not a measured UNA claim. Exact final-result caching already
exists on current main; a fair repeated-identical-job baseline must enable it.
The new opportunity is intermediate routing traces and individual metric
identities across DISTINCT jobs, such as beta/weight/KNN studies.

## Integration priorities

1. Verify these prototypes against the actual frozen kernels using
   `verify_checkout.py`; rebuild/repair the oracle on divergence, never loosen
   tolerances. Preserve una_legacy, madina_legacy and corrected_v1 separately.
2. Add an opt-in internal sparse route for large local queries. Keep current
   paths for small/dense/unsupported inputs. Do not delete full public ODMs.
3. Separate routing identity from metric identity and artifact identity. Cache
   ordered sparse distance/ID traces, not a global O*D matrix. Include graph
   ordering, typed seed values, cutoff, cost/obstacle/elevation state, profile,
   code/math implementation identity. Preserve input validation and mutations.
4. Plan equivalent batch rows as a cohort. Share immutable preparation/trace
   data while keeping mutable outputs and ordered publication separate. Build
   or retrieve a trace once, then replay the actual ordered metric operations.
5. For flow, retain existing F2 locality. Investigate sparse-output bounded
   shortest paths upstream of it and exact routing/update tapes across repeated
   OD configurations. Preserve predecessor/tie semantics and logical stripes.
   A pre-summed unit-flow matrix times changed weights is NOT generally bitwise
   equivalent to the current ordered accumulation.
6. Native CPU/GPU schedules should execute this reduced-work plan, not merely
   port the global scans. Retain trusted delicate math or validate exact device
   math. Charge transfers, compilation, conversion, sync and outputs.
7. Measure full installed RunAccessibility and public RunBatch with real inputs,
   required artifacts, per-stage counters, simultaneous process-tree memory,
   cold/warm cache states and equal CPU/thread budgets. Record new raw bytes.
   Freeze the 10x target before final selection; require the appropriate paired
   confidence bound, not a kernel ratio or a tuned historical sequential arm.

## Commands used in this container

Run from this packet directory with the versions in environment.json:

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 NUMBA_NUM_THREADS=4
python run_sweep.py
python probe.py --side 512 --destinations 262144 --origins 8192 \
  --radius 4 --threads 4 --reps 5 \
  --out raw/local_large_many_origins_H4.json
python reuse_probe.py
python memo_probe.py
```

The `compile_warmup_s` field is warm-up/specialization service, not an isolated
cold compiler benchmark. The sweep runs tests first; many dispatchers are warm.

Next command on an actual pinned checkout (NOT RUN HERE):

```bash
python verify_checkout.py --repo /absolute/path/to/una-x \
  --out /absolute/path/to/new-evidence/kernel_bridge.json
```

Do not run against dirty user source or relax the blob checks to make it pass.
The scripts write only the supplied output paths and local Numba caches; they do
not touch the remote repository. Tests do not automatically authorize promotion.

## Source basis

- Current reference: https://github.com/AlanSynn/una-x/tree/395cdc5f683894b6f2ba460f7dbcefee99981ba3
- `Engines/_large_access_scratch.py`: A3 initialization and two-phase heap schedule.
- `Engines/AccessibilityWElevation.py`: destination adjustment, metrics and result-cache identity.
- `kernels/compacted.py`: recorded speed-neutral compacted producer, not sparse enumeration.
- `cache/stages.py`: current .npy byte serialization and fresh array restoration.
- `Engines/AggregateFlow.py`: existing local overlap kernel and dense SciPy gradient producer.

Source-derived code retains the upstream MIT notice in LICENSE.
