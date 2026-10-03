# Prespecified 10x benchmark and selection protocol

## Baselines and ablations
B0 = pinned current main 395cdc5f..., already optimized. B_current = later main only if runtime
drift is real and stronger; retained for final comparison. A = strongest verified configuration
of B0/B_current on this workload with existing cache and parallel public API capabilities.
B = sparse queries only. C = routing trace reuse only. D = C + per-metric reuse. E = integrated
sparse+reuse. N/G = SAME reduced-work plan on native/GPU. Benchmark each accepted delta at matched
configuration first, then compare strongest arm-specific configurations under equal total budget.
Do not multiply ratios. Portable Python oracles are never performance baselines.

## Cache regimes and fairness
1. FIRST_COHORT: process/JIT may be warm as declared; numerical routing trace AND final results
   start empty for both arms. Candidate trace build/pack/key/commit is inside timing. Existing
   graph/geometry cache is either identically warm or identically cold as frozen. This is the
   primary distinct-job comparison; the new trace cache is not prepopulated off the clock.
2. RESIDENT_DISTINCT: traces resident and metric parameters genuinely changed, both arms' allowed
   graph/data caches primed equally. Report build/amortization separately; never mix with (1).
3. REPEAT_IDENTICAL: both arms use their best existing full-result caches. Secondary only.
4. COLD: fresh process, empty arm-specific JIT cache, input conversion and final output included.
Explicit workload/profile/code/cache identities and hash_seed recorded per run. Never key JIT
caches by a guessed worktree substring. No shared mutable output paths or caches that mix arms.

## Primary timer and successful throughput
Timer starts immediately before the required public RunBatch (cohort) or public RunAccessibility
(single) call and ends after it returns with required files, state, validation and checkpoint/
cache/publication guarantees satisfied. Do not pre-run AddNetwork, snapshot hashing, planning,
trace construction, result conversion or state rehydration outside this boundary if the normal
public call would do that work. Record outer process/import/pool/JIT overhead separately AND
report process-start-to-completion as the cold boundary. No second diagnostic constructor in
service totals. Stage times are exclusive or explicitly nested, never summed twice.

Post-boundary independent comparison hashes may be outside service timing but must be timed,
retained and identical-policy across arms. Only validated jobs count. Any error, missing expected
artifact, skipped required computation, numerical mismatch or unverified installed route makes
a qualification window invalid. Report failures and reason, never speed from submitted jobs.
N_valid/service_wall is the throughput. Also report validation-inclusive wall for audit. Workers
must finish computation and synchronization, not merely enqueue asynchronous device work.

## Freeze before observing candidate results
Write PRIMARY.json using templates/primary.json after baseline-only calibration. Fix workload
bytes/order, exact beta list, output policy, profile/environment, baseline admission, run count,
per-window job count/cohort repeats, ordering, seed and comparison method. Calibrate using baseline
only, not until a desired ratio appears. Each window contains a fixed count of complete cohorts
or jobs. Target >=60 s baseline aggregate service and >=10 s expected fast-arm measurement using
the prior 50x upper sensitivity, subject to resource budget; this is scheduling policy, not a
prediction. If too large, record a short-window limitation before selection and use repeated
independent windows, never hide duration. Exact same N for A and B in a pair. Where repeating
cohorts within a window, clear trace/final numerical cache to the frozen initial regime between
cohorts outside per-public-call service, record reset time separately, and sum complete public
call times; also report outer wall. Do not count one initial trace build followed by identical
answer hits as FIRST_COHORT. A changed reset strategy creates a different benchmark regime.

## Selection and confirmation
L3 screening: two paired windows per admitted candidate/config; first tune at most four feasible
process/thread configs based on measured single-job memory. Choose at most two candidate finalists
plus A. Keep logical flow reduction stripes fixed even when physical workers change.
Final: freeze ONE immutable candidate and configuration, then seven independent paired windows
(minimum). Order AB,BA,AB,BA,AB,BA,AB. Fix seed 20261003 and 20,000 bootstrap resamples over paired
log ratios. Statistic: median(log(A_wall/B_wall)); report exp(statistic). One-sided 95% lower
bound: exponentiate the 5th percentile of bootstrap median log ratios. tools/gate_10x.py implements
the frozen interpolation. Median >=10 and lower bound >=10 required. This is a within-workload
scope claim, not a guarantee for every future city. No optional stopping after favorable samples.
One FULL repeat block permitted only for a preregistered invalidation (external load/fault), with
all blocks retained; no selective reruns of slow candidate windows. Each pair is the unit, not
individual jobs from the same warm pool. No pooling unrelated families to cross the gate.

## Protected cells and default dispatch
Five pairs for tiny GIS, search-dominated/dense and cold process. Default candidate B/A wall-ratio
median <=1.05 and one-sided 95% upper <=1.10, identical required behavior. Failure means refine a
conservative pre-execution dispatcher and revalidate, or keep explicit scoped mode, not blanket
default. Final dispatch thresholds are frozen before held-out runs. Profile instrumentation is
separate from timings. Count touches, candidate/kept destinations, heap work, unique trace keys,
metric reuse, bytes, serialization and fallback engagement outside final timing as diagnostics.

## Memory and artifacts
Measure simultaneous coordinator+all descendants RSS (PSS/USS where available), cache residency,
device peak and staging; interval and missed-short-peak limitations recorded. Post-run RSS is
not a phase peak. Required CSV/GeoJSON/Feather outputs remain in the public boundary, same fs/
durability. File bytes are not physical I/O traffic. No asymmetric /tmp vs network storage.
Source/wheel/binary/toolchain/harness hashes and complete raw records must be retained and linked.
