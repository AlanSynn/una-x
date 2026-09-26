# H02: implement and prove the new benchmark harness

Existing benchmarks/una_cpu is historical. Create benchmarks/large_e2e and tests/large_e2e/harness. Preserve old raw evidence. The new harness must run identically for B0 and every candidate. H02 is a delivered implementation task, not an already-existing runner.

## Frozen command interface to implement
Run the script by absolute path using each ARM'S venv Python, from a neutral working directory:

```text
<arm-python> <repo>/benchmarks/large_e2e/run.py \
  --manifest <workload.json> --arm <label> --identity <arm_identity.json> \
  --mode single|batch --jobs <K> --workers <W> --numba-threads <H> \
  --flow-stripes <Kflow-or-default> --queue-depth <Q> --writer-limit <L> \
  --cpu-budget <C> --memory-budget-mib <M> --timeout-s <T> \
  --cache-root <unique-cache-root> --out <new-run-directory>
```

Every argument is validated: positive integral counts, nonempty unique output directory, existing verified manifest, legal analysis, installed identity, enough memory/disk, actual pool admission. A missing parameter or malformed manifest fails before job execution. No implicit absolute laptop paths. Unknown analysis errors instead of defaulting to accessibility. No hidden source/auto-fallback arm in qualification; source diagnostics use a distinctly named diagnostic mode and cannot emit installed-qualified records.

## Mandatory implementation behavior
1. Dispatcher reads manifest analysis and explicitly calls RunAccessibility, RunFlow or RunODM; unsupported type fails. Batch and single paths use ONE shared dispatcher, not separately maintained logic. Return an engine/result signature appropriate to the requested analysis. Tests mock each method and assert exact call count, including zero calls to wrong methods.
2. Every worker validates resolved package path is under that arm venv's site-packages, using Path.resolve/is_relative_to or os.path.commonpath (not string-prefix matching). Validate all imported urban_network_analysis modules, wheel/source identity manifest and installed file hashes. No PYTHONPATH, editable installs, source .pth or repository import shadowing. Two arms can share version number; version string alone is not identity. Record interpreter path, package paths and module hashes.
3. Set environment/pool limits before importing numerical libraries. Record actual Numba layer/count after initialization; inspect flow topology pool count, BLAS/OpenMP/Arrow and turn/cluster pools. Reject insufficient budget; requested counts are not proof of effective counts. Frozen flow stripe count is numerical identity.
4. Spawn fresh workers. Each job owns a new UNA/Settings/Topology and unique output directory. Workers never share live engine objects. Warm-up matches the analysis/profile specialization or explicitly records additional first timed compilation. Warm-up outputs are separate and discarded only after hashes/metadata are recorded.
5. Use a bounded input queue AND bounded result channel. Limit in-flight work and interleave dispatch with result draining. Do not enqueue all jobs before reading a bounded result queue. Monitor worker liveness during dispatch, wait and shutdown. Put/get operations have short polling timeouts plus an overall monotonic deadline. Abrupt worker exit fails the run promptly, cancels outstanding owned jobs and retains results already received. No infinite readiness wait or interpreter-exit hang. Terminate/join only campaign-owned children; no blanket killall.
6. Writer concurrency is independently bounded by a shared semaphore around the existing export boundary in a harness adapter, not a changed production export policy. For final no-instrumentation public-path timing, use L=W unless a reviewed external pipeline preserves the exact writer semantics. Do not claim a writer-limit sweep happened if the adapter never controls writes. Record actual peak active writers.
7. Simultaneously sample coordinator and all descendants recursively at a fixed declared interval. Record summed RSS as a conservative resident measure (shared pages can double-count), plus USS/PSS where available, available RAM, swap/compression indicators when available, total child count and native/JIT memory. Do not sum different-time peaks and call that a simultaneous peak. Sampler overhead is measured; use identical sampling in both arms. Enforce watchdog/cancel thresholds and report sampled, not guaranteed, peaks.
8. Rehash all input files against the manifest before the session, outside primary application timing but inside cold/whole-session accounting. Immutable filenames alone are insufficient. Reject mutated inputs. Prevent outputs or warmups from colliding with inputs or previous runs.
9. Time the REAL public method, not a manually reconstructed chain. Diagnostic profiling may wrap named functions without extra invocations, preserve arguments/results/exceptions and log call counts; compare profile output against uninstrumented output. Inclusive spans and exclusive stage partitions are separate. Count the constructor once; never pre-resolve gravity cap outside RunFlow to measure RunFlow's original behavior.
10. Workers report application start/end, hash/validation start/end, startup/import/JIT time, output counts/bytes/signatures, analysis, status, paths, source/wheel/cache identities, requested/effective configuration and resource observations. Ensure result records are emitted on failure as well as success. Unexpected exceptions fail loudly; no fallback from candidate import to baseline.

## Time definitions
application_ns: immediately before UNA construction through return from the requested public method including synchronous required exports; Settings assembly included separately and in whole job.
cold_process_ns: before spawning interpreter through exit, includes import/JIT/conversion/output/verification; fresh empty cache for designated compile-cold test.
warm_batch_application_wall_ns: timed dispatch through final application-completion event received.
validated_batch_wall_ns: timed dispatch through receipt and successful verification of all job results. Primary successful throughput = validated_jobs / validated_batch_wall_seconds. Hashing is reported separately; because workers block for it before the next job, do not pretend it has zero throughput cost. Also report total_jobs/application window descriptively when all jobs validate.
whole_session_ns: process/pool setup, warm-up, work, validation and shutdown. Warm-up is not hidden in a headline cold end-to-end result.

Start barrier records every worker's ready identity and warm-up. End requires all expected jobs, no duplicates, correct analysis/results/artifact signatures, no failures/timeouts, all writers returned and workers shut down. Failed jobs never enter successful throughput; any failure disqualifies that comparison block. No throughput from submitted-job count.

## Required negative tests before benchmarking
- flow request invokes flow, accessibility request invokes accessibility, ODM invokes ODM; invalid analysis fails.
- one worker returns wrong engine signature, zero-size placeholder output, duplicate job ID, missing required companion file or altered bytes: run is INVALID.
- one worker raises, exits before ready, dies while coordinator puts, hangs after output or never sends sentinel: bounded cancellation and no surviving owned processes.
- full input and full result queues: no deadlock, bounded memory.
- sibling 'site-packages-fake' prefix and editable .pth import: wheel guard fails.
- cached filename with changed bytes: input guard fails.
- parent+grandchild allocate simultaneously: sampler includes both; declared memory metric matches implementation.
- NUMBA_NUM_THREADS=1 with larger flow executor: admission detects mismatch.
- changed cache/module source hash: invalidates warm identity.
- nonexistent manifest/zero budget/negative worker count/output collision: fail before work.
- final run with one failed job: no qualified status, successful count reflects actual validated completions.

Keep expected-error fixtures outside production data. Review H02 code and tests independently. Running a self-test that only inspects JSON structure is not enough; use tiny subprocess workers to exercise shutdown, queues, import guards and memory accounting.
