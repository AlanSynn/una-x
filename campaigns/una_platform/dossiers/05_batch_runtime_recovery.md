# D05: Batch runtime, cancellation, and durable recovery

## Runtime schedule
Use spawn-safe module-level worker targets and a guarded entry point. Workers must be reusable
without retaining mutable state from previous jobs. Budget all pools, including Numba, BLAS,
AggregateFlow logical stripes, native threads and device queues. Logical numerical stripes
never derive from transient available cores. Physical workers schedule fixed stripes/jobs.

Parent concurrently feeds and drains queues. Do not enqueue the whole workload before reading
results when either queue is bounded. Use bounded pending+running+completed slots, per-job IDs,
ready/failure/result messages, timeouts and worker exit monitoring. Verify message schema,
unique completion IDs and that all expected outputs are present before success credit.

Workers return descriptors for immutable artifacts/state; avoid serializing entire repeated
GeoDataFrames or dense OD tensors. Share only immutable ownership-checked snapshots. Measure
serialization, startup, cache attach and writer time in the relevant throughput boundary.
A device owner accepts bounded work requests rather than creating a CUDA context per CPU worker.

## Transaction states
PLANNED -> RUNNING -> STAGED -> VALIDATED -> COMMITTED; terminal FAILED/CANCELLED are distinct.
Checkpoint records code/model/profile/input identity, row ordinal, dependencies, output digests,
logical reduction progress and transaction phase. Only COMMITTED units are reusable results.
Use temporary files in the destination filesystem, flush, atomic rename, manifest commit marker
and directory synchronization where the platform supports the claimed durability. Document
weaker remote filesystem semantics. Multi-file publication needs a manifest/generation protocol;
several os.replace calls are not one atomic multi-file transaction.

## Recovery invariants
On restart, reject changed input/profile/required-output identity; verify every committed
artifact before reuse; quarantine partial/corrupt generations; resume from exact completed
logical units without double-counting sums or replaying state mutations twice. Checkpoints of
floating reductions store exact bits and the next ordered item. A rejected cache record is
not automatically a reusable checkpoint. Never load untrusted pickle to recover a job.

## Cancellation and failure injection
Inject worker kill before readiness, during compute, between artifact renames, after manifest
commit but before acknowledgement, disk-full, invalid result message, OOM, device loss and
KeyboardInterrupt while queues are full. Parent must stop accepting work, cancel owned workers,
reap processes, release cache locks/device buffers and preserve diagnostic capsules. Existing
user files are never recursive-cleanup targets. Locks need safe stale-owner handling with PID
reuse/generation detection, not a PID-only heuristic.

## Public behavior
Default serial RunBatch remains valid. parallel=True has documented cancellation/deadline and
error semantics, deterministic ordered publication, and bounded memory. Add expert partial/
streaming APIs only with explicit status; never change normal return to a generator. Re-running
a completed resumed batch must not create duplicate composite columns or extra trip volumes.

## Acceptance
End-to-end public RunBatch test uses actual exports and post-row state. Require functional
parallel overlap, equal bits across admitted worker configurations, bounded simultaneous process
memory, clean failure cancellation and crash-recovery idempotence. A benchmark-only worker pool
does not satisfy this task.
