# D04: Public RunBatch planning and semantics

## Public contract
Add `RunBatch(analysis, pairing_file=None, *, parallel=False, workers=None, execution=None)`
without breaking old calls or return-None behavior. `parallel=True` must actually overlap
independent rows. Expose a structured `self.batch_report` with requested/effective concurrency,
serial barriers, backend choices, successful/failed/cancelled rows, timings and resume identity.
Existing serial path remains callable and serves as the state/artifact reference.

## Why simply submitting current Run* is wrong
RunBatch mutates self.settings, Settings snapshots in projects, topology, final engine fields,
readiness flags, output-folder defaults, names, gravity-cap writeback and composite capture.
Rows may reuse output paths, read earlier outputs, alias Settings objects or rely on callbacks.
Parallel calls on one UNA instance race. Independent computation must be separated from ordered
observable state transitions and publication, not assumed from distinct row numbers.

## Planner algorithm
1. In caller order, snapshot input descriptors and detect object aliasing before deep copies.
   Capture the scalar settings/export naming decisions at their original logical boundaries.
2. Build dependency edges for explicit dependencies, canonical input paths read after prior
   writes, colliding output artifacts, shared mutable callbacks/subclasses, and known carried
   state. Resolve symlinks/case policy consistently; a different spelling is not independent.
3. Hold row validation errors as ordered outcomes. Do not raise an error from row 8 before
   row 2 runs merely because the planner inspected it first. Missing-field skips stay skips.
4. Admit only components proven independent for worker execution. Unsupported dynamic behavior
   executes serially with a reason in batch_report. Do not silently ignore parallel=True for
   all normal rows, or reject every case to claim correctness.
5. Each worker owns fresh mutable UNA/Settings/Topology and private staging output. It receives
   immutable descriptors or admitted read-only shared snapshots, not live parent objects.
6. Completed results enter a bounded reorder buffer. The coordinator commits row i only after
   all required predecessors and every earlier observable row have committed or failed.
7. Rehydrate final public engine/topology/settings state without rerunning the scientific job.
   Apply original composite row capture/fold order. Return None as before.

## Formal observable-prefix invariant
Let S_i be public state and published artifacts after serial rows [0,i). After the parallel
coordinator has committed i rows, require public state and artifacts equal S_i under the profile.
Speculative rows i+1... may exist only in private storage and must not affect reads/warnings
or published outputs. Check after EVERY committed row, not just at batch completion.

## Failure ordering and resource policy
If row k fails, finish/commit the valid serial prefix, expose the earliest ordered failure,
cancel descendants and speculative later work, and clean private artifacts by ownership.
Never count failed rows as throughput. Exceptions retain supported type/message/row context;
custom Python exceptions or nonserializable callback state may require serial fallback.
Timestamp values are compared under a controlled clock; preserve naming policy and original
format. Do not pretend faster elapsed wall time can reproduce historical timestamps literally.

## Tests
Two independent normal rows must overlap (barrier-based test, not noisy speed assertion).
Compare serial/parallel with W=1,2,4 and repeated Settings aliases, empty/missing rows, changed
Network_File, gravity caps, obstacles cleared between rows, composite sums, shared output name,
row consuming prior output, callback, failure before/after writer, no-output flags and restart.
Intentionally publish out of order or sum composites by completion order; both mutants must fail.
