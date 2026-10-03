# 01: direct engine bridge and honest application harness

Owner: parity/harness engineers, separate scopes. Gate: no candidate selection until S02/H01 pass.

S01 audits current source and unfinished platform work. Do NOT execute historical scripts against
a moving dirty checkout. Use detached B0 plus fresh experiment checkout. Copy the original 28-file
packet to an external scratch directory when reproducing its sweeps; never rewrite prior/raw/.
The preserved verify_checkout.py pins two source blobs and tests 1/4 threads on selected fixtures.
Run it in a supervised process with explicit CPU/RAM/deadline. It hardcodes some fixture settings
and is only a first bridge. If actual source differs, retain failed receipt and reconcile, never
edit pins until the source change and its numerical closure have been independently reviewed.

Create fresh baseline-vs-baseline-vs-candidate tests. For legacy, compare actual compiled A3 and
full integrated_scope_access; for corrected use the independent corrected reference; Madina
coverage remains separate. Include count/shape/dtype/order, finite bits, zero signs and admitted
nonfinite handling. Do not treat portable oracle agreement as direct-main parity. Freeze the
compiler/profile tuple. Capture per-pop/row/write traces for tiny cases outside timing.

Audit current benchmarks before reusing them. Earlier bugs may already be repaired: inspect source
and prove the new behavior rather than mechanically apply an old fix list. Required harness tests:
- Dispatch access/flow/ODM as requested; prove engine/profile/output engagement and forbid wrong
  analysis success. Batch/public mode must truly call UNA.RunBatch, not an external substitute.
- Assert every worker imports the selected installed module/binary hashes; cwd cannot shadow.
- Full bounded queues plus a dead worker do not hang. Result collection runs concurrently with
  dispatch, byte limits cap pending descriptors, supervisors reap process groups on timeout.
- Validate source/workload/cache/profile manifests before timing, but include all input work the
  public call normally performs inside the public boundary. Use separate diagnostic timer spans.
- Zero jobs, partial completion, failed output, unpicklable errors, interrupted writer/device and
  corrupted checkpoint produce explicit invalid/failed outcomes, not an inflated successful N.
- Record simultaneous whole-process-tree memory, real thread pools, output bytes and complete
  calls. Hash results without masking filenames/schemas or storing into historical tracked paths.

Deliver reference_bridge.json, direct test logs/counterexamples, harness negative tests, full
command/env records and a public-boundary diagram. Abort optimization on a reference mismatch
until explained. No hypothetical 10x extrapolation is labeled measured by this bridge.
