# Bit, state, artifact and operational validation

L0: tiny exhaustive/state-machine/algebra checks, dtype/layout/size admissions, seeded bit patterns.
L1: actual compiled pinned baseline vs candidate, adversarial graphs and metric specializations.
L2: real public API chronology over small GIS, public RunBatch with failures and required exports.
L3: observed medium/large selection, peak memory and backend/crossover tests.
L4: only the final immutable candidate on the unreduced user actual workload, when provided.
Observed proxies stay L3/final-proxy, not L4. No allclose in exact gates.

## Oracle discipline
Run baseline twice first; record true variability rather than masking it. Use immutable baseline
source/wheel and isolated environments. Prototype and portable characterization code are NOT
independent correctness authorities until bridged to actual engines. Prevent cwd/PYTHONPATH or
historical cache-artifact reuse from substituting the candidate for its reference. Instrumented
traces are diagnostics; final timings run without profiling hooks. Pin actual executed modules.

## Sparse tests
Exhaustive small multigraphs; forward/reverse duplicates; identical neighbor/weight entries;
self-loops; two seeds on one node (last assignment wins); isolated/dead-end nodes; flags=False;
seeds equal/above cutoff; strict vs closed cutoff and nextafter; zero costs and signed zero;
empty O/D; unsorted/noncontiguous/read-only arrays; integer boundary counts; mismatched shapes;
NaN/Inf/negative prepared values refusal; rounded sentinel==R refusal; overflow-safe admission;
epoch near rollover; interrupted lane reused; two concurrent callers; reentrant callbacks.
Trace popped (weight,node), eligible offset/cost sequence, logical label writes and queue events
against baseline. Timestamps, private buffers and thread completion order are not numeric oracles.
Reconstruct full dense labels at each trace boundary outside timing. Never claim final-only
comparison proves chronological parity. Add a test with equal parallel improvements accepted
TWICE before either write, then both replayed. Do not add a stale-heap-entry guard as cleanup.

## Metric/compiler tests
Compare full compiled fold and any split/replay specialization for 0/1/2/3/4/7/8/9/15/16/17/
31/32/33/64/65/257 retained counts, ties with unequal weights, fractional reach, all-zero and
subnormal weights, extreme finite costs/parameters, signed zero, branch boundaries, each KNN
branch, arrays/layouts and compiler profiles. Characterize FMA/reduction code where necessary.
If separate exponential-only code differs by even one required bit, retain the combined fold
for that profile or implement the characterized ordered plan. Not evidence that tolerances are
needed. Never replace argsort with top-k/argpartition without its exact tie permutation proof.

## Trace/cache tests
Change every routing dependency singly -> miss; change only proven metric input -> routing hit;
change only artifact name -> shared numeric trace but fresh correct output. Mutate in-memory
geometry/cost/weights through every supported public route. Same pathname changed bytes -> miss;
same bytes different provenance -> numeric reuse with current metadata. Cache disabled, cold,
hit, eviction, corrupt/truncated blob, schema/code/profile change, single-flight race, owner death,
read-only directory, symlink/path traversal, timeout and resume. No pickle for durable untrusted
cache. Actual arrays after a hit must not alias another job's mutable results.

## Public batch tests
Serial reference vs parallel cohort at EACH ordered commit: effective settings, last topology/
engine state, cap writeback, flags, skipped rows, composite reductions and public files. Use a
clock fixture to test timestamp policy; do not normalize arbitrary output names away. Failure in
an early row while later numerical work completes cannot publish that later result. Kill worker
or writer at each stage, fill queues, cancel during search/tape/replay/commit, and resume. Retries
must not duplicate published files or composite updates. State transfer and hashing remain in
public service. Different metric outputs must genuinely differ in distinct-job performance cases.

## Installed/backend tests
Build wheel/sdist from exact final SHA. Install outside source and assert per-module/binary hashes,
profile/capability identity and executed optimized path; force fallback separately. Source-tree
pass is not wheel pass. CPU-only install remains functional. Native loads a real binary and runs
substantive work; GPU runs real device work and synchronizes. Validate host/device error and
partial-output handling. No fallback-only/simulated qualification. Full-repo tests must run from a
clean disposable frame without reading hardcoded old worktrees or overwriting historical files.

Retain actual decisive raw arrays, bytes and logs. Required check skipped/unavailable means not
passed. Scope a successful result to the tested profile/hardware; do not imply cross-ISA bit parity
from one host. Independent reviewer validates all proof, runtime-path and claim boundaries.
