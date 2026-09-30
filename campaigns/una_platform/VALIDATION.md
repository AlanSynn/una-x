# Validation ladder and negative controls

L0: tiny exhaustive state/graph cases, bit patterns, dtypes, identities, schemas and error cases.
L1: installed kernel/adapter vs immutable reference; queue/pop/OD/state traces; native sanitizers.
L2: genuine public APIs and workflows on small observed data, all requested artifacts and state.
L3: medium observed networks for profiling, schedules, cache/backend crossover and fault tests.
L4: only immutable finalists on preregistered full large workloads/held-out cities. A user
production workload remains unavailable until supplied; a large observed proxy is not L4
production qualification. No hidden reduction of inputs, alternatives or outputs.

## Required numerical and API gates
Baseline-vs-baseline determinism first. Full declared profile matrices, API signatures/defaults,
return value/container/metadata, mutable object state and ID lineage, finite bits, signed zero,
nonfinite policy, heap/tie/path order, corrected reference invariants. Compare full results
where available, streaming hashes with retained blocks for huge arrays. No output-only check
for stateful algorithms. Empty test sets, all-skipped suites and fallback-only runs cannot pass.

## Negative controls that must fail
- change one finite bit, zero sign, NaN payload on a preserved-copy field, dtype or row order;
- change alpha/decay/turn/cost/profile but reuse a cache key;
- request flow but dispatch accessibility; relabel a source import as installed;
- disable native/GPU entry and claim acceleration from a returned capability flag;
- drop one failed job from success accounting; omit one export or destination;
- merge partial sums in completion order; reuse candidate data as oracle;
- mutate a cached public GeoDataFrame and observe another job;
- run tests from another worktree while importing hardcoded primary source;
- worker dies while queues full, writer crashes between formats, device allocation fails;
- restored checkpoint has incompatible schema/profile/input hash.

## Historical-test hardening
Audit the recorded absolute-path and evidence-rewrite sites, not just one representative.
Tests take explicit reference/candidate roots and private output directories. They assert
module __file__ and complete source-tree digest, never rely on cwd/import coincidence.
Pin/rebuild reference goldens with an audited dependency bridge. Updating a stale structural
source test is not deleting a numerical regression. Fix NUMBA_DISABLE_JIT behavior explicitly.
Run the complete suite from a clean disposable tree without modifying committed evidence.

## Installed/native/GPU
Build exact wheels in clean environments, clear PYTHONPATH/source shadowing, verify import
locations, binary hashes, loader flags and backend counters on real work. Run CPU-only install,
native ARM/x86 supported matrix and actual device smoke/integration. Emulators/simulators
are diagnostics, not GPU performance or hardware parity qualification. Explicit requests must
not silently fallback. Unsupported automatic routes test the trusted fallback and reason.

## Bugs and city failures
Failing reproducer before patch, minimized causally diagnostic case, reviewed expected result,
then patched reference and every admitted backend. Separate input defects, dependency failures,
algorithmic explosion, numerical bugs, races/deadlocks, memory exhaustion and publication I/O.
A timeout is censored data, not a proved deadlock. No claimed fix without reproducing the
mechanism or a precise statement that only a class-level synthetic reproducer was available.

## Evidence reuse and review
Reuse only with unchanged source/fixture/reference/compiler/math/profile/thread policy/harness
closure. Composition needs fresh combined tests. Implementation owner and independent reviewer
are separate. If review capability is unavailable, record it. Required release/platform checks
are never inferred from successful local unit tests or untested CI YAML.
