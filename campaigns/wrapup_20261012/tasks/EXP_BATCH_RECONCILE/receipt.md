# EXP_BATCH_RECONCILE — semantic merge receipt

Branch `perf/exp-batch-reconcile` (worktree `/storage/scratch1/1/dsynn6/una-x-reconcile-20261012`).
Merge commit `208ab71851e2ed19e9a04376ff019078064e726e` (parents `be56059` + `4658344`).
Merge: `git merge --no-ff 4658344` onto `be56059`; merge base `c1f8614a2733`.
(An initial merge commit `8952fde` accidentally omitted the wedge-test harmonization
hunk — it was staged before that edit — and was rebuilt as `208ab718` with identical
parents before any report; the committed tree's five resolved-file digests were
re-verified against `resolved_files_sha256.txt` after the rebuild.)
Parents: batchdef = `perf/batch-def24-major1` @ `be56059` (re-review ACCEPT-WITH-FINDINGS 0 MAJOR/0 MINOR/3 NOTE);
exp-lazy = `perf/exp-lazy-1` @ `4658344` (M2 gate ACCEPTED 1.10654x median, LB 1.07908).
Interpreter alan, `PYTHONHASHSEED=0`. Raw evidence: `evidence/tasks/EXP_BATCH_RECONCILE/rcn-20261012T0451Z/` (untracked, main repo).

**STATUS: FULLY RESOLVED. The section-5 contradiction was decided by the
team-lead (option A, user-authorized execution 2026-10-12) and closed by the
follow-up commit: exp-lazy's one red test was retyped to the reviewed contract
(name kept, expected exception + arity changed), island now 99 passed / 3
skipped / 0 failed. See section 5 for the decision record.**

## 1. Hunk resolution decisions (all 11 hand-resolved regions)

| # | Location | batchdef content | exp-lazy content | Resolution |
|---|----------|------------------|------------------|------------|
| 1 | `checkpoint.py` `create()` error message (~L164-169) | "resume by re-invoking RunBatch with the same checkpoint directory (ExecutionOptions.checkpoint)..." | "resume loads it via load() and admits it only through verify_identity()..." | batchdef **verbatim** (evidence-bound to the re-review verdict; pinned by `test_checkpoint_create_error_points_to_the_real_resume_path`). The exp-lazy admission LAW is not lost: both branches' runtime journal-admission region (`doc = journal.load()` / `create()` only when `doc is None` / else `verify_identity(identity)`) is byte-identical, so both semantics hold in one message. |
| 2 | `runtime.py` file-fsync helper | `_fsync_published` + MINOR-4 docstring | `_fsync_file` + pre-rename docstring | exp-lazy helper NAME `_fsync_file` (its checkpoint test monkeypatches `rt._fsync_file` by name) carrying batchdef's reviewed docstring. |
| 3 | `runtime.py` `_fsync_dir` docstring | one-liner, reviewed | longer una_10x variant | batchdef (reviewed base; nothing binds the docstring). |
| 4 | `runtime.py` `_publish_staged` body | fsync **dest after** `os.replace`; `sorted(dirs)` | fsync **src before** `os.replace`; first-seen dir order | batchdef DISCIPLINE (post-rename dest fsync — pinned by batchdef's ordering test `fsynced[:2] == [public/a.feather, public/sub/b.csv]`) + exp-lazy first-seen dir ORDER (`dict.fromkeys` — pinned by exp-lazy's checkpoint test `["sub", "public"]`; batchdef's own dir assertion is set-based and order-free) + batchdef's MINOR-4 comment. Auto-merged exp-lazy docstring corrected to the resolved discipline. Both fsync tests green. |
| 5 | `runtime.py` `_verify_staged` | `(staging_dir, files, row: int)`, raises `InvariantViolation(row, detail)` | `(staging_dir, files)`, raises `BatchCheckpointError` | batchdef **verbatim** (reviewed bytes; pinned by `test_staged_digest_verification_blocks_publishing` `ei.value.row == 0` + `test_staged_vanish_is_row_attributed_integrity_error` `row=3`). **This is the contradiction site — section 5.** |
| 6 | `runtime.py` `_commit_worker_row` signature+head | plain; `_verify_staged(..., row=i)` | `rec: Optional[PhaseRecorder]`; `_ph` span decomposition | exp-lazy span structure + batchdef's `row=i` inside the `commit_staged_verify` span. |
| 7 | `runtime.py` composite finalization | MAJOR-1+MINOR-1 law: on resume-with-finalized, re-fold+re-finalize+re-record `files=_composite_digest_map(exported)` iff `folded_this_call or not journal.verify_composite()`; else-branch always finalizes+records (gen+1 on republication, clean resume idempotent) | una_10x variant: inline journal refold iff re-ran after finalized; `record_composite_finalized()` only when not already finalized (no digest map, no verify gate) | batchdef LAW (strict superset: adds the `verify_composite()` repair trigger and the digest-map republication record) + exp-lazy's `_ph("composite_finalize")` span around both `_finalize_batch_composite()` calls. exp-lazy's `test_resume_after_finalize_refolds_full_capture_set` green under it; batchdef's `test_runtime_composite_resume.py` (138 lines, clean add) green. |
| 8 | `test_runtime_checkpoint.py` appended region @382 | +1 test (create-error message) | +3 tests (full-refold resume; `_verify_staged` unit; publish-fsync) | UNION: all 4 verbatim. exp-lazy's `isfile()` filters in `test_tampered_artifact_forces_row_rerun` / `test_completed_batch_rerun_is_idempotent` (auto-merged) are REQUIRED on the merged tree (cohort/prepared store dirs appear in `os.listdir(out)`). |
| 9 | `test_runtime_ordering.py` staged-digest docstring | row-attributed InvariantViolation wording | BatchCheckpointError wording | batchdef (matches the resolved raise). |
| 10 | `test_runtime_ordering.py` staged-digest body | `pytest.raises(InvariantViolation)` + `ei.value.row == 0` | `pytest.raises(BatchCheckpointError)` + sibling-class probe | batchdef raises+row (reviewed contract) + exp-lazy's sibling-class probe RETAINED (class-level law, non-contradictory — `BatchCheckpointError`/`BatchCancelledError` siblinghood holds in both trees). The exp-lazy raises-type assertion is the same contradiction as #5; its standalone unit-level form is kept verbatim (red) per union law. |
| 11 | `test_runtime_cancel_wedge.py` `_no_staging_residue` (file auto-merged; harmonization found by island run) | predicate flags ALL `.una-batch*` | (file batchdef-only) | Excluded exactly `.una-batch-cohort`/`.una-batch-prepared` (exp-lazy's `runtime_residue` convention): on the merged tree they persist by design beside row outputs and survive aborts; staging (`.una-batch-staging-*`) residue still fails the test. Wedge law itself verified working (MemoryError surfaced, bounded cancel, trio 3 passed/3 skipped = reviewed counts). |

Plus one mechanical cleanup: `BatchCheckpointError` removed from `runtime.py`'s
report import (dead after resolution #5; batchdef's reviewed import line restored).

## 2. Auto-merged semantic groups (verified present, no hand edits)

batchdef-only: `_Pool.cancel` DEF-24 wedge law (`cancel_join_thread()` on inbox+ready before `terminate()`, runtime.py ~L446-454); `_refold_composite_from_journal` + `_composite_digest_map` helpers; `record_composite_finalized(files)` + `verify_composite()` (checkpoint.py, +39 lines vs exp-lazy); `Engines/Base.py` `collected=` kwarg (+12); `cache/identity.py` NaN canonical-tag docstring (+13); tests `test_runtime_cancel_wedge.py` + `test_runtime_composite_resume.py` (new files).
exp-lazy-only: telemetry (`PhaseRecorder`/`_null_phase`/spans), cohort route (runtime.py ~L1027), prepared route (~L1050), `worker_tel`, UNA.py prepared-session seam + `CohortGraphEngine` arming, lazy imports (Topology.py:1191 `from sklearn.cluster import KMeans` inside BuildClusters; `Engines/Flow.py:1115`, `Engines/AggregateFlow.py:920` function-level `import networkx`; zero module-level sklearn/networkx imports remain), conftest residue helpers, tests/tenx additions.
Both (disjoint regions, auto-merged): UNA.py (batchdef composite return plumbing ~L817-942 vs exp-lazy seams ~L321/~L388).

## 3. Verification battery (merged tree; alan, PYTHONHASHSEED=0)

| Suite | Result | Baseline | Verdict |
|-------|--------|----------|---------|
| tests/batch (island) run 1 (pre-wedge-harmonization) | 97 passed / 3 skipped / 2 failed, 173.89 s | — | wedge failure diagnosed (table #11), contradiction failure expected |
| tests/batch (island) run 2 (final bytes) | **98 passed / 3 skipped / 1 failed**, 178.90 s | batchdef 96/3; exp-lazy 91 | +3 exp-lazy tests = 99 collected = 98+1; arithmetic closes; 1 failure = the contradiction (section 5) |
| tests/batch (island) run 3 (final bytes, log captured) | **98 passed / 3 skipped / 1 failed**, 170.77 s — identical failure set | — | reproducible; log `tests_batch_island_run2.log` |
| tests/batch (island) run 4 (post-option-A bytes, FINAL) | **99 passed / 3 skipped / 0 failed**, 174.77 s | 99/3/0 | contradiction closed; log `tests_batch_island_run3_optionA.log` |
| tests/cache | 162 passed (46.91 s cold; 12.27 s logged re-run) | 162 | baseline-exact; log `tests_cache.log` |
| tests/execution | 107 passed (7.33 s; 5.55 s logged re-run) | 107 | baseline-exact; log `tests_execution.log` |
| tests/cpu_kernels | 97 passed / 4 skipped / 0 failed (41.42 s cold; 5.51 s warm-JIT re-run) | 97/4/0 | baseline-exact; log `tests_cpu_kernels.log` |
| Lazy-import law | `import urban_network_analysis` then sklearn*/networkx* ABSENT from sys.modules (3/3 fresh interpreters, both probe rounds) | M2 law | holds; log `lazy_import_probe.log` |
| Lazy-import wall | ~0.001 s per fresh import from the worktree `src` tree (informational: worktree `__init__` is light; the gate's 3.3→1.5 s numbers were installed-context) | ~1.5 s target | law (absence) is the binding check; holds |
| BuildClusters / RunFlow | exercised by the batch island (make_batch path) + lazy import sites present at function level | M2 probes | holds |

Bootstrap 107-file byte census intentionally NOT run (gate's job, per tasking). Paired throughput gate NOT run (heavy-owner lane).

## 4. Bidirectional diff proofs

- `git diff HEAD -- src tests` (vs batchdef `be56059`): exactly exp-lazy's non-conflicting content + the 11 resolutions — 80 files, +14664/−326 (`diff_vs_batchdef_befixed.exp_lazy_content_plus_resolutions.patch`). `checkpoint.py` EMPTY vs batchdef (kept verbatim).
- `git diff 4658344 -- src tests` (vs exp-lazy): exactly batchdef's content + resolutions — 5 src files +173/−66, 4 test files +436/−5 (`diff_vs_explazy_4658344.batchdef_content_plus_resolutions.patch`).
- Per-file resolution patches + sha256 manifest: `resolved_*.patch`, `resolved_files_sha256.txt` in the raw evidence dir. Key digests: runtime.py `a289e350...4bd536`, checkpoint.py `e7b24c1f...99abc`.

## 5. STOP-CONDITION CONTRADICTION — DECIDED: OPTION A EXECUTED

`_verify_staged` staged-digest failure must raise **`InvariantViolation`** (batchdef,
re-reviewed: row-attributed, `plan.InvariantViolation(AssertionError)`, 3-arg `row`)
or **`BatchCheckpointError`** (exp-lazy's una_10x lineage: `report.BatchCheckpointError(CapabilityError)`,
2-arg). The two classes are unrelated siblings on BOTH parents; no subclass
relation exists, so one raise cannot satisfy both assertions.

Machine demonstration on the merged tree (batchdef bytes in place, exp-lazy test
kept verbatim per union law):
`tests/batch/runtime/test_runtime_checkpoint.py::test_verify_staged_failure_is_integrity_not_cancelled`
FAILS — `TypeError: _verify_staged() missing 1 required positional argument: 'row'`
(log `tests_batch_island_run2.log`). The exp-lazy ordering-test variant of the same
assertion (raises-type around a full RunBatch) was superseded in the union by
batchdef's reviewed body; its non-contradictory sibling-class probe was retained.

Options (were the lead's call):
- **A (recommended): batchdef bytes stand.** Rewrite the one exp-lazy unit test to
  the reviewed contract (3-arg `row=`, expect `InvariantViolation`, assert `.row`);
  its intent (corruption is not cancellation; handlers must not swallow it) is fully
  preserved — only the type name and arity change. The re-reviewed ACCEPT verdict is
  the stronger evidence; the una_10x variant is superseded.
- **B: exp-lazy contract stands.** Reverts MINOR-2 to the unreviewed-for-this-surface
  variant; breaks batchdef's two reviewed tests and the row-attribution the
  re-review accepted.
- **C: diamond class `X(InvariantViolation, BatchCheckpointError)`.** Satisfies all
  tests mechanically but invents typed-contract semantics neither branch reviewed;
  would need its own review.

**DECISION (team-lead, user-authorized execution 2026-10-12): OPTION A.**
Rationale: the adversarially re-reviewed verdict (ACCEPT-WITH-FINDINGS 0 MAJOR /
0 MINOR / 3 NOTE) is the stronger evidence; the una_10x-era variant is
superseded; the test's intent law is preserved verbatim; B was rejected (breaks
two reviewed tests + the accepted row attribution), C was rejected (invents
unreviewed typed-contract semantics).

Execution: `test_verify_staged_failure_is_integrity_not_cancelled` retyped in
place — NAME KEPT (it states the preserved intent: staged failure is integrity,
not cancellation); expected exception changed to `InvariantViolation` with
3-arg `row=` calls, row attribution asserted (`ei.value.row == 4`,
`ev.value.row == 1`), docstring records the retyping provenance; the
sibling-class assertions (`BatchCheckpointError`/`BatchCancelledError`) kept
verbatim. Amended file sha256
`f1b243601876ef1717848273d18040972e0550ba95624570af6ed46e767e3df9`
(prev `24672986...19bacda`).

Post-decision verification (amended bytes, alan, PYTHONHASHSEED=0):
- tests/batch island: **99 passed / 3 skipped / 0 failed**, 174.77 s
  (log `tests_batch_island_run3_optionA.log`) — 96 batchdef + 3 exp-lazy = 99,
  zero failures, as predicted.
- tests/cache: 162 passed (`tests_cache_optionA.log`); tests/execution: 107
  passed (`tests_execution_optionA.log`) — both baseline-exact.
- Re-review notification owed (batchdef's reviewed test surface now carries the
  retyped test); recorded as an obligation in section 6.

## 6. What the gate re-run must cover (not executed here)

1. Paired AB/BA throughput windows for the merged tree vs the strongest current-main
   arm, per the EXPW preregistration law (heavy-owner lane) — runs on these final
   option-A bytes.
2. DONE (option A): island re-run on the amended bytes = 99/3/0; exp-lazy's test
   retyped (name kept). Still owed: re-review notification, since batchdef's
   reviewed test surface now carries the retyped test.
3. The M2 import-census byte law (107-file) remains the M2 owner's obligation on
   the shipped bytes.

**CLOSURE: item 1 EXECUTED by the gate owner (G4RC gate-reconf) on these bytes and
landed on this branch by ff-only handoff — `dd9c5acf91bfba818e4121cb95f7af0588036594`
(parent `9d82a0d`, evidence-only delta verified from the object before the ff: the
3-file trio under `campaigns/una_10x/evidence/tasks/G4RC/g4rc-20261012T0512Z/`,
803 insertions, zero src/tests paths). Verdict CONFIRMED: median 3.076x, one-sided
95% LB 3.03499x, 7/7 windows, tier-1 export byte law equal on all 8 counted pairs
(Amendment A1 two-tier law); guards on the reconciled bytes: G1 import law,
G2a/G2b, tests/batch island 99 passed / 3 skipped / 0 failed.**
Item 3 (M2 census) remains with the M2 owner; item 2's re-review notification
remains owed.

## 7. Scope

Only the reconcile worktree touched; no pushes, no merges to main, no releases, no
purchases, no credential/provider changes. Historical evidence untouched. The
preserved prior packet was not loaded as a reference for any resolution decision.
