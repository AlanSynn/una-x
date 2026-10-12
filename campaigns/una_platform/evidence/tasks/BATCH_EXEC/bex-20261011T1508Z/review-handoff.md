# BATCH_EXEC fix-round handoff — re-review package for BATCH_REVIEW rev-20261003T0105Z

- Run id: `bex-20261011T1508Z` (UTC)
- Branch: `perf/batch-def24-major1` (isolated worktree
  `/storage/scratch1/1/dsynn6/una-x-batchfix-20261011`, base `c1f8614`)
- Commits: `bb0684f` (W1: DEF-24 adaptive port), `06ffac0` (W2: review fixes)
- Raw evidence (untracked): `evidence/tasks/BATCH_EXEC/bex-20261011T1508Z/`
- Conditional approval addressed: MAJOR-1 fix (this round) — re-review of
  the composite resume path is the remaining condition; MINOR-2/3/4 are
  also closed here ("before the batch-recovery surface ships").

## W1 — DEF-24 wedge fix (adaptive port of perf/una-10x `ab357c6`)

`_Pool.cancel` (`src/urban_network_analysis/batch/runtime.py`) called
`terminate()` on the workers BEFORE the queue teardown's
`join_thread()`. With a backlogged inbox (queued bytes beyond the ~64 KiB
pipe buffer), the inbox feeder blocks in `send_bytes` forever (the only
remaining read end is the coordinator's own undrained inbox reader), so
`join_thread()` wedged the cancellation and the caller's failure path —
which re-raises the original worker-row exception — never ran.

Port: `cancel_join_thread()` on `inbox` and `ready` BEFORE the
termination loop. Adaptation vs upstream: una_platform's outbox is the
synchronous `_ResultChannel` (SimpleQueue; `join_thread` is a no-op), so
it takes no `cancel_join_thread` — same semantics as upstream, matched
to this file's channel classes. Teardown loop unchanged.

Regression: new `tests/batch/runtime/test_runtime_cancel_wedge.py`
(3 runner + 3 child-guarded scenarios; >64 KiB backlog pins; faulthandler
dump at 60 s, hard kill at 150 s via `start_new_session`+`killpg`).

- Pre-fix (hunk reverted, restore sha256-verified): **3 failed in
  457.42 s** — every guarded child wedged to its 150 s kill bound
  (`batch_def24_prefix_demo_console.txt`; second demonstration earlier in
  the session: 3 failed in 460.60 s).
- Post-fix (final bytes): **3 passed / 3 skipped in 17.50 s**
  (`batch_def24_postfix_console.txt`).

## W2 — review findings

### MAJOR-1 — resume-after-finalize rebuilt the composite from a subset. FIXED by full rebuild.

Semantics chosen: **rebuild the full composite from the journal's
captures** (the reviewer's first suggested disposition). When a
finalized-composite batch resumes and anything folded this call, the
runtime calls new `_refold_composite_from_journal` — every committed
row's capture payload, digest-verified on read, folded in
`plan.fold_order` (caller order; the fold is float order-sensitive) —
re-exports, and re-records the composite artifacts (generation bump: a
republication is a new finalized generation).

Rationale for rebuild-over-refuse: the journal already holds verified
capture payloads for every committed row (resume verification re-runs
rows whose payloads fail), so the correct composite is always
reconstructible; refusing would deny the user the documented
crash-recovery outcome and add a new error surface for a recoverable
state. Refuse-loudly remains the fallback for the unrecoverable case:
`read_capture` raises `BatchCheckpointError` if a payload fails its
digest at fold time.

Gate mapping:
- `runtime.py` finalize section: `if journal is not None and
  composite_already_finalized:` → rebuild when `folded_this_call` or the
  published files fail `verify_composite()`; clean path (nothing folded,
  digests verify) stays byte-idempotent — this preserves the reviewed
  idempotence contract pinned by
  `test_resume_folds_composite_once_no_duplicate_columns` (mtime-identical
  composite files on a completed-batch rerun), which still passes.
- `runtime.py` `_refold_composite_from_journal` + `_composite_digest_map`
  (new helpers).
- UNA.py: `_finalize_batch_composite`/`_export_batch_composite` return
  the published paths (serial route ignores the return);
  `Engines/Base.py write_gdf_outputs` gained an additive optional
  `collected` list (all existing engine callers unchanged).

Regression: `test_resume_after_finalize_rebuilds_full_composite`
(3-row composite batch, tampered row0 artifact, resume ⇒ composite
columns/arrays/FILES byte-identical to run 1, generation +1).
Pre-fix demonstration: **1 failed in 23.87 s** on the original gate;
post-fix passes.

### MINOR-1 — composite integrity on resume. FIXED by journal + verify + rebuild.

`record_composite_finalized(files=...)` records the published composite
digest map (absolute paths — the composite folder may be timestamped, so
it cannot be re-derived from Settings); new `CheckpointJournal.
verify_composite()` checks size+sha256 at resume; a failure triggers the
MAJOR-1 rebuild. Additive-optional: journals finalized before this field
exist keep resuming unchanged (nothing recorded, nothing to verify).

Regression: `test_resume_restores_corrupted_composite_files` (composite
file corrupted post-run, no row re-runs ⇒ resume restores the published
bytes). Pre-fix demonstration: **1 failed in 14.84 s** with
`verify_composite` disabled; post-fix passes.

### MINOR-2 — integrity-as-cancel conflation. FIXED.

`_verify_staged(..., row=i)` raises the row-attributed `InvariantViolation`
(the reviewer's named carrier of row attribution), never
`BatchCancelledError`: corruption and cancellation are different failure
classes with different replay semantics. The existing
`test_staged_digest_verification_blocks_publishing` expectation was
updated accordingly (legitimate change — the review directs the
conflation removal) and now also asserts `row == 0`; new
`test_staged_vanish_is_row_attributed_integrity_error` pins the vanished
branch with `row == 3`. Pre-fix: both selectors fail (10.09 s / 1.33 s).

### MINOR-3 — dead `load_resume()` hint. FIXED.

The `create()` error now points to the real entry: re-invoke RunBatch
with the same checkpoint directory (`ExecutionOptions.checkpoint`).
Regression: `test_checkpoint_create_error_points_to_the_real_resume_path`
(asserts "RunBatch" present, "load_resume" absent). Pre-fix: 0.13 s fail.

### MINOR-4 — publication not fsynced. FIXED.

`_publish_staged` fsyncs every published file and each touched parent
directory at commit (mirrors `CheckpointJournal._write`); dir fsync
tolerates filesystems that refuse it. Regression:
`test_publish_staged_fsyncs_files_and_directories` (unit frame on the
publish seam: per-file + per-dir fsync order and targets). Pre-fix:
0.15 s fail.

### NOTE-4 — NaN docstring. FIXED (doc-only).

`cache/identity.py`: module docstring and the `_enc` float comment now
state that NaN scalars collapse to ONE canonical tag (payloads not
distinguished), citing the pinning test
`test_nan_payloads_get_one_canonical_key`. No behavior change;
tests/cache 162 passed.

## Verification summary (final bytes, alan env py3.11.4, PYTHONHASHSEED=0)

- tests/batch: 96 passed / 3 skipped (reviewed baseline 88 + 8 new)
- tests/execution + tests/platform_geometry: 112 passed / 32 skipped
  (baseline 112/32; skips are .refs-absence artifacts)
- tests/cache: 162 passed (baseline 162)
- Whole-repo `pytest tests/ -q`: see receipt.json / evidence.json
- Mutation battery: 9/9 SELECTED (2 wedge mutants at the 150 s kill
  bound; every restore sha256-verified) — `mutants.json`
- Per-finding pre-fix failure demonstrations: see above; every temporary
  revert was restored byte-identically (sha256-checked in-session).

## Notes for the re-reviewer

- The composite rebuild uses ONLY journal payloads (`read_capture`
  digest-verifies each); the live accumulator is discarded on that path,
  so a re-run row's fresh capture is picked up from the journal, not
  mixed with reused rows' in-memory state.
- `record_composite_finalized` generation semantics: clean finalized
  resume does NOT bump (idempotent rerun); rebuild paths DO bump (one
  bump per republication). `test_resume_after_finalize_rebuilds_full_composite`
  pins gen+1; `test_completed_batch_rerun_is_idempotent` (unchanged)
  pins stability of the clean path.
- The absolute-path choice for composite artifact records is deliberate
  (timestamped composite folders are not re-derivable); row artifacts
  remain relpath-based.
