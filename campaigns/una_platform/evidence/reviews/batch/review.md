# BATCH_REVIEW — independent review of BATCH_EXEC + CACHE_GRAPH

- Task: `campaigns/una_platform` BATCH_REVIEW (independent reviewer)
- Reviewer: independent reviewer agent, fresh context, **not** an
  implementation owner of either task; adversarial review with
  self-designed probes.
- Run id: `rev-20261003T0105Z` (UTC), branch `worktree-batch-review`
  (isolated review worktree off the main checkout).
- Reviewed commits (full SHAs):
  - BATCH_EXEC: `0acbb11120eed1b77f316da82e857afb16562c54`
    (implementation), `22de4466bd9339e29812ef5c4071e864123d8a3f`
    (fix round), `3aa61ece2d4407a8c19b1379d652948024c376d9` (closure);
  - CACHE_GRAPH: `e61632632847a7fe7a4dc049ea467fb945e6a96d`
    (implementation), `a95ddfc2cdc6cee2e4d899e523ba5cb42fd1cefa`
    (fix round), `e2d790b096d088a11ce9fd8f6330d6be4ad5c446`
    (closure = reviewed HEAD).
- Environment: alan micromamba env Python 3.11.4, numpy 2.4.6,
  pandas 3.0.6, 4-core shared node, `PYTHONHASHSEED=0`. All probe
  commands run against the reviewed bytes at `e2d790b` (worktree
  clean except this review's own artifacts).

## VERDICT

**ACCEPT-WITH-FINDINGS** — 1 MAJOR, 4 MINOR, 4 NOTE.

Completion-condition mapping:

- *No live shared mutable UNA/Settings* — supported. Workers build a
  fresh UNA and restore an invariant-preserving settings delta
  (`src/urban_network_analysis/batch/runtime.py:410-436` — frozen
  dataclasses are never swapped for their dict projection); result
  transport is a synchronous channel with SIGKILL-grade truncation
  resistance (probe2); no cross-worker or coordinator/worker shared
  mutable object was found; resume replaces topology/engine as one
  consistent bundle (NOTE-1, not live sharing).
- *No double publish* — supported. Single commit cursor in caller
  order, staging dirs + `os.replace`, staged bytes digest-verified
  before commit; probe5(a): report.output_manifest == disk exactly;
  probe6: no duplicate cache-store entries. MAJOR-1 below is a
  *wrong-content single publish* of the composite artifact, not a
  double publish.
- *No numerical repartition* — supported. probe6: parallel cold-cache
  and parallel warm runs are byte-identical to the serial reference;
  probe3: stage digests stable across reuse; mode='off' fully inert.
- *Faults and invalidation evidence approved* — **recommended for
  approval** (probe2/3/4/5/6/7 + 357 suite tests), **subject to**
  MAJOR-1 fix + re-review of the composite resume path. MINOR-2/3/4
  should be fixed before the batch-recovery surface is released;
  they do not undermine the parity claims.

---

## Findings (most severe first)

### MAJOR-1 — Resume-after-finalize silently overwrites the published composite with a single-row rebuild

- Where: `src/urban_network_analysis/batch/runtime.py:1037-1047`
  (capture re-fold gated on `not composite_already_finalized`),
  `runtime.py:1113-1116` (trailing `_finalize_batch_composite()` when
  any row folded this call), `src/urban_network_analysis/UNA.py:826-843`
  (`_compute_batch_composite` rebuilds from `self._composite_captured`
  only), `UNA.py:822-824` + `_export_batch_composite` (overwrite).
- Mechanism: a checkpointed composite batch that has completed has
  `composite_already_finalized=True`. On resume, committed rows'
  captures are NOT re-folded (the gate at runtime.py:1039). If any row
  re-runs (e.g. its artifact was damaged), that row folds its capture
  and trips the runtime.py:1113 gate, so `_finalize_batch_composite()`
  runs with `_composite_captured` holding ONLY the re-run row — and
  `_export_batch_composite` overwrites the composite files with a
  one-column composite. No warning; report shows success.
- Demonstrated (probe1, `probes/probe1_composite_resume_refold.py`,
  re-run 2026-10-03, log `probes/log/probe1.log`, exit 1, 22 s):
  3-row composite batch; tamper `row0.feather`; resume completes with
  rows `[(0,'ran'),(1,'ran'),(2,'ran')]` and note "2 committed row(s)
  reused"; composite columns shrink
  `['Count','OID','geometry','reach_row0','reach_row1','reach_row2','composite_sum']`
  → `['Count','OID','geometry','reach_row0','composite_sum']`.
- Concrete failure: silent loss of published data on the documented
  crash-recovery path, with a success report — the worst kind of
  recovery failure (undetectable by the user).
- Suggested disposition: when `composite_already_finalized` and any
  row re-folds, re-fold ALL committed captures from the journal
  (`journal.read_capture` per committed row) before re-finalizing —
  or refuse loudly (drop the composite phase from the journal so the
  full composite re-runs). Add a regression test: tampered artifact +
  composite output + resume ⇒ composite columns unchanged.

### MINOR-1 — Composite artifacts have no integrity verification on resume

- Where: `src/urban_network_analysis/batch/checkpoint.py`
  (`verify_row_artifacts`/`verify_row_capture` cover row artifacts and
  captures only); `runtime.py:1113-1116` (no re-export when nothing
  re-folds).
- Mechanism: if a resumed batch re-runs no row, the composite files on
  disk are neither digest-checked nor rebuilt; a corrupted composite
  persists with a success report. Distinct facet of the composite
  verification gap from MAJOR-1 (that one is wrong-content overwrite
  when a row DOES re-run).
- Concrete failure: post-run corruption of a composite file survives a
  resume cycle silently; downstream consumers read corrupt data under
  a fresh success report.
- Suggested disposition: journal the composite artifacts like row
  artifacts and digest-verify at resume; or always re-export the
  composite from journal captures on resume.

### MINOR-2 — Staged-artifact integrity failures masquerade as cancellations

- Where: `src/urban_network_analysis/batch/runtime.py:255-266` — a
  staged artifact that vanished or fails digest verification raises
  `BatchCancelledError`.
- Mechanism: cancellation and silent corruption are different failure
  classes with different replay semantics; the public error contract
  conflates them. A caller implementing "on cancel, retry" loops
  forever on permanent corruption; the error message does not name the
  row.
- Concrete failure: disk-level bitrot on one staged file converts a
  diagnosable integrity error into "a row worker process died …"
  semantics.
- Suggested disposition: raise a dedicated integrity error (an
  `InvariantViolation` with `row=i` already exists and carries row
  attribution) instead of `BatchCancelledError`.

### MINOR-3 — Checkpoint error message references a non-existent `load_resume()`

- Where: `src/urban_network_analysis/batch/checkpoint.py:166-169` —
  the `BatchCheckpointError` literal tells the user that "resume must
  go through `load_resume()`". Repo-wide grep: this is the ONLY
  occurrence of the name; no such function exists.
- Concrete failure: a user hitting the "journal already holds a
  checkpoint" error follows the message and dead-ends.
- Suggested disposition: point to the real entry (re-invoke RunBatch
  with the same checkpoint directory).

### MINOR-4 — Publication is neither fsynced nor generation-atomic

- Where: `runtime.py` `_publish_staged` (per-file `os.replace`, no
  fsync of published files or the output directory) — in contrast to
  `CheckpointJournal._write`, which fsyncs file and dir.
- Mechanism: a crash/power-loss window can leave torn or
  mixed-generation published outputs. Bounded: resume digest-
  verification forces re-run of damaged rows (reviewed test
  `test_tampered_artifact_forces_row_rerun`; probe1's tamper was also
  detected), so recovery exists — but the report==published invariant
  holds only at commit time, not against later torn state.
- Suggested disposition: fsync published files + parent dir at commit
  (cheap, mirrors the journal), and/or document the window and the
  recovery semantics next to dossier 05.

### NOTE-1 — Resume replaces `una.topology`/`una.accessibility`/`una.flow` object identities

`runtime.py:768-777` rehydrates the final row's topology/engine as one
consistent bundle by assignment; the serial loop mutates in place.
Internally consistent; only callers holding cross-call object
references could notice. No action required; worth a line in dossier 05.

### NOTE-2 — `_ResultChannel.get` depends on private `SimpleQueue._poll`

`runtime.py:293` uses `self._q._poll(timeout)` (CPython 3.11.4
verified). Probe2 pins current behavior under SIGKILL on this
interpreter. Suggested: a compat shim or a documented poll mechanism.

### NOTE-3 — Worker death between `put()` and `get()` can surface as bare `EOFError`

Residual from probe2(a): the synchronous channel delivered every put
that completed before SIGKILL, and probe2(b) shows worker death
resolving to a loud `BatchCancelledError` via the `any_dead()` cancel
path. The narrow EOFError window never escaped in any probe; recorded
as a residual risk only.

### NOTE-4 — `cache/identity.py` NaN-scalar docstring overclaims

The implementation collapses NaN scalars to one canonical tag and a
dedicated reviewed test pins that intent; the docstring wording
suggests payload preservation. Doc-only fix.

---

## Probe log (all re-run 2026-10-03 UTC against `e2d790b`)

Interpreter: alan env Python 3.11.4, `PYTHONHASHSEED=0`; probe
workdirs under `/tmp/batch_review_probe*` (probe7:
`/tmp/bpr7_instr`). Verbatim stdout per probe:
`probes/log/probe<N>.log` — kept UNTRACKED as raw evidence (campaign
convention; the logs embed the worktree's own absolute paths), while
this file and `probes/*.py` are the tracked record.

| probe | command | exit | elapsed | result |
|---|---|---|---|---|
| 1 | `python probes/probe1_composite_resume_refold.py` | 1 | 22 s | **FAIL → MAJOR-1** (composite shrinks to re-run row's column; report success) |
| 2 | `python probes/probe2_transport_death.py` | 0 | 12 s | PASS — (a) all 5×48 MiB results intact incl. both puts returned before SIGKILL; (b) row4 worker death → `BatchCancelledError` in 5.8 s, committed prefix rows 0-3 in caller order, report None, staging clean, no stray workers |
| 3 | `python probes/probe3_identity_invalidation.py` | 0 | 6 s | PASS — A1 producers cold/warm/again (1,0,0) digest stable (independently confirms CACHE_GRAPH MAJOR-1 closure); A2 mtime-only → reuse; B/C new-bytes (mtime pinned) → miss; D float32 knn_weights → miss; E profile flip → miss; F output-only → reuse identical arrays; G mode=off inert (0 key computations, no cache dir, byte-identical) |
| 4 | `python probes/probe4_public_errors.py` | 0 | 11 s | PASS — mid-batch missing input raises the serial route's exact exception type+message after row0 committed/published; `exc.batch_rows=[0]`; report None; skipped-row status sequence and bind state serial-identical |
| 5 | `python probes/probe5_deadline_published.py` | 0 | 30 s | PASS — (a) report.output_manifest == disk exactly (6==6); (b) 20 s deadline fired at 20.5 s with committed prefix [0,1] == disk rows, no staging leftovers, no stray workers, report None; resume without deadline completes all 6 rows |
| 6 | `python probes/probe6_parallel_cache_race.py` | 0 | 11 s | PASS — parallel cold cache: 2 workers contending on shared stage keys, 6 artifacts byte-identical to serial, store 42 entries, 0 tmp/lock; parallel warm: byte-identical again, 0 store entries rewritten/added |
| 7 | `python probes/probe7_row_cache_dispatch.py` | 0 | 7 s | PASS — dispatch snapshot carries per-row `cache.mode=disk` into spawned workers; store populated (42 files); post-run row settings round-trip disk (closes the probe6-v1 harness bug: setting cache options on live Settings AFTER `SaveSettingsToProject` does not reach rows — v1's "store empty" was that, not a product bug) |

Suite evidence (this session, reviewed bytes): `python -m pytest
tests/batch tests/cache tests/execution -q` → **357 passed / 0 failed
in 272.62 s** (alan env, `PYTHONHASHSEED=0`, shared node). Whole-repo
regression 694 passed / 348 skipped / 0 failed is cited from the
`e2d790b` closure receipt (not re-run by this review).

## Coverage vs the five probe classes

1. **Concurrency / no live shared mutable state / transport
   truncation** — probe2 (unit channel + worker-death integration),
   probe5(b) stray-worker classification, probe6 (multi-worker store
   race), probe7 (dispatch snapshot). Static: single commit cursor in
   caller order; staging isolation + digest gate; nested-batch guard
   (`runtime.py:795+`); identity-preserving settings writeback
   (`runtime.py:410-436`); `_LockSafe` transport for lock-bearing
   bundles (`worker.py:202-262`).
2. **Determinism / publication** — probe5(a) manifest==disk; probe6
   serial/cold/warm byte equality; probe5(b) resume idempotence;
   probe1 (found the violation, MAJOR-1).
3. **Cache identity** — probe3 A-G invalidation matrix (content
   addressing, mtime irrelevance, dtype/profile closure,
   output-independence, mode=off inertness).
4. **Cache/checkpoint interaction** — probe1 (MAJOR-1), probe3 A1
   (their MAJOR-1 cold double-compute closure independently
   reproduced), probe6/7 (cross-process store sharing, per-row
   options), journal capture/artifact digest verification (static).
5. **Public errors** — probe4 (serial-exact type+message, row
   attribution, skipped parity incl. bind state), probe5 (deadline →
   `BatchCancelledError` with committed prefix == disk), MINOR-2
   (integrity-as-cancel conflation), MINOR-3 (bad resume hint).

## Limitations

- 4-core shared node; probe timings are single-run observations, not
  performance claims (no paired-window comparisons were needed — no
  timing-based conclusions are drawn).
- Fault injection uses the documented `inject_specs` seam (same
  mechanism as `tests/batch/runtime/conftest.py`); production rows do
  not carry it and it does not alter results.
- GPU/native backends: no qualified capability in this build
  (admission raises before any row runs — verified statically); not
  exercised at runtime, matching task scope.
- The raw evidence dirs of the reviewed tasks were not needed; every
  claim above rests on this review's own probes, suites, and static
  reads of the reviewed bytes.

## Reviewed files (repo-relative)

- `src/urban_network_analysis/batch/runtime.py` (1215 lines, full)
- `src/urban_network_analysis/batch/checkpoint.py` (338 lines, full)
- `src/urban_network_analysis/batch/worker.py` (482 lines, full)
- `src/urban_network_analysis/UNA.py` (batch/composite sections)
- `src/urban_network_analysis/Settings.py` (serialization/composite)
- `src/urban_network_analysis/Execution.py` (`CacheOptions`,
  admission)
- `src/urban_network_analysis/cache/identity.py`,
  `src/urban_network_analysis/cache/stages.py` (full)
- `src/urban_network_analysis/Topology.py`,
  `src/urban_network_analysis/Engines/Accessibility.py`,
  `src/urban_network_analysis/Engines/AccessibilityWElevation.py`,
  `src/urban_network_analysis/Engines/AccessibilityWTurns.py`
  (cache wiring + producer seams)
- `tests/batch/runtime/conftest.py`,
  `tests/batch/runtime/test_runtime_checkpoint.py` (test-scope check:
  MAJOR-1 is untested territory — the reviewed tamper test has no
  composite rows)
