# Independent review — BATCH_PLAN (bpt-20261001T1330Z)

Reviewer: independent-reviewer (fresh context, not the implementation owner).
Worktree: `/storage/scratch1/1/dsynn6/una-x/.claude/worktrees/batch-plan`
(branch `worktree-batch-plan`, base `51fdb1f`). Read-only on the repo except this
report; all probes live under `/tmp/bpt_probes/`.

## Verdict: ACCEPT-WITH-FINDINGS

1 MAJOR / 2 MINOR / 6 NOTE. The planner core (ordered decisions, ordered
validation, output-conflict/alias/read-write hazards over the four modeled input
fields, admission, executable prefix invariant) is sound and verified against a
real serial RunBatch. MAJOR-1 is a bounded model gap (one supported input field
missing from the input set) that must be returned to the owner before
BATCH_EXEC consumes `plan.worker_admissible`.

---

## Findings

### MAJOR-1 — `observer_points_file` is a per-row input but is absent from the planner's input model
- `src/urban_network_analysis/batch/plan.py:223-224` — `_INPUT_FIELDS = ("network_file", "origins_file", "destinations_file", "obstacle_points_file")`.
- `src/urban_network_analysis/UNA.py:561-562` — `RunFlow` loads observers per row when `observer_points_file` is set; `src/urban_network_analysis/Topology.py:246-249` resolves it exactly like the other four fields (`os.path.join(settings.data_folder, file)`).
- Consequence: a flow row whose observer file is an earlier row's published artifact gets **no** `read_write` hazard and both rows are admitted to workers. This is precisely the dossier-04 "row consuming prior output" scenario, for a supported Settings field.
- Probe (`/tmp/bpt_probes/probe_observer_field.py`, output below): writer stem `obs` → `shared/obs.feather`; consumer flow row with `observer_points_file="obs.feather"`, `data_folder=shared` → planner: no hazards, `worker_admissible == (0, 1)`. Control: the same path spelled as `network_file` IS caught (`read_write:0`).
- Required fix: add the field to `_INPUT_FIELDS` plus a dependency test. Cheap, but the "proven independent" admission claim is not complete without it.

### MINOR-1 — write-set model breaks for stems containing a path separator (missed read-write AND write-write hazards)
- Model assumption: `plan.py:210-216` (`_stems_overlap`) and `plan.py:366-397` (hazard derivation) treat every written basename as `<stem>`/`<stem>_<suffix>` **directly under the output folder**.
- A stem with `/` or `..` makes the written basename's location diverge from the model:
  - `"../evil"` in folder `out/` actually writes `<base>/evil*.feather` (verified path computation);
  - reader of `<base>/evil_network_nodes.feather` → **no** hazard, both rows `worker` (probe (c));
  - write-write: `../evil` from `out/` and `evil` from `<base>` both write `<base>/evil.feather`, planner admits both (probe V2);
  - subdirectory stem `sub/evil` writes `out/sub/evil.feather`; reader of that exact path is missed because `basename("sub/evil.feather")` does not start with `"sub/evil"` (probe V1).
- The serial loop accepts such stems and performs exactly these writes, so this is a real (if adversarial/unvalidated-input) hole in "proven independent". Suggested fix: any row whose `output_file_name` contains `os.sep`/`os.altsep` is unprovable → serial with a reason.

### MINOR-2 — `plan_batch` raises at plan time on truthy non-string `data_folder`, contradicting its own docstring
- `plan.py:241-244` claims "Never raises: an exotic non-string data_folder degrades to '' here" — but `(getattr(...) or '')` only catches **falsy** non-strings; `os.path.join(data_folder, ...)` at `plan.py:235-236` and `plan.py:251-252` raises `TypeError` for a truthy non-string (`int`, `object()`, a `Path` would raise `AttributeError` on `.strip()` at `plan.py:245`).
- The serial loop raises nothing at plan time (it would fail at the offending row, in order), so this violates dossier step 3 / `plan.py:305-307` ("never for a per-row problem") for exotic rows. Same class: an exotic object with a raising `output_folder` property. (A raising `execution` property is serial-faithful — the serial call also reads `s.execution.backend` before any row, `UNA.py:232-238`.)
- Probe (e) outputs below. Fix: guard the join with `str(...)`/isinstance and degrade to the serial failure path, or hold it as a row outcome.

### NOTE-1 — descriptor records the name substitution as fired for skipped rows
`plan.py:258-262` computes `substituted`/`stem` unconditionally; the serial loop substitutes only **after** the skip gate (`UNA.py:253-265`). Real-run probe: the skipped row's live `output_file_name` stayed `'Results'` while its descriptor recorded `stem='skippy', substituted=True`. Harmless today (skipped rows are not hazard sources and don't commit), but a runtime that replays recorded decisions at the named transition boundary must not substitute on a skipped row — `ROW_TRANSITIONS` ordering already implies this; a word in the descriptor docstring would pin it.

### NOTE-2 — skipped rows DO mutate serial state; the runtime must still apply bind+resolve for them
Serial defaulting (`UNA.py:249-251`) runs **before** the missing-field check, and the probe showed a skipped row's (shared) Settings object left with `output_folder` defaulted after a real serial batch. `ROW_TRANSITIONS` (bind → resolve → validate) encodes the order, but `plan.py:346-348` says skipped rows "are hazard sources for nothing" and `commit_order` excludes them — the BATCH_EXEC handoff should state explicitly that a skipped row still performs `bind_settings` + `resolve_output_folder` (observable on aliased/shared objects and on the rehydrated final state).

### NOTE-3 — admission reason mislabels every non-exact-type object as "a custom Settings subclass"
`plan.py:409-413`; probe: a duck-typed `types.SimpleNamespace` gets reason "row settings are a custom Settings subclass (types.SimpleNamespace…)". Cosmetic; the admission decision (serial) is correct.

### NOTE-4 — dossier step-2 "shared mutable callbacks … known carried state" edges are only partially realized
Subclasses are handled (exact-type check). Settings carries no callback fields (grep over `Settings.py`: none) so there is nothing to detect today. Carried UNA/Topology instance state is delegated to the dossier step-5 rule (each worker owns fresh instances), but neither `ROW_TRANSITIONS`/`BATCH_TRANSITIONS` nor the plan notes state the fresh-instance requirement — pin it in the BATCH_EXEC handoff so the runtime cannot reuse a parent instance.

### NOTE-5 — the batch-composite write-set is not in the per-row model
`_export_batch_composite` writes `composite` / `composite_<group>` under the **last row's** resolved folder (`UNA.py:868-900`). No per-row hazard involves it, and `BATCH_TRANSITIONS` orders `finalize_composite` after `run_rows`, which is what makes this safe; worth one line in `plan.notes` so the runtime does not overlap finalize with speculative work.

### NOTE-6 — `__pycache__/` directories exist inside the untracked delivered dirs
Gitignored (`.gitignore:2`), so they will not be committed. No action.

---

## Attacks that FAILED to break the planner (soundness evidence)

- **(a) stem-collision brute force** (`/tmp/bpt_probes/probe_soundness.py`): over the real basename suffix set extracted from `Engines/Base.py` (`""`, `_network_nodes`, `_observer_points`, `_obstacle_points_usage`, `_destinations_used_origins`, `_routes`) and ~4,000 candidate stems: **no** non-prefix stem pair produces colliding files. A collision `x+sx == y+sy` with `sx≠sy` requires one suffix to be a proper suffix of another; none is. The prefix rule is **complete** for this writer set.
- **(b) folder spellings**: trailing slash, `a/../out`, relative vs absolute, real dir vs symlink-to-itself — all correctly collapsed to one canonical folder (`output_conflict` raised for same stem); reader input through a symlinked output directory → `read_write` raised. All serialized.
- **(c2)** input inside the writer's folder with a non-prefix basename: no hazard, correctly — the writer can only ever write stem-prefixed basenames there.
- **(d) case tricks**: `'City'` vs reader of `'city_network_nodes.feather'` → serialized. Case-insensitive comparison is many-to-one onto lowercase, so equality detection is complete; on ext4 it can only over-serialize, never under-serialize. Confirmed sound-by-construction.
- **wStamp**: `resolve_output_folder` (`Engines/Base.py:35-40`) only adds a timestamped subdirectory; basenames remain stem-derived. Literal timestamped input paths are covered conservatively by `_under(p, folder)` + basename prefix.
- **empty stem** `""`: `_stems_overlap` returns True against everything (`sb.startswith("")`) — conservative.
- **`output_file_name=None`**: planner does not raise; stem `None` compares as `"none"`; a runtime serial execution fails exactly where serial fails.

## Serial-parity verification (real RunBatch, not a transcription)

`/tmp/bpt_probes/probe_serial_parity.py` ran the **real** serial `RunBatch("accessibility")` on the committed 3x3 smoke grid (`tests/execution/fixtures/`) over 4 rows (row folder set / missing-fields skip / same folder distinct stems / defaults-only with substitution) and compared against `plan_batch` on identical rows:

```
planner: row0 folder=<base>/shared_out defaulted=False stem=cityA subst=False ok
         row1 folder=<FIX>/Results        defaulted=True  stem=skippy subst=True SKIPPED missing=('origins_file','destinations_file')
         row2 folder=<base>/shared_out    defaulted=False stem=cityB subst=False ok
         row3 folder=<base>/defdata/Results defaulted=True stem=defrow subst=True ok
commit_order (0, 2, 3)
serial outcomes:        [(0, ran), (1, skipped 'missing required fields: [origins_file, destinations_file]'), (2, ran), (3, ran)]
serial-applied objects: stemA→folder kept/cityA ; skippy→folder=<FIX>/Results, file_name stays 'Results' ;
                        stemB→kept/cityB ; defrow→<base>/defdata/Results, file_name='defrow'
files written:          shared_out/{cityA.feather,cityA.geojson,cityB.feather,cityB.geojson}
                        defdata/Results/{defrow.feather,defrow.geojson}
shared-object skip:     shared_s.output_folder AFTER serial batch = <FIX>/Results  (skipped row DID mutate)
planner on trio before: admissions [skipped, skipped, worker]; hazards []; commit_order (2,)
```

Every planner decision equals the serial-applied value on all four boundaries (precedence row > script > `data_folder/Results`; substitution; missing list content+order; skip semantics), and the on-disk basenames confirm the `<stem>.<ext>` write-set model for accessibility rows.

## Mutant re-run (2 of the receipt's 7, independently applied)

Pristine `plan.py` sha256 `63cd6e812e1ab099cf3f74060e8b01fbc8c2bd924a9dcda4c5279aaef9ed0490` (matches receipt `content_hashes_at_commit`).

| Mutant | Change | Suite result | Selected |
|---|---|---|---|
| M4 | `_stems_overlap` → `return False` | 5 failed / 30 passed (`test_same_folder_same_stem_conflicts`, `test_prefix_stems_conflict`, `test_case_insensitive_stems_are_not_independent`, `test_independent_rows_admitted_despite_serialized_neighbors`, `test_serial_note_and_timestamp_note`) | YES |
| M1 | `commit_order` reversed (`[::-1]`) | 6 failed / 29 passed (incl. `test_invariant_holds_after_every_commit`, `test_composite_folds_in_caller_order`, `test_commit_and_fold_order_are_caller_order`) | YES |

Restored from `/tmp/bpt_probes/plan.py.pristine` after each; sha256 re-verified `63cd6e81…` after each restore; suite green again (35 passed). The other five mutants were not independently re-run (receipt lists all seven selected; the two dossier-mandated ones are covered below).

### Dossier-mandated mutants (publish-out-of-order; fold-by-completion)
- `test_mutant_publishing_out_of_order_is_selected` builds a coordinator committing `(1, 0)` and asserts `InvariantViolation` with `row == 0`; the suite is green, so the raise genuinely happened (a non-raising `pytest.raises` fails the test). Not vacuous.
- Harness non-vacuity: the simulator's `_effect` folds a dependency marker only when already committed, so reordered commits produce observably different prefixes — demonstrated by `test_mutant_independent_rows_published_out_of_order_still_violates` (independent rows, reorder still violates via the row log).
- Fold-by-completion: `test_composite_folds_in_caller_order` asserts `fold_order == (0,1,2) == commit_order` (selects the plan-level mutant), and `test_mutant_folding_by_completion_order_is_selected` proves order-sensitivity is real arithmetic (`1e16 + -1e16 + 1.0 == 1.0` vs `1.0 + 1e16 + -1e16 == 0.0`), so the discipline is load-bearing, not decorative.

## Run log (my own executions)

| Command (alan interpreter, `PYTHONHASHSEED=0`, worktree root) | Outcome |
|---|---|
| `pytest tests/batch/plan/ -q` | **35 passed** in 0.86 s |
| `pytest tests/batch/plan/ tests/execution/ tests/perf_contract/ tests/cache/store/ -q` | **268 passed, 12 skipped** in 37.34 s |
| `pytest tests/execution/ tests/perf_contract/ tests/cache/store/ -rs -q` | 233 passed, 12 skipped; skips: `SKIPPED [6] tests/perf_contract/test_l1_engine_arms.py:89` and `[6] :95` — "UNA_BASELINE_SRC not set; L1 needs the immutable baseline source tree" → confirmed **.refs-baseline-absence artifacts of worktrees**, exactly as the receipt claims |
| `pytest tests/batch/plan/ --collect-only -q` | 35 tests collected |
| `sha256sum` of all 9 delivered files | all equal to the receipt's `content_hashes_at_commit` |
| `find tests -name 'test_*.py' -printf '%f\n' \| sort \| uniq -d` | **empty** — all test module basenames unique across `tests/` |
| `git status --short` / `git diff --stat` / `git diff --cached --stat` | only `?? campaigns/una_platform/evidence/tasks/BATCH_PLAN/`, `?? src/urban_network_analysis/batch/`, `?? tests/batch/`; **no tracked file modified** |
| probes in `/tmp/bpt_probes/` (4 scripts) | outputs quoted above |
| mutants M4, M1 + restores | see table above; worktree left pristine |

Import hygiene: `test_plan_import_hygiene.py` is a **real** subprocess probe (`subprocess.run([sys.executable, "-c", PROBE])` with `PYTHONPATH=<repo>/src`; no mocks/patches in the file) asserting `loaded_una == [una, una.Execution, una.Settings, una.batch, una.batch.plan]`, `heavy == []` (pandas/geopandas/shapely/networkx/numba/pyproj absent), `numpy == ["numpy"]`. Passing. `plan.py` imports only `os`, `dataclasses`, `typing`, `..Settings` — nothing from UNA/Topology/Engines, nothing heavier than numpy. `grep -c "setattr\|\.output_folder =\|\.output_file_name =" plan.py` → 0 (no mutation of live objects). `grep RunBatch tests/batch/` → no hits (planner-only scope; no runtime wiring attempted).

## Dossier-04 clause compliance

| Clause | Status | Evidence |
|---|---|---|
| §1 snapshot descriptors + alias before copies | Met | descriptor fields record decided values; live objects untouched (test + grep 0 mutations); alias by `id()` before snapshotting |
| §2 edges: canonical input paths read-after-write | **Partially** — MAJOR-1 (`observer_points_file` missing), MINOR-1 (separator stems) | probes |
| §2 edges: colliding output artifacts | Met | prefix rule proven complete for the real suffix set (attack (a)); canonical folder equality incl. symlinks (attack (b)) |
| §2 edges: shared mutable callbacks/subclasses | Subclasses met; callbacks N/A (no such Settings fields); carried state delegated to runtime (NOTE-4) | grep + admission test |
| §3 ordered outcomes, no cross-row raise | Met for supported rows; MINOR-2 for exotic truthy non-string `data_folder` | real-run parity + probe (e) |
| §4 admit only proven-independent; never reject wholesale; serial-with-reason | Met | mixed-batch test; subclass reason; probe (b)/(d) |
| §5-7 runtime (fresh instances, reorder buffer, rehydrate) | Out of scope for this task, delivered as `ROW_TRANSITIONS`/`BATCH_TRANSITIONS`/`verify_prefix` spec — acceptable for owned files; NOTES 2/4/5 pin what BATCH_EXEC must honor | spec tests |
| Formal observable-prefix invariant, checked after every row | Met | `verify_prefix` earliest-row violation; loop-over-prefixes test; both mutants selected |
| Failure ordering (earliest ordered failure, cancel descendants) | Runtime's dossier section (receipt `limitations` states this honestly) | receipt |
| Tests: both mutants must fail | Met | suite + my M1 re-run |

## Receipt claim-by-claim audit

| Receipt claim | Audit |
|---|---|
| "35 tests, 35 passed / 0 failed (alan, PYTHONHASHSEED=0)" | Verified independently (35 collected, 35 passed). |
| "combined neighbors 268 passed / 12 skipped" | Verified (268/12, 37.34 s). |
| "skips are .refs-absence artifacts of worktrees" | Verified via `-rs`: 12× `UNA_BASELINE_SRC not set` in `tests/perf_contract/test_l1_engine_arms.py`. |
| "7/7 mutants selected, restore byte-identical (sha 63cd6e81…)" | 2/7 independently re-run (M1, M4) — both selected; restore verified by sha256 each time; other 5 not re-run (stated here as not independently verified). |
| "serial parity: precedence, substitution recorded-not-applied, missing-field computation, trio defaults non-empty" | Independently reproduced **against a real RunBatch** on the smoke grid — all boundaries match; `Settings` trio defaults confirmed non-empty (`Settings.py:39-41`). |
| "write-set model: basenames `<stem>`/`<stem>_<suffix>` per Engines/Base.py call sites" | Verified by reading every writer (`Base.py:71/77/84/181-201/261-278/357/382/404/447/514`, `UNA.py:897`) and by on-disk outputs; **complete only for separator-free stems** (MINOR-1). |
| "case-insensitive = conservative direction; cannot corrupt" | Confirmed sound (attack (d)). |
| "skipped rows … hazard sources for nothing" | True for inter-row hazards (aliasing forces equal validation), but skipped rows do mutate (NOTE-2) — wording corrected in this report. |
| "content_hashes_at_commit" | All 9 files match at review time. |
| "scope: only batch/** + tests/batch/plan/** + evidence; nothing else touched" | Verified (`git status`, empty diffs); basenames unique across `tests/`. |
| objective/completion_condition ("serial prefix invariant after every row; validation/skips/errors respect serial order; normal independent rows admitted") | Met at planner scope, with MAJOR-1 qualifying the "I-O dependency" completeness claim. |

## Items I could NOT verify
- The five un-re-run mutants (M2, M3, M5, M6, M7) — receipt's selection claims accepted on the strength of the two re-run representatives plus the suite's structure.
- Whole-repo closure regression on the merged tree — explicitly deferred by the receipt to post-merge in the main checkout; out of scope here.
- Runtime behaviors (worker overlap, cancellation, rehydration) — not part of this task.

## Worktree state at review end
`git status --short` shows exactly the three pre-existing untracked paths; no tracked file modified; mutants reverted with byte-identical sha256. Probe scripts preserved under `/tmp/bpt_probes/` (`probe_serial_parity.py`, `probe_soundness.py`, `probe_traversal_variants.py`, `probe_observer_field.py`, `plan.py.pristine`).
