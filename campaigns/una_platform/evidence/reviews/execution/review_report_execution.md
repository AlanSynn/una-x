# Independent review — campaign task EXECUTION (una-platform-2026-09)

- Reviewer: independent-reviewer role (fresh context; not an implementation owner)
- run_id: `exe-review-20261001T0900Z`
- Reviewed state: worktree `/storage/scratch1/1/dsynn6/una-x/.claude/worktrees/execution`,
  branch `worktree-execution`, HEAD `99469d419b12fd37ec718ba8bf9296c5ad8cae8f`
  **plus uncommitted working-tree changes** (modified: `src/urban_network_analysis/Settings.py`,
  `src/urban_network_analysis/UNA.py`, `src/urban_network_analysis/__init__.py`; untracked:
  `src/urban_network_analysis/Execution.py`, `src/urban_network_analysis/backends/{__init__,contracts}.py`,
  `tests/execution/**`).
- Review executed 2026-10-01 08:35–09:35 EDT. **The tree mutated during review**
  (see F5); all numbers below are from the settled tree (no relevant file modified
  after 08:56:22 EDT except as explicitly timestamped).

## 1. Scope and method

Adversarial verification of the delivery against:
`campaigns/una_platform/evidence/contract/contract.md` §6 (binding, "implement exactly"),
`campaigns/una_platform/dossiers/10_backend_policy.md` (kernel ABI),
`campaigns/una_platform/TASKS_CLAUDE.yaml` EXECUTION entry (owned files, completion condition),
`campaigns/una_platform/AUTHORITY.md` (integrator authority over shared schema files).

All claims below are from commands I ran in the worktree (alan interpreter
`/storage/home/hcoda1/1/dsynn6/micromamba/envs/alan/bin/python`, `PYTHONHASHSEED=0`),
plus scratch probes in `/tmp` (repo left untouched). Raw evidence retained under
`campaigns/una_platform/evidence/tasks/EXECUTION/exe-review-20261001T0900Z/`.

## 2. Contract §6 field-by-field audit — PASS (exact)

Audited by direct read of `src/urban_network_analysis/Execution.py` against contract §6 text:

- `ExecutionOptions` (Execution.py:137–191): 13 fields, contract order, contract names,
  contract defaults — `semantic_profile="una_legacy"`, `backend="auto"`,
  `device=None`, `cpu_budget=None`, `threads_per_worker=1`,
  `logical_reduction_plan="canonical"`, `memory_limit_bytes=None`,
  `workspace_limit_bytes=None`, `queue_depth=64`, `writer_concurrency=1`,
  `timeout_s=None`, `cache=CacheOptions()`, `checkpoint=None`. No extra, missing,
  renamed, or redefaulted field. `frozen=True, kw_only=True` enforced (probe + tests).
- `CacheOptions` (Execution.py:101–134): 6 fields exact — `mode="off"`,
  `directory=".una-cache"`, `max_memory_bytes=None`, `max_disk_bytes=None`,
  `verification="on_read"`, `schema_version=1`.
- `UNA.execution` attribute default `ExecutionOptions()` (UNA.py:35, :62);
  `UNA.RunBatch(analysis, pairing_file=None, *, parallel=False, workers=None,
  execution=None)` — signature matches §6 verbatim (UNA.py:113–115); returns None.
- Errors: `BackendNotAvailableError`/`ExecutionNotAdmittedError` under
  `CapabilityError(RuntimeError)` (Execution.py:47–70); forced native/gpu raise before
  scientific execution; auto falls back with recorded reason; no silent fallback.

Validation strictness (timeout_s>0, positive-int budgets, bool rejection, whitespace-only
rejection) is additive and not contract-forbidden; gaps are recorded in F3.

## 3. Serialization audit — PASS

Live probes (my own, `/tmp/exe_rev_serial_probe.py`, all observed values recorded):

- `ToDict(compact=True)` on a default Settings emits exactly the four dotted keys with
  explicit defaults (`una_legacy`, `auto`, `off`, `.una-cache`); no `execution` or
  `execution.cache` nested key. Custom values reflected in compact mode too.
- `ExportProjectAsCSV` then `pd.read_csv(..., dtype=str)` → `ApplyRow` round-trips the
  four keys exactly (CUSTOM == CUSTOM). Same via `ExportProjectAsJSON` (raw JSON carries
  exactly the four dotted keys) and via `Settings.Save`/`Load`.
- TSV pairing row (`execution.semantic_profile`/`execution.backend` columns) applies.
- `Reset()` restores the default `ExecutionOptions` (fresh instance);
  `copy.deepcopy(Settings)` carries and keeps independence.
- Invalid payloads: per-key convention verified — an invalid `execution.backend="warp"`
  cell keeps the default backend (warning printed) while other valid keys in the same row
  still apply; invalid `cache.mode` / empty profile keep defaults entirely.
- `Settings._version` bumped `1.0` → `1.1` with rationale comment (Settings.py:31).
  `_version` is written, never read (pre-existing behavior, also true at HEAD).
- Backward/forward compatibility: a v1.0 file without the keys loads with defaults kept;
  a v1.1 file under old code yields "not a recognized Settings attribute" warnings (graceful).

## 4. Admission ordering — PASS

- Code order in `RunBatch` (UNA.py:160–188, :222–236, :246–262): analysis validation →
  call-level `admit_execution` → `parallel` typed raise → workers checks → stale-report
  clear → pairing load → row-level native/gpu admission loop → composite init → row loop.
  A rejected call (parallel/native/gpu/non-ExecutionOptions/workers<1/row-forced native)
  raises before `_init_batch_compositor` and before any row runs; output dirs stay empty
  (tests assert this; I verified the code paths directly).
- Rejected calls preserve prior state: the last ADMITTED call's `batch_report` survives an
  admission failure (deliberate, documented, tested — `test_report_is_cleared_at_admission_of_a_new_call`);
  a mid-run failure leaves `batch_report is None` (cleared at admission).
- Serial row loop vs HEAD: byte-identical apart from the `RowOutcome` appends and two
  trailing-whitespace cleanups (`engine = self.flow` / `self.accessibility`).
- Independent real-run regression check: I rebuilt the HEAD sources into `/tmp`
  (HEAD `UNA.py`/`Settings.py`/`__init__.py` over a copy of this tree's `src`) and ran the
  same one-row accessibility batch under HEAD code and current code on the committed smoke
  fixture: output bytes identical —
  `r1.feather` sha256 `0e12f16b1e5c5677aee32e585e075c49fa4dfd85fe02efcbb1918cb65d6825c8`
  and `r1.geojson` sha256 `e7553065cfb65fb5baa3b4ff7a7d691c5f364f4f1def78908960c5166b155d9b`
  in both runs. Existing positional/keyword call shapes bind unchanged (test + signature).

## 5. Kernel ABI (dossier 10) — PASS

Probes in `/tmp/exe_rev_abi_probe.py` (all observed):

- Structural requirement: omitting `stage`, `semantic_profile`, `logical_reduction_plan`
  or `owner` from `KernelInputs` raises `TypeError` ("missing 1 required keyword-only
  argument"); empty/whitespace values raise `ValueError`. Construction IS the enforcement.
- Duplicate array names rejected (inputs and outputs).
- `kernel_inputs_fingerprint` equals my independent sha256 recomputation over canonical
  JSON (stage, semantic_profile, logical_reduction_plan, per-array
  name/dtype/shape/layout/content_hash); changes with profile, plan, stage, array content,
  layout, and array ORDER; identical across owners (owner excluded, documented).
- Restart contract: `status="incomplete"` without `next_logical_state` raises
  `ValueError`; unknown status rejected; `'complete'` never requires restart state.
- `KernelCapabilities`/`CapabilityRegistryEntry`/`ExecutionContext` carry every dossier-10
  named binding (semantic+math version, binary fingerprint, ISA/device, workload features,
  memory estimate, evidence/qualification, crossover, cancellation slot, memory budgets,
  stage/plan/owner). Unqualified-by-default (`qualified=False`) keeps forced routes raising
  and `auto` on reference with the empty registry — no stub execution routes exist anywhere
  (verified: no execute/dispatch code in `backends/`).
- Import hygiene: fresh-interpreter probes — `import urban_network_analysis` and
  `import urban_network_analysis.backends` pull neither numba, sklearn, geopandas nor
  pandas; `Execution.py` imports only dataclasses/typing; `backends/contracts.py` only
  numpy. The five new export names never hit the pre-existing lazy-export shadowing quirk
  (probe: classes stay classes after submodule import); the Settings/UNA/Topology shadowing
  quirk is pre-existing, unchanged, and now documented in tests.

## 6. Fixtures — PASS

sha256 of `tests/execution/fixtures/{network,origins,destinations}.geojson` match the
HARNESS frozen manifest
(`campaigns/una_platform/evidence/tasks/HARNESS/har-20260930T153809Z/manifest.json`,
`workload.input_hashes`) exactly: `1e7f8052…`, `d988681b…`, `93b1d4bb…`. The suite's real
serial runs use this fixture (no mocks).

## 7. Suites I ran (actual numbers)

| command | result |
|---|---|
| `pytest tests/execution/ -q` (settled tree, 09:0x) | **102 passed**, 0 failed, 14.9s |
| same, mid-review tree state 08:54 (pre-settle) | 99 passed, **1 failed** (see F5) |
| `pytest tests/perf_contract/ tests/platform_geometry/ -q` | 29 passed, 44 skipped (`.refs` absent — expected) |
| `pytest tests/ -q` (whole repo incl. execution) | **6 failed**, 275 passed, 506 skipped |
| `pytest tests/ -q --ignore=tests/execution` | **1 failed**, 178 passed, 506 skipped |
| /tmp copy, execution suite renamed to sort LAST, full `tests/` | 3 failed (facade, bug_triples, +1 harness test that is a relocation artifact of the copy — passes in the worktree); the 4 cutoff failures disappear |
| single reruns | `test_facade_import_writes_no_environment_variables` PASSES in isolation; the 4 `test_cutoff_gate_decision_parity` failures reproduce deterministically only after the execution suite has run in-process |

Whole-repo failure attribution (all six classified, none is a library-behavior regression):

1. `tests/science/test_bug_triples.py::TestBtnStats::test_legacy_retains_pinned_defect_pattern`
   — `FileNotFoundError` on `.refs/madina/src/madina/una/betweenness.py`. The worktree has
   no `.refs/` (per briefing, expected). Unrelated to the diff; fails identically with the
   execution suite excluded.
2. `tests/platform_geometry/...::test_facade_import_writes_no_environment_variables` —
   subprocess inherits the pytest parent env; the execution suite's in-process
   `import urban_network_analysis` triggers the pre-existing `Topology.py:5`
   `os.environ['USE_PYGEOS'] = '0'` write, so the un-scrubbed subprocess sees
   `USE_PYGEOS`. PASSES in isolation; fails whenever the execution suite was collected
   first (position-independent — also fails with the suite sorted last). The madina_api
   sibling test scrubs the inherited env (`tests/madina_api/zonal/test_zonal_facade_env.py:25`);
   this one does not.
3.–6. `tests/science/test_jit_stub_fix.py::test_cutoff_gate_decision_parity[4 params]` —
   `RuntimeError: Cannot set NUMBA_NUM_THREADS to a different value once the threads have
   been launched`. I reproduced the exact mechanism in a single interpreter
   (`/tmp/exe_rev_numba_probe.py`): numba first imported with ambient env (config=4) →
   `tests/large_e2e/*/conftest.py` collection-time `os.environ.setdefault("NUMBA_NUM_THREADS","2")`
   → the science test's njit compile-time read of `nb.config.NUMBA_NUM_THREADS`
   (`Engines/_large_access_scratch.py:57`) re-processes the environ and raises. The
   execution suite perturbs numba import/pool-launch timing (it imports UNA→Engines at
   collection and runs real engines first); without it, the pre-existing suites are
   accidentally consistent. Runs with the execution suite sorted last do not hit it.

The delivered source diff contains **no** `os.environ` writes, no numba configuration, and
no global mutation (grepped the diff and the new modules). These five failures are a
cross-suite isolation interaction between the new suite and pre-existing fragile global
state — see F1.

## 8. Dossier-10 cross-check

Every dossier-named requirement has a corresponding field/constraint:
dtype/layout (`ArrayDescriptor`), logical reduction plan (`KernelInputs`,
`ExecutionContext`), stage identity (`KernelInputs.stage`, `KernelCapabilities.stage`),
owner (`KernelInputs`, `ArrayDescriptor`, `ExecutionContext`), cancellation
(`ExecutionContext.cancellation`, `None` until FAULTS/BATCH_EXEC), memory budget
(`ExecutionContext.memory_limit_bytes`/`workspace_limit_bytes`,
`CapabilityRegistryEntry.memory_estimate_bytes`), registry bindings (semantic+math
version, binary fingerprint, ISA/device, workload features, evidence, qualification,
crossover), honest completion (`KernelOutputs.status`/`next_logical_state`), profile
segregation of fingerprints, empty-registry fail-closed admission, no silent fallback.

Fields beyond the dossier text, each acceptable: `KernelCapabilities.known_refusals`
(uncommented but consistent with the dossier's refusal test matrix), `layout="geometry"`
literal, `ExecutionContext.provenance`/`backend` (effective route — dossier requires
recording it), `ArrayResult` (structural part of Outputs). The dossier's auto-selection
feature list (frontier estimates, turn state size, route model, requested outputs,
cache/residency, batch length) has no dedicated fields; `workload_features: Mapping`
represents them and selection calibration belongs to later DAG tasks. No dossier
requirement is unmet.

## 9. Declared deviations — adjudicated

1. **UNA.py edited despite BATCH_EXEC interest**: legitimate. EXECUTION's role is
   `integrator`; AUTHORITY.md ("The integrator alone edits shared
   API/default/schema/dispatch files") and contract §6's header (schema approved for
   EXECUTION to implement exactly) authorize it. The UNA.py diff is surgical/additive
   (imports, class attr, init default, RunBatch keyword-only params, admission block,
   RowOutcome appends, batch_report) and behavior-preserving (byte-identical outputs vs
   HEAD, §3/§4 above).
2. **tests/execution/ not in DAG owned_files**: acceptable as the task's verification
   home; no other owner writes there. Should be recorded in the receipt (it is in mine).
3. **Zonal execution control not touched**: correct — `compat/` is unmodified (verified
   via git status/diff); the facade bullet of §6 belongs to the zonal owners
   (MADINA_ZONAL/TOPOLOGY). The handoff MUST be recorded in the implementation receipt;
   at review time `evidence/tasks/EXECUTION/` contained no implementation receipt yet.
4. **contracts.py validation tightening**: real but incomplete — see F3.
5. **Pre-existing lazy-export shadowing quirk**: verified unchanged and documented
   (`tests/execution/test_lazy_imports.py`), not relied upon by the new tests.

## 10. Findings

### MINOR

- **F1 — Cross-suite isolation: whole-repo `pytest tests/` is not green with the new
  suite present** (files: `tests/execution/conftest.py` trigger; pre-existing
  `src/urban_network_analysis/Topology.py:5`, `tests/large_e2e/*/conftest.py`
  collection-time env writes, `tests/platform_geometry/test_errors_and_unsupported.py:49`
  un-scrubbed subprocess env, `src/urban_network_analysis/Engines/_large_access_scratch.py:57`
  compile-time numba config read). Concrete failure: `pytest tests/` in this worktree →
  6 failed vs 1 failed with `--ignore=tests/execution`; mechanism proven by my single-
  interpreter reproduction (exact RuntimeError string) and the sorted-last experiment.
  The delivery's source diff introduces no global state mutation, and the completion
  condition does not require whole-repo green — but every prior campaign closure used a
  green whole-repo run as evidence, so this must be dispositioned (fix belongs with the
  fragile pre-existing tests' owners — env scrubbing in the facade test as its
  madina_api sibling already does, and numba/env pinning for the science/large_e2e
  interaction), not left silent.
- **F2 — Literal `execution` key/column bypasses validation on external input**
  (`src/urban_network_analysis/Settings.py:379–412` ApplyRow, `:433–449` Load). A
  hand-written/externally-edited project file containing a top-level `execution` key
  (dict via Load, string via a CSV column) is set raw onto `Settings.execution` — both
  probes accepted it (`dict` / `str`) — because the field name is a valid dataclass field
  while the dotted-key branch only recognizes the four serialized keys. Failure scenario:
  `Load()` prints no warning (key is "recognized"), then the next `ToDict()`
  (Settings.py:660 `ex.semantic_profile`) raises `AttributeError: 'dict' object has no
  attribute 'semantic_profile'`, or RunBatch row admission (UNA.py:229 `s.execution.backend`)
  raises `AttributeError`. Files produced by this build never contain the key (ToDict
  skips the field), so this is contained to externally-edited input; recommend rejecting
  the bare key with the standard warning.
- **F3 — contracts.py validation inconsistent with the declared tightening**
  (`src/urban_network_analysis/backends/contracts.py`). Probes: `content_hash=""` accepted
  on `ArrayDescriptor` (:54–56 validated only name/dtype/owner — yet content_hash is "the
  identity a cache key or a parity check consumes" per its own docstring);
  `ExecutionContext(backend="totally-bogus")` accepted (:217, no literal check, unlike
  `KernelCapabilities.backend` :175–177); empty-string members accepted inside
  `KernelCapabilities.supported_profiles` / `isa_or_device_requirements`. Contained
  (producers are campaign-owned; parity checks downstream), but the declared
  "_require_non_empty everywhere" tightening should be completed, and content_hash should
  at minimum be checked as 64-hex at construction.
- **F4 — Scratch probe left in the tree**: untracked `_probe_export.py` at the repo root
  (implementation-session leftover, self-labeled "Scratch probe"), not covered by
  `.gitignore`. A later `git add -A` would commit it. Remove before any commit.

### NOTE

- **F5 — Tree mutated during review**: UNA.py rewritten at 08:53:44 and
  `test_admission.py`/`test_runbatch_serial.py` at 08:55:22/33 while this review was
  running; the intermediate state failed
  `test_instance_execution_is_the_default_request` deterministically (1 failed / 99 passed,
  reproduced in isolation) because rows reported `effective.options.semantic_profile`
  while the test expected the instance profile. The settled implementation switched to
  per-row profile identity (`s.execution.semantic_profile`, UNA.py:255/:275) with the test
  rewritten as `test_instance_execution_is_the_call_level_request` plus two new tests —
  settled suite 102/102. The review brief's "100 tests passing" matched the earlier tree;
  the final count is 102. Review covers the settled state.
- **F6 — RowOutcome mixed provenance**: `semantic_profile` is the row's own requested
  identity (`s.execution.semantic_profile`) while `backend` is the call-level effective
  route (`effective.backend`). Deliberate, documented in-code and in tests; today no
  engine consumes profiles, so no numerical fact is misattributed. Forward obligation:
  when engines consume profiles (SCIENCE/BATCH_EXEC), per-row `backend` must become
  per-row factual or the fields must be renamed `*_requested`.
- **F7 — `EffectiveExecution` imported but unused in `UNA.py:11`** (dead import added by
  the diff).
- **F8 — `workers="4"` (non-int) raises `TypeError` from the `<` comparison
  (UNA.py workers check) rather than the documented `ValueError` path.**
- **F9 — No implementation receipt existed at `evidence/tasks/EXECUTION/` when this
  review ran** (08:35–09:35 EDT); this review's evidence directory is the first occupant.
  The implementation receipt must land with the Zonal-execution handoff (deviation 3),
  the final test count (102), and the whole-repo disposition of F1.
- **F10 — `Settings._version` is never read** (bump is informational; pre-existing
  behavior, unchanged).
- **F11 — Documented pre-existing quirk confirmed, not regressed**: Settings/UNA/Topology
  lazy-export shadowing (probe + dedicated test); new export names avoid the collision.

## 11. Completion condition

"Existing positional calls work; new options roundtrip/reset; profile/model/partition and
ownership sent to every kernel." — **SATISFIED** for this delivery: positional calls proven
byte-identical on a real run; options roundtrip/reset proven across ToDict/CSV/TSV/JSON/
Save/Load/ApplyRow/Reset/deepcopy; KernelInputs structurally requires
profile/partition/owner (and stage) on every kernel request — engine-side consumption is
honestly staged to later DAG tasks (NATIVE_CORE/GPU_DEVICE/CACHE_GRAPH/CPU_LAYOUT/
BATCH_EXEC), and nothing in the delivery pretends otherwise (no stub execution routes,
empty registry keeps forced routes raising).

## 12. Verdict

**APPROVED** — no MAJOR findings. Four MINOR findings (F1–F4) require disposition
(F1 before the campaign's next whole-repo closure claim; F2–F4 in the implementation
commit), none of which invalidates the delivered contract surface.

Verification status: implementation behavior VERIFIED as described above; whole-repo
regression NOT green in this worktree (6 failed with the suite, 1 failed without — all
six attributed in §7; five are pre-existing-fragility interactions, one is the `.refs`
environment artifact).
