# API_REVIEW — independent review report (numerical-architect stance)

- Task: API_REVIEW (campaign una-platform-2026-09)
- Run id: mda-20261001T1215Z
- Reviewer: API_REVIEW task owner (independent; not an implementation owner; read-only outside the
  API_REVIEW evidence scopes)
- Tree reviewed: `/storage/scratch1/1/dsynn6/una-x`, branch `perf/una-platform`, HEAD
  `99469d419b12fd37ec718ba8bf9296c5ad8cae8f` (MADINA_WORKFLOWS closure commit). Working tree at
  review time: only `.claude/commands/goal.md` modified (user file, untouched).
- Upstream pin: City-Form-Lab/madina @ `8b5c3bd3b1c0048ae8b04054daed92bfb201b9d6`
  (`.refs/madina_ref`, read-only).
- Date: 2026-10-01.
- Canonical location of this report and its siblings:
  `/storage/scratch1/1/dsynn6/una-x/campaigns/una_platform/evidence/reviews/api/` and
  `/storage/scratch1/1/dsynn6/una-x/campaigns/una_platform/evidence/tasks/API_REVIEW/mda-20261001T1215Z/`
  (identical copies were staged through this session's isolated worktree because the harness
  confined tool writes there; see N5).

## 1. Verdict

**ACCEPT — with three explicitly returned rows.** Per the task completion condition ("No ignored
supported parameter or superficial alias; matrix evidence approved or missing mandatory rows
explicitly return to owners"):

1. **No ignored supported parameter** — verified (Section 4.1). The una surface (tools.py,
   betweenness.py, paths.py) is byte-verbatim upstream code, so no parameter can be silently
   dropped; the zonal bridge is diff/AST-verified equal to upstream modulo exactly the four
   ledgered deltas D1–D4, and every dossier-named parameter is exercised positively in the suites.
2. **No superficial alias** — verified (Section 4.2). Live namespace enumeration shows
   `compat.madina.una` **exactly equals** upstream `madina.una` (0 missing / 0 extra) and
   `compat.madina.zonal` (package) equals upstream minus exactly `{os, pdk}` (both ledgered D3/D4
   consequences). There are no forwarding shims; `zonal/adapter.py` is additive private
   infrastructure, not star-exported.
3. **Matrix evidence is NOT approved as-is** — `evidence/api/api_matrix.json` predates every
   parity delivery (INVENTORY ran before the MADINA_* tasks): all 18 families still
   `required_unimplemented_or_unverified`, `tests: []` for all 18, and M01–M11 have zero
   `expanded_symbols`. **Rows are RETURNED TO INVENTORY** (F2/R2 below, named per family).
4. **One real parity gap found and returned** — the top-level reexport block of
   `src/urban_network_analysis/compat/madina/__init__.py` was never completed after the una
   delivery: upstream `madina/__init__.py` does `from .una import *`; the facade does not, so 30
   upstream top-level names are absent and the file's own docstring ("not present yet, not
   silently stubbed") is now stale. **RETURNED TO OWNERS as F1/R1** (highest-priority handoff).
5. **Shim parity is deferred to PACKAGE with evidence** — no shim distribution exists in this
   checkout (verified). Facade parity is verified; shim parity is not claimed. DEFERRED as F3/R3.

Consequently: "API parity complete" claims are supported **for the `zonal` and `una.*` subpath
surface only**. They are NOT supported for (a) the top-level `madina.<una-name>` reexport layer
until F1 closes, (b) drop-in `import madina` shim parity until PACKAGE delivers and verifies the
shim distribution, and (c) the matrix until INVENTORY writes the verification rows back.

## 2. What I independently re-ran (all observed this session)

| # | Command (cwd `/storage/scratch1/1/dsynn6/una-x`) | Outcome |
|---|---|---|
| 1 | `PYTHONHASHSEED=0 /storage/home/hcoda1/1/dsynn6/micromamba/envs/alan/bin/python -m pytest tests/madina_api/workflows/ -v` | **35 passed in 174.54 s**, exit 0 (log: `reviews/api/scratch/api_review_workflows_suite.log`) |
| 2 | same driver, `-q`, suites `zonal`, `paths`, `access`, `flow` sequentially | **21 passed (100.52 s)**, **19 passed (69.32 s)**, **24 passed (105.83 s)**, **27 passed (230.51 s)** — all exit 0 |
| 3 | byte-diff of facade `una/tools.py`, `una/paths.py`, `una/betweenness.py` bodies (after docstring header) vs `.refs/madina_ref/src/madina/una/*` | **byte-equal** (tools header 14 lines, betweenness header 102 lines, paths no header) |
| 4 | reverse-D4 reconstruction of `una/workflows.py` (strip 106-line header, remove lazy pydeck block in `Logger.flow_map_template_1`, restore the two module-top imports) | **byte-equal upstream**, reversed md5 `f5f02b214dd4a6e82a1b41d9b5673f0c` == upstream md5; facade md5 `684300bb9fe87cac25519f1e323816ff` matches the MADINA_WORKFLOWS receipt |
| 5 | line-diff of all five `zonal/*.py` vs upstream, raw and rstrip-normalized; AST parse; non-module docstring comparison | residual deltas = exactly headers + D1 + D3 + D4 (+ trailing-whitespace normalization; see N1); docstring constants differ only by stripped trailing spaces in `load_layer`/`describe`/`create_graph` |
| 6 | live namespace enumeration: upstream in `.refs/venv_madina_legacy`, facade in the alan env; set compare at `madina`, `madina.zonal` (package), `madina.una` levels | una: **0/0**; zonal pkg: missing `{os, pdk}` only; top: **30 upstream names missing** (F1) |
| 7 | mutant-registry and mechanism reads in all five `_scenario.py` modules + selection/integrity test protocol reads | all four dossier mutants present and selected; forms documented in N2 |
| 8 | digest artifact spot-reads (`artifacts/arm_runs/facade/wf_knn_weights.json` etc.) | full schema pinning: `columns` (11 names), `dtypes`, `shape`, per-column values, md5 per file |
| 9 | shim search: `pyproject.toml`, `packaging/`, repo grep for shim distribution | **no shim distribution exists** in this checkout |
| 10 | `grep "import madina|from madina" src/urban_network_analysis/compat/` | only a docstring migration example (`una/workflows.py:82`); **no runtime upstream dependency** |

I did not re-run the whole-repo regression (relied on MADINA_WORKFLOWS closure run 3 on this exact
tree: 368 passed / 348 skipped / 0 failed, per its receipt — **not re-verified by me**).

## 3. Family-by-family matrix verdict (18 families)

Legend: DELIVERED = symbol present in `src/urban_network_analysis/compat/madina/**` with real
verification evidence; MATRIX ROW = rows returned to INVENTORY regardless (F2), since every family
still has empty `tests` and unexpanded rows in `api_matrix.json`.

| Family | Delivered? | Verification evidence (independently checked) | Family verdict |
|---|---|---|---|
| M01 `madina.zonal.Zonal` | yes | zonal suite 21 passed (my run); constants/`__getitem__`/state verbatim modulo ledgered deltas (diff/AST compare); reflection artifacts `MADINA_ZONAL/mdz-20260930T2124Z/artifacts/reflection/` | DELIVERED + matrix row RETURNED |
| M02 `Zonal.load_layer` | yes | same suite; body verbatim modulo D3/D4; RNG-consumption-order pin in module header; positional-id reset documented | DELIVERED + matrix row RETURNED |
| M03 `Zonal.create_street_network` | yes | zonal `id_semantics` scenario (repeated call replaces network, `discard` default, weight-column behavior, `parent_street_id` pinned) | DELIVERED + matrix row RETURNED |
| M04 `Zonal.set_turn_parameters` | yes | zonal `turn_params` + paths `turn_params` scenarios | DELIVERED + matrix row RETURNED |
| M05 `Zonal.insert_node` | yes | zonal `insert_multinode_edge` (multi-node-per-edge pin, add/remove roundtrip digests); weight_attribute profile split pinned by workflows `knn_alpha_2` | DELIVERED + matrix row RETURNED |
| M06 `Zonal.create_graph` | yes | zonal + flow scenarios (light/d/od prerequisite pinned as quirk) | DELIVERED + matrix row RETURNED |
| M07 `Zonal.clear_nodes` | yes | flow `flow_repeats_mutations` (`clear_nodes()` + reinsert equals fresh build) | DELIVERED + matrix row RETURNED |
| M08 `Zonal.create_map` | yes | zonal `map_deck` scenario (seeded colors + uuid4 control, deck structure); D4 lazy pydeck with actionable ImportError | DELIVERED + matrix row RETURNED |
| M09 `Zonal.describe` | yes | zonal `describe_output` scenario (dead geo_center branch pinned) | DELIVERED + matrix row RETURNED |
| M10 Layer/Layers | yes | zonal suite + reflection (surface identical beyond ledgered annotation); `set_style` AttributeError pinned (BUG-LAYER-SET-STYLE-ATTRERROR) | DELIVERED + matrix row RETURNED |
| M11 Network | yes | zonal suite (graph mutations, roundtrips); unsupported stubs (`visualize_graph`, `_get_nodes_at_*`, `scan_for_intersections`, `fuse_degree_2_nodes`) raise NotImplementedError exactly as upstream — documented, not invented | DELIVERED + matrix row RETURNED |
| M12 `una.tools.accessibility` | yes | tools.py byte-verbatim (my diff); access suite 24 passed (my run); alpha/beta/knn_weights/knn_plateau/closest_facility/save_*/num_cores/turns all exercised; `alpha_swap` behavioral mutant + 5 comparator controls selected | DELIVERED + matrix row RETURNED |
| M13 `una.tools.service_area` | yes | access `service_areas` scenario incl. `turn_penalty=True`, silent int-miss empty result, negative-radius validation | DELIVERED + matrix row RETURNED |
| M14 `una.tools.alternative_paths` | yes | paths.py byte-verbatim; paths suite 19 passed (my run); no-hidden-K independent enumeration (6 routes, engine==independent); empty route set + tiny-radius pins | DELIVERED + matrix row RETURNED |
| M15 `una.tools.betweenness` | yes | betweenness.py byte-verbatim; flow suite 27 passed (my run); 4/4 mutants selected; pinned upstream defects carried, never hidden | DELIVERED + matrix row RETURNED |
| M16 `una.workflows.betweenness_flow_simulation` | yes | workflows.py reverse-D4 == upstream (my reconstruction); workflows suite 35 passed (my run); 11 scenarios, 5 mutants selected+integral; CSV defaults ('pairings.csv' vs 'pairing.csv') pinned per profile | DELIVERED + matrix row RETURNED |
| M17 `una.workflows.KNN_accessibility` | yes | same suite: `wf_knn_*` scenarios; per-row Network_File reload KeyError and zero-reach KeyError 'reach' pinned as upstream defects; closest-destination stats UnboundLocalError pinned with healthy Huff mirror | DELIVERED + matrix row RETURNED |
| M18 Public helpers and reexports | **PARTIAL** | 70 helper rows (module-qualified) all delivered: una namespace 0/0 vs upstream; zonal classes/methods verified. **Top-level reexport block MISSING** (F1): 30 upstream top-level names absent; stale docstring | **RETURNED — named rows F1/R1 + matrix row F2/R2** |

## 4. Dossier-01 completeness checks

### 4.1 No ignored supported parameter (sampled deeply)

- `accessibility`: `alpha` (16 scenario uses + workflows arm), `beta` (34), `knn_weights` (17),
  `knn_plateau` (27), `closest_facility` (39), the `save_reach_as`/`save_gravity_as`/`save_knn_as`/
  `save_dist_as` family (30), `num_cores` (86 incl. nc>1 structural policies), turns
  (`turn_penalty`/`turns`, 36), `search_radius` (106), `detour` (76), `weight_attribute` (15),
  `exponent` (31). tools.py/betweenness.py are byte-verbatim upstream bodies, so parameter
  plumbing is upstream's own; the suites prove engagement (alpha profile split pinned end-to-end;
  `alpha_swap` flips full reach/gravity state).
- Workflows pairing-CSV surface: 20 per-row columns consumed by the verbatim module (`Beta`,
  `Closest_destination`, `Decay`, `Decay_Mode`, `Destination_File`, `Destination_Name`,
  `Destination_Weight`, `Detour`, `Elastic_Weights`, `Exposure_Attribute`, `Flow_Name`,
  `KNN_Weight`, `Origin_File`, `Origin_Name`, `Origin_Weight`, `Plateau`, `Radius`,
  `Turn_Penalty`, `Turns`, `Turn_Threshold`); per-row changes tested (`wf_knn_changes`,
  `wf_flow_two_pairings`, `wf_flow_cost_change`); CSV default names pinned per profile by
  `test_csv_defaults_correct_by_profile`.
- Zonal mutation methods, Layer indexing, Network mutation: verified through the zonal/flow
  scenarios and the reflection artifacts (class surfaces identical beyond the ledgered annotation).
- **No ignored parameter and no superficial alias found.**

### 4.2 Superficial-alias sweep

Live enumeration (Section 2 #6): `compat.madina.una` == upstream `madina.una` exactly (including
the upstream submodule-attribute quirks `betweenness`/`paths` landing in `una`); `compat.madina.zonal`
== upstream package minus `{os, pdk}` (ledgered D3/D4; zero extra names). The only additive module,
`zonal/adapter.py` (immutable `NetworkState`/`GraphState` snapshots), is not star-exported and adds
no public alias. `compat.madina` restores the `zonal` package binding that upstream's star-import
shadows with the inner module — a ledgered, documented, strict-superset divergence (pinned by the
ZONAL reflection artifacts: "upstream attribute = inner zonal.py module, compat restores package").

### 4.3 The four dossier-required mutants — what actually exists

| Dossier mutant | Where | Form | Selected? |
|---|---|---|---|
| alpha=1 <-> 2 swap | workflows `knn_alpha_2` (source replacement in EACH arm's own workflows.py, `alpha=1,` -> `alpha=2,` exactly once, enforced by assert); also ACCESS `alpha_swap` (behavioral, in-scenario) | **SOURCE-level** | yes — selected AND integral (mutant facade != clean facade; mutant facade == mutant reference); inert-on-Count / engages-on-real-weights profile split pinned, with the alpha=999 sensitivity clone showing the inert profile bitwise-stable |
| reverse closest-facility direction | ACCESS `reverse_closest_facility` (swaps `closest_facility`/`closest_facility_distance` between the two reachable destinations in the recorded digest; distances 100 vs 200 so guaranteed selected); workflows `knn_closest_facility_true` (source replacement `closest_facility = False,` -> `True,` at the KNN call site) | **DIGEST-level** (ACCESS) + **SOURCE-level flag flip** (workflows) | yes both. There is NO engine-interior source mutant that reverses the argmin direction inside `one_access`; the ACCESS form proves the comparator catches a reversed assignment (honestly labeled digest-level by the ACCESS review), and the ACCESS dev-run failure log shows a genuine direction assertion (`test_completion_condition_closest_facility_direction`: d1 -> source 2 at exactly 100.0; tie -> first processed; unreachable NaN) once failed before being satisfied |
| drop a split edge | FLOW `split_edge_drop` (drops one half of a split street — first row with shared `parent_street_id`, guarded to really be a split edge — from the recorded edge digests, with matching betweenness rekey) | **DIGEST-level** | yes — selected |
| omit an exposure column | WORKFLOWS `omit_exposure_column` (drops exposure/`decayed_mean_hazzad` columns from the `wf_flow_exposure_row` record) + FLOW `exposure_column_drop` (same column, flow suite) | **DIGEST-level** | yes both |

Assessment: the dossier's "each mutant must fail" is satisfied — every mutant is selected by its
comparator, and two of the four exist additionally/instead as true source-level mutants run
end-to-end through both arms under the selected+integral protocol. The digest-level forms are
honestly labeled as comparator-sensitivity controls in the ACCESS evidence and review; the
workflows suite docstring states the source-mutant protocol ("mutants patched into EACH arm's OWN
workflows.py must (a) be SELECTED ... and (b) be INTEGRITY-checked"). I record as N2 that no
engine-interior direction-reversal source mutant exists; the composite evidence (direction
assertion test + digest swap + flag-flip source mutant) covers the requirement's intent.

### 4.4 Required test combinations — which suite covers which

| Required combination | Covered by | Evidence |
|---|---|---|
| repeated calls on the same Zonal | zonal `id_semantics` ("network replacement on repeated calls"); paths `edge_cases` (d) (origin weight consistent across repeated calls); workflows per-row scenarios | scenario code read by me; suites green |
| changed layer weights | zonal (two distinct `weight_col` fixtures 50+10i / 100−5i); workflows `wf_flow_cost_change`; flow weight columns | same |
| clear/reinsert | flow `flow_repeats_mutations` (`z3.clear_nodes()`; clear/reinsert round-trip equals fresh build) | same |
| mixed IDs | flow `flow_repeats_mutations` (mixed origin source layers, join receiver pinned) | same |
| no reachable destination | workflows `wf_knn_zero_reach` (KeyError 'reach' pinned); paths `edge_cases` (isolated component -> empty route set with pinned schema; tiny radius -> nothing reachable); access unreachable-destination NaN survival pin | same |
| empty output | paths empty-route-set schema pin; access silent-empty int miss + empty-Zonal validation probes; flow `retained_paths == {}` pins | same |
| all output names | structural: every scenario digest pins the FULL frame schema — `columns`, `dtypes`, `shape`, per-column values (verified in retained digests, e.g. `wf_knn_weights` origin_record: 11 columns); workflows `test_every_api_row_covered` additionally asserts the manifest against the live module surface | digest artifacts read |
| invalid arguments | access `validation` (negative radius, empty Zonal, empty layers, `validate_zonal_ready` probes), paths `validation`, flow `flow_validation`, workflows `wf_flow_errors` + `test_knn_guard_and_valueerror_matrix` (city_name-only guard, NaN knn_weight list message, EmptyDataError, FileNotFoundError) | same |

No combination gap found. (ACCESS review M4 noted `accessibility(turn_penalty=True)` positive
coverage was added post-review; the flow/access turn scenarios cover the plumbing.)

### 4.5 Unsupported upstream stubs — documented, not invented

Facade `zonal/network.py` raises NotImplementedError for `visualize_graph`,
`_get_nodes_at_geometric_distance`, `_get_nodes_at_network_distance`, `_get_nodes_at_bf_distance`,
`scan_for_intersections`, `fuse_degree_2_nodes` — exactly the upstream pinned stubs, enumerated in
the module header and in INVENTORY's receipt (`unsupported_stubs` list matches). `Layer.set_style`
AttributeError and the d_graph/light_graph prerequisite are carried as registered pinned quirks
with the failure behavior as the parity behavior. No invented features found.

### 4.6 Facade/shim contract

- **Facade**: delivered and verified (this report). CONTRACTS.md's "report facade parity and shim
  parity separately" is satisfiable today only for the facade half.
- **Shim**: dossier-01 requires "a separately installed drop-in shim distribution for unchanged
  `import madina` code; never install overlapping `madina` files into an environment containing
  upstream Madina". Verified: **no shim distribution exists in this checkout** — `pyproject.toml`
  has no shim/`madina` packaging configuration, there is no `packaging/` directory, and no
  packaging task has run (TASKS_CLAUDE.yaml PACKAGE: "Build portable/native/GPU/shim install
  artifacts"; depends_on PARITY + API_REVIEW). Disposition: **deferred PACKAGE item with
  evidence**, not a return-to-owner against API_REVIEW — the DAG places PACKAGE downstream of this
  review, so requiring shim verification now would deadlock; the requirement is recorded as a
  mandatory open cell that PACKAGE must close, and CLOSE/PARITY must not represent shim parity as
  verified before then. The workflows facade header already carries the separate-environment shim
  rule and migration examples (`import madina` -> `urban_network_analysis.compat.madina...`).
  Note: F1 below materially affects shim drop-in fidelity for top-level usage patterns.
- **No global `sys.modules` aliasing**: the facade-env probes assert `madina_leaked is False`
  (no `madina*` module ever enters `sys.modules` in the facade process) — verified in the workflows
  facade-env test source and green runs.

### 4.7 MIT attribution and runtime independence

MIT attribution (City-Form-Lab/madina @ 8b5c3bd, Copyright (c) 2023 MIT City Form Lab) is carried
in `compat/madina/__init__.py`, `zonal/__init__.py`, and `una/__init__.py`. Runtime dependency on
upstream madina: none (grep: only a docstring migration example at `una/workflows.py:82`; the
facade-env probes would fail on any `madina*` import). The reference arm runs upstream from
`.refs/madina_ref` inside `.refs/venv_madina_legacy` only, as oracle.

### 4.8 Delta ledger honesty (workflows D4 and beyond)

- Reverse-D4 identity **re-proven independently** (Section 2 #4): body modulo the single D4 delta
  byte-equals upstream; facade md5 matches the receipt. The lazy import preserves semantics with
  pydeck present (parity arms run with pydeck 0.9.3 and pass bitwise) and raises an actionable
  ImportError without it (facade-env test).
- `una/__init__.py` ledger rows verified against reality for tools/paths/betweenness/workflows
  (byte-verbatim claims true; star-import order betweenness-before-paths matches upstream;
  submodule-attr quirk reproduced).
- zonal ledger: D1/D2 (network_utils), D3 (no env write), D4 (lazy pydeck) all found exactly where
  the ledger says; nothing else semantic differs (rstrip-normalized diff + AST + docstring
  comparison). See N1 for the byte-precision caveat.
- **Stale text found**: `compat/madina/__init__.py` docstring still says the una star-import is
  "not present yet" — false since the MADINA_* deliveries; part of F1.

## 5. Findings

| ID | Severity | Finding | Disposition |
|---|---|---|---|
| F1 | **MAJOR (return-to-owner)** | Top-level reexport block of `src/urban_network_analysis/compat/madina/__init__.py` never completed after the una delivery: upstream does `from .una import *`, the facade does not. 30 upstream top-level names absent from `compat.madina`: 17 functional (`parallel_betweenness`, `one_betweenness_2`, `betweenness`, `betweenness_exposure`, `paralell_betweenness_exposure`, `one_access`, `parallel_access`, `get_origin_properties`, `clockwiseangle_and_distance`, `path_generator`, `turn_o_scope`, `turn_penalty_value`, `bfs_subgraph_generation`, `bfs_paths_many_targets_iterative`, `bfs_path_edges_many_targets_iterative`, `wandering_messenger`, `angle_deviation_between_two_lines`), 12 upstream leakage/attr names (`concurrent`, `deque`, `futures`, `getsizeof`, `heappop`, `heappush`, `math`, `mp`, `os`, `psutil`, `time`, `paths` submodule attr), plus `pdk` (whose absence is the ledgered D4 consequence and is CORRECT to omit). Also the `m.una` attribute is unavailable without an explicit submodule import, breaking the `import madina; madina.una.tools.accessibility(...)` attribute-chain pattern that upstream's init supports. Docstring "not present yet, not silently stubbed" is stale. INVENTORY's own M18 semantic note records the upstream layout ("madina/__init__: from .zonal import *; from .una import *"), so this is a gap against the campaign's recorded census, not a new interpretation. | RETURNED (R1). Fix is small and verifiable: add the star import (handling the submodule-attr quirk the way the `zonal` binding is restored), update the docstring, extend a reflection/namespace test to pin the top-level set, then refresh the M18 matrix rows. Route via the lead: no remaining task owns `compat/madina/__init__.py` (MADINA_WORKFLOWS owned `una/workflows.py` only) — an unowned integration seam. |
| F2 | **MAJOR (return-to-owner)** | `evidence/api/api_matrix.json` is stale: 18/18 families `required_unimplemented_or_unverified`, `tests: []` for all 18, M01–M11 `expanded_symbols: []`. It predates all five parity deliveries by construction of the DAG (INVENTORY depends only on BOOT; its receipt honestly flags "source observations only" and "MADINA_* tasks add scenario rows"), but no downstream task owns the write-back. | RETURNED (R2) to INVENTORY (owner of `evidence/api/**`): expand M01–M11 (per-symbol defaults/positional-kw/return/mutation/errors/schemas/ordering/geometry/optional-deps/scenario/oracle), add `tests` rows pointing at the five suites' concrete test ids, flip statuses to the verified state, and record M18's F1 split (helpers verified / top-level reexport open). |
| F3 | **BLOCKED/DEFERRED (PACKAGE)** | `installed_shim` required check (api_matrix `required_checks`) is unmet: no shim distribution exists in the checkout. | DEFERRED with evidence (R3): PACKAGE owns shim install artifacts and sits downstream of this review; shim parity must be delivered and verified there, reported separately from facade parity; until then no drop-in-`import madina` claim is supported. |
| N1 | NOTE | zonal/ modules are not byte-verbatim: trailing whitespace is stripped throughout, including inside three docstrings (`load_layer`, `describe`, `create_graph` — trailing spaces only; content otherwise identical). The una trio IS byte-verbatim after header (tools 14-line header, betweenness 102-line, paths none), and workflows is byte-exact modulo D4. "Verbatim port" language in the zonal ledger is accurate at AST/semantic level, not byte level. | Recorded; no action required (semantically inert; behavioral bitwise evidence does not rest on byte claims). |
| N2 | NOTE | Mutant forms: alpha swap = source-level; closest-facility direction = digest-level (ACCESS) + source-level flag flip (workflows); split-edge drop and exposure-column omission = digest-level. No engine-interior direction-reversal source mutant exists. All digest-level forms are honestly labeled. | Recorded; composite evidence covers the dossier intent. Optional strengthening: an engine-interior source reversal mutant in the ACCESS suite. |
| N3 | NOTE | Namespace verification summary: una 0/0 vs upstream; zonal package minus `{os, pdk}` (ledgered); `madina.zonal` attribute identity divergence (package vs inner module) is ledgered and strictly more capable; top-level gap is F1 only. | Recorded. |
| N4 | NOTE | Whole-repo regression NOT re-run by this review; relied on MADINA_WORKFLOWS closure run 3 (368 passed / 348 skipped / 0 failed) on this exact tree. | Recorded as not-verified-by-me. |
| N5 | NOTE | This review session was moved into a harness worktree (`.claude/worktrees/execution`, branch `worktree-execution`) mid-task; the worktree HEAD equals the reviewed HEAD `99469d4` (same commit, 224-commit history), so post-switch relative reads were content-identical. All review artifacts were written by absolute path into the main checkout; the five suite re-runs executed against the main checkout at HEAD `99469d4`. | Recorded for custody transparency. |

## 6. Named return-to-owner rows

- **R1 -> compat namespace owner (route via campaign lead; unowned seam)**: complete the top-level
  reexport layer in `src/urban_network_analysis/compat/madina/__init__.py` (add the una star
  import with the same submodule-attr handling discipline used for `zonal`; omit `pdk` deliberately
  as ledgered D4), fix the stale docstring, add a top-level namespace pin to a reflection test,
  re-run the affected suites, and hand the row back for matrix refresh. Gate: must close before
  CLOSE may claim a complete API matrix or before PACKAGE's shim can honestly claim drop-in
  fidelity for top-level upstream usage patterns.
- **R2 -> INVENTORY**: matrix refresh per F2 (all 18 families: expanded_symbols, tests rows,
  status flips, evidence pointers to the five MADINA_* task dirs and to this review).
- **R3 -> PACKAGE**: installed_shim cell per F3, with facade/shim parity reported separately.

## 7. Command log (actual, this session)

1. `git log` / `git status --porcelain` — HEAD `99469d4`; only `.claude/commands/goal.md` dirty.
2. `PYTHONHASHSEED=0 /storage/home/hcoda1/1/dsynn6/micromamba/envs/alan/bin/python -m pytest tests/madina_api/workflows/ -v` — 35 passed in 174.54 s, exit 0.
3. Same driver, `-q`, sequential `tests/madina_api/{zonal,paths,access,flow}/` — 21/19/24/27 passed, exits 0.
4. Python byte-diff probes: facade `una/{tools,paths,betweenness}.py` bodies vs `.refs/madina_ref` — all byte-equal; `una/workflows.py` reverse-D4 reconstruction — byte-equal (md5s above).
5. zonal/ five-module raw + rstrip diffs, AST parse, non-module docstring comparison — deltas = headers + D1/D3/D4 + whitespace only.
6. Namespace enumeration probes in `.refs/venv_madina_legacy` (upstream) and the alan env (facade); set comparisons at three levels — results in Sections 2/5.
7. Reads: five task receipts + evidence.json indices, sabotage registries and mechanisms in all five `_scenario.py` modules, mutant selection/integrity tests, facade-env tests, retained digest artifacts, INVENTORY receipt, ACCESS review report (sabotage-honesty section), `api_matrix.json` families and required_checks, `pyproject.toml`/`packaging/` shim search, compat `__init__` files and `adapter.py`.

— API_REVIEW owner, mda-20261001T1215Z
