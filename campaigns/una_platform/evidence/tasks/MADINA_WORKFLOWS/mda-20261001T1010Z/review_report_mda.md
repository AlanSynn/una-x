# Independent adversarial review — MADINA_WORKFLOWS (mda-20261001T1010Z)

Reviewer: fresh-context adversarial reviewer. All claims verified from raw
artifacts and upstream/source; nothing taken from evidence.json narratives
without re-derivation.

## VERDICT

**ACCEPT** — 0 MAJOR / 3 MINOR / 6 NOTE.

The parity claim holds and is independently re-proven: the facade is the
pinned upstream file byte-for-byte behind a 70-line docstring header plus
exactly the ledgered D4 delta (-2/+12), and my own fresh suite run
reproduced every retained digest byte-for-byte in BOTH arms. The three
MINOR findings are suite-documentation/robustness defects, none of which
invalidates the parity claim or a completion-condition clause.

## FINDINGS

**F1. MINOR — the "reference x2 determinism control" is vacuous.**
`tests/madina_api/workflows/test_workflows_surface_parity.py:37-46,109-114`:
`_digest` caches by `(scenario, arm, sabotage)` in the module-level
`_DIGESTS` dict, so the second `_digest(..., "reference")` call in
`test_scenario_bitwise_parity` returns the cached dict and `ref_a == ref_b`
cannot fail. The test docstring ("Reference determinism control
(reference x2)") and evidence.json's protocol paragraph ("Every scenario
runs the reference TWICE (determinism control) then the facade once") are
both false as implemented — exactly one reference subprocess runs per
scenario. Concrete failure scenario: a reference arm that is
nondeterministic run-to-run would not be detected in-suite; facade ==
reference would then be a single-draw comparison. Mitigation (verified, not
assumed): my fresh run in a separate process reproduced all 16 facade +
16 reference + 2 reflection retained digests byte-for-byte (md5-identical
files), which independently establishes run-to-run determinism of both
arms; `facade == reference modulo arm tags` was also re-derived for all 16
scenarios. The parity claim itself is not invalidated. Fix is trivial
(call `run_arm` twice for the reference, or drop the cache for the second
reference call) and the docstring/evidence wording should match reality.

**F2. MINOR — documented digest policy not implemented for
geometry-bearing CSVs ({bytes}-only digest, no md5, no structure).**
`tests/madina_api/workflows/_workflows_scenario.py:26-27` claims "every
other produced file is digested structurally (columns, dtypes, IEEE-754
value hex, WKB geometry hashes) AND by raw md5". In reality every produced
CSV containing a geometry column fails inside `_digest_frame`
(`_geom_wkb` on a WKT string -> `AttributeError: 'str' object has no
attribute 'wkb'`) and is caught by `_digest_file`'s broad except
(:246-247), leaving `{bytes, parse_error}` — no md5, no structure.
Verified in the digests: `betweenness_record.csv` in all six flow
scenarios and `origin_record.csv` in wf_knn_defaults/wf_knn_weights.
Concrete failure scenario: a data regression confined to a CSV whose bytes
stay the same length AND whose geoJSON twin were absent would pass; today
the twins (`betweenness_record.geoJSON`, `origin_record.geoJSON`) are
always written from the same gdf in the same run and ARE md5-digested
(e.g. knn `origin_record.geoJSON` md5 `5d83b3346b67de158d86fd2acce28b93`,
no time columns, all values hex-compared), so no parity coverage is
actually lost. The defect is that the documented policy overstates what is
enforced; a maintainer would wrongly assume the CSVs are md5-pinned.

**F3. MINOR — the nc3 scenario's "structural-only, never the edge bits"
policy is not what the digest implements.**
`_workflows_scenario.py:28-33` and evidence.json (`wf_flow_nc3_structural`,
limits[2]) state the num_cores=8 scenario digests the streets/edge record
STRUCTURALLY (exists/columns/rows/dtypes) and "never the edge bits".
Implemented reality: `_digest_tree` fully digests
`betweenness_record.geoJSON` (md5 `41cbac402fe59a5439cc8253f127b056` plus
per-value hex of id/length/weight/huff_flow/geometry); the
`record_structural` extra (:541-545) is ADDITIONAL, not a replacement.
The enforced check is stricter than documented, so no parity claim is
weakened — but the suite documents a policy it does not enforce. Concrete
failure scenario: if the exposure engine's nc>1 queue-partition
nondeterminism (pinned in MADINA_FLOW, "no bitwise-stability contract")
ever manifests on this fixture, this scenario fails intermittently even
though the facade is bit-exact — contradicting the suite's own documented
expectation and inviting a wrong-diagnosis "fix". Empirically the bits are
stable here: edge md5 identical across the retained run + 3 fresh runs I
executed (`nc3_rep{1,2,3}.json`, all `41cbac402fe59a5439cc8253f127b056`).

**F4. MINOR — the `knn_alpha_2` sabotage comment misstates the mechanism
and contradicts the evidence it cites.**
`_workflows_scenario.py:789-791`: "It selects because
destination_weight=None forwards the destination layer's 'weight' column
(NOT unit weights, despite the accessibility docstring's "equal weight of
1" default claim — pinned in wf_knn_defaults' alpha sensitivity probe)."
Both halves are wrong. (a) Through the KNN workflow, `accessibility`'s
`destination_weight` is hardcoded `None` in BOTH profiles (upstream
workflows.py:535); the differentiator is `insert_node`'s
`weight_attribute` (upstream workflows.py:528-531, "Count" -> None vs the
attribute name). (b) The cited wf_knn_defaults probe pins the OPPOSITE of
the comment's claim: `alpha_999_deterministic_outputs_equal: True`, i.e.
unit node weights and alpha-inertness on the Count profile. The mutant
does genuinely select on wf_knn_weights (re-derived from the retained
digests) — the defect is the comment, which would send a maintainer to
the wrong call site and the wrong conclusion. The facade header's version
of the same fact (workflows.py:63-69) is accurate.

**F5. NOTE — facade header line-range imprecision (substance correct).**
`src/urban_network_analysis/compat/madina/una/workflows.py:31` cites
"betweenness.py:858-859" (exact) and ":612" (exact) but ":585-588" for the
closest-destination branch, which is actually 582-586
(`destination_probabilities = [1]` at 584); :69 cites "zonal.py:88-89" for
the index reset, which is zonal/zonal.py:89-90.

**F6. NOTE — probe raw outputs not retained; one corner of a stated 2x2
grid never probed.** The header (:40-42) pins "byte-identical across
num_cores in {1, 3} and Turn_Penalty in {0, 30} ... (evidence probes 1-2)"
and "probes C1/C2" (pydeck HTML nondeterminism), but only the probe
SCRIPTS (probes/probe_wf1.py, probe_wf2.py) plus one output CSV are
retained — no probe result JSON. I re-ran probe_wf2.py myself: record_md5
identical `7726c4f3a32f31b328895fd63efaa1a3` across A(nc1,tp0)/B(nc3,tp30)
/C1/C2, and all four HTML md5s differ between the identical-config C1/C2
runs — both header claims TRUE (my run retained at
reviewer_artifacts/probe_wf2_rerun.json). The (nc=1, tp=30) cell of the
stated grid was not probed (3 of 4 cells).

**F7. NOTE — evidence.json `completion_condition.exports` wording is
looser than what is pinned.** "Logger, betweenness_flow_simulation,
KNN_accessibility importable from the compat package": they are importable
from the compat package's `workflows` SUBMODULE; `una/__init__` does not
re-export them — correctly matching upstream `madina/una/__init__.py`
(star-imports `.betweenness` + `.paths` only, verified byte-for-byte).
The pinning test (`test_una_package_lazy_surface`) asserts the correct,
parity-faithful behavior.

**F8. NOTE — scenario docstring "empty pairings: headers only, zero rows"**
(`_workflows_scenario.py:566-567`): `pd.DataFrame([]).to_csv` writes an
empty (0-byte-class) file, not headers-only; the observed
`EmptyDataError` matches the facade header's accurate description.
Cosmetic.

**F9. NOTE — digest helper edge cases (theoretical on this fixture).**
`_float_hex` collapses every NaN payload to "nan"
(:135-137); `_is_time_col` endswith matching (:161-167) would exclude any
user column whose name merely ends with a time-column name; `_norm_err`
replaces only the arm scratch root (correct scope — verified both arms
normalize to `<scratch>` and remain comparable). `run_arm` raises
AssertionError on any `scenario_error` (conftest.py:80-83), so two-arm
identical crashes cannot pass silently — the masking path I probed for is
closed.

**F10. NOTE — dead probe field.** `una_exports_tools`
(test_workflows_facade_env.py:33-37) is computed but never asserted by any
test.

## CHECK LOG

1. **Independent delta verification: PASS.** Own script
   `reviewer_artifacts/verify_delta.py` (difflib opcode ground truth +
   byte-level reconstruction, implementer scripts not reused): facade md5
   `f4c2f593da563f1e968b1df6ba231477` (30714 B, 639 lines), upstream md5
   `f5f02b214dd4a6e82a1b41d9b5673f0c` (26053 B, 559 lines); header is
   exactly lines 1-70 and `ast.parse` confirms a single bare string
   expression; body vs upstream has exactly 3 non-equal opcodes = delete
   upstream line 13, delete upstream line 19, insert 12 lines (11 D4 lines
   + 1 blank separator) at facade lines 273-284 — diff shape **-2/+12
   exact**; reconstruction (strip header, remove the 12 lines, restore
   `import pydeck as pdk` / `from pydeck.types import String` at their
   original positions) reproduces upstream byte-for-byte
   (`f5f02b214dd4a6e82a1b41d9b5673f0c`). pdk/String have no other
   consumer (grep), so "flow_map_template_1 is the only consumer" holds;
   the D4 ImportError message and `create_deckGL_map` convention
   (zonal/utils.py:118-126) verified.
2. **Suite run: PASS (35 passed in 98.96s, 0 failed)** —
   `UNA_LARGE_E2E_ARTIFACTS=/tmp/una_wf_review_artifacts .../alan/bin/python
   -m pytest tests/madina_api/workflows/ -v`; log at
   `reviewer_artifacts/review_suite_run.log`. Digest comparison (own
   `reviewer_artifacts/compare_digests.py`): all 16 facade + 16 reference
   scenario digests and both reflection digests are byte-identical (md5)
   between my run and the retained set (sample md5s: facade
   wf_flow_defaults `aa56a24898ccf94e03a54949112c4a8a`, reference
   wf_flow_defaults `d7681cafd88a157c7c2b1967bab111ed`, facade wf_knn_changes
   `bd5a65f36f04c67b9c88fd7dcde0896d`, facade
   wf_knn_defaults__knn_closest_facility_true
   `d41d590738bb740e07ece4a11ee2b5c0`), and facade == reference content
   modulo the arm/scenario/sabotage tags for all 16. Arms confirmed to load
   the intended files (venv probe: facade ->
   `.../compat/madina/una/workflows.py`, reference ->
   `.refs/madina_ref/src/madina/una/workflows.py`).
3. **Digest helper audit: done** — findings F2, F3, F9. No float tolerance
   or rounding (IEEE-754 `struct.pack('>d')` only); no value sorting or
   row reordering anywhere; time-column exclusion limited to the documented
   columns (values of all other columns still hex-compared); `_norm_err`
   scoped to the scratch root; no error-string over-matching found beyond
   F9; the parse_error swallow (F2) is the one real gap.
4. **Mutant audit: PASS.** All 4 replacement literals counted == 1 in BOTH
   files by my own count; the patched clone replaces
   `_ARM_MARKER["workflows"]` before the scenario runs and `_WF()` returns
   it (a no-op patch would leave mutant == clean, and all four select);
   selection (facade-mutant != clean) and integrity (reference-mutant ==
   facade-mutant) re-derived by me from the retained digests for all 4
   source mutants + the digest mutant — all True. `knn_alpha_2` inertness
   on wf_knn_defaults is pinned, not hidden: `alpha_sensitivity_error:
   None`, `alpha_999_deterministic_outputs_equal: True` (a stronger
   perturbation than alpha=2), and the engaging profile wf_knn_weights
   selects the mutant. The registry comment is defective (F4).
5. **Completion-test honesty: PASS.** `test_csv_defaults_correct_by_profile`
   / `test_per_row_changes_correct_by_profile` bind to real recorded digest
   fields (verified present and True in the digests, e.g.
   `record_csv_absent_by_default`, `both_flow_columns`,
   time_log events 1 load / 2 builds / 1 origin load matching the upstream
   branch conditions at workflows.py:331-350);
   `test_every_api_row_covered` computes the real module surface
   (`public == [KNN_accessibility, Logger, betweenness_flow_simulation]`,
   Logger methods == 4) and would fail on surface growth, though the
   manifest rows themselves are prose (NOTE, folded into F7/F8
   context). Pinned-crash strings verified verbatim against upstream:
   KNN Network_File crash = zonal/layer.py:134 exact message, KeyError
   'reach' = tools.py:197 `node_gdf.loc[..., 'reach']` before any mkdir/
   to_csv, noargs ValueError = workflows.py:312 exact, KNN guard =
   workflows.py:443 exact, knn_weight list message = tools.py:134 exact,
   EmptyDataError/FileNotFoundError confirmed in digests.
6. **Header/ledger audit: PASS with 2 line-range nits (F5).** Verified
   against upstream source: keep_diagnostics=True and
   `num_cores=min(origin rows, num_cores)` (workflows.py:376-378),
   KNN city_name-only guard (:442-443), singular "pairing.csv" default
   (:421) vs flow "pairings.csv" (:293), explicit
   `redundant_edge_treatment='discard'` (KNN only, :514), raw KNN_Weight
   forwarding + NaN verbatim error, `decay=False if Elastic_Weights`
   suppression (:372), `save_path_exposure_as="exposure_"+Flow_Name`
   (:395-396), betweenness.py:858-859 read / :612 only assignment /
   bare `except:` swallow at :880-884 (mechanism confirmed live: my nc3
   probe re-run printed the UnboundLocalError traceback at :858),
   `decayed_mean_hazzad` typo (betweenness.py:854), zonal id reset
   (zonal/zonal.py:89-90). Ledger row (__init__.py:52-76, md5
   `ac7707f3fde72ffd49412baf3485fe13` — matches evidence.json) carries the
   same facts; "upstream imports pydeck at module top (lines 13/19)" exact.
7. **Scope audit: PASS.** `git status --short` shows exactly the 5 expected
   entries (M .claude/commands/goal.md, M
   src/.../compat/madina/una/__init__.py, untracked evidence dir,
   untracked workflows.py, untracked tests/madina_api/workflows/);
   `git diff --cached` empty (nothing staged); HEAD == f547856 before and
   after my review; no commits, no pushes, no stash/clean performed;
   goal.md left modified/unstaged.
8. **Evidence spot-check: PASS (all 8 claims verified).** (a) -2/+12 D4
   diff and `body_reverse_d4_equals_upstream: true` — re-derived
   independently (check 1), artifacts/facade_delta consistent with my md5s
   (facade `f4c2f5...`, upstream `f5f02b...`, pre-extension md5 superseded
   transparently in summary.json). (b) sabotage matrix selections —
   re-derived from retained digests, all 5 selected+integral. (c)
   `stats_columns_present == 4` on wf_flow_huff_elastic — verified in
   digest (4 exact names). (d) `alpha_999_deterministic_outputs_equal` —
   True in digest. (e) wall-clock/pydeck suppression policies — verified
   in digests (`presence_only_pydeck_nondeterministic`,
   `structured_only_wall_clock_columns`,
   `md5_excluded_wall_clock_columns_in_frame`) and justified by my own
   probe re-run (HTML md5s differ between identical-config runs). (f)
   retained file count 298, suite log "35 passed in 129.32s", test-file
   line counts 117/116/36/383/96/913, __init__ md5 — all match. (g) venv
   versions 3.11.4 / gpd 1.2.0 / shapely 2.1.2 / pandas 3.0.6 / numpy
   2.4.6 / nx 3.6.1 / pydeck 0.9.3 — all match. (h) upstream line cites
   :311-312, :442, :463-467 — exact.

## Commands run and outcomes

1. `git status --short`, `git log --oneline -3`, `git diff --cached --stat`
   — scope as claimed; nothing staged; HEAD f547856.
2. `/storage/home/hcoda1/1/dsynn6/micromamba/envs/alan/bin/python
   reviewer_artifacts/verify_delta.py` — byte equality PASS (md5s above).
3. `UNA_LARGE_E2E_ARTIFACTS=/tmp/una_wf_review_artifacts
   /storage/home/hcoda1/1/dsynn6/micromamba/envs/alan/bin/python -m pytest
   tests/madina_api/workflows/ -v` — **35 passed in 98.96s**.
4. `.../alan/bin/python reviewer_artifacts/compare_digests.py` — 16/16
   digests: my run == retained (byte), facade == reference modulo tags;
   reflection identical.
5. `.../alan/bin/python` inline digests for the mutant matrix — 5/5
   selected + integral.
6. `.refs/venv_madina_legacy/bin/python
   probes/probe_wf2.py /storage/scratch1/1/dsynn6/una-x` — header probe
   claims confirmed (record md5 stable, HTML md5s differ); output retained
   at `reviewer_artifacts/probe_wf2_rerun.json` (wrote only to /tmp and
   reviewer_artifacts).
7. 3x `.refs/venv_madina_legacy/bin/python
   tests/madina_api/workflows/_workflows_scenario.py --arm reference
   --scenario wf_flow_nc3_structural ...` — edge md5 stable across runs
   (`41cbac40...`); UnboundLocalError at betweenness.py:858 observed live.
8. venv module-resolution probe + version probe — arms load the intended
   files; version claims match.
9. Not run: whole-repo regression (implementer owns the concurrent run, per
   instructions), build_facade.py / refresh_facade_delta.py (would write
   tracked-tree files; identity already proven independently), network
   operations of any kind.

No repository file outside `reviewer_artifacts/` and this report was
created, modified, or deleted by this review.
