# MADINA_FLOW independent review (attempt 3) — 2026-10-01T08:27Z



## Check A — facade diff vs upstream (.refs/madina_ref @ 8b5c3bd)
- `diff` betweenness.py: +99 / -0 (pure docstring header). tools.py: +14 / -0 (pure docstring). Matches expectation exactly.
- Header bullets spot-read: documents unseeded origin shuffling, exception swallowing, closest-facility direction, in-place decay reassignment, 0.01 clamp, double destination_count increment, cocentric-chunks hardcode, 'decayed_mean_hazzad' typo pin, degenerate edge attribution, and parallel num_cores facts. tools.py header correctly states betweenness/paralell_betweenness_exposure are re-imported from .betweenness mirroring upstream line 7.
- Header-vs-suite cross-check deferred to check F (evidence.json) — to be resolved below.

## Check C — harness digest helpers (_flow_scenario.py)
- `_f` = struct.pack('>d').hex() — bitwise IEEE-754, no rounding/tolerance/allclose anywhere in the file (grep for round/atol/rtol/allclose: only hits are dtype handling).
- Only exclusions are the 4 documented TIME_COLUMNS (line 70), and their PRESENCE is recorded per frame (`excluded_time_columns_present`, line 162) plus `stats_columns_present` (line 164) — exclusion is not silent. Layer digests drop only the `bt_`-prefixed time columns (line 792-803). Object columns go through `str(v)` (line 130) — a theoretical str()-masking window, but identical to the prior paths/zonal/access suite discipline; NOTE only.
- Flags audited for vacuity in scenario_flow_lowlevel_engine: upstream default `path_detour_penalty="equal"` (betweenness.py:38), so base(=equal,1.0) vs exponent_single_path(=exponent,1.0), equal_multi vs exponent vs power (all @1.5), and huff(1.0) vs huff_detour_15(1.5) are ALL parameter-differing pairs. No run-vs-itself pairing. The fixed `lowlevel_detour_engages` = huff_detour_15 != huff is a true 1.5-vs-1.0 pair (lines 578-586).
- scenario_flow_closest_huff: `models_distinguish` = closest(closest_destination=True) vs huff(False) — real param delta. decay_matrix engage flags (exponent/power/cap/detour, lines 512-530), plateau_engages (k=100 vs unbounded knn), weight_change_engages (edge weight mutated between runs), turn_engages (turn penalty vs no_turn) — all cross-run with genuine state/param deltas. NO vacuous flag found.

## Check D — sabotages
- 2 parameter-level: `decay_method_swap` (scenario line 460 swaps exponent/power method strings actually passed to paralell_betweenness_exposure) and `detour_swap` (line 548 swaps r_small/r_large so base/huff run at 1.5 while kw-pinned runs stay 1.5). Both genuinely change recorded state (pinned by `power_differs_from_exponent`/`lowlevel_detour_engages` = True in clean runs).
- 2 digest-level: `split_edge_drop` drops the shared-parent edge row from edges/parents/betweenness_by_edge with a guard that raises if the row is not actually split; `exposure_column_drop` removes decayed_mean_hazzad from columns + stats ledger, guard raises if absent.
- `test_sabotaged_digests_are_selected` (test_flow_surface_parity.py:409) asserts broken != honest on the reference arm for all 4 — a no-op mutant fails it. Deeper assertions pin row counts, dropped keys, and column shrinkage (lines 444-451, 463-467), so a partially-effective mutant is also caught. Candidate breakage is caught by the byte-equality parity comparator plus flag pins (`lowlevel_detour_engages is True` at line 184 — the known-history fix IS asserted).

## Check E — custody + workspace hygiene
- All 13 md5s in custody_md5s_post_fix.txt match the working tree exactly (`CUSTODY_MATCH`; covers 4 facade files incl. paths.py, 6 flow test files, 3 carried-over access/paths reflection files).
- `git status --porcelain`: exactly the expected set — M goal.md (user state, unstaged), M una/{__init__,betweenness,tools}.py, M 3 access/paths reflection files, ?? evidence/tasks/MADINA_FLOW/, ?? tests/madina_api/flow/. No unexpected files.

## Check B — suite run + digest custody (my own run)
- `pytest tests/madina_api/flow/ -q` (UNA_LARGE_E2E_ARTIFACTS=/tmp/mda_flow_review3/artifacts): **27 passed, 0 failed** in 243.72 s, exit 0.
- All 24 digests (9 facade, 13 reference incl. 4 sabotages, 2 reflection) are **byte-identical md5-for-md5** with the retained evidence artifacts/madina_flow/ — including the "arm" field (same arms, absolute ARTIFACTS_OUT override only).
- Independent cross-arm equality (my run, strip "arm"): all 9 scenarios EQUAL facade-vs-reference, verified outside the test framework.
- conftest arm provenance is genuine: reference arm = .refs/venv_madina_legacy python + sys.path[0]=.refs/madina_ref/src importing `madina.*`; facade arm = REPO/src importing `urban_network_analysis.compat.madina.*`. Reflection digests prove distinct module identity: 8/8 betweenness signatures identical after stripping only the type-annotation prefix (`urban_network_analysis.compat.madina.` vs `madina.`), same for the 6 tools signatures; key sets equal. Not a facade-vs-facade vacuity.

## Check F — evidence.json audit vs observations
- scenario_matrix: 9 rows = exactly the 9 scenario functions in _flow_scenario.py; descriptions match the flags I read (models_distinguish, nc2_all_zero, layer_state_equal_across_cores, etc.).
- sabotage_matrix: 4 mutants, all "SELECTED"; consistent with test_sabotaged_digests_are_selected passing and with the two dedicated integrity tests.
- new_pinned_upstream_facts: `lowlevel_detour_ratio_filters_path_set` correctly documents the REVIEW CATCH and retraction (`huff_detour_moot_bitwise` marked RETRACTED pre-delivery, pointing at lowlevel_detour_ratio_filters_path_set + retained prefix artifacts). Fact text matches the code I read (betweenness.py:297 filter; run() pairs 1.0 vs 1.5).
- delta_ledger: matches Check A exactly (betweenness.py full module behind a 99-line docstring; tools.py docstring-only; __init__ rows; access/paths reflection docstring-only changes — those 3 files are in custody and I diffed nothing else changed).
- limits: honest (upstream defects carried verbatim/pinned, nc>1 exposure nondeterminism documented, TIME columns policy, split-street construction rationale, 8-edge grid scope).
- whole_repo_regression (302 passed/0 failed) and independent_review_outcome=PENDING: skipped per instructions.

## Additional observations
- Comparator: test_facade_matches_bridged_reference_bitwise runs the reference arm TWICE (determinism guard) before facade-vs-reference, with a _first_difference path reporter; digests compare ["state"] so the "arm" field cannot mask anything.
- RNG control (_seeded_origin_order, line 310): seeds the global numpy RNG, draws the unseeded upstream sample(frac=1) permutation once to record it, then re-seeds so the call consumes the identical permutation — and my run reproduced the evidence run byte-identically on all 24 digests, empirically validating determinism.

## Findings
- MAJOR: none.
- MINOR: none.
- NOTE 1 (_flow_scenario.py:130): non-numeric/object columns are digested via str(v); two distinct objects with equal str() would collide. Consistent with the paths/zonal/access suite discipline; no concrete instance in this suite (the heavy columns are floats digested bitwise).
- NOTE 2 (evidence new_pinned_upstream_facts/exposure_nc_gt1_partition_nondeterminism): the "per-edge float-addition order varies run to run" claim is code-reading-based (manager.Queue at betweenness.py:1045/1305 + as_completed at 1085/1361); no retained experiment demonstrates an actual bit diff across two nc=2 runs, and the suite digests nc=2 edges only structurally, so if bits were in fact stable the nc=2 edge parity would be structurally-only. Mitigated: per-origin layer state IS compared bitwise at nc=2, and nc>1 is a pinned-defective upstream surface (corrected_v1 scope).
- NOTE 3 (reflection "identity" section is just package_has_path=true): thin, but module provenance is proven more strongly by the type-annotation prefixes in the surface section (verified above).

## Checks completed
A facade diff (+99/-0, +14/-0; headers read) / B pytest 27 passed 243.72s + 24/24 md5 byte-identical + independent 9/9 state equality / C digest helpers + all flag pairings audited, no vacuity / D 4 mutators + selection assertions read / E 13/13 custody md5 match + git status clean-set / F evidence.json cross-audited (regression + review-outcome sections skipped as instructed).

## Checks NOT run
- whole_repo_regression re-run (302 passed claim taken from evidence.json, unverified).
- An empirical multi-run nc=2 exposure bit-stability experiment (see NOTE 2).
- No re-run of the pre-fix sabotage comparison (retained artifacts inspected for existence only).

## VERDICT: ACCEPT
