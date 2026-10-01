# MADINA_ACCESS independent adversarial review — archived verbatim

Reviewer: mda-reviewer agent (fresh context, read-only, adversarial),
launched 2026-10-01T02:20Z against custody snapshot
/tmp/mda_review_snapshot.txt (10 delivery files). The reviewer's
harness blocked its own report-file write, so its three transmitted
messages below ARE the complete report, archived verbatim and
unabridged in delivery order. Reviewer's own fresh artifacts are
preserved at `reviewer_artifacts/` in this run directory.

---

## Transmission 1 (final-result notification)

VERDICT: **ACCEPT**

No MAJOR findings. 4 MINOR, 6 NOTE. All required checks A-G completed. Custody 10/10; suite reproduced green independently; all 24 digests byte-identical to retained evidence; reflection ledgers verified against real surfaces with zero unaccounted names.

## Custody
Recomputed md5 on all 10 delivery files: all match `/tmp/mda_review_snapshot.txt` exactly (`betweenness.py 68d91055...`, `tools.py 6b041e76...`, `__init__.py 124d4c94...`, `paths.py ba8d262b...` which also still equals upstream byte-for-byte, plus the 6 test files). Re-verified the 4 facade sources again at review end after noticing concurrent activity: unchanged.

## Check A — verbatim ports: VERIFIED
- `tools.py`: difflib of facade :29-51 / :53-245 / :247-367 vs upstream :10-32 / :36-228 / :231-351 → **0 diff lines each** (byte-exact, whitespace included).
- `betweenness.py`: facade :31-368 vs upstream[:1-28]+[:1016-1323] → only 2 whitespace insertions (one blank line at the slice junction facade:59, one extra trailing blank at EOF). Import block byte-exact including the trailing-space `## when one_betweenness_2...` comment; trio byte-exact. Upstream betweenness.py is exactly 1323 lines, so the trio really ends at :1323; upstream tools.py defs at :10/:36/:231 match the claimed ranges as full function bodies.
- `__init__` star-import order matches upstream (betweenness before paths; upstream `__init__` is exactly those 2 lines). Reflection confirms star-import semantics equal (module-level imports np/pd/gpd/mp/nx/... present in `una_public` on both sides).

## Check B — suite run: REPRODUCED
`UNA_LARGE_E2E_ARTIFACTS=/tmp/mda_review_work/artifacts .refs/venv_madina_legacy/bin/python -m pytest tests/madina_api/access/ -q` → **24 passed in 125.68s**, exit 0.
- 24 digests written to my private dir; **all 24 md5s mutually distinct** (no fabrication signature); **all 24 byte-identical to the retained ones** (24/24 SAME).
- Facade state == reference state for all 8 scenarios in both retained and my fresh artifacts (checked directly, independent of pytest asserts); the only top-level JSON difference between arms is the `"arm"` field (3 bytes — explains the uniform 3-byte size delta).
- Facade-arm integrity: `urban_network_analysis` is NOT installed in venv_madina_legacy site-packages (only `distutils-precedence.pth`); with src at sys.path[0] the facade resolves to `/storage/scratch1/1/dsynn6/una-x/src/...` — no stale-copy or upstream-leak cheat path. Retained log header shows the pytest driver was alan while the arms ran under the venv via conftest's hard-coded MADINA_PYTHON — consistent with evidence.json (which claims venv for the arms only).

## Check C — comparator honesty: SOUND
All 6 sabotage digests differ from honest references, verified in content: skip_shuffle flips order [11,10,9]→[9,10,11] and the d2 tie winner source 2→0 at 200 (0x4069000000000000); perturb_gravity_bit zeroes gravity bits 402a...→0000000000000000; fillna sets layer cfd[-1] 7ff8...→0000000000000000; hide_validation_error → NO_ERROR vs honest `ValueError: Parameter 'search_radius': Cannot be negative. search_radius=-5 was given.`. reverse/fillna/perturb/hide are digest-level controls — and evidence.json says exactly that, honestly; alpha_swap and skip_shuffle are true behavioral mutants. The comparator is full-state JSON equality, so anything recorded is selected. Named catch mechanisms: wrong closest-facility winner → `closest_facility_by_destination_source_id` + node digest + explicit `assert cf["0"][0] == 2 and _float(cf["0"][1]) == 100.0` (test_access_surface_parity.py:72); dropped layer join → per-call layer digests + `final["columns"][-3:] == ["reach","grav","reach2"]` (:97); changed alpha default → knn scenario omits alpha so gravity is computed under the default and pinned bitwise, plus the verbatim signature; swapped origin order → `shuffled_origin_order` recorded per call + tie asserts; empty-vs-nonempty scope → `o3_scope_empty`, below_radius/int_miss len==0, scop

[continued in Transmission 2]

---

## Transmission 2 (message, part 1 of 2)

MADINA_ACCESS review (mda-reviewer2), PART 1 of 2 — remainder of check C + checks D and E. VERDICT: ACCEPT. Custody, check A, check B already transmitted (custody 10/10 md5 match; verbatim diffs byte-exact; my own suite run: 24 passed in 125.68s, all 24 fresh digests md5-distinct from each other and byte-identical to the retained ones; facade state == reference state for all 8 scenarios in both retained and fresh artifacts; facade arm provably imports /storage/scratch1/1/dsynn6/una-x/src — urban_network_analysis is NOT installed in venv_madina_legacy site-packages, so no stale-copy cheat path).

REST OF CHECK C (comparator honesty — SOUND):
- Sabotage honesty: reverse_closest_facility, fillna_closest_distance, perturb_gravity_bit, hide_validation_error are DIGEST-LEVEL controls (they mutate the digest dict after the scenario runs); alpha_swap and skip_shuffle are TRUE behavioral mutants applied inside the scenario body. evidence.json sabotage_matrix states exactly this split ("alpha_swap and skip_shuffle are true behavioral mutants" / "reverse/fillna/perturb/hide are comparator-sensitivity controls") — honest labeling, no inflation.
- None is a no-op: I verified each sabotaged retained digest differs from its honest reference in the intended field: skip_shuffle: shuffled_origin_order [11,10,9] -> [9,10,11] and d2 tie winner source 2 -> 0 at 0x4069000000000000 (200.0); reverse_closest_facility: node values closest_facility_distance swapped between d1/d2 rows (100 vs 200 bits differ, so selected); fillna_closest_distance: destination layer cfd[-1] 7ff8000000000000 -> 0000000000000000; perturb_gravity_bit: gravity[0] 402a000000000000 -> 0000000000000000; hide_validation_error: probe becomes "NO_ERROR" vs honest "ValueError: Parameter 'search_radius': Cannot be negative. search_radius=-5 was given."; alpha_swap: full reach/gravity state differs (parameter-level).
- The comparator itself is full-state JSON equality (assert facade["state"] == ref_a["state"]), so anything recorded is selected; separately, test_validation_matrix_all_real_errors asserts all 32 probes != NO_ERROR plus exact error-type counts (15 TypeError / 13 ValueError / 2 AttributeError / 2 IndexError — count verified against scenario_validation's 32 probes: 21 accessibility + 4 structural + 4 service_area + 3 vzr).
- Named catch mechanisms (the specific assertion that catches each): wrong closest-facility winner -> closest_facility_by_destination_source_id map + full node-gdf digest + explicit `assert cf["0"][0] == 2 and _float(cf["0"][1]) == 100.0` (test_access_surface_parity.py:72); dropped layer join -> per-call layer gdf digests in reach_gravity/knn/closest_facility/layer_mutations states + `final["columns"][-3:] == ["reach", "grav", "reach2"]` (:97) + `origin_layer_index_int == "0"`; changed alpha default -> the knn scenario calls accessibility WITHOUT alpha so gravity is computed under the default and pinned bitwise in the node digest, plus the verbatim signature diff (check A) makes a default change impossible without detection; swapped origin order -> shuffled_origin_order recorded per call + tie-winner asserts + skip_shuffle proves order-sensitivity; empty-vs-nonempty scope -> `o3_scope_empty` True, below_radius/int_miss len==0 asserts, scope_gdf row counts 2 (single) / 6 (all origins); an off-by-one in scope rows would change reach/gravity sums -> IEEE-754 bit digest mismatch.

CHECK D (reflection) — VERIFIED INDEPENDENTLY. I ran tests/madina_api/access/_access_reflection.py myself: reference arm under .refs/venv_madina_legacy/bin/python, facade arm under the alan python. Raw results:
- una_public raw diff: only-ref = ['betweenness_exposure', 'clockwiseangle_and_distance', 'one_betweenness_2', 'paralell_betweenness_exposure', 'parallel_betweenness'] == REFERENCE_ONLY_UNA_NAMES exactly; only-facade = [] (FACADE_ONLY empty, correct).
- tools_public raw diff: only-ref = ['betweenness', 'paralell_betweenness_exposure'] == REFERENCE_ONLY_TOOLS_NAMES exactly; only-facade = [].
- betweenness module namespace: only-ref = same five names; only-facade = [].
- Signatures equal after the documented alias strip (only the Zonal qualname prefix differs: 'madina.zonal.zonal.Zonal' vs 'urban_network_analysis.compat.madina.zonal.zonal.Zonal'); e.g. accessibility sig identical through `turn_penalty: bool = False, num_cores: int = 1) -> None`.
- star_exports_are_betweenness_functions True/True; una_level_tools_bindings all False on BOTH arms (pinned equal — neither __init__ binds tools fns at una level, matching upstream).
- Zero unaccounted names: MAJOR-hunting found nothing.

CHECK E (evidence spot-checks) — results with raw evidence:
1. bug_registry.json: exactly 11 bugs (listed ids BUG-MADINA-BTN-STATS-NAMEERROR, BUG-FRAC-WEIGHT-TRUNC, BUG-SAME-EDGE-OD, BUG-COINCIDENT-SEEDS, BUG-NONFINITE-NOVALIDATION, BUG-SVC-GEOMCOLL, BUG-PREP-ZERO-COORD-DROP, BUG-JIT-DISABLED-STUB, BUG-LAYER-SET-STYLE-ATTRERROR, BUG-NETWORK-TO-LAYER-TYPEERROR, BUG-TURN-ELEV-AUDIT). Grep for lexsort/node_builder/truncation hits only BUG-FRAC-WEIGHT-TRUNC's title word "truncated" (fractional weights — different defect). "Not covered by the 11 existing entries" HOLDS.
2. Upstream code verified at .refs/madina_ref/src/madina/zonal/network_utils.py: the dead statement is `np.append(node_first_occurance, point_count - 1)` at :123 (result discarded — np.append is not in-place), the final range is `node_first_occurance[-1] + 1` in the loop at :128, function def at :80, return at :140. So the MECHANISM in evidence.json is CORRECT (final occurrence range truncated to one; remaining occurrences of the lexicographic max keep zeros init), but the CITED lines ":126-149" and ":135" are wrong (facade copy has them at :95/:138/:143) — that is finding M2.
3. Logs: logs/access_suite.log ends "24 passed in 120.92s (0:02:00)"; logs/access_suite_dev_run_failures.log ends "3 failed, 21 passed in 129.19s" with FAILED test_reflection_matches_pinned_reference, FAILED test_star_exports_and_submodule_identity, FAILED test_completion_condition_closest_facility_direction — matches the retained-first-run claim.
4. Claimed completion values all present in retained digests: reach bits ['4014000000000000' x3] = 5.0; gravity ['402a000000000000' x3] = 13.0; NaN-weight reach ['4000000000000000' x3] = 2.0; joint reach sorted [0.0, 0.0, 5.0]; shuffled order [11, 10, 9]; iso destination cf [None, '7ff8000000000000'] in node gdf AND layer cfd[-1] '7ff8000000000000'.
5. Independent hand-recomputation (different failure mode, per policy): gravity = 2^2 + 3^2 = 13.0 under alpha=2/beta=0 with named weights [2,3]; NaN destination weight -> fillna(0) -> 2+0 = 2.0; d2 is a genuine three-way 200 m tie from the fixture geometry (o1 at (-50,0), o2 (50,0), o3 (100,50) each reach (0,150) via A-D-E at 200 m); o3->d1 = 50+50 = 100 m strictly closest. The digest numbers are geometrically correct, not self-consistent fabrications.

---

## Transmission 3 (message, part 2 of 2)

MADINA_ACCESS review (mda-reviewer2), PART 2 of 2 — checks F, G, findings, limitations.

CHECK F (assertion coverage / owned-file substitution):
- Clause mapping for "Closest-facility direction, named joins, return types, exact numeric fields, degenerate geometry fixes and all supported parameters tested": closest-facility direction asserted (test_completion_condition_closest_facility_direction: d1->source 2 at exactly 100.0; tie -> first processed; unreachable NaN; joint reach [0,0,5]); named joins asserted (columns[-3:] == [reach, grav, reach2]; int layer index); return types asserted structurally (set(rec) == {destinations, network_edges, scope_gdf}; a changed return arity/type would break the tuple unpack in _access_scenario.py:454 or the key-set assert); exact numeric fields asserted as IEEE-754 hex (5.0/13.0/2.0/knn); degenerate geometry delivered as BEHAVIOR PINS (below-radius empty structures, silent int-miss, unreachable-destination NaN survival, node-builder broken fixture) — a documented reinterpretation of the dossier's word "fixes", consistent with campaign bitparty rules (fixes belong to corrected_v1, parity profile reproduces upstream bits); all supported parameters PARTIAL: every param exercised positively EXCEPT accessibility(turn_penalty=True) — only service_area turn_penalty=True (_access_scenario.py:469) and the turn_penalty_str validation probe cover it. That is finding M4 (depth gap, not hollowness: the turn_o_scope plumbing is exercised elsewhere and the accessibility text is verbatim-diff-verified).
- owned_files substitution (dossier said una/accessibility.py; delivery puts code in una/betweenness.py + una/tools.py): SOUND, not a scope dodge. Verified upstream madina/una contains {__init__.py, betweenness.py, paths.py, tools.py, workflows.py} — no accessibility module exists upstream; a facade-only accessibility.py would deviate from the pinned module layout and complicate star-import semantics. Documented in evidence.json delta_ledger "owned-file path note". All dossier substance (validate_zonal_ready/accessibility/service_area + engine trio) is delivered under upstream filenames.

CHECK G (documented intentional deltas — all verified as described, none falsely flagged):
1. INTERIM single-name tools import: tools.py:23-25 (`# INTERIM: upstream line 7 also imports paralell_betweenness_exposure...` + `from .betweenness import parallel_access  # noqa: F401`); upstream line 7 confirmed to be the two-name form; ledgered in evidence.json + tools docstring.
2. Five missing betweenness functions: verified by my own reflection run — exactly 5 reference-only names, zero facade-only.
3. JSON round-trip tuple->list in digests: cf_by_source values arrive as lists (e.g. '0': [2, '4059000000000000']) — affects both arms identically.
4. '<Zonal>' normalization: _access_scenario.py:521 `error.replace(str(mz.Zonal), "<Zonal>")` — replaces each arm's own class repr, applied to both arms; only validate_zonal_ready's TypeError actually contains it.
5. Progress-print timing excluded, only line counts kept (_progress_line_count, :260-263).
6. conftest .resolve() relative-vs-absolute ARTIFACTS_OUT fix: conftest.py:55-63 with the explanatory comment; exercised by my absolute path landing at exactly /tmp/mda_review_work/artifacts/madina_access/{arm} with no spurious nesting.

FINDINGS (complete list; no MAJOR):
MINOR
- M1 src/urban_network_analysis/compat/madina/una/tools.py:369-371 — UNLEDGERED structural delta: import block duplicated mid-file (`from .paths import turn_o_scope, path_generator`, `from .betweenness import parallel_access  # noqa: F401  (INTERIM: ...)`, `from ..zonal import Zonal` appear a second time before alternative_paths). Evidence: `git show HEAD:...tools.py` is 104 lines with a single import block at 14-20; current file re-imports at 369-371. Idempotent and semantically inert (tools_public equal modulo ledger), but una/__init__.py:7 claims the ledger is a "complete list of deviations". Fix: one ledger sentence documenting the interim composite file layout (or restructure when MADINA_FLOW lands).
- M2 evidence.json new_pinned_upstream_facts.node_builder_lexsort_truncation + receipt.json handoffs — WRONG LINE CITATIONS: claims upstream network_utils.py ":126-149" and dead append "at :135"; actual upstream def :80, dead np.append :123, occurrence loop :128, return :140 (facade copy has them at :95/:138/:143). The mechanism description itself is correct. Fix: correct the two citations before the bug-registry handoff.
- M3 src/urban_network_analysis/compat/madina/una/__init__.py:31-38 — tools.py ledger row lists `paralell_betweenness_exposure` among the names tools.py "re-imports" in present tense, but the current file imports only parallel_access (INTERIM form documented in the tools docstring and evidence.json, not in this row). Misleading as written. Fix: add "(INTERIM until MADINA_FLOW)" to that row.
- M4 tests/madina_api/access/_access_scenario.py — "all supported parameters tested" gap: no positive accessibility(turn_penalty=True) exercise (only service_area :469 and the type-validation probe). Fix: one positive digest scenario (e.g. turn_penalty=True reach/gravity on the fixture).
NOTE
- N1 betweenness.py:59 + EOF — two whitespace-only insertions (blank line at the import-block/trio junction; extra trailing blank), outside both claimed verbatim slices; no parse/semantic effect. Fix: optional ledger word.
- N2 Whole-repo regression (background task, log logs/whole_repo_regression_mda.log) still in flight at review time; its progress already shows at least one "F" at the 70% mark — outcome undetermined. Receipt honestly records it PENDING; closure must NOT claim no-regression until it finishes green.
- N3 evidence.json independent_review_outcome.verdict = "PENDING — filled at closure" — correctly not self-approved.
- N4 Concurrent actor modified tests/madina_api/paths/test_paths_reflection.py at 22:16:58 during my review window (git diff: 17 insertions, 24 deletions; not a delivery file; I ran no writes — facade sources re-verified unchanged: betweenness 68d91055..., tools 6b041e76..., __init__ 124d4c94..., paths ba8d262b...). Working tree moved under review — relevant to closure sequencing.
- N5 Dev-failure narrative slightly loose: the retained log shows 3 failed = 2 reflection tests + 1 closest-facility assertion; evidence.json describes "two test-assertion bugs ... and the reflection probe's wrong star-export assumption" (one assumption plausibly causing both reflection failures). Raw log retained, so no evidence gap.
- N6 Upstream madina/una/workflows.py has no facade counterpart — within the documented "workflows not in scope" limit; noted so the absence stays visible until the workflows task.

LIMITATIONS (unrun checks — limitations, not passes): (1) did not re-run the whole-repo regression (owned by the concurrent process); (2) did not re-derive the node-builder truncation from first principles beyond reading upstream :110-140 and the scenario's pinned broken edge table (edge 2 = start 4, end 0, length 100.0; 6 edges; o3 scope empty) — that table is asserted in test_node_builder_truncation_pinned and reproduced bitwise in my own run; (3) the retained run's "relative UNA_LARGE_E2E_ARTIFACTS" claim is consistent with where the retained artifacts landed (repo evidence dir) but not provable from the log alone (no env echo) — immaterial to parity. Note: /tmp/mda_review_work/report.md could not be created — my harness blocks subagent report-file writes (it rejected the Write); these two transmitted messages are the complete report.

---

## Post-review disposition (by the implementer, not the reviewer)

Verdict ACCEPT. All four MINOR findings fixed before closure:

- M1 FIXED: duplicate mid-file import block removed from tools.py (the
  top import block at :17-26 already carries every name); single
  top-of-file import block now matches upstream layout. Reviewer
  offered "ledger or restructure" — restructured.
- M2 FIXED: evidence.json line citations corrected to the verified
  upstream lines (vectorized_node_edge_builder def :80, lexsort :96,
  dead np.append :123, occurrence loop :128, return :140).
- M3 FIXED: __init__ ledger row reworded (INTERIM re-import scope made
  explicit).
- M4 FIXED: reach_gravity scenario gained a positive
  accessibility(turn_penalty=True) call (state key
  reach_turn_penalty with turn_engages), asserted in
  test_completion_condition_exact_numeric_fields; probe run green on
  both arms before the full post-fix suite.

NOTEs: N1 ledgered in evidence.json delta_ledger; N2/N3 handled by
this closure (regression completed and verdict filled); N4 is this
task's own regression-driven paths-ledger fix (documented in
whole_repo_regression); N5 evidence.json wording tightened; N6 stays
visible as the documented MADINA_WORKFLOWS limit.

Post-fix evidence: the pre-fix digests the reviewer verified
byte-identical are preserved at
`artifacts/madina_access_prereview_fix/`; the post-fix suite rerun
regenerated `artifacts/madina_access/` (logs/access_suite.log; the
pre-fix retained run log is `logs/access_suite_prereview_fix.log`).
The reviewer's own fresh artifacts (byte-identical to the pre-fix
retained ones per Transmission 1) are preserved at
`reviewer_artifacts/`.
