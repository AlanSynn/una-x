# MADINA_PATHS independent adversarial review — mdp-reviewer
2026-10-01, branch perf/una-platform, HEAD d6ccc9a. Custody: /tmp/mdp_review_snapshot.txt
(2026-10-01T00:43:46Z) — all 9 reviewed files md5-verified OK before and after review.
Read-only review; no repository file was created, modified, or deleted; all review artifacts
under /tmp.

## Verdict: ACCEPT

No MAJOR findings. Two MINOR evidence-metadata errors (both in the not-yet-committed evidence
JSON, both trivially fixable before commit). The core deliverables — verbatim-port claims,
ledger honesty, test substance, and the retained evidence — all verified independently,
including a full fresh re-execution that reproduced every retained digest byte-for-byte.

## Findings

### MAJOR — none.

### MINOR

M1. Digest count misstated in evidence metadata.
- `campaigns/una_platform/evidence/tasks/MADINA_PATHS/mdp-20261001T0037Z/receipt.json` line 20
  ("the 21 digests written by the retained run") and deliverables "21 retained digests";
  `evidence.json` environment.artifacts says "19 scenario+sabotage digests ... reflection = 2".
- Actual on disk: 6 facade + 12 reference (6 clean + 6 sabotage) = 18 scenario+sabotage,
  + 2 reflection = 20 files. The "19" is the passing-test count conflated with the artifact
  count; the receipt's "21" inherits it.
- Why it matters: evidence documents are part of the auditable record; a number that is wrong
  by one and guessable-as-a-conflation is exactly the kind of detail an auditor uses to discount
  the rest. No parity claim is affected (all 20 files were verified, below).

M2. `evidence.json` scenario_matrix `alt_detour` is described as "detour_ratio=2.5
turn_penalty=True". False: `tests/madina_api/paths/_paths_scenario.py:338-386`
(`scenario_alt_detour`) uses `turn_penalty=False` on every call (turn_o_scope, the manual
bfs_path_edges call, and `alternative_paths` via default). Turn-penalty engagement is pinned by
`turn_params`, not `alt_detour`. Misdescription of what the artifact pins; the artifact itself is
honest.

### NOTE

N1. "Bitwise-strict" has a hair-thin gap: a few state leaves are raw Python floats compared by
`==` rather than `_f()` bit strings — `turn_probes[*]["penalty"]`
(_paths_scenario.py:424) and `net_params_before/after` (:391-392, :427-428). `==` differs from
bit equality only for -0.0 vs 0.0 and NaN, neither reachable on this fixture (penalty 0.0,
thresholds 45/30/91/15). Every GDF float, route distance, d_idx, scope weight, and angle IS
bit-ified. Cosmetic overstatement of the blanket claim, not a parity hole.

N2. `--seed 20260930` is recorded in every digest but consumed by nothing — paths.py has no RNG;
PYTHONHASHSEED=0 (forced in both `run_arm` and `run_reflection`) is the real determinism
control, and the reference-runs-twice determinism control is real. Inert plumbing, honestly
recorded.

N3. `origin_id_list` probe records only NO_ERROR and discards the produced GeoDataFrame
(_paths_scenario.py:562-563). The upstream acceptance fact itself is confirmed in source
(`.refs/madina_ref/src/madina/una/tools.py:51` — `isinstance(origin_id, (int, list))`), and the
membership line (`origin_id not in origin_gdf['source_id'].values`) is an elementwise ndarray
comparison that [0] passes when source_id 0 exists, so NO_ERROR is genuine, not an omission.
Thin but accurate.

N4. `__init__.py` ledger line 11 calls paths.py "pure math/collections/heapq/networkx/pandas" —
paths.py imports math, collections.deque, heapq, networkx (line 265), and `..zonal.Network`;
no pandas import (pandas enters only via tools.py's GeoDataFrame materialization). Cosmetic
docstring inaccuracy inside a correct ledger.

N5. `_independent_paths` inherits the engine's own scope (`turn_o_scope`) and shortest-distance
bounds for its enumeration. This is explicitly disclosed ("over the engine's own scope",
_paths_scenario.py:235; evidence.json all_paths_completion_condition), and the scope itself is
pinned bitwise across arms (`turn_o_scope_scope`, `scope_size`), so the no-hidden-K-limit claim
is validated within a correctly-pinned scope — not a global-optimality proof. Appropriately
scoped and honestly labeled; noted so nobody later reads it as more than it is.

N6. Zonal conftest cross-scope touch (`tests/madina_api/zonal/conftest.py`, 2 resolve() calls +
honest comments): same shared defect class via the common `tests/large_e2e/_legacy_paths.py`
ARTIFACTS_OUT, absolute-path usage had masked it, and the zonal suite is green with the touch
(21 passed in my execution). Justified, minimal, correctly attributed.

## What I executed (vs. only read)

Executed:
1. `md5sum -c /tmp/mdp_review_snapshot.txt` — 9/9 OK (custody intact).
2. `diff` facade `paths.py` vs `.refs/madina_ref/src/madina/una/paths.py` — byte-identical
   (same md5 ba8d262b...); confirmed .refs checkout is at d6ccc9a (same as branch HEAD).
3. Ranged extraction + diff of upstream `tools.py:355-435` vs facade `alternative_paths`
   body — byte-identical (sole upstream-slice delta is the trailing blank separator line 436).
4. Full fresh suite run:
   `UNA_LARGE_E2E_ARTIFACTS=/tmp/mdp_review_artifacts <alan-python> -m pytest tests/madina_api/paths/ -q`
   — 19 passed in 64.70s.
5. Cross-run reproducibility: every retained digest `state` (6 facade + 6 reference clean +
   6 reference sabotage + 2 reflection) compared against my fresh run's output — ALL
   byte-equal. (This also makes retained-artifact fabrication moot: any forgery would have to
   be a perfect predictor of a fresh independent run.)
6. `tests/madina_api/zonal/` with the touched conftest — 21 passed in 67.62s.
7. Relative-env fix exercised as the incident describes: ran the suite from a NON-repo cwd
   (/tmp/mdp_relcwd) with a relative `UNA_LARGE_E2E_ARTIFACTS=mdp_rel_art` — test passed and
   the digest landed at exactly `<cwd>/mdp_rel_art/madina_paths/facade/turn_params.json`
   (one level; no out_dir/campaigns/... nesting). Without the fix this nests.
8. Direct reproduction of the validation scenario under the reference arm: rc=0, stderr contains
   the `AttributeError ... 'NoneType' object has no attribute 'graph'` traceback, digest records
   `no_graph: UnboundLocalError`. This resolved my one suspicion about the retained failure log:
   the stderr traceback in `paths_suite_relative_artifacts_failure.log` is emitted by UPSTREAM's
   own `add_node_to_graph` (`madina/zonal/network.py:94-118` prints the exception via
   `traceback.print_exc()`, swallows it, then `neigoboring_nodes` is unbound → UnboundLocalError).
   The delivered script therefore CAN produce that exact log — the first-run evidence is
   consistent with the delivered code, and the incident narrative needs no correction.
9. Programmatic verification of retained artifacts: facade==reference `state` for all 6
   scenarios; all 6 sabotage digests differ from their honest digest; validation matrix = 10
   errors + `origin_id_list: NO_ERROR`, facade arm identical; `engine_equals_independent: True`
   with per_dest {9: 4, 10: 2}, scope_size 7, full_route_set 6, coherence
   gdf_counts {0:4, 1:2} == independent counts; reflection ledger sets match the ACTUAL
   namespace diffs exactly (21 ref-only una names, 6 ref-only tools names, zero facade-only on
   either side); raw signature deltas are exactly the bridged annotation qualifiers, stripped
   symmetrically longest-first (ordering is required and correct).

Only read (verified by inspection, not execution): upstream `una/__init__.py` star-import order
(betweenness BEFORE paths — ledger claim correct); upstream `paths.py`
bfs_path_edges semantics backing the representation matching ("if len(visited) == 1:
neighbor_edges = []" — first hop never stored; arriving edge appended-before-never-stored;
`edges.copy() if edge_id in edges else edges + [edge_id]` — first-occurrence parent-id dedupe)
all present verbatim in upstream source as the test claims; upstream `path_generator` bound
chain (`d_allowed = d_idxs * detour_ratio`, `<= 0.00001` acceptance) matching the independent
walk's `d_idxs*detour_ratio + 1e-5`; facade tools.py import sufficiency (gpd, GeometryCollection,
path_generator, Zonal all imported; alternative_paths touches nothing else); no try/except
anywhere in facade una/ (grep); test-count arithmetic 6+1+6+1+1+2+2 = 19; sabotage mutations
each alter real recorded state; `_capture_stdout` records type+message of pinned errors.

## Verbatim/parity claims — independently confirmed vs. not

Independently CONFIRMED by execution/diff:
- paths.py is a byte-identical copy of the pinned upstream file (zero deltas — ledger claim TRUE).
- `alternative_paths` is byte-identical to upstream tools.py:355-435; the facade tools.py import
  block equals upstream's minus exactly the ledgered `.betweenness` import; imports suffice at
  runtime (facade arm executed it in my run).
- `__init__.py` ledger matches reality: only `from .paths import *`; upstream star-import order
  claim correct; no hidden deltas; reflection digests show exactly the ledgered interim
  namespace deltas and no facade-only names.
- Validation matrix honest: 10 real errors + 1 legitimate NO_ERROR (upstream source accepts
  int-or-list origin_id).
- 19/19 tests pass in a fresh independent run; retained evidence is byte-reproducible.
- The retained first-run failure log is consistent with the delivered code (upstream-printed
  traceback), so the "relative ARTIFACTS_OUT incident, fixed in both conftests" narrative holds;
  the fix verified behaviorally.
- Zonal conftest touch safe (21 passed).

Could NOT confirm (no evidence either way, none claimed dishonestly):
- That the retained green run used a relative (vs absolute) UNA_LARGE_E2E_ARTIFACTS — the log
  does not record the env var. Immaterial: I verified the fix itself works with a relative var,
  and the digests are byte-reproducible regardless.
- Whole-repo regression status (receipt marks it PENDING in background) — outside my scope; I
  ran the two affected suites instead.
- City-scale / weights-bound behavior — explicitly out of scope in the receipt's limits, honestly.
