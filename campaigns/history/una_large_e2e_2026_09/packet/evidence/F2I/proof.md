# F2I proof — local flow overlap + scratch reset (non-turns pipeline)

Task F2I (#30), implementation round. Executed under h04's "F2I
PREIMPLEMENTATION PROOF APPROVAL — GRANTED" (relayed by team-lead
2026-09-27) with binding conditions (a)–(h). This document is written
against the actual compiled source: every citation below is a line
number of the candidate module
`src/urban_network_analysis/Engines/AggregateFlow.py`,
sha256
`1f3cf4d2725320569c1b19bf339a39d0bf013d1cb7b6f8e5d72526a77531797c`
(fresh tool output at authoring time; identical to the sha pinned by
`test_artifacts.json.aggregate_flow_sha256_at_flush`, flushed
2026-09-27T17:58:18Z). Diff vs HEAD `9a88eabe2eb2ab21b7772b8a1a6d1ffe1e8cc6a8`:
471 insertions, 12 deletions, hunks at @@ 389 (F2 block insert after
the baseline kernel), @@ 1110/1132/1149 (driver gating, counters,
scratch), @@ 1205 (call site), @@ 1271/1277 (stats + log line). The
baseline kernel `_accumulate_od_flow` (lines 194–387) and the entire
turns pipeline (kernel at module scope, driver `_process_stripe` at
line 1903) are byte-untouched.

Test evidence: `tests/large_e2e/F2/` — 6 files, 15 tests, all green in
one process (2026-09-27T17:58:16Z, `venvs/campaign` interpreter:
python 3.11.16, numba 0.67.0, numpy 2.4.6, scipy 1.17.1, pytest
9.1.1). Harness flush: `evidence/F2I/test_artifacts.json` (sha256
`3753642dfcd1fcdb4b60c46a2e2d3ca906c30b139613d7c0089607825faa4aa1`),
60 records, `first_divergences` empty.

---

## 1. Scope

F2 applies ONLY to the non-turns pipeline `_process_origins_aggregate`
(line 1422). The turns variant calls a different kernel
(`_accumulate_od_flow_turns`) from its own `_process_stripe` (line
1903); neither contains any F2 symbol. Pinned structurally by
`test_f2_driver.py::test_turns_pipeline_has_no_f2_wrapper`
(inspect.getsource of `_process_origins_aggregate_turns` contains
neither `_accumulate_od_flow_local` nor `_use_local_route`) and by the
hunk map (no insertion inside the turns driver). F1's nogil stripe
mechanism and F3's gradient workspace are consumed unchanged; the F2
driver code adds symbols around them, never modifies them.

## 2. Mechanism

For each OD the driver scatters the destination's sparse gradient
slice `cols = g_nodes[s0:s1]` into the reusable dense buffers
(lines 1576–1581) and then routes:

* **Route decision** (line 1583):
  `_use_local_route(local_route_ok, s1 - s0, n_total_nodes)`
  (lines 440–446) = precondition AND `slice_len <= max(1, n_total //
  _F2_LOCAL_SLICE_FRAC)` with `_F2_LOCAL_SLICE_FRAC = 8` (line 413).
  Deterministic, no state.
* **Fast route** (lines 1592–1641): `_accumulate_od_flow_local`
  (lines 449–736) on per-stripe preallocated scratch allocated once
  per `_process_stripe` invocation (five arrays: `sc_reach/sc_cont_o/
  sc_cont_d` bool, `sc_acc_o/sc_acc_d` f64, plus `sc_reach_nodes` and
  five int64 touched lists sized `n_total` / `n_edges + n_total`).
* **Fallback on exception** (lines 1612–1637): host-side `.fill()`
  re-zero of all five scratch arrays, then the unchanged baseline
  kernel serves the OD; `local_n_fallback[slot] += 1`.
* **Baseline route** (lines 1642–1655): the untouched baseline kernel,
  as before F2.

The kernel returns `(delivered, of)`; `delivered` feeds the existing
gap accounting (line 1661) exactly as the baseline's return always
did; `of` and the counters surface in `self._f2_stats`
(lines 1711–1720) and the `[F2]` log line (lines 1728–1734). The
stats are observability only — no arithmetic reads them.

## 3. Exactness (transition proof)

Claim: for every OD where the route admits the fast kernel, its
outputs (`out_AB`, `out_BA`, `out_node_flow`, `delivered`) are
bitwise equal to `_accumulate_od_flow` on the same inputs.

1. **Reach-set equality.** The baseline reach scan (lines 224–235)
   admits `v` iff `d_o[v] < inf and d_d[v] < inf and d_o[v] + d_d[v]
   <= budget`. The driver scatters finite `d_d`/`pred_d` values into
   `dd_buf`/`pd_buf` exactly on `cols` and leaves `+inf`/`-9999`
   elsewhere (F3 sparse-gradient contract; scatter at 1580–1581,
   restore at 1657–1659), so `dd_buf < inf` holds exactly on `cols`.
   The local kernel (lines 492–507) applies the identical predicate
   restricted to `cols`. Hence the admitted sets are equal. This is
   the load-bearing leg and it is differential-tested two ways:
   `test_slice_covers_full_scan_reach` (numpy replication of the full
   scan vs the cols-filtered predicate on every micro case) and the
   admission probe's reach mask+list equality on 187/187 real capture
   ODs (f2i_admission_125153).
2. **Iteration-order equality.** `cols` is ascending (CSR `g_nodes`
   slice; precondition §6), so `reach_nodes[0:n_r]` is ascending —
   the same array the baseline fills (lines 237–242). `order_o`/
   `order_d` (lines 519–520) are `np.argsort` of `d_o[reach_nodes]`/
   `d_d[reach_nodes]` on the identical input arrays as the baseline's
   lines 244–245; identical inputs to the same numba-compiled argsort
   give identical outputs, ties included (identical input ⇒ identical
   output regardless of tie policy; pinned by `m_ties` and the fuzz).
3. **Sections 2–7 statement identity.** Contamination marking
   (522–566 vs baseline 247–276), pass 1 (568–589 vs 278–299), pass 2
   (610–650 vs 305–338), origin legs (652–674 vs 340–359), destination
   legs (676–698 vs 361–378), node flow (700–705 vs 380–385): each
   loop body is token-identical to the baseline except for the
   appended touched-list record statements, which write only to the
   caller-owned integer lists and never influence the arithmetic.
   Identical statements over identical inputs with identical
   `fastmath=True` reassociation give bitwise-identical results.
4. **Delivered equality.** `delivered` is captured at line 712
   BEFORE the reset epilogue and returned at line 736 — the same
   pre-reset `acc_d[dest_virtual_node]` the baseline returns at line
   387. (See §9: the first port read it after the reset; the battery
   caught it.)

Exactness is asserted bitwise (`.tobytes()` equality on all four
outputs) by `test_micro_battery_bitwise` (14 adversarial micro cases),
`test_fuzz_bitwise_first_divergence` (24 seeded random digraphs, ≥12
callable), `test_route_toggle_engine_bitwise` and
`test_multithread_engine_bitwise` (engine level, K=1 and K=3), and
was validated against the real engine kernel on 187/187 capture ODs
at admission. One bitwise divergence rejects the domain; zero
divergence records exist in `test_artifacts.json`.

## 4. Observation: every exit restores pristine scratch

The scratch contract (docstring lines 479–482): caller-owned arrays,
pristine (all False / +0.0) on entry and on exit.
`reach_nodes[0:n_r]` is overwritten each call and needs no reset.

* **Virtual early return** (lines 509–517): only `reach` was written;
  touched-list reset of `reach` (full re-zero if `of`), return
  `(0.0, of)` — matches baseline line 234–235's `return 0.0`.
* **q_sum <= 0** (lines 591–607): `reach` + `cont_o` + `cont_d`
  written; touched-list reset of all three (full re-zero of all five
  arrays if `of`), return `(0.0, of)` — matches baseline 301–302.
  Structurally unreachable in-sample (both virtuals must be reached
  while every via-arc check fails), so it has a dedicated adversarial
  fixture (`m_qsum_zero`: slice pinned to the two virtual nodes,
  tight budget) asserting the baseline really takes this exit, the
  local kernel delivers 0.0 bitwise, the scratch is pristine, and a
  subsequent OD served on the SAME scratch is still exact
  (`test_qsum_zero_adversarial_reset`). Source identity alone is not
  the coverage — the dedicated case is.
* **Full path** (lines 707–735): epilogue resets via touched lists;
  if `of` (any list overflowed) it re-zeros all five arrays in full
  (§5). `delivered` was captured at line 712, before any reset store.
* **Kernel exception** (driver lines 1612–1637): the wrapper
  `.fill()`s all five arrays host-side and serves the OD on the
  baseline kernel; the worker and its scratch continue. Covered by
  `test_exception_safety_injected_error_then_continue`: an injected
  failure on the second default-route fast call yields exactly one
  fallback, byte-identical final engine outputs vs the forced-baseline
  reference, and a working fast route on prior calls. (The wrapper
  covers exceptions raised at the call boundary; a hard kernel crash
  is not catchable by any caller in any route.)
* **Between ODs**: sequential reuse of the SAME scratch across
  different ODs is exercised directly
  (`test_sequential_reuse_cross_od`: base → contam → base on one
  scratch; `qsum_zero_then_base_reuse`; `overflow_then_reuse`), and
  structurally by every engine run (per-stripe scratch serves many
  ODs; engine outputs are byte-identical to forced-baseline runs).

Pristineness is asserted by `assert_pristine` — byte-comparison of
all five arrays against fresh zero allocations after every kernel
call in every micro test.

## 5. Overflow policy (binding condition b)

An overflowed touched list is TRUNCATED, never wrapped: the record
sites (e.g. lines 501–505, 532–536, 641–650, 670–674, 694–698) set
`of = 1` and drop the record. Appends are passive records — the
arithmetic never reads the lists — so the OD still completes with
bitwise-exact outputs. The epilogue (lines 708–716) treats `of` as
authoritative: ANY overflow ⇒ full re-zero of all five arrays; the
truncated lists can never drive a (partial) reset. A partial reset
never happens on any path. The driver only counts overflow events
(lines 1640–1641). Covered by
`test_overflow_full_rezero_never_partial` (caps of 1 force `of == 1`
mid-recording; outputs bitwise exact; all five arrays byte-equal
pristine allocations; scratch then reused exactly). At admission:
0 overflow events over 187 ODs with production caps.

## 6. Precondition (binding condition c)

`_gradient_slices_strictly_ascending` (lines 416–437) is dossier
step 1: within EVERY destination slice, `g_nodes` must be strictly
ascending (⇒ unique global ids). One vectorized pass: `np.diff >
0`, then unconditional exemption of the adjacency pair ending at
each interior slice start (a boundary at 0 has no left pair; a
boundary at `len(g_nodes)` is an empty final slice whose left pair
lies inside the previous slice — neither is exempted). Any
within-slice violation ⇒ `False` ⇒ `local_route_ok = False` for the
WHOLE run (line 1465) ⇒ every OD on the unchanged baseline kernel.

Evidence:
* `test_check_matches_bruteforce_adversarial`: 12 hand-built
  boundary cases differential-tested against a per-slice brute-force
  reference, including the two traps — `violation_before_empty_slice`
  (a violation in the last pair of a slice followed by an empty slice
  must NOT be swallowed by the exemption) and
  `empty_slice_legal_cross` (a legal cross-boundary descent must NOT
  false-positive when an empty slice sits between non-empty ones).
* `test_check_matches_bruteforce_random`: 200 seeded random
  indptr/nodes trials, exact agreement (this is where the empty-final-
  slice bounds bug was found, §9).
* `test_precondition_disabled_falls_back`: engine-level — forced
  `False` precondition yields `local_route_ok: False`,
  `fast_calls == 0`, outputs byte-identical to the forced-baseline
  reference.
* The engine verifies the REAL g_nodes construction every run at
  line 1465; an unsupported construction silently degrades to the
  exact prior behavior, never to a wrong answer.

## 7. Dense-overlap crossover (binding condition f)

`_use_local_route` (lines 440–446): the fast route is admitted iff
the precondition holds AND `slice_len <= max(1, n_total //
_F2_LOCAL_SLICE_FRAC)`, `_F2_LOCAL_SLICE_FRAC = 8`. Deterministic in
`(local_ok, slice_len, n_total)`; no state, no measurement.
`test_use_local_route_crossover` pins the threshold semantics
(inclusive at the threshold, exclusive one past, floor at 1 for
degenerate CSRs, never-on-precondition-false). Engine-level both
sides, engaged through the PRODUCTION decision path (settings-only
overrides, no patched decisions): on the shared oracle stub the
default crossover refuses every OD (`slice_max = 2` < every slice —
`test_route_toggle_engine_bitwise` asserts `fast_calls == 0` there);
on `big_flow_spec()` (fixtures_f2, n=120 path, oracle module
untouched) at `search_radius 4.0` the default crossover admits real
ODs (`slice_max = 15`, probed `fast_calls = 2`, nonzero flow) and all
three arms — forced-baseline, forced-fast, default — are
byte-identical.

## 8. Adversarial battery (binding condition e)

Classes → where covered (`fixtures_f2.micro_cases()` unless noted):
weight ties (`m_ties` + fuzz); contaminated snap edges (`m_contam`,
both directions of leg contamination); u-turn / dead-end exclusion
(`m_uturn_leaf`); -9999 missing predecessors (every case's `pd_buf`
off-slice contract; `test_slice_covers_full_scan_reach` asserts the
scatter discipline); zero node-output (`m_zero_trip`, and
`m_base_nonode` for `out_node_flow.shape == 0`); repeated destination
/ duplicate reset targets (`test_sequential_reuse_cross_od`, fuzz
naturally generates repeated `acc` indices); disjoint envelopes and
pv-outside-reach (fuzz seeds); all-overlap dense slice (shared-stub
engine arms + `m_base`); virtual early return (`m_unreach_dest` +
`test_early_exit_pristine`); q_sum <= 0 (`m_qsum_zero` +
`test_qsum_zero_adversarial_reset`); touched-list overflow
(`test_overflow_full_rezero_never_partial`); injected early error
then continued worker (`test_exception_safety_injected_error_then_continue`);
decay curves (equal/exponential/logistic, `m_decay_*`); budget and
grad-limit variation (`m_budget_tight`, `m_sparse_slice`); 24-seed
fuzz (`test_fuzz_bitwise_first_divergence`). Driver level:
route toggle both sides, precondition fallback, exception safety,
K=3 multithread bitwise, turns source pin
(`test_f2_driver.py`, 5 tests). Precondition level: 12 adversarial +
200 random + crossover (3 tests).

## 9. Defect ledger (found and fixed in this round)

Honest record; each entry names the fire that caught it (lease log
windows 6–10, log sha256 at close of test phase
`4ff95097663fd11f7469a9921ed9dd0bdca201d7531560402fefef512c8cfe9d`).

1. **delivered-after-reset** (kernel, porting order): the production
   port returned `acc_d[dest_virtual_node]` AFTER the reset epilogue,
   so `delivered` was always 0.0 on the full-pipeline path. Outputs
   were bitwise fine (the accumulator at the destination virtual is
   terminal in the reversed tree pass and invisible to all three
   output arrays), but the driver's delivery-gap accounting
   (line 1661) would have counted every fast-route OD as undelivered.
   The admission probe's prototype did not have this defect (it read
   the accumulator before its reset); the defect was introduced by
   the port and caught by the FIRST battery execution
   (17:47:45Z, `m_base delivered_equal False` with all output arrays
   equal). Fix: capture at line 712 before the epilogue. The
   micro battery exists precisely to catch this class.
2. **precondition bounds** (`_gradient_slices_strictly_ascending`):
   `ok[starts - 1] = True` raised IndexError when an interior slice
   start equals `len(g_nodes)` (empty final slice — no left pair
   inside the array). Caught by the random differential trial in the
   same fire. Fix: the filter at line 435 plus the case's semantics
   (an empty final slice's left pair is a WITHIN-slice pair of the
   previous slice and must stay checked).
3. **fixture `limit=None`**: scipy 1.17.1 `dijkstra` rejects
   `limit=None` (collection error, 17:47:27Z, zero tests executed).
   Fixture fixed to omit the kwarg when None. No production impact.
4. **lost test binding**: `assert_engine_bytes` unqualified after the
   import-cleanup pass (driver fires 17:52:13Z). Fixed by qualifying
   through `fixtures_f2`. No production impact.

No defect touched the baseline kernel or the turns pipeline (their
bytes are unchanged; the F1/F3 suites in §10 re-ran green against the
candidate module).

## 10. Regression evidence

Runs against the F2-candidate module, `venvs/campaign` interpreter,
REV-9 windows, lease-logged (windows 12–14):

* **F2 suite**: 15/15 passed (17:58:16Z, full suite in one process).
* **F1 suite**: 28 passed / 2 failed (17:58:29Z). The 2 failures are
  F1-round MODULE-IDENTITY instruments vs the frozen pre-F1 B0
  reference and are superseded BY DESIGN by the additive F2
  insertion: `test_t7_targetoptions_census` fails with symmetric
  difference EXACTLY `{'_accumulate_od_flow_local'}` (the one new
  dispatcher — fresh tool output), and `test_t8_source_diff_single_
  decorator_line` fails because the candidate now differs from B0 by
  the F1 decorator line PLUS the F2 insertion. Every F1 BEHAVIORAL
  suite passes: kernel differentials, stripes, turns, drivers,
  observed, specializations (`t7_specialization_sets_equal` — the
  pre-existing kernels' specialization sets are unchanged), nrt.
  F1 test bytes were not modified.
* **F3 suite**: 22/22 passed (17:59:45Z) — the gradient workspace
  region this diff sits adjacent to.
* Regression surface completeness: the diff touches only
  AggregateFlow.py; F1 + F3 + F2 suites cover it (A1/A2 target other
  modules).

CUSTODY DISCLOSURE (crossing, corrected in-session): the F1 and F3
harnesses flush on sessionfinish with no artifact-path override, so
those two regression runs overwrote the COMMITTED
`evidence/F1I/test_artifacts.json` and
`evidence/F3I/test_artifacts.json`. The accidental post-run states
were preserved out-of-repo BEFORE restoration as
`campaign_data/interim_f1I_test_artifacts_a7fdd636.json` (sha256
`a7fdd636f54761c1a77975af8d3aa2fddca6b91a69671be7881bf71a080e4d45`,
6,150 B) and
`campaign_data/interim_f3I_test_artifacts_fe038582.json` (sha256
`fe038582271e4b80fd64b1c5bf4c63b228e98dd470ddc05e52fdfe09dc5b9c76`,
25,385 B) — these are the F2-candidate regression artifacts and are
offered as interim custody evidence of §10's runs. Both committed
files were then restored to HEAD bytes and verified:
F1I sha256 `00a5264d6962389159557ba2f50c165ee21b1b19705139986a08c960fb8001ed`,
F3I sha256 `ebfc94a87d8bdcfc49f8214f0fc6afe989cfcefc10309e1f123d56658560791e`
(matching HEAD). Porcelain at filing time contains no F1I/F3I entries.

## 11. Performance framing (binding condition g)

All admission figures (f2i_admission_125153: removable_wall_share_
ceiling 0.59286 in the measured cell FLOW_sel1024_od vs gate 0.05;
chosen-cell context 0.05478) are REMOVABILITY evidence — they
establish that the fixed per-OD full-V′ work the F2 route removes is
real and dominant (attribution: scan2 + alloc4 = 92.1% of the median
baseline kernel wall). The single-thread per-OD wall figures (median
t_base 48,417 ns) are NOT transferable per-call cost estimates: the
flagged 27x session kernel-mean discrepancy (single-thread session
kernel means vs the campaign cell's kernel mean) stands, and F2R's
complete-job screen is the instrument that resolves realized speedup.
No performance claim in this filing extends beyond removability.

Prose-grade note (no verdict impact), carried from the admission
ruling: the strict no-model bound is
min(f × kernel_cpu, stage_wall) / window = 0.6581736602179776; the
admission's 0.5928632945308131 additionally assumes proportional
transfer and is a conservative ESTIMATE, not an assured bound.

## 12. Filing and custody state

* Custody copies (out-of-repo, campaign_data root, byte copies of the
  h04-verified pre-amendment states, taken BEFORE regeneration):
  `interim_admission_42f48906.json` (sha256 `42f48906…`, 13,799 B)
  and `interim_result_8c016c0c.json` (sha256 `8c016c0c…`, 1,686 B).
* admission.json carries the sole smoke-row fold-in (path
  `campaign_data/f2i_probes/smoke_f2i.py`, sha256 `010da3d5…`,
  6,342 B, role exactly as ruled) + refreshed lease pin; result.json
  carries the two custody-ref rows + refreshed lease pin — via the
  AMENDED generator (supersedes `make_admission.py` 27bac4c6…;
  probe-era prose and date pin preserved byte-identical,
  supersession carried by the custody pins). Regenerated finals
  (two runs byte-identical, fresh tool output):
  admission.json sha256
  `16bbe0a3b4e72a3194b1e6c9daaf29bf6080a9e520888bb04560566c5341e4ad`
  (14,095 B); result.json sha256
  `d2ac05ef1cfb5b87eb6b46cc382f6f40e1d3015762885c62adc959d8f0fcd26a`
  (2,245 B). Exact-delta verification vs the interim custody copies:
  admission = {artifacts row added, lease pin refreshed}; result =
  {custody_copies added, lease pin refreshed}; NOTHING ELSE.
* Lease log at close of the implementation round:
  `campaign_data/f2i_lease_log.json` sha256
  `4ff95097663fd11f7469a9921ed9dd0bdca201d7531560402fefef512c8cfe9d`,
  15 windows (5 probe-era + 10 implementation-era: 2 crashed fires,
  2 failed driver intermediates, 1 defect-discovery fire, 3 green F2
  fires, 2 regression runs), all diagnostics-class, uncharged,
  gate_eligible false. The Q00 cold-variance prereg obligation rides
  to F2R.
* Census fold-ins riding to the successor manifest / F2R closeout:
  smoke_f2i.py (now also pinned in admission.json) and the
  `nbc_f2i_probe/` compiled-cache root (14 files).

## 13. Test inventory (tests/large_e2e/F2/)

| file | tests |
|---|---|
| conftest.py | dedicated NUMBA cache root (`nbc_f2`), candidate-root sys.path, oracle sys.path, sessionfinish harness flush |
| fixtures_f2.py | engine builders (+ `big_flow_spec`), MicroCase scratch contract, 14 micro cases, seeded fuzz generator |
| harness_f2.py | artifact records + first-divergence capture + flush (UNA_F2_ARTIFACTS override) |
| test_f2_kernel.py | 7 tests: micro battery, slice-covers-reach, early exit, q_sum adversarial, overflow full re-zero, cross-OD reuse, 24-seed fuzz |
| test_f2_precondition.py | 3 tests: 12 adversarial + 200 random + crossover |
| test_f2_driver.py | 5 tests: route toggle both sides, precondition fallback, exception safety, K=3 multithread, turns source pin |

15/15 green. All conditions (a)–(h) are discharged: (a) this
document; (b) §5; (c) §6; (d) §4; (e) §8; (f) §7; (g) §11; (h) lease
discipline throughout (15 windows), Q00 obligation riding to F2R.
