# Independent adversarial review — S10 sparse epoch projection

Reviewer: indep-s10 (independent of the S10 author). Worktree:
`/storage/scratch1/1/dsynn6/una-x/.claude/worktrees/cpu-kernels`.
Every check below was EXECUTED under
`/storage/home/hcoda1/1/dsynn6/micromamba/envs/alan/bin/python` with
`NUMBA_CACHE_DIR=/storage/scratch1/1/dsynn6/campaign_data/numba_cache_h01`.
Scripts and raw outputs live in this directory. No campaign file was
edited; the probe re-runs overwrote `sparse_admission.json` in place
(its designed behavior) and the fresh bytes are identical to the
committed receipt (snapshot preserved here as
`committed_sparse_admission.json`).

## VERDICT: ACCEPT_WITH_FINDINGS (2 MAJOR, 9 MINOR, 7 NOTE)

The projection theorem survives every hostile attempt to break it: on
all 138,165 probe cases (freshly reproduced), 19 new hostile cases, and
both observed comparison laws, the projection `K'` is bitwise-equal to
the frozen oracle `K` — including identical NONTERMINATION under a pop
cap. The two pre-existing engine findings F1/F2 reproduce byte-for-byte
from the audited worktree sources. However, the probe's shared-epoch
reconstruction deviates from the proof's own §4/T4 specification
(findable counterexample preserved), and the proof claims one check
that does not exist in the probe. Both are evidence defects, not
theorem breaks; both carry concrete S11 obligations listed at the end.

## Baseline and identity checks (all pass)

- sha256(probe) = `4fe2445a90662c35e58b45cced186bac78c78a6b90495d4bd83cb8edd739fd15`
  == `sparse_admission.json.probe_sha256` (recomputed, not copied).
- Findings artifacts re-hashed: all four sha256 values in
  `findings/manifest.json` match the bytes on disk.
- Authority sources byte-identical to the pin:
  `git diff 395cdc5..HEAD` empty for
  `src/urban_network_analysis/kernels/scope_access.py`,
  `kernels/SPEC.md`, `Engines/AccessibilityWElevation.py`,
  `Engines/_large_access_scratch.py`; `git status` shows `src/` clean.
  Import paths asserted at runtime to resolve into this worktree
  (`cpu-kernels` in `__file__`).

## CHECK A — receipt reproduction: PASS

Fresh full probe re-run: `passed: 138165 | failed: 0`, 44.55 s wall,
344 MB max RSS (`rerun_stdout.log`, `rerun_stderr.log`). The fresh
`sparse_admission.json` is byte-identical to the committed receipt
(counts e1_cases=138128, boundary=138155, coincide=138153, divergent=2,
a1_nan_divergences=1, e5_pump_pops=400, e9_cases=120, all finding tags
and hex bytes, probe_sha256). Count arithmetic independently
recomputed: 32+1024+131072 origin-pair x mask cases + 6000 random =
138128; +27 boundary = 138155; +e5(3)+e7(3)+e8(1)+e9(1)+e10(1) =
138165.

## CHECK B — probe-vs-proof fidelity

- B(i) Mirror faithfulness: `trace_schedule(stamp=False)` (probe
  L40-109) was diffed by text and AST against
  `kernels/scope_access.py::scope_search_schedule` (L304-373). Deltas
  are exactly: `float()`/`int()` value-equal casts, event appends,
  pop_cap guard, tuple return. Decisive gates verified present and in
  the same order in both: two-phase row scan, strict fast-law seed
  pushes END-THEN-START (probe L81-84 = oracle L339-343), eligibility
  `fast_le(c, cutoff) and fast_lt(c, L(v))` (probe L99 = oracle
  L359), leaf degree gate `> 1` (probe L106 = oracle L370),
  closed-radius `c <= cutoff`. Additionally the probe bitwise-checks
  mirror-vs-oracle AND Kp-vs-oracle on every case-matrix entry
  (probe L434-435) — 138,155/138,155 green in the reproduced run.
- B(ii) stamp=True deltas: only `L(v)` (probe L57-58) and `W(v,c)`
  (L60-65). No asymmetric pop_cap, no touched-order difference. BUT
  see F-MAJOR-1: the reconstruction consumer (`Kp_schedule` L122-126)
  deviates from proof §4/T4 in the shared-epoch regime.
- B(iii) §4 reconstruction form: `np.full(n, fl(1.0+cutoff))` +
  gather-over-touched matches the proof text on single-epoch arenas
  (the regime all 138,165 cases actually exercise).

## CHECK C — hostile execution: 19 new cases, ZERO projection!=oracle

Driver `hostile_cases.py` imports the probe via importlib and runs the
probe's own `case()` plus cap-bounded comparisons
(`hostile_results.json`, `hostile_stdout.log`). Hard acceptance
(projection bitwise == oracle, both laws, events NaN-canonically
equal) held on every case:

- h1/h1b dead-end (out-degree 0) nodes in and beside cycles, incl. a
  NaN candidate from the cycle — OK.
- h2/h2b cutoff=NaN (cap-bounded; admission refuses NaN cutoff) —
  identical event sequences. h2/h2b terminate (finite candidates bound
  writes); my initial always-true-gate conjecture was falsified by
  execution.
- h3/h3b negative cutoffs (incl. negative seeds entering the heap) —
  OK.
- h4/h4b int64 cutoff 2^63-1 with NaN seeds; int64 min — OK bitwise
  vs oracle AND vs a1 (h4 is law-divergent: fast/IEEE mirrors
  disagree, recorded, not dropped).
- h5/h5b duplicated arcs with NaN weights — h5 OK (law-divergent).
  h5b (NaN arc weight + NaN seed): see F-MINOR-2 — the oracle itself
  nonterminates; capped pop sequences of oracle and Kp are identical
  (both capped, events equal) = nontermination preserved.
- h6/h6b 1-node graph, self-loop, same-node NaN seeds — OK vs oracle;
  h6 is a NEW a1-divergent instance of the F2 same-node class (see
  F-MINOR-8); h6b (finite) fully clean.
- h7/h7b seeds exactly at/above cutoff with closed-radius row writes —
  OK.
- h8/h8b/h8c inf seed, +inf arc, -inf arc — OK.
- h9 a1 domain (d_count=4, domain='a1', destination tail): kp == oracle
  == a1 bitwise, tail `[4,8)` all-sentinel `0x4004000000000000` — the
  probe never exercises this domain (see F-MINOR-1).
- h10 dirty-arena continuation with a genuinely MID-ORIGIN injected
  exception (numpy subclass trap in the weights array): initial run
  reported divergences; attribution resolved below (one part harness
  bug, one part REAL probe defect F-MAJOR-1); with the proof-faithful
  reconstruction the full sequence incl. interrupt+continuation is
  bitwise clean (`final_checks.py`: `interrupted_fired=True bad=NONE`).
- h11 same-node NaN/NaN seeds — OK vs oracle (law-divergent, a1
  clean).

## CHECK D — epoch/lane claims (§5/§7 E7/E9)

See F-MAJOR-1, F-MINOR-4/5/6. The uint8 rollover test performs a manual
stamp reset at wrap (probe L224-225) and is green over 300 epochs, but
its fixture cannot detect a missing reset (no negative control). E9
uses per-rep fresh arenas at epoch=1 (probe L265-266), not "distinct
epochs per call" as §7 states; threads are GIL-serialized (weak but
honestly L0).

## CHECK E — admitted-domain narrowness

The current bytes make NO compact-parity claim on NaN cases: §6
(sparse_proof.md L228-234) scopes compact parity to the law-coinciding
domain and §7 (L252-256) says "no cross-law claim is made" on NaN-seed
cases. The early-draft defect the charter warned about is NOT present
in the current bytes. F1/F2 counterexample bytes in §6 match the
findings artifacts and the admission JSON exactly, and reproduce on
re-run.

## CHECK F — F1/F2 verification: PASS

`findings/_s10_nan_probe.py` re-run against the audited worktree src:
`oracle [nan, 0.6, nan] / engine [0.1, 0.6, nan]` and
`oracle [1.5, nan, 0.5] / engine [nan, 2.5, 0.5]` — hex bytes identical
to the committed `.out` (`f1_rerun.out`). `_s10_nan_probe2.py` re-run:
a1 matches the oracle bitwise on both e4-style NaN-seed cases and the
recorded F2 counterexample reproduces (oracle `[0.5, 1.0]` vs a1
`[0.5, nan]`; admission JSON `a1_nan_counterexamples` hex
`0x3fe0000000000000/0x3ff0000000000000` vs
`0x3fe0000000000000/0x7ff8000000000000`) — reproduced identically by
the fresh probe run. The compact engine's compiled NaN behavior is
exotic (the `[nan, 2.5, 0.5]` vector is not reachable by any
hand-derivable pop sequence under either scalar law) but is
deterministically reproducible, which is what the evidence needs.

# FINDINGS

## MAJOR

**M1. The probe's shared-epoch reconstruction violates the proof's own
§4/T4; real counterexample preserved.** `Kp_schedule`
(probe_sparse_projection_tiny.py L122-126) gathers over the arena's
CUMULATIVE `touched` list without filtering by `stamp == epoch`, while
proof §4 (sparse_proof.md L135-140) and T4 (L110-115) require exactly
the current epoch's touched set. Counterexample (executed,
`e7_gap.py`, `debug_h10.py`): shared arenas; epoch 1 origin (0,1)
touches node 0 (label 0.1); epoch 2 origin (2,2) on the DAG 0→1→2→3
does NOT touch nodes 0-1; probe returns
`[0.1, 0.6, 0.9, 0.35]` vs oracle `[2.5, 2.5, 0.9, 0.35]` — stale
epoch-1 labels copied into the fresh dense output in place of the
sentinel `fl(1.0+1.5)=2.5`. Not a theorem break (§4's mechanism with a
per-epoch touched set is sound): the stamp==epoch-filtered
reconstruction is bitwise-correct on the counterexample AND on a full
interrupt-mid-origin + dirty-arena continuation sequence
(`confirm_t4.py`, `final_checks.py`). The probe's E7 never sees this
because its fixture provably touches EVERY node on EVERY epoch
(`e7_gap.py`: final stamps `[9,9,9]`). All 138,165 passing cases are
single-epoch (fresh arenas) and unaffected. S11 RISK: an implementer
copying the probe's reconstruction verbatim ships a cross-origin
staleness bug in the shared-arena path.

**M2. The proof claims a check that does not exist.** §7 E1
(sparse_proof.md L266-268): "with full dense label reconstruction
compared at every trace boundary on the trace subset" — no such
comparison exists anywhere in the probe (grep: the only "boundary" in
the probe is the e2_e6_boundary function name). The probe compares one
final reconstruction per case. The claimed-but-absent check is exactly
of the class that would have caught M1. Every other §7 E-item I audited
does exist in code (E1-E10 all mapped to functions; see D for
weakenesses).

## MINOR

**m1. The A1 destination tail (domain='a1', d_count>0) is never
exercised by the probe.** Every `case()` call uses the default
a3/d_count=0 (grep: no `domain='a1'` call site). §3's tail sentence
("both return S there") rests on SPEC §2 plus tests/cpu_kernels, plus
my single hostile spot-check h9 (passed). The probe's L0 evidence does
not cover the label-domain size the A1 consumer actually allocates.

**m2. New nontermination class: NaN ARC weights.**
`isolate_h5b.py`: on 0→1 duplicated arcs with weights [NaN, 0.5, NaN,
0.5], seeds (NaN@0, 0.5@1), cutoff 1.5 — the frozen oracle AND the
staged a1 kernel do not return (>20 s, thread-watch); compact and the
IEEE mirrors return `[nan, 0.5]`. Capped comparison
(`hostile_results.json` h5b, both_capped=true): oracle and Kp pump
IDENTICALLY (events equal at cap 2000) — the theorem's
nontermination-preservation holds. But SPEC §3's NONTERMINATION LAW
text covers NaN on the seed SET, not NaN-arc pumps, and the probe's
`case()` path has NO pop cap: hostile input HANGS the probe (observed
>600 s twice) instead of failing loudly. Outside the admitted domain
(admission refuses NaN arc weights: scope_access.py L190), so no false
claim results — but the probe is not safe on hostile input, and the
E4 "candidates" wording (§7 L276-278) suggests NaN-candidate coverage
that is not there.

**m3. Event-trace equality is validated on 138,155/138,165 cases.**
`case()` compares events (probe L443-445); e7 (L200-233), e9
(L262-277), e10 (L280-293) compare vectors only (e10 discards
`ev_p`). The §1 GOAL "event traces identical element-for-element"
therefore holds as executed only on the case matrix + e5.

**m4. E7 "interrupted-origin (injected exception mid-origin)" is not
mid-origin.** `raise_after_pops` raises AFTER `trace_schedule`
returns, pre-reconstruction (probe L114-121, L208-212). The state
tested is post-search/pre-reconstruction, not mid-origin. I ran the
genuinely mid-origin variant (weights-array trap): with repair-A
reconstruction it is bitwise clean, so the epoch mechanism survives
the stronger test — but the probe does not run it.

**m5. E9 text vs code.** §7 L295-297 says "no cross-call stamping
(distinct epochs per call)"; the code gives every call epoch=1 on
distinct arenas (probe L265-266). Safety-equivalent (private arenas),
but the stated mechanism is not the implemented one. Threads are
GIL-serialized; no true parallelism is exercised (acceptable at L0,
worth stating).

**m6. E7 uint8 rollover has no negative control.** The fixture
resets stamps at wrap (probe L224-225) and compares 300 epochs
green, but a missing reset would also pass: after epoch 255 the
stamps hold 255, never equal to the reused epoch 1. The test pins
"reset+reuse works", not "stale stamps are caught".

**m7. Dangling evidence citation.** sparse_proof.md L5-7: "engine
sources at this worktree verified byte-identical to the pin; see
preflight in the run log" — neither `probe_run.log` nor
`probe_run5_dual_law.log` contains any preflight record. The
underlying claim is TRUE (I verified the pin myself), but the cited
artifact does not contain it.

**m8. The a1 NaN exception in code is broader than F2's stated
class.** `strict_a1` (probe L433) exempts ALL NaN-seed cases from the
hard a1 check; the proof (L248-251) excepts only "NaN-input same-node
seed collisions". Observed divergences are confined to the same-node
class so far (e2/same00/nan/0.5 and my new h6 1-node self-loop
instance), so the claim is empirically true — but a different-node
NaN-seed a1 divergence would be recorded, not failed. Also the
admission JSON records exactly ONE F2-class instance while the class
is demonstrably broader (h6 adds one).

**m9. Small text-vs-artifact imprecisions.** (a) §6 F1 (L205-207)
says "3-cycle 0→1→2→0, weights 0.5" — the findings probe's weights are
[0.5, 0.5, 1.0] (counterexample bytes themselves correct and
reproduced). (b) `findings/manifest.json` command
`python _s10_nan_probe.py` is not reproducible as written: the probe
inserts a nonexistent `HERE/src` and `urban_network_analysis` is not
importable from the findings directory (ModuleNotFoundError verified);
my re-runs required `PYTHONPATH=<worktree>/src`.

## NOTE

**n1.** The law-coincide gate compares the two PROJECTIONS
(`bit_eq(r['kpf'], r['ki'])`, probe L446) while §6 L229-232 says
"mirrors" — equivalent given the per-case mirror/oracle checks, but
the wording differs from the code.

**n2.** `e10_touched` in the receipt is `[0, 1, "2"]` — mixed int/str
(np.int64 → str via json default=str; 0 and 1 are the Python-int seed
writes). Cosmetic; a downstream equality check against `[0,1,2]`
would mismatch.

**n3.** E8 boundary set differs from §6 L192-194's list: the code
(probe L237) omits int64 1 and -1 and adds float cases (1.5, 0.0,
-0.0, nextafter(0.0,1.0), 1e308). Both forms still pinned equal on
everything tested.

**n4.** "138,165 cases" mixes granularities: e1's 138,128 are
individual graphs; e7's 300-iteration uint8 loop counts as 1 case;
e8 is a sentinel-form check, not a projection case.

**n5.** New divergence-class instances discovered by this review
(extend, do not contradict, F1/F2; recorded in
`hostile_results.json`): law-divergent h4 (int64max cutoff + NaN
seeds), h5 (NaN arc weights, finite seeds), h11 (same-node NaN/NaN);
a1-divergent h6 (1-node self-loop, same-node NaN — new F2-class
instance).

**n6.** Mirror-vs-oracle bitwise validation exists and runs on every
case-matrix entry (probe L434); e5/e7/e9/e10 validate Kp-vs-oracle
only (same body transitively).

**n7.** The engine NaN behaviors (F1 compact vs F2 a1 pop-order
inversion) are accepted here as REPRODUCIBLE BYTES, not from codegen
analysis; my hand-models of both engines' NaN semantics were falsified
by execution more than once. The evidence does not depend on the
mechanism being understood.

## Checks NOT run (unrun — not claimed)

- `tests/cpu_kernels/` conformance battery re-run (the §3 tail claim
  "pinned by SPEC §2 and the conformance tests" rests on it; I ran
  only my own h9 spot-check).
- Numba codegen-level explanation of the compact/a1 NaN laws.
- True multi-core execution for E9 (GIL-bound in-process threads).
- Direct observation of WHERE the first 600 s hostile run hung
  (output was buffered and lost); h5b was isolated deterministically
  afterwards and explains it.
- The probe under CPython 3.9 (a stale 3.9 `.pyc` exists; charter
  specifies the alan interpreter only).
- Any performance/timing measurement (S10 claims none; none made).

# THE GATE QUESTION

**Is the narrow admitted domain — projection ≡ frozen oracle
everywhere tested, engine authorities scoped per findings F1/F2 —
sound and adequately evidenced for implementation to proceed (S11)?**

**YES, with binding conditions.** The core claim executed and
reproduced: K' is bitwise-equal to the frozen oracle on all 138,165
probe cases (fresh run, receipt byte-identical), under BOTH observed
comparison laws, across 19 additional hostile cases including dead
ends, NaN/negative/int64-extreme/NaN cutoffs, NaN and infinite arc
weights, 1-node self-loops, seeds at/above cutoff, the A1 destination
tail, and IDENTICAL nontermination under a pop cap. F1/F2 reproduce
byte-for-byte from the audited pin, are correctly scoped in the
current proof bytes (no compact-parity claim on NaN cases anywhere in
§6/§7), and are not caused or masked by S10. Conditions S11 MUST
carry: (1) implement §4 reconstruction with the per-epoch touched set
(stamp==epoch filter or per-epoch count reset) — the probe's
unfiltered gather is defective (M1) and must not be copied; (2) add
the touched-then-untouched shared-epoch case and the a1-tail (V+d)
case to the S11 test plan, closing m1/M1's coverage holes; (3) treat
NaN-arc-weight nontermination (m2) as a known outside-admitted-domain
hazard and keep pop-cap protection in any hostile-domain probe; (4)
F1/F2 remain open dispositions owned by the SPEC/engine owners,
unchanged by S10. With those conditions the S10 correctness claim is
adequately evidenced; nothing in this review blocks S11.
