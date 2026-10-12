# S10 — Sparse epoch projection proof (dossier 02 over SPEC §3)

Status: ACCEPTED_WITH_FINDINGS — independent adversarial review
2026-10-03 (`review_indep/verdict.md`, `verdict.json`): the narrow
admitted domain is accepted for S11 implementation. Binding conditions
(per-epoch reconstruction filter, touched-then-untouched + A1-tail
tests, pop caps for hostile probing, engine findings stay with their
owners) are implemented in the probe and listed in §7/§8.
Baseline: `395cdc5f683894b6f2ba460f7dbcefee99981ba3` (engine sources
verified byte-identical to the pin by the independent review's own
git-diff check, `review_indep/`, and the H01 preflight receipts).
Kernel contract: `kernels/SPEC.md` §3 (frozen CPU_ALGORITHMS).
Task: prove that replacing the dense per-origin label initialization of
`scope_search_schedule` (SPEC §3 state `label` initialized to the
sentinel `S = fl(1.0 + cutoff)`) with epoch-stamped private arenas
preserves every observable of the schedule exactly, on the whole
admitted domain.

## 0. What is replaced, and what is not

Baseline cost structure (per origin): `label = np.ones(V(+d), float64) + cutoff`
allocates and writes the whole label domain before the first heap pop —
O(V) work per origin, H·V across a cohort of H origins.

Replacement (dossier 02 "State projection and schedule"): per-lane
private arenas

```
label[V(+d)] float64   stamp[V(+d)] epoch-width    touched[V] + count
row_offset[max_degree] int64   row_cost[max_degree] float64
```

with read projection `L(v) = label[v] if stamp[v] == e else S` and write
`W(v, c): if stamp[v] != e: touched[count++] = v; stamp[v] = e; label[v] = c`.

NOT replaced: the public `compact_vector_node_view_scope` full-array
return (its signature and bytes are pinned); the route ladder, admission
guards, heap semantics, fold/reduction plans, profiles, error contract.
The projection lives strictly inside a private consumer boundary
(`integrated_scope_access`'s per-origin loop) and every public array is
materialized by reconstruction (§4). This proof claims nothing for the
compact fallback route's code shape, only for its observable return
values, which §3 pins bitwise.

## 1. Definitions

- `K`: the SPEC §3 schedule with dense labels, as compiled in
  `Engines/AccessibilityWElevation.py::compact_vector_node_view_scope`
  (and its A1/A3 staged twins in `Engines/_large_access_scratch.py`,
  same schedule). Its portable conformance oracle is
  `kernels/scope_access.py::scope_search_schedule`, pinned bitwise to
  the compiled engines by `tests/cpu_kernels/` (frozen contract).
- `K'`: the same schedule with the label state held as stamp-projected
  arenas above. `K'` differs from `K` in exactly two places: (i) the
  value read for `label[v]` becomes `L(v)`; (ii) each assignment
  `label[v] = c` becomes `W(v, c)`. All heap operations, comparisons,
  row scans, gates, and their order are unchanged source text.
- `e`: the epoch — a positive integer unique to (call, lane, origin).
  First implementation: counter starts at 1 per call, +1 per origin,
  stamps zeroed once per call. `e` is never used as a float.
- `S`: the sentinel, the bit-exact value of `np.ones(n, dtype=w.dtype)
  + cutoff` for the admitted initializer types (float64 cutoff; int64
  cutoff under the engine's promotion — §6).
- Observation maps: for an execution of `K` with inputs `X`, the
  observable is the returned label array (`label_K`), plus (for the
  oracle only) the event trace. `K'`'s observable is the reconstructed
  array `label_K'` (§4) and its event trace. Theorem GOAL:
  `label_K'` is bitwise equal to `label_K` for every admitted `X`,
  and the event traces are identical element-for-element.

## 2. Lemmas

T1 (read-value equality). At every algorithmic step k of a paired
execution, `label_K[v] == L_K'(v)` for all v in the label domain.
Proof by induction over the (finite or nonterminating) step sequence:
- Before any write: `K` holds `S` everywhere (its only initialization);
  `K'` holds `stamp=0 != e`, so `L(v) = S`. Equal.
- Seeds: `K` executes `label[o_start] = w0; label[o_end] = w1`
  (unconditional, in this order). `K'` executes `W(o_start, w0);
  W(o_end, w1)` — same order; after both, `L(o_start) = w0`,
  `L(o_end) = w1`. Equal. (If `o_start == o_end` the second write wins
  in both; `W` re-stamps the same epoch without a second touched entry.)
- Inductive step: assume equality holds at the start of heap step k.
  The pop sequence is a function of the heap contents and the tuple
  order (both implementations push/pop through the same heap text with
  the same pinned tuple law), so the same `(weight, node)` pops in both.
  Phase-1 eligibility for offset i compares `c_i <= cutoff` and
  `c_i < L(v)` (K') against `c_i < label_K[v]` (K) with `c_i` computed
  by the same compiled add from the same row weights. By the induction
  hypothesis the compared values are bit-equal, and the comparison
  operators are the same compiled comparisons (same njit flags, same
  source expressions — the projection adds a branch, not arithmetic),
  so every eligibility decision and stored `(offset, c_i)` pair is
  identical, including all characterized NaN/signed-zero behaviors
  (SPEC §3 compiled laws). Phase-2 replays the identical pairs: `K`
  writes `label_K[v] = c_i`; `K'` executes `W(v, c_i)`, after which
  `L(v) = c_i`. Reads in later rows of the same pop and all later pops
  see equal values. ∎

T2 (write-sequence equality). The sequences of (v, c) label writes are
identical in both implementations. Direct from T1's inductive step
(identical eligible pairs, identical replay). The touched sequence of
`K'` is the subsequence of first writes per epoch: `W` appends v iff
`stamp[v] != e`, then sets it; within a lane the arena is private and
the lane is sequential, so no interleaving can append twice. ∎

T3 (heap-sequence equality). Every push/pop tuple sequence is
identical. Pushes are decided by the identical eligible pairs, the
identical flag column and the identical leaf-degree gate (plain integer
comparisons on the same CSR); pops are a function of heap contents and
the pinned tuple law. Hence the pop order, the tie permutation
(ascending node id at equal finite weights, NaN-first law), and any
nontermination are identical. ∎

T4 (reconstruction soundness/completeness). At return:
`v in touched_e  iff  stamp[v] == e  iff  K' wrote v this epoch`
`iff K wrote v this epoch` (T2). Completeness of touched: every v with
`label_K[v] != S` was written by `K` this epoch (labels start at S and
only writes change them), hence is touched. Soundness: touched v were
written by `K` too. ∎

## 3. Theorem (bitwise output equality)

`label_K' == label_K` bitwise on the whole admitted domain.
For touched v: both hold the last `c` of the identical write sequences
(T1, T2 — copies, no arithmetic). For untouched v: `K` holds `S`;
`label_K'`'s reconstruction assigns exactly `S` (§4 — the same bit
value by construction). The A1 destination tail `[V, V+d)` is admitted
only for in-range terminals (typed guard), so it is never written by
either implementation (pinned by SPEC §2 and the conformance tests);
both return `S` there. ∎

Corollary (downstream equality): `adjust_destination_distances` and
every fold/reduction of SPEC §4–§5 consume the label array; they
receive bitwise-equal input from both implementations and their plans
are unchanged, so all four per-origin outputs and the assembled (o,)
arrays are bitwise equal. Route attribution is unchanged (admission is
input-only, §6).

## 4. Reconstruction (the private consumer's materialization)

```
dense = np.full(domain_size, S, dtype=float64)
dense[touched[:count]] = label[touched[:count]]
```

Touched here means the PER-EPOCH set of §1/T4 — `{v : stamp[v] == e}`.
An implementation must filter by the epoch stamp (or reset touched per
epoch); gathering a cumulative touched list without the stamp filter
copies stale labels from earlier epochs into untouched slots (review
finding M1 — the pre-review probe had exactly this defect; regression
pinned by probe E7b).

`np.full` broadcasts the scalar `S = fl(1.0 + cutoff)` — the same bit
value `np.ones(n) + cutoff` produces (single IEEE add, no reassociation
class exists for one add; both initializer types pinned in §6). The
gather is a copy of bit patterns in ascending touched order; positions
not covered keep `S`. Cost O(count + domain). Reconstruction happens
outside any timing-relevant claim change — S10 charges it in the S11+
gate like any other stage (it replaces the baseline's per-origin O(V)
init; net effect is measured, not assumed).

## 5. Epoch, lane, and lifetime discipline

- Epochs per call: 1, 2, ..., H. int64 counter; overflow requires
  H ≥ 2^63 origins — unreachable (inputs are memory-bounded); the
  counter still asserts `e >= 1` and `< 2^63` (checked, per dossier).
  Stamps zeroed once per call — O(V(+d)) per CALL, not per origin.
- Lane exclusivity: arenas are per-lane allocations; no lane reads or
  writes another's; lane ID is not a thread ID (prange over logical
  lanes in S11; the sequential oracle here is the 1-lane case). No
  global arena is shared across concurrent calls (first implementation
  allocates per call).
- Exception/lease: arenas live only for the call; any exception
  abandons them; the next caller sees pristine zeroed stamps. A thrown
  exception can never leak a stamped arena into another call (T6 test).
- Persistent arenas (S11+ evolution): NOT claimed by this proof. When
  introduced they require exclusive leases, reset stamps and
  destination marks BEFORE counter rollover with no active reader, and
  the reduced-width exhaustion test (§7 E7) pins the rollover
  semantics. Until then, rollover is unreachable by construction.
- Cost charged (S10 honesty): stamp zeroing V(+d) per call, touched
  bookkeeping per write, reconstruction O(count + domain) per origin,
  arena allocation per call. Nothing is hidden; the S11+ benchmarks
  include them (VALIDATION: "Charge initialization H*V and lifetime
  overhead in timing and memory").

## 6. Admission, initializer types, sentinel bits

- The typed guard is unchanged: `K'` admits exactly `K`'s signatures
  (same dtypes/shapes/lengths/native-order/layout checks, same CSR
  invariants, same terminal bounds) and refuses the same inputs, before
  any allocation. The projection introduces no new acceptance.
- Sentinel for float64 cutoff: `S = fl(1.0) + cutoff` — bit-identical
  to the kernel's `np.ones(n, float64) + cutoff` (same single add).
- Sentinel for int64 cutoff (engine admits it via the compile-time
  gate): the kernel's expression promotes `cutoff` to float64 exactly
  as the scalar add does; `np.ones(n, float64) + int64_scalar` and
  `np.full(n, fl(1.0) + float64(int64_cutoff))` produce the same bits
  for every int64 value with an exact float64 image, and for large
  int64 values the SAME promotion applies in both expressions (the
  oracle computes `S` by the kernel's own expression form — never by a
  re-derived one). Test E8 pins bitwise equality of the two forms over
  an int64 boundary set (0, 1, -1, 2^53, 2^53+1, 2^62, int64 max,
  int64 min — noting the kernel's own overflow-reach behavior is
  inherited, not fixed).
- Domain: finite and non-finite admitted bits alike (the legacy warts
  are pinned, §7 E5/E6). No new restriction is introduced by S10; the
  sparse CONSUMER's assumptions (touched ⇒ within closed radius) hold
  because every write is gated by `c_i <= cutoff` or is a seed —
  seed labels are not radius-filtered in `K` either (unconditional
  writes; the adjust/fold membership does its own filtering).
- Finding F1 (pre-existing compact-fallback NaN law; NOT introduced or
  masked by S10): with NaN seed weights, the compiled
  `compact_vector_node_view_scope` gates act IEEE-ordered
  (`nan < cutoff` false, finite `c < nan` false) and its outputs
  diverge from the frozen schedule. Counterexamples (3-cycle
  0→1→2→0, weights 0.5, cutoff 1.5): seeds (0.1, NaN) → oracle
  [NaN, 0.6, NaN] vs compact [0.1, 0.6, NaN]; seeds (NaN, 0.5) →
  oracle [1.5, NaN, 0.5] vs compact [NaN, 2.5, 0.5] (hex-dump probes
  preserved under `findings/`, tags `e4/warts/0.1/nan`,
  `e4/warts/nan/0.5` in `sparse_admission.json`). The STAGED
  production kernel `_a1_scope_search` matches the oracle bitwise on
  both cases (the fast law holds on the production route). SPEC §3's
  blanket compiled-law sentence therefore holds for the staged kernels
  but NOT for the compact fallback; disposition belongs to the SPEC /
  CPU_ALGORITHMS owners — recorded here, not patched.
- Finding F2 (pre-existing staged-kernel same-node seed collision; NOT
  introduced by S10): with BOTH seeds on the SAME node and a NaN seed
  weight, `_a1_scope_search` diverges from the frozen oracle AND the
  compact kernel. Counterexample (2-cycle 0→1→0, weights 0.5, cutoff
  1.5, seeds (NaN, 0.5) at node 0): oracle label bytes
  [0x3fe0000000000000, 0x3ff0000000000000] (= [0.5, 1.0]) vs
  `_a1_scope_search` [0x3fe0000000000000, 0x7ff8000000000000]
  (= [0.5, NaN]) — tag `e2/same00/nan/0.5`, exact bytes in
  `sparse_admission.json` `findings.a1_nan_counterexamples`. Its
  seed-write collision resolution differs from the schedule's
  unconditional start-then-end writes; NaN-free same-node seeds agree
  bitwise. Same disposition as F1: engine owners, not S10.
- Finding F3 (new nontermination class, found by the independent
  review; recorded, NOT fixed here): NaN ARC WEIGHTS — not seeds —
  pump the frozen oracle AND the staged A1 kernel indefinitely, while
  the compact/IEEE route terminates; SPEC §3's nontermination
  characterization (NaN on both seeds) does not cover this class. The
  probe never invokes uncapped kernels on NaN-arc inputs (docstring
  warning); hostile-case evidence: `review_indep/hostile_*`.
- S10 claim domain (narrow, admitted): `K'` ≡ oracle (events, bitwise)
  on EVERY tested case (138,165/138,165); ≡ `_a1_scope_search` bitwise
  except F2 tags; ≡ compact bitwise on the law-coinciding domain —
  every case where the fast and IEEE mirrors return bitwise-equal
  vectors (138,153/138,155 case matrix entries; 2 tagged divergent).
  Law-divergent cases are counted and tagged in the admission JSON,
  never silently dropped.

## 7. Tiny-test plan (L0; probe `probe_sparse_projection_tiny.py`)

Each test evaluates the projection oracle `K'` (exact schedule mirror,
only the two deltas of §0) under BOTH observed comparison laws of the
shipped engines, each against its own byte-exact schedule mirror —
the fast-law mirror and an IEEE-law mirror (identical text, plain
`<`/`<=`), with event traces compared NaN-canonically (IEEE tuple
equality treats NaN as unequal to itself). Authorities per case:

- frozen portable oracle `K` (`kernels/scope_access.py`) — fast-law
  bitwise + event authority on EVERY case;
- staged production kernel `_a1_scope_search`
  (`Engines/_large_access_scratch.py`) — fast-law bitwise authority,
  NaN-input same-node seed collisions excepted (finding F2, §6: those
  tags are recorded, and on NaN inputs the gate degrades to a recorded
  divergence instead of a hard failure);
- compiled fallback kernel `compact_vector_node_view_scope` — bitwise
  authority ONLY on the law-coinciding domain (cases where the two
  mirrors return bitwise-equal vectors — every finite-input case
  tested); on NaN-seed cases its IEEE-ordered gate law diverges from
  the frozen schedule (finding F1, §6) and no cross-law claim is made.

The uncapped kernels are never invoked on the nontermination probe
(E5); capped mirrors carry that comparison. VALIDATION.md sparse list
coverage:

- E1 exhaustive structures: n ∈ {2,3,4}, every arc-subset of the
  complete directed multigraph with replacement (self-loops, parallel
  arcs), fixed cycled weights + seeded random weights (20k samples);
  final vectors bitwise vs engine AND vs oracle; event traces equal on
  EVERY case; full dense label reconstruction compared at EVERY
  chronological pop boundary by the E11 lockstep sweep (bounded
  subset — repair of review finding M2).
- E2 seed laws: same-node start==end (equal and unequal weights — end
  write wins), seeds at/above cutoff (strict `<` gates), nextafter
  cutoff boundaries both sides, negative-zero weights retained as
  non-negative bits.
- E3 boundary values: costs/cutoffs from {0, -0.0, subnormal, nextafter
  around eligibility edges, max float}; strict `<` vs closed `<=`
  asymmetry pinned per SPEC §3 step 4/6.
- E4 legacy warts (terminating subset): NaN/negative/inf seeds and
  candidates — `nan_seed_end` and single-seed NaN stay terminating and
  bitwise; negative seeds enter when `< cutoff`.
- E5 nontermination preserved: NaN on both seeds — bounded-pop-cap
  oracle comparison reproduces the pinned oscillation for BOTH `K` and
  `K'` (identical pop sequences under the cap). The compiled engine is
  NOT run on this case (pinned NONTERMINATION law; excluded from the
  shared battery by CPU_ALGORITHMS).
- E6 duplicate/self-loop semantics: equal-weight parallel arcs —
  first-in-row wins the snapshot; self-loop rewrite/push gates.
- E7 epoch discipline: H consecutive origins on shared arenas (epochs
  1..H, no reset between origins) — per-origin outputs bitwise equal to
  fresh-arena runs; interrupted-origin (injected exception mid-origin)
  then fresh call — pristine outputs; reduced-width counter (uint8)
  exhaustion — wrap resets stamps before counter rollover, no
  stale-stamp read (semantics pin for future persistent arenas).
- E8 sentinel/init equality: `np.full(n, fl(1.0)+cutoff)` vs
  `np.ones(n)+cutoff` bitwise over float64 and int64 cutoff boundary
  sets (§6).
- E9 concurrency: two concurrent calls (threads) over interleaved
  random graphs, per-call private arenas (fresh allocations, default
  epoch) — outputs bitwise equal to sequential; no cross-call state.
- E10 untouched reconstruction: graphs with most nodes untouched —
  reconstruction equals the engine's full array bitwise, including
  untouched seeds-adjacent nodes and the A1 tail when present.
- E7b touched-then-untouched epochs on SHARED arenas, plus dirty-arena
  continuation after a mid-origin interrupt (repair of review finding
  M1: reconstruction gathers the stamp==epoch set, never the
  cumulative touched list).
- E11 lockstep boundary sweep: one schedule execution advancing BOTH
  mirrors; at every pop boundary the epoch reconstruction must be
  bitwise-equal to the plain label array, and the lockstep mirror
  itself is pinned against the frozen oracle per case (196-case
  bounded subset; repair of review finding M2).
- E12 A1 reduced-domain tail (d_count > 0): four origins across the
  full authority matrix on domain 'a1' (review MINOR: the tail was
  previously unexercised; reviewer spot-check h9 passed).

Acceptance: all E1–E12 green under the alan interpreter. Independent
review reproduction: 138,165 passed / 0 failed in 44.55 s with a
byte-identical receipt (`review_indep/committed_sparse_admission.json`).
Post-repair run (M1/M2 repairs + E7b/E11/E12 implemented): 138,368
passed / 0 failed in 42.9 s — log `probe_run6_postrepair.log`; the
admission JSON `probe_sha256` pins the exact repaired probe bytes. The
run emits the executed matrix, law-coincide/diverge counts, and the
recorded findings (tags + counterexample bytes). The typed refusal
matrix is unchanged (S10 adds no acceptance; `scope_route_admits`
behavior is pinned by the CPU_ALGORITHMS battery, not re-tested here).

## 8. What this proof does NOT claim

- No public dense-view replacement (dossier 02 scope guard).
- No fix, patch, or disposition of findings F1/F2 (§6) is claimed or
  performed here: they are pre-existing engine characteristics,
  preserved with counterexample bytes for the SPEC and engine owners;
  S10 neither causes nor conceals them.
- No performance claim of any size — S10 is correctness-only; the 10x
  arithmetic is measured later through the public gate with all §5
  costs charged.
- No persistent-arena, cancellation, or cross-call residency semantics
  (those arrive with S11 and carry their own proof obligations).
- No reduction/fold change; profiles stay split per SPEC §0.
- No claim on non-admitted inputs beyond unchanged typed refusal.
