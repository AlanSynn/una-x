# A1 proof: snapshot-preserving search scratch (pre-implementation, for independent review)

Task A1I. Author: implementation-owner. Source identity at proof time:
worktree `/Users/alansynn/orca/workspaces/una-x/wt-large-e2e`, branch
`perf/una-large-e2e`, HEAD `6205086ef64773440c4f705bee2139a4c40b49a5`,
`git rev-parse HEAD:src` = `a5883dddcef3afb8debdb4438a6338da886add6f`
(the B0 `361928e` reviewed source tree), working tree clean. Admission
recheck: `admission.json` (all H05 A1 facts re-verified at this source).

Everything below is written against the ACTUAL source lines cited, not
a mental model. Dossier: `dossiers/A1_snapshot_scratch.md`. Reviewer
approval of THIS document is required before implementation starts.

---

## 0. Scope

Writable: `src/urban_network_analysis/Engines/Accessibility.py`,
`src/urban_network_analysis/Engines/AccessibilityWElevation.py`,
new `src/urban_network_analysis/Engines/_large_access_*.py`,
`tests/large_e2e/A1/**`, `campaigns/una_large_e2e/evidence/A1I/**`.
Read-only: everything else, including `_ordered_csr.py` (never
modified), the oracle package (imported, not changed), h05 instrument,
evidence of other tasks. NO git writes.

The dossier names the kernel `compact_vector_node_view_scope` in both
engine files as the target. Both jitted drivers
`od_compact_vector_node_view_scope` (`Accessibility.py:207-242`) and
`integrated_scope_access` (`Accessibility.py:251-327`; mirror line
numbers in `AccessibilityWElevation.py`: kernel 106-159, od driver
203-238, integrated driver 247-312) are the kernel's only callers, so
the private route is dispatched from the drivers, per origin, with the
guard evaluated once per driver call. Both drivers get the identical
treatment; the observed O2 fixture records cover both
(`observed_o2_kernels.hashes.json` calls keys
`integrated_scope_access` (2 records, module AccessibilityWElevation)
and `od_compact_vector_node_view_scope`).

Public surfaces, defaults, warning/exception behavior, result-array
ownership, and the original kernels/functions are unchanged. The
original kernel `compact_vector_node_view_scope` stays in both files
BYTE-IDENTICAL as reference and fallback.

## 1. Baseline transition (actual code, `Accessibility.py:143-161`;
`AccessibilityWElevation.py:139-157` is textually identical apart from
a 4-line offset)

```python
while queue:                                                                                    # L143
    weight, node = heappop(queue)                                                               # L144
    node_start_pointer = adjacency_pointer[node]                                                # L146
    node_end_pointer = adjacency_pointer[node + 1]                                              # L147
    weights_neighbors = adjacency_vector_weights[node_start_pointer:node_end_pointer] + weight   # L148

    queue_neighbors = np.nonzero(                                                                # L150
        (weights_neighbors <= cutoff)                                                            # L151
        & (weights_neighbors < o_scope_weights[adjacency_vector[node_start_pointer:node_end_pointer]])  # L152
    )[0]

    for i in queue_neighbors:                                                                    # L155
        neighbor_weight = weights_neighbors[i]
        neighbor_node = adjacency_vector[node_start_pointer + i]
        o_scope_weights[neighbor_node] = neighbor_weight                                         # L158
        if adjacynct_vector_network_node[node_start_pointer + i]:                                # L159
            if adjacency_pointer[neighbor_node + 1] - adjacency_pointer[neighbor_node] > 1:      # L160
                heappush(queue, (neighbor_weight, neighbor_node))                                # L161
```

Prologue (L121-140, kept byte-identical): `nd_node_count = d_count +
adjacency_pointer.shape[0] - 1`; labels
`o_scope_weights = np.ones(nd_node_count, dtype=o_terminal_weights.dtype) + cutoff`;
`o_scope_pred = np.empty(0, dtype=o_terminal_idxs.dtype)`; unconditional
terminal assignments `S[o_idx_start]=w_start` then
`S[o_idx_end]=w_end` (that order); seed `queue=[(w_start, o_start)]`;
seed popped AND DISCARDED; then `if w_end < cutoff: push (w_end,
o_idx_end)`; `if w_start < cutoff: push (w_start, o_idx_start)`.
Return `(o_scope_weights, o_scope_pred)`.

Observation semantics pinned by the baseline:

* **Row snapshot**: L152 gathers `o_scope_weights` at the row's
  neighbor ids into a temporary BEFORE any write of this row, so every
  eligibility test compares against the pre-row label vector S0. With
  duplicate destinations in one row (two offsets to the same node),
  BOTH tests read the same S0[v]; phase two then writes in increasing
  offset order, so the later (possibly larger) value wins. This
  staging is exactly what `trace_access.trace_scope` journals and what
  the H03 mutants `mutant_scope_immediate_update` /
  `mutant_scope_terminal_order` prove discriminative.
* **Eligibility**: `c_i <= R AND c_i < S0[neighbor_i]`, with `c_i =
  weights[p0+i] + w` (one float64 add per incidence), evaluated for ALL
  offsets of the row before any assignment; surviving offsets are
  visited in increasing row order.
* **Push rule**: push `(c_i, neighbor_i)` iff the incidence's
  network-node flag is true AND `deg(neighbor_i) > 1`. Degree-one
  nodes are never enqueued; non-network (False flag) incidences are
  never enqueued.
* **Stale entries**: a popped `(w,u)` with `w > S[u]` is re-scanned
  normally; its row eligibility is tested against CURRENT labels
  (which the row snapshot makes pre-row values for this row only).
* **No validation, no early exits**: the kernel has no guards and no
  early return; every conditional is listed above (two prologue pushes,
  the while condition, the row eligibility, the two push conditions).
  Numba boundscheck is OFF, so out-of-domain indices are UB, not
  exceptions. The public entries raise only where B0 raises
  (`Accessibility.Centrality` L594-595 `ValueError` when the graph
  engine is missing; `OD_Matrix` L649-650 likewise). The candidate
  adds no new exception, warning, or early exit on any input.

Per-pop allocations (the A1 target, verified in compiled IR:
12 dynamic NRT alloc sites in the kernel,
`evidence/A1I/ir_callsites_recheck.json`): L148 row add (1), L152
adjacency gather (1) + label gather (1), L151/L152 two compare bools (2)
+ `&` result (1), L150 nonzero output (1+); plus per call: L127 labels
(V+D), L128 empty pred. At the observed O2 scale this measures as
9,563,127 NRT allocations per single-origin integrated call, all freed
in-call (`memory_lifetimes.json runtime_nrt_evidence`).

## 2. Candidate transition (dossier "Proposed algorithm", compiled form)

New shared helper module `src/urban_network_analysis/Engines/_large_access_scratch.py`
(follows the `_ordered_csr` precedent of a shared engine helper; NUMBA
flag constants identical to the engine files: `cache=True, nogil=True,
fastmath=True`, and `parallel=False` — the search kernel is called
inside the drivers' `prange` exactly like the original kernel, whose
`parallel` is `NUMBA_PARALLEL=False`).

Scratch search kernel `_a1_scope_search(...)`: the prologue of
Section 1 byte-identical; then

```
while queue:
    weight, node = heappop(queue)                     # identical
    p0 = adjacency_pointer[node]; p1 = adjacency_pointer[node + 1]
    n_eligible = 0
    for i in range(p1 - p0):                          # phase one: NO label writes
        c = adjacency_vector_weights[p0 + i] + weight # scalar typed add
        v = adjacency_vector[p0 + i]
        if c <= cutoff and c < o_scope_weights[v]:    # scalar compares vs pre-row S
            eligible_offset[n_eligible] = p0 + i      # absolute adjacency offset
            eligible_weight[n_eligible] = c
            n_eligible += 1
    for j in range(n_eligible):                       # phase two: NO eligibility recomputation
        off = eligible_offset[j]; c = eligible_weight[j]
        v = adjacency_vector[off]
        o_scope_weights[v] = c
        if adjacynct_vector_network_node[off]:
            if adjacency_pointer[v + 1] - adjacency_pointer[v] > 1:
                heappush(queue, (c, v))
return o_scope_weights, o_scope_pred                  # identical shapes and values
```

Design decisions within the dossier text (each is a choice the reviewer
should confirm):

* D1. `eligible_offset` stores the ABSOLUTE adjacency offset
  (`p0+i`); the row-local offset is `off - p0`. The dossier's
  `original_neighbor_at(i)`/`original_is_network(i)` notation indexes
  the global adjacency arrays, so absolute offsets avoid any phase-two
  address recomputation beyond a load. Row-local values remain
  derivable for journal comparison (`off - p0`).
* D2. Phase one short-circuits the second comparison when
  `c <= cutoff` is false (scalar `and`). The skipped operation is a
  pure in-bounds array read (domain: guard section 3), so the eligible
  set and stored values are identical to the baseline's materialized
  bool arrays.
* D3. Scratch is TWO buffers `eligible_offset:int64[max_degree]`,
  `eligible_weight:float64[max_degree]`, allocated per origin INSIDE
  each `prange` iteration body. "Every independent origin/search owns
  its buffers; no module-global or shared Numba array" is satisfied in
  the strongest form: no two searches, concurrent or sequential, ever
  share a buffer; nothing survives the iteration. (A per-thread
  slotted table was rejected: it would introduce thread-identity
  assumptions numba's scheduler does not document, for no measurable
  gain — the hot path is per-pop, and per-origin cost is 2 small
  allocations against the retained per-origin V+D label array that the
  dossier requires to stay.)
* D4. `max_degree = max over v in [0,V) of pointer[v+1]-pointer[v]`
  is computed ONCE per driver call from the unchanged pointer array,
  inside the same pass as the guard, before `prange`. Row length is
  always <= max_degree because the guard verifies the pointer array
  monotone (section 3), so scratch indexing stays in-bounds.
* D5. Capacity bound: with `H = nb.get_num_threads()` (queried once in
  the driver prologue), peak live scratch is `H * max_degree * 16`
  bytes. If that exceeds `_A1_SCRATCH_CAP_BYTES = 268435456` (256 MiB;
  1/4 of the 1 GiB pressure-stop threshold, 1/40 of the 10.5 GiB
  campaign ceiling, H00 resources.json) the guard refuses and the
  original path runs. At the observed scale this is a few KB, so the
  bound only fires on pathological high-degree graphs — it exists to
  honor "excessive scratch capacity use original behavior" and "bound
  multiplied active-search scratch under RESOURCES".
* D6. Both drivers dispatch. The original prange loops in both drivers
  remain in the file, textually unchanged, as the fallback branch.

## 3. Admitted domain and refusals (dossier Admission)

One guard, `_a1_scope_admits(...)`, jitted, executed ONCE per driver
call (charged to the full job; "do it once per graph ... not every
pop"), before any allocation beyond the driver's own output arrays. It
performs only shape/dtype equality checks and in-bounds reads; it
allocates nothing, casts nothing, warns nothing, raises nothing. On
ANY failed condition it returns false and the driver executes the
ORIGINAL code path — the same compiled functions B0 runs — so every
refusal is byte-identical to B0 by construction.

Refuse (fallback) unless ALL hold:

1. `adjacency_pointer`: ndim 1, dtype int64, length >= 1.
2. `adjacency_vector`: ndim 1, dtype int64.
3. `adjacency_vector_weights`: ndim 1, dtype float64.
4. `adjacynct_vector_network_node`: ndim 1, dtype bool.
5. `o_terminal_idxs`: ndim 2, dtype int64, shape[1] == 2.
6. `o_terminal_weights`: ndim 2, dtype float64, shape ==
   o_terminal_idxs.shape.
7. `cutoff`: finite and >= 0 (`cutoff == cutoff`, `cutoff < inf`,
   `cutoff >= 0`).
8. CSR validity: `pointer[0] == 0`; `pointer` nondecreasing;
   `pointer[V] == len(adjacency_vector)` where `V = len(pointer)-1`.
9. Endpoint validity: every `adjacency_vector[j]` in `[0, V)`.
10. Cost domain: every `adjacency_vector_weights[j]` finite and >= 0
    (signed zero admitted; -0.0 >= 0 is true).
11. Origin terminals: every `o_terminal_idxs[o,k]` in `[0, V)`.
12. Scratch capacity (D5).

Not guarded (and why): `d_terminal_idxs/d_terminal_weights/d_weights`
feed ONLY `adjust_destination_distances` and
`reach_gravity_knn_access`, which the candidate does not modify and
which run identically on both routes; `o_terminal_weights` values
(only dtype/shape guarded) enter labels by direct assignment in
identical prologue code — no candidate arithmetic differs; the math/
ISA profile is fixed by construction (same module flag constants, same
frozen toolchain, verified by compiled behavioral tests and IR dump at
A1R — section 7). The transition proof of section 5 is in fact
independent of these value domains; the guarded domain is the
dossier's admission contract for the private route.

## 4. Snapshot, lifetime and cleanup semantics

* Scratch holds ONLY row-local eligibility results. Phase one reads
  labels and writes scratch (disjoint storage); phase two reads scratch
  and writes labels/queue. So every label read in phase one is a
  pre-row S0 value — the same values L152's gather materializes in B0.
* Per-origin allocation (D3) gives: first-call == every-later-call ==
  any-call-in-any-thread-state, because there is no state to reuse:
  buffers are freshly allocated (np.empty, contents written before any
  read: only indices `[0, n_eligible)` are ever read back) and die
  with the iteration. Results are bit-identical regardless of scratch
  reuse history — pinned by test T12 (origin permutation + repetition).
* No module-global or shared Numba array exists in the new module; no
  buffer is returned or retained; outputs are freshly allocated per
  call exactly as B0 (no aliasing of outputs with inputs or scratch —
  pinned by T2).
* Cleanup obligations ("Private scratch cleanup on early return/error
  must prevent contamination of future calls"): there are no early
  returns or caught errors on the private route. NRT frees the
  per-iteration buffers by refcount when the iteration exits, including
  on exception unwind; no cleanup action is required or possible to
  get wrong. The prologue/return values are freshly allocated per call
  as in B0.
* Thread safety: scratch is iteration-local; labels/queue are
  iteration-local; output cells `reach[o_pos]` etc. are disjoint by
  origin; adjacency/terminal arrays are read-only on both routes
  (as in B0). No atomics, no sharing.

## 5. Transition proof (induction on popped heap events)

Claim: for any input in the admitted domain (section 3), the candidate
search kernel returns arrays byte-identical to the baseline kernel of
section 1, and both traverse the identical sequence of heap states.

Invariant: at the top of each while-iteration, (S, Q) — the label
vector and the queue — are identical in baseline and candidate.

Base: identical by construction. The prologue is the same code:
same nd_node_count, same `np.ones(nd, dtype)+cutoff` sentinel
expression, same dtype, same two unconditional terminal assignments in
the same order, same seed tuple, same discarded pop, same two
conditional pushes in the same order with the same comparisons.

Step: assume (S, Q) identical. Both pop the same tuple `(w,u)` (same Q,
same deterministic heap order for (float64,int64) tuples). The row is
offsets `[p0, p1)`.

a) Identical c values. Baseline: `weights[p0:p1] + w` — an array
expression of one float64 add per element under fastmath. Candidate:
`weights[p0+i] + w` — one float64 add per element under the same
fastmath flags. A single binary add admits no reassociation and no FMA
(no multiply operand); the per-element value is the same float64 in
any compilation (verified at IR and behavior level, section 7).

b) Identical eligibility. Baseline compares the add results against
`cutoff` and against the pre-row labels gathered at the row's
neighbors. Candidate compares the same add results against `cutoff`
and `S[v]`, reading S before ANY phase-two write (section 4), so the
values read are exactly the pre-row gather's values. The comparison
predicates are the same ordered `<`,`<=` on float64: NaN operands
yield false on both sides; ±0.0 compares equal; the `and` short-circuit
(D2) skips only a pure read. Hence the eligible offset set, in
ascending row order, and the stored c values are identical. Scratch
capacity covers the row (D4), so no truncation occurs.

c) Identical phase two. Candidate visits `j = 0..n_eligible-1`, i.e.
the same ascending offsets, reading back the SAME stored c values (no
recomputation, no second rounding), performing the same label writes
to the same nodes in the same order — including duplicate-destination
later-larger overwrites — and the same conditional pushes with the
same keys into the same queue. "Do not sort eligible neighbors; do not
skip stale entries; do not mark settled; no `<=`/`<` change; no
terminal overwrite change; no degree-one enqueue" — none of these
transformations is present.

d) Therefore S and Q after the row are identical, re-establishing the
invariant.

Termination and return: `while queue` ends at the same iteration on
both sides (identical queue evolution); the returned pair is the same
freshly built `(o_scope_weights, o_scope_pred)` — same shape, dtype,
bytes.

This is an execution argument about THIS compiled pair, not
shortest-path optimality; it makes no claim that labels are globally
optimal or that the queue is duplicate-free.

Driver-level composition: per origin, the driver consumes
`scope_weights[0]` identically (same `adjust_destination_distances`
and `reach_gravity_knn_access` calls with the same arguments, both
unchanged kernels), then writes the same output cells. Origins are
independent in both versions (`prange` over disjoint cells), so
per-origin equality implies whole-array equality of
`reach/gravity_exponential/gravity_logistic/knn_access` (int64 reach
dtype from `o_terminal_idxs.dtype` preserved) and of
`od_distances` rows.

## 6. Fallback

The original kernel functions and the original driver prange loops
remain in the source, textually unchanged; the guard only selects
between two branches inside the driver. For every refusing input
(section 3) the executed code is the same compiled code B0 executes —
byte-identical outputs, errors, warnings (none), and performance
profile by identity. Fallback reachability is pinned by tests T3/T4/T13
(refusing inputs produce B0-identical bytes AND exhibit B0's
allocation-rate signature, proving the private route did not engage).
A guard that refuses too much only shrinks the optimized domain and
can never change behavior; a guard that admits outside section 3 is a
rejection.

## 7. Compiler obligation (dossier "Proof required before code promotion")

Source-level equivalence of expressions is NOT claimed as sufficient.
Delivered at implementation time, before promotion:

1. **Compiled behavioral equality** on every L0 corpus case, all
   kernel-scope cases, and the new A1 boundary fixtures (cutoff
   nextafter, signed zero, subnormals, near-overflow finite, TRUE
   overflow to +inf refused by the cutoff compare, R+1==R sentinel,
   max-float cutoff, duplicate/parallel edges both orders, self-loop,
   stale entry, degree-one, max-degree row, empty-edge CSR):
   candidate driver vs B0 driver, byte equality
   (`comparator.assert_array_bytes_equal`; dtype+shape+bytes, first
   divergence named). NO tolerance anywhere; the H03 ULP<=4 rule
   applies ONLY to pure-Python-trace vs compiled float sums
   (comparator.py:160-169) and is NOT available to any
   candidate-vs-B0 comparison.
2. **Observed-scale replay** (H04-N4 fixture, already captured):
   `replay_integrated_scope_access` and the OD driver records from
   `golden/observed/` executed with the candidate namespace
   (`b0_import.load_arm(UNA_ORACLE_ARM_B=candidate src)`) must
   reproduce every stored output byte-exactly.
3. **IR evidence**: `inspect_llvm` of the candidate search kernel and
   the original kernel at the observed signature; the eligibility
   region's fadd/fcmp instructions recorded side-by-side in
   `evidence/A1I/` for the reviewer (same fastmath flag set; no FMA in
   the changed region — there is no multiply; no reassociation site —
   single binary add). Any mismatch in semantics between the compiled
   pair on ANY fixture blocks promotion regardless of the induction.
4. **No flag change**: NUMBA flag constants and decorator options of
   every touched function unchanged (`parallel` stays False for the
   search kernel, True for the two drivers; fastmath/nogil/cache
   unchanged). No new fastmath, no precision, no FMA instruction, no
   ISA targeting.

## 8. Inherited B0 behaviors preserved (H04 finding N9)

Not fixed, not counted as regressions, expected on BOTH arms in any
chronology comparison: `UNA.has_flow_results` never set True by
RunFlow; RunAccessibility does not clear a previous RunFlow's stored
engine/results (stale `una.flow` remains); RunFlow re-reads the full
network GeoJSON. A1 touches no engine attribute, no Settings field,
and no state outside the kernel-local scope buffers. Also preserved:
integer-typed `reach` (`o_terminal_idxs.dtype`), empty predecessor
result, sentinel `1.0 + cutoff` construction and its R+1==R rounding
domain, nonfinite input hazards (identical in both arms; hazard probes
remain bounded children only).

## 9. Test pins (tests/large_e2e/A1/**; each name maps to a proof obligation)

* **T1 kernel battery**: compiled candidate scratch kernel vs compiled
  B0 kernel on all `fixtures.kernel_scope_cases()` (K1-K9) and the
  `l0_empty_edges` kernel arrays: final labels + pred byte equality.
* **T2 engine battery**: candidate engine (both `Accessibility` and
  `AccessibilityWElevation`) vs B0 engine on all
  `fixtures.engine_cases()`: reach/gravity_exponential/gravity_logistic/
  knn_access and OD matrix (full and pairwise) byte equality; input
  arrays unchanged; outputs freshly owned (no aliasing); repeated
  sequential Centrality on one engine byte-identical.
* **T3 specialization**: int32-endpoint engine case (engine converts
  to int64 -> private route) byte-equal; direct int32 kernel call ->
  refusal -> fallback byte-equal.
* **T4 float domain**: flt_nextafter_cutoff, flt_signed_zero,
  flt_subnormals, flt_near_overflow_finite, NEW true-overflow case
  (sum -> +inf, ineligible in both), flt_sentinel_two_pow_53,
  flt_maxfloat_cutoff: byte equality; flt_inf_incidence_weight ->
  refusal -> byte equality.
* **T5 staged duplicate overwrite**: k_dup_dest_cheap_first/last byte
  equality; FIRST-DIVERGENCE negative: a compiled immediate-update
  mutant (candidate-shaped, phase two eliminated) MUST be caught by
  the byte comparator on the duplicate fixtures; first differing byte
  recorded as evidence.
* **T6 stale entry**: k_stale_queue byte equality.
* **T7 degree-one / flag-false**: k_degree_one_no_push,
  k_flags_false_row byte equality.
* **T8 terminal quirks**: l0_same_terminal_unequal_costs,
  k_same_terminal_overwrite, k_seed_equals_sentinel byte equality.
* **T9 empty-edge CSR**: byte equality with zero-length scratch.
* **T10 max-degree row**: NEW case where max_degree exceeds every
  scanned row (buffer larger than any row) byte equality.
* **T11 reuse-state independence**: candidate driver output identical
  across first call, repeated calls, and different origin permutations
  (per-origin rows byte-equal across orders).
* **T12 multi-thread**: candidate vs B0 with NUMBA_NUM_THREADS=4 in a
  bounded subprocess (compare_arms-style, unique cache roots):
  per-origin byte equality under concurrency; repeated.
* **T13 refusal routing**: for each refusal class (dtype, nonfinite
  weight, negative weight, NaN/inf/negative cutoff, broken pointer
  monotonicity, OOB neighbor, OOB terminal, scratch cap): outputs
  byte-equal to B0 AND allocation-count signature of the original
  path (bounded child, NUMBA_NRT_STATS), proving non-engagement.
* **T14 admitted routing**: allocation count on an admitted input is
  per-call-bounded (no per-pop growth), the A1 target made observable.
* **T15 OD path**: candidate od driver vs B0 od driver byte equality
  on engine cases (dense + pairwise conversion untouched).
* **T16 observed replay**: `replay_integrated_scope_access` (2
  records) and >=2 captured OD driver records, candidate namespace vs
  stored arrays, byte-exact (bounded: O2-scale replay ~seconds, within
  the light-run allowance).
* **T17 warnings/errors silence**: no new warnings before baseline
  validation (logger/log capture empty on both routes), no new
  exception types on refusal or admitted inputs.
* Negative mutation policy: every negative test asserts the comparator
  RAISES (OracleMismatch) and records the first divergence; mutants
  live only under tests/large_e2e/A1/ and are never golden sources.

## 10. Explicit non-goals and bounds

No composition changes beyond the two drivers + new helper module; no
`_ordered_csr` change; no public API/signature/defaults change; no GPU,
no native backend, no relaxed math; no new warnings; no caching of
graph metadata across calls (guard recomputed per driver call — it is
O(V+E) once, charged to the job); no benchmark claims in A1I (the
marginal screen belongs to A1R under DECISION.md). One design, and at
most one performance revision after the initial screened version
(dossier Benchmark and stop).

## 11. Deviations from dossier text

None. Interpretations D1-D6 (section 2) are within the dossier's
language and are flagged for reviewer confirmation. The admission
correction A1I-N1 (`admission.json`: H05's "adjust 0 / caller-buffered"
sub-fact was an empty-IR artifact; fresh compile shows 2 dynamic sites;
load-bearing facts unaffected) is recorded and does not alter the
admitted hypothesis.
