# A3I proof.md — preregistration and structural proof, design T (tail-free private scope)

role: implementation-owner (h05). Status: PREREG — pre-implementation, pre-measurement.
Standing admission record: evidence/A3I/admission.json, sha256
39b0806c27951cedff7d73f7f38f2b824db66783661fc500c9aa2b45b85b8af1 (disposition ADMITTED
on the pinned serial-equivalent reading 0.08915417305677027 >= 0.05, margin
1.7830834611354054; capacity clause unreached; 32/32 origins validated).

This document is authored BEFORE any A3 candidate measurement, per T5 term 1. Every
code claim below carries a line citation verified in this session against the worktree
at HEAD 9b1340eb008accdff45fbdf57c6009dd5a6b5146 (accessibility triple byte-identical
to the admitted base; wheel == worktree per admission.json source_pins).

## 1. Design selection (preregistered)

ONE design is selected: **Option T — D-tail removal in a new private integration
routine.** The public `compact_vector_node_view_scope`, `od_compact_vector_node_view_scope`,
`integrated_scope_access`, `adjust_destination_distances`, `o_scope`, `o_access` and all
exposed graph/results remain unchanged; the only new code is a private search kernel +
private admission-check in `_large_access_scratch.py` plus a guarded dispatch branch in
each driver.

Option E (epoch/marks workspace) is NOT selected. Per T5 term 1 its displacement
criterion is fixed HERE, before any measurement:

- Quantity M_E: serial-equivalent private-init share of the complete-job application
  window measured on the T-built candidate — numerator = mean per-origin init time of
  the tail-free routine x O, denominator = warm application window (same estimand as
  the pinned T3 reading; evidence source = the A3R screen legs, h05 harness, not ad-hoc
  timing).
- E may displace T only if BOTH hold on that candidate build:
  (i) M_E >= 0.05 (the residual init is still admission-grade work), AND
  (ii) amortized per-search reset overhead (epoch/marks maintenance + touched-list
  reset, leg-C-style microbenchmark on the same build) <= 0.5 x the per-origin init
  time it replaces.
- Even if both hold, E is a NEW gated cycle with its own proof and hard-gate approval —
  the dossier's "at most ONE design may be selected after proof review" governs this
  task, and this prereg selects T. No post-hoc judgment can substitute E here.

## 2. The vector and its index domain (structural facts)

`nd_node_count = d_count + adjacency_pointer.shape[0] - 1` (scratch :198; twin
Accessibility.py :122, AWE :118) = D + V with V = network node count,
`adjacency_pointer.shape[0] - 1`. The private label vector
`o_scope_weights = np.ones(nd_node_count, dtype=o_terminal_weights.dtype) + cutoff`
(scratch :204; Accessibility :128; AWE :124) therefore has a **tail** `[V, nd)` of
exactly d_count float64 slots beyond the network-node index range.

Every index that reaches `o_scope_weights` inside the kernel is in `[0, V)`:

- `o_scope_weights[o_idx_start]`, `o_scope_weights[o_idx_end]` (scratch :208-209;
  twins Accessibility :132-133, AWE :128-129): values of `o_terminal_idxs[o]`; the
  admission guard validates ALL origins' terminal values in `[0, node_count)` over
  the full `o_terminal_idxs` array (scratch :153-158, condition 11; block comment
  :152) — once per driver call, over the whole domain, not per sample.
- eligibility read `o_scope_weights[neighbor_node]` (scratch :232; Accessibility
  :153; AWE :149) and label write `o_scope_weights[neighbor_node] = neighbor_weight`
  (scratch :243; Accessibility :159; AWE :155): `neighbor_node` is an
  `adjacency_vector` value; condition 9 validates EVERY edge endpoint of the ENTIRE
  adjacency in `[0, node_count)` (scratch :143-150) — again total-domain, per call.

Both conditions hold in EVERY firing of the kernel on admitted inputs, because the
guard runs once per driver call and validates the whole graph and all origins before
any origin search runs. The heap sequence, pushes, and eligibility depend only on
label values at `[0, V)` slots. **The tail `[V, nd)` is never read and never written
by the kernel in any admitted firing.** The sample finding (zero destination-segment
touches, admission.json) is NOT relied on for this claim — it is motivation only.

## 3. Reader enumeration (complete)

Repo census: `grep -rl "scope_weights" src/` returns exactly three files —
`Accessibility.py`, `AccessibilityWElevation.py`, `_large_access_scratch.py`. All
readers:

1. In-kernel (all engines/paths): the four index sites above (scratch :208, :209,
   :232, :243; twins Accessibility :132, :133, :153, :159; AWE :128, :129, :149,
   :155 — init at AWE :124, return at AWE :160).
2. Driver unwrap: `scope_weights = <search>(...)[0]` — `o_scope_pred` (element `[1]`,
   an `np.empty(0)`) is dropped at every call site: Accessibility :253, :264-273,
   :346-357, :386-395; AWE :238-249, :332-343, :260/:269, :371/:380.
3. Downstream consumer: `adjust_destination_distances` (Accessibility :173-199; AWE
   :169-198) — called from all four A1-route and all four fallback sites. Its element
   reads are exactly `scope_weights[start_node]` and `scope_weights[end_node]`
   (Accessibility :194-195; AWE :190-191), where `start_node = d_terminal_idxs[i, 0]`,
   `end_node = d_terminal_idxs[i, 1]` for `i in range(d_count)`.
4. Nothing else: no global, no attribute capture, no other module touches the vector.
   The private vector never escapes its `prange` iteration; `od_distances[i]` /
   `d_distance` receive `adjust`'s output, not the vector.

`n_count = scope_weights.shape[0] - d_count` (Accessibility :185; AWE :181) is
computed once and never used — a tool count shows it is the ONLY occurrence of
`n_count` inside `adjust_destination_distances` in each file. Per the dossier's
explicit warning, this deadness is NOT part of the proof; it is recorded only because
the tail-free vector changes that dead local's VALUE (from `V` to `V - d_count`), and
the proof must state that no behavior depends on it: the function's output depends
only on the element reads of item 3.

### 3.1 The residual risk and its enforcement (typed guard)

Item 3's reads are safe on a tail-free vector only if every `d_terminal_idxs` value
is `< V`. Structural argument: `d_terminal_idxs` is built from
`topology.destinations.edge_start_node` / `.edge_end_node`
(Accessibility :571-572 -> :651; AWE :617 -> :734) — the destination's HOST-EDGE
endpoints in `topology.network`, the same tables whose `start_nodes`/`end_nodes`
generate the CSR over `node_points.shape[0]` nodes (Accessibility :560-562; AWE
:556). Destination terminals are therefore drawn from the network node id domain.

The construction argument alone is not the proof's load wall: the design adds a
**typed runtime check**. `_a3_tail_admits(d_terminal_idxs, d_count, node_count)`,
run once per driver call, refuses the tail-free route unless
`d_terminal_idxs.shape == (d_count, 2)` and every value is in `[0, node_count)`.
Dispatch shape per driver:

    a1_admitted, max_degree = _a1_scope_admits(...)      # existing, unchanged
    if a1_admitted:
        if _a3_tail_admits(...):  -> tail-free private search (new)
        else:                     -> A1 route exactly as today
    else:                         -> original loop exactly as today

A3-refused inputs keep byte-identical A1 behavior; nothing outside the new conjunct
changes routes. Per the binding guard-probe posture, review includes semantic probes
of `_a3_tail_admits` on the firing case and the refusal cases.

## 4. A2 `scope[V]` sentinel dependency (explicit)

A2's (NOT_ADMITTED, unimplemented) local-pruning design reads `scope[V]` — the FIRST
tail slot, holding the untouched typed sentinel b = 1 + cutoff — to learn the
baseline sentinel at runtime (dossier A2_destination_locality.md :12). Removing the
tail eliminates that slot on the PRIVATE route. Statements:

- No A2 composition exists or is combined in this task (A2R closed NOT_ADMITTED;
  source unchanged). The tail-free route and any future A2 route are mutually
  exclusive until composed.
- The PUBLIC kernels keep their full nd tail, so the sentinel slot exists unchanged
  on every public path; A2's dependency on public-path bytes is untouched.
- If A2 is ever revived and composed with A3, the dossier's alternative — an
  independently verified typed sentinel production — is REQUIRED; combining must not
  leave that read out of bounds (dossier A3 :8). This constraint is recorded here as
  the dependency statement T5 requires.

## 5. Design spec (implementation contract)

- New kernel `_a3_scope_search_tailless` in `_large_access_scratch.py`: line-for-line
  copy of `_a1_scope_search` with the single init change
  `o_scope_weights = np.ones(adjacency_pointer.shape[0] - 1, dtype=o_terminal_weights.dtype) + cutoff`
  — no `d_count` term. Same decorator consts (NUMBA_PARALLEL/CACHE/NOGIL/FASTMATH
  module pins), same signature minus nothing (d_count stays a parameter for
  interface symmetry or is dropped — implementer's choice, fixed at review).
- New guard `_a3_tail_admits` in the same module (plain `@nb.njit`, value scan,
  no writes, raises nothing, returns bool).
- Driver branches in ALL FOUR drivers — `od_compact_vector_node_view_scope` and
  `integrated_scope_access` in BOTH Accessibility.py and AccessibilityWElevation.py —
  per the dispatch shape in section 3.1. `adjust_destination_distances` is REUSED
  unchanged (its reads are guard-enforced in-range; section 3).
- Ownership: per-origin, unchanged — each `prange` iteration allocates its own
  private vector (now V slots); no cross-thread sharing, no workspace reuse, no new
  package-level global cache (module constants only, matching A1's pattern).
- Public surface: names, signatures, and outputs of all public functions unchanged;
  outputs (`od_distances`, `reach`, `gravity_exponential`, `gravity_logistic`,
  `knn_access`) byte-identical on every input (admitted or not).
- Bit-equality argument: label values, write order, heap sequence, and eligibility
  snapshots depend only on `[0, V)` slots, which behave identically; `adjust` reads
  guard-enforced in-range slots; therefore outputs are bit-identical. Tests force
  this (section 7).

## 6. Saving projection and replacement-work count (tool-computed)

All figures from the admission record's exact integers (D = 6,661, V = 49,159,
nd = 55,820, O = 14,751, window = 2,474,174,042 ns; exact-fraction arithmetic):

- Tail fraction of the init: 6661/55820 = 0.11932998925116446 (this is the UPPER
  BOUND of what T can save — the head init, fraction 0.8806700107488356, remains
  per-origin).
- Projected window-fraction saving, serial-equivalent reading:
  0.08915417305677027 x 0.11932998925116446 = 0.010638766512560852 (~1.06%).
  Wall-honest reading: 0.008915417305677027 x 0.11932998925116446 =
  0.0010638766512560851 (~0.11%).
- Projected post-T init share (serial-equivalent): 0.07851540654420941 — still above
  the 0.05 bar; this is the honest statement that T does NOT exhaust the admitted
  init cost (E's criterion (i) would hold; criterion (ii) and the new-cycle rule
  govern any E pursuit).
- Allocation removed per live search: 6,661 x 8 B = 53,288 B; peak across the
  10-thread pool = 532,880 B (no capacity claim is made).
- Replacement work: NONE added on the search path — the guard's added
  `d_terminal_idxs` scan is once per driver call, O(d_count), against per-origin
  init work of O(nd) x O. No clearing, no reverse-index state, no O(V) reset exists
  in T (that is E's machinery). Compile cost is one extra kernel, cached.
- Materiality context (NOT a gate, per T5 term 4): median init / median per-origin
  search = 0.5158010995282457.

## 7. Test plan (dossier list, adapted)

- T1 Compiled bit equality: `od_distances` / access outputs byte-identical vs the
  pre-change build on admitted synthetic and L0/L1 fixture inputs, multiple seeds;
  origins sharing terminals; disjoint consecutive regions; all-unvisited and D = 0
  cases (tail-free routine degenerates to ones(V), adjust loops zero times).
- T2 First-divergence mutants: (a) tail-free kernel reading slot V (sentinel read)
  must diverge or raise; (b) init missing `+ cutoff` must diverge; (c) disabled
  `d_terminal` range check with an out-of-range synthetic must produce a guard
  refusal when enabled.
- T3 Guard semantic probes (binding posture): `_a3_tail_admits` firing case and
  refusal cases (wrong shape, value == node_count, negative, d_count mismatch);
  `_a1_scope_admits` still refuses its domain; A3 refusal falls through to a
  byte-identical A1 route (regression pin).
- T4 A1 suite regression: all A1/A2-state test pins green; A1 kernel bytes untouched.
- T5 Read-only inputs: kernel and guard write nothing outside the private vector
  (input array hashes unchanged around calls).
- T6 Concurrent owners: prange per-origin allocation under the pool; no shared
  mutable state (existing A1 concurrency pattern).
- T7 Early error then another origin: an exception in one iteration leaves no
  residue (per-iteration allocation; nothing to reset).
- T8 Capacity fallback: A1 capacity condition unchanged and still effective; A3
  refused above the cap.
- T9 LLVM/math flags and cache: decorators pinned to the module consts; compile
  flags identical to engine siblings; cached kernels load without recompile in a
  fresh process.
- T10 Public-path invariance: with the tail-free route forcibly refused, public
  outputs byte-identical to the pre-change build (guards the unchanged surface).

## 8. Failure policy and claims discipline

Any red test stops implementation; evidence is retained; stop-and-route. No
throughput or capacity claim is made by A3I — A3R's screen decides survival against
the campaign's screen gates, using the admission record's estimands. The measured
admission was for the FULL nd init; T's capture is bounded by the tail fraction
(section 6), stated here before measurement so the screen result is read against the
preregistered bound, not against optimism.
