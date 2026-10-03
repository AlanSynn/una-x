# Kernel contract freeze — integrated scope access (task CPU_ALGORITHMS)

This document is the frozen algorithm/kernel contract for the
accessibility scope-access kernel family, for the AOT native (dossier
08) and GPU (dossier 09) ports. It pins the *characterized compiler
reference* — the reviewed numba kernels in
`Engines/_large_access_scratch.py` + `Engines/AccessibilityWElevation.py`
as of baseline `16bf404d2cc5e776a78dc073c80405f6d186da29` line — at the
level of detail required by dossier 08 ("Algorithm schedule"): heap
tuples, row snapshot updates, tie ordering, sentinels, duplicate edges
and final reductions. `kernels/scope_access.py` executes the same
schedule as a portable characterization oracle (numpy + plain loops, no
numba), and `kernels/reductions.py` pins the final-reduction plans per
semantic profile. Conformance is machine-checked by
`tests/cpu_kernels/` (oracle vs compiled engines, bitwise).

Companion artifacts delivered by other tasks and consumed, not
redefined, here: the backend-neutral ABI types
(`urban_network_analysis.backends.contracts`, task EXECUTION per
dossier 10) and the corrected_v1 independent reference
(`urban_network_analysis.reference.*`, task SCIENCE per dossier 02).

## 0. Profiles

| profile | FP contract | arithmetic authority |
|---|---|---|
| `una_legacy` | numba `fastmath=True` compilation of the exact reference source; operation order as compiled on the qualification host | the engines (pinned behavior, warts included: NaN seeds propagate, overwrite seed combination) |
| `corrected_v1` | explicit binary64, no reassociation/contraction; declared index orders; invalid domains refused pre-search | `urban_network_analysis.reference.*` (independent scalar reference) |

A port must reproduce its profile's profile-specific observable
contract bitwise on the qualification host. Cross-profile outputs are
NOT expected to agree (corrected_v1 fixes registered bug classes; see
`reference/__init__.py` and NUMERICS.md). Profile is part of every
stage identity and cache key (CACHE_GRAPH identity.py); it is an error
to compare or reuse across profiles.

## 1. Inputs (typed; ABI types in `backends.contracts`)

| name | type/shape | constraints (admission) |
|---|---|---|
| adjacency_pointer | int64 (V+1) | pointer[0]==0, pointer[V]==E, nondecreasing |
| adjacency_vector | int64 (E) | 0 ≤ neighbor < V |
| adjacency_vector_weights | float64 (E) | finite, ≥ 0 |
| adjacynct_vector_network_node | bool_ (E) | len E; the repo's historical spelling is part of the characterized ABI; do not rename in ports |
| o_terminal_idxs | int64 (o,2) | 0 ≤ t < V |
| o_terminal_weights | float64 (o,2) | unconstrained (seeds; NaN/negative legal in legacy) |
| d_terminal_idxs | int64 (d,2) | 0 ≤ t < V (A3 admission; A1 admits a d_count tail instead) |
| d_terminal_weights | float64 (d,2) | unconstrained in legacy (any bits; min() semantics) |
| d_weights | float64 (d) | fold weights, len d |
| cutoff | float64 | A1 admission requires finite ≥ 0 |

Duplicate edges, self-loops and parallel arcs are legal inputs and part
of the characterization (the schedule processes every row entry; see
§3 phase semantics — this is exactly the duplicate-neighbor semantic
counterexample that rejected the 2026-09 "scalar immediate relaxation"
candidate; register, una_cpu_2026_09).

## 2. Route ladder (fixed; same profile-specific outputs on every admitted route)

```
integrated_scope_access(o…, d…, profile scalars…):
  A3 tail-free route   if A1 admits (structure, weights, cutoff,
                       o-terminal bounds, scratch cap) AND _a3_tail_admits
                       (d_terminal_idxs shape (d,2), 0 ≤ t < V)
  A1 route             if A1 admits (label domain V + d_count; the
                       destination tail [V, V+d) is allocated but never
                       read on admitted inputs)
  compact fallback     otherwise (compact_vector_node_view_scope;
                       legacy-behavior anchor for refused domains)
```

Admission is a typed-guard decision computed from values only (no
global state). A port may re-derive admission internally but MUST
preserve the ladder: every input the reference admits must produce
bitwise-identical outputs on the same route the reference selects;
refused inputs fall to the compact fallback whose behavior is likewise
pinned. There is no "broader domain because random tests passed"
(dossier 10).

## 3. Scope search schedule (one origin; the characterized algorithm)

State: `label[v]` float64 over the label domain (A3: V; A1: V+d_count),
initialized to the sentinel `S = fl(1.0 + cutoff)` (bit-exact value of
`np.ones(V, float64) + cutoff`).

1. `label[o_start] = o_weights[0]`; `label[o_end] = o_weights[1]`
   (unconditional overwrite, any bits, evaluated in this order).
2. Heap H ← [(o_weights[0], o_start)] (single entry).
3. Pop once: `weight, node = heappop(H)` (H is a binary heap with
   CPython/numba heapq tuple semantics: lexicographic (weight, node);
   equal finite weights pop ascending node id — this tie order is part
   of the contract).  The compiled tuple law (characterized against
   the engine's heapq): a NaN-weight entry compares LESS than every
   entry (NaN on the left → True; NaN on the right → False), so a
   NaN-weighted item is popped FIRST.
4. If `o_weights[1] < cutoff`: push `(o_weights[1], o_end)`.
   If `o_weights[0] < cutoff`: push `(o_weights[0], o_start)`.
   Note the STRICT `<` here vs the `≤ cutoff` eligibility test in
   step 6 — the asymmetry is characterized behavior.  COMPILED
   (fastmath) LAW: each scalar `<` executes as `not (b <= a)`, so any
   NaN-involved comparison is True — NaN seeds PASS these gates and
   are pushed; NaN end-weights also overwrite their seed label
   (step 1 is unconditional either way).
5. While H nonempty: pop `(weight, node)`.
6. Two-phase row scan over adjacency row [pointer[node], pointer[node+1))
   in ascending offset order:
   - Phase 1 (snapshot): for each offset i in the row, candidate
     `c_i = weights[i] + weight`; collect (i, c_i) into scratch in row
     order iff `c_i <= cutoff AND c_i < label[neighbor(i)]` —
     eligibility judged against the PRE-ROW label snapshot; no label
     writes in this pass ("row snapshot updates").  COMPILED LAW: `<=`
     executes as `not (cutoff < c_i)` and `<` as `not (label <= c_i)`,
     so NaN candidates are ELIGIBLE and NaN labels are IMPROVABLE —
     exactly as observed on the engine (a NaN seed label replaced by a
     finite candidate, NaN candidates propagated).
   - Phase 2 (assign/push): replay collected pairs in stored order:
     `label[neighbor(i)] = c_i`; push `(c_i, neighbor(i))` iff
     `adjacynct_vector_network_node[i]` AND row degree of neighbor(i) > 1
     (the leaf-node push gate).
7. Return `label` (and an empty predecessor array, part of the
   signature).

Duplicate edges: both row entries are evaluated; each writes its own
candidate; the later write wins only by the snapshot rule (a second
equal candidate fails the strict `<` snapshot test, so first-in-row
order wins equal candidates — observable with duplicate arcs of equal
weight). Self-loops are legal; a self-candidate can only rewrite
label[node] if strictly smaller, then is pushed only if the flag and
degree gate pass, terminating by §3's nonnegative-cost induction.

Termination: nonnegative finite arc weights (admission) + strict
snapshot improvement ⇒ the heap drains (A1I/A3I proof, campaign
history).  Seed/candidate NaN and negative bits are NOT excluded:
under the compiled comparison law NaN seeds and NaN candidates pass
the strict gates (characterized; negative seeds enter whenever they
compare `< cutoff`).  These are pinned legacy warts — `corrected_v1`
replaces the comparison law wholesale (reference package), never mix.

NONTERMINATION LAW (characterized on the compiled engines; the oracle
faithfully reproduces it under a bounded pop cap): with NaN present on
the seed set, the finite/NaN label pair can OSCILLATE without decrease
— a NaN candidate improves ANY finite label (`not (finite <= NaN)` is
True) and any finite candidate improves a NaN label (`not (NaN <=
finite)` is True) — so no monotone potential remains and the heap
pumps indefinitely whenever a gated cycle connects them.  The shared
case battery therefore EXCLUDES the NaN-on-both-seeds case
(`build_case(..., nan_seed_start=True)`); the compiled engine was
observed not to return within 280 s on it (cached-compile kernel;
raw evidence `engine_nan_seed_nontermination/` in this task's run
dir), and `test_nan_seed_pair_nontermination_bounded` pins the
oscillation on the oracle.  NaN on the END seed alone
(`nan_seed_end`) terminates and stays in the battery.

## 4. Destination adjust (per origin)

```
dist[i] = min(label[t0_i] + w0_i, label[t1_i] + w1_i)   (builtin min,
        operand order (start, end), NaN per Python min semantics:
        min(NaN, x) = NaN, min(x, NaN) = x)
```

## 5. Final reductions (canonical plans; see kernels/reductions.py)

Membership: `dist <= cutoff` (closed radius; NaN membership False).
Kept arrays compact in ascending destination index order (mask +
boolean gathers in the reference; a compacting producer producing the
same values in the same order is an admissible exact substitute —
measured as CPU_LAYOUT candidate C4, bitwise-verified, wall-neutral,
preserved in `kernels/compacted.py::scope_access_compacted`).

- reach = Σ kept weights — `np.sum` in numpy's own reduction order
  over the kept array.  The FOLD returns float64; the int64 reach
  dtype (o_terminal_idxs.dtype) is a DRIVER-side cast (C truncation
  toward zero), not the fold's.
- gravity_exponential = Σ (w / exp(β·max(0, d−plateau))) — elementwise
  expression then `np.sum`, in that composition order.
- gravity_logistic = Σ (w · (1 − 1/(1 + exp(−γ·(d−plateau−mid))))).
- knn: k = min(len_kept, len(knn_weights)); if k == 0 → 0.0; else
  `np.argsort(distances)` — default introsort, tie permutation is
  OBSERVABLE (equal distances with differing weights) and part of the
  contract; sort, crop to k, then the decay branch:
  exponential/logistic/other (passthrough Σ coef·w).
- Outputs per origin: (reach, gravity_exponential, gravity_logistic,
  knn_access), assembled into (o,) arrays: reach has dtype
  o_terminal_idxs.dtype (int64), the three others float64.

Profile split for the reductions (characterized; this is a REAL
observable profile delta, not an implementation accident):

- `una_legacy` folds compile with fastmath and their bits INCLUDE LLVM
  fastmath reassociation of the vectorized elementwise/sum pipelines —
  NOT reproducible by plain numpy (demonstrated 1-ulp delta, logistic
  branch, n=4: engine 0x40170f5f9ced3661 vs numpy 0x40170f5f9ced3660).
  The una_legacy parity artifact is therefore the COMPILED
  transcription (`reductions.py::fold_reach_gravity_knn_legacy_plan`,
  njit with the engine's own flags); tests pin it to the engines
  bitwise on the qualification host.  Exception inside the compiled
  fold: the `np.where` membership and array `<=` ufunc stay IEEE
  (closed radius, NaN distances EXCLUDED) — array ufuncs were not
  canonicalized by fastmath; only scalar comparisons in the source
  were.
- `corrected_v1` semantics are the plain-numpy plans (binary64, no
  reassociation, numpy reduction order) — declared, self-pinned, and
  NOT claimed to reproduce legacy bits.

Membership: `dist <= cutoff` (closed radius; NaN membership False) in
BOTH profiles.

## 6. Ownership, scratch, concurrency (port requirements, dossier 08 §Binding)

- Inputs are read-only shared buffers; validation (§1) precedes any
  GIL release / kernel launch.
- Per-origin parallelism (prange in the reference): per-worker scratch
  is `eligible_offset int64[max_degree]`, `eligible_weight
  float64[max_degree]` (A1 additionally admits scratch caps — 256 MiB
  pattern, `backends.contracts` memory budget), plus (in the compacted
  arm) per-worker compacted destination buffers.
- No worker writes another worker's scratch; outputs are
  origin-partitioned (each origin writes only its own output slots —
  no ordered cross-origin reduction exists).
- Cancellation: checked at bounded safe boundaries (between origins);
  no partial public writes (batch/checkpoint contract, dossier 05).
- Work engagement counters (expand count, heap pushes/pops, rows
  scanned, kept counts) are part of the KernelOutputs metadata and
  must be reported by every route (dossier 10: "reports work
  engagement").

## 7. Error contract

Typed refusals before execution (no partial writes): malformed CSR,
out-of-range terminals, non-finite arc weights (corrected_v1 additionally
refuses NaN/negative destination terminal weights), unsupported
dtypes/layouts, scratch-cap overflow (admission refusal, not error),
and capability errors for unregistered backends
(`backends.contracts` ABI + `Execution.admit_execution` policy).
Once admitted work starts, internal mismatch fails loudly (no silent
fallback mid-kernel).

## 8. Port checklist (dossier 08 mapping)

1. Reproduce §3 exactly, including heap tie order (test:
   `test_schedule_pinning.py::test_heap_tie_order_ascending_node`).
2. Reproduce the seed gates and ≤/< asymmetry (tests: seed-gate,
   nextafter-cutoff boundary).
3. Duplicate edges/self-loops: row-order snapshot semantics
   (test: duplicate-neighbor counterexample).
4. Vectorize across origins/stripes, never inside an ordered row scan
   or an ordered reduction (§5/§6).
5. -ffast-math is forbidden as a global flag; per-region FP contract
   per profile (§0); never fix a mismatch with tolerance.
6. Bitwise conformance per platform vs `kernels.scope_access` oracle
   AND vs the compiled engines (tests/cpu_kernels run both).
7. Record binary hash, compiler flags, ISA, timings (dossier 08 gates)
   in the NATIVE_CORE receipt referencing THIS spec version.
