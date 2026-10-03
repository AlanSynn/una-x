# CPU_LAYOUT candidate register (run cpul-20261002T2305Z)

Continuation of the campaign candidate-register convention
(campaigns/history/una_cpu_2026_09/packet/CANDIDATE_REGISTER.md).
Statuses: REJECTED (measured), NOT ATTEMPTED (bounded), PRESERVED.

## C-G exact redundant guard/graph-preparation reuse
REJECTED pre-implementation. A1 admission scan O(V+E) ≈ 38 K iterations
≈ tens of µs per engine call (≤0.03% of the kernel). No other
redundant per-call preparation exists inside owned scope.

## C5 owned per-origin/per-stripe scratch with touched resets
REJECTED (measured regression). Implemented (`_c_scope_search_scratch`,
arena label/flag/tlist, compiled touched-list reset). 4-thread routed
0.341 s vs legacy 0.243 s (0.71×); thread scaling 1.81× vs 2.27×;
serial ablation: recording+reset overhead exceeds the removed init
passes. Re-confirms 2026-09 register C3 (init share ~1%, attempt
refused) with a compiled falsification instead of a share estimate.

## C6 local destination enumeration from touched terminal reverse indexes
REJECTED (measured). V-scan variant (`_c_mark_touched`: sequential
label≠init touched detection + two per-side reverse-index CSR mark
passes): +81 ms serial for the mark stage vs 13 ms for the
`adjust_destination_distances` pass it enables removing (6.2× at 11.9%
touched sparsity). Guard consequence: without marking, no
weight-domain constraint is needed; the marking variant requires
finite nonnegative destination terminal weights (negative-weight
counterexample, proof §3).

## C4 producer/consumer fusion (single-pass ordered collect + kept fold)
REJECTED as default; PRESERVED for CPU_ALGORITHMS. Bitwise-exact
(same search kernel, same builtin-min predicate and operand order, same
ascending destination order, same fold arithmetic; 22/22 tests incl.
admitted NaN/negative/inf weights, interpreted mode, public-path cache
parity). Paired alternating windows: W3 4T median 0.9895× (first
window 0.998×), W3 1T 1.0029×, W1 4T 0.8512×, W1 1T ≈ 1.1× — no
positive paired improvement (BENCHMARKS.md gate). Strictly less
allocation (removes the fold's mask + two boolean-gather temporaries
and the 20 KB `d_distances` per origin). One clean-window 1.54×
observation (scaling harness) did not reproduce in steady state;
documented as a cold-allocator transient. The compacted-producer form
is the natural AOT/GPU port shape — CPU_ALGORITHMS should evaluate it
when freezing the kernel contract.

## C-init np.full single-pass label init (legacy search kernel)
NOT ATTEMPTED (bounded sub-gate). Halves init traffic (~0.23 GB serial)
but the init wall share is ~1–3% of the kernel (bounded by the C5
ablation), and the edit touches the audited shared legacy kernel.
Same disposition class as 2026-09 C3.

## Exactness blocks recorded for downstream tasks
- argsort → selection replacement in the knn fold: tie permutation is
  observable (equal distances, differing weights); blocked.
- Fold summation reordering (touched/heap order): float addition is
  not associative bit-wise; blocked.
- Marking-based enumeration on this fixture: measured additive
  (C6 above).

## Frontier conclusion
Within dossier-07's authorized candidate classes, no exact
transformation of the accessibility kernel yields a positive paired
improvement on the decision fixture. The kernel (83% search) is the
designated NATIVE_CORE/GPU port target via the CPU_ALGORITHMS contract
freeze; the next-largest job layers (file loading 26%, AddNetwork 11%)
are outside CPU_LAYOUT's owned files and are recorded here for the
integrator.
