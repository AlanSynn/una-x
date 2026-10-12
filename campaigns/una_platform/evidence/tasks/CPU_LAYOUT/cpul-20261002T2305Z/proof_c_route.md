# CPU_LAYOUT frontier record — accessibility kernel (dossier 07)

Task CPU_LAYOUT (dossier 07, "Profile current installed pipeline and
implement most valuable exact reduced-work/layout algorithms"). This
run implemented, proved, bitwise-tested, and measured four candidate
families against the accessibility kernel. **Disposition: all rejected
by measurement; the engine ships unchanged** (git-diff-empty vs the
audited baseline for `AccessibilityWElevation.py`), per BENCHMARKS.md
("a default performance change needs a positive paired improvement")
and dossier 07 ("Accept only exact outputs/state and complete-region
improvement under equal resources"). The complete implementation,
proof draft, and a 22-test bitwise parity/unit suite are preserved
under `rejected_implementation/` in this directory.

## 1. Profiling attribution (fixture W3_medium_proxy: o=2560, d=2560, V=7360, E=30800, cutoff=1200)

Full-job attribution (warm, cProfile, `cpulayout_jobprof.json`):

| layer | wall | share | in CPU_LAYOUT scope? |
|---|---|---|---|
| accessibility kernel (`integrated_scope_access`) | 0.32 s | 35% | yes |
| layer/point file loads (`Topology._get_gdf`, 3 calls) | 0.24 s | 26% | no (Topology.py) |
| `AddNetwork` own processing | ~0.10 s | 11% | no |
| `ExportAccessibilityResults` (gpd/feather I/O) | 0.13 s | 14% | yes (Engines/Base.py), public output formats frozen |
| remainder | ~0.14 s | 14% | — |

Kernel serial split (`cpulayout_kernelprof.py`): search 0.458 s (83%),
destination adjust 0.016 s (2.8%), reach/gravity/knn fold 0.077 s
(14%). Scan counts (`cpulayout_scancounts.py`): mean in-scope
destinations 305.8/2560 = 11.9%; mean touched nodes 873.7/7360 = 11.9%;
label-init traffic ≈ 0.452 GB serial per run.

The kernel is the only large in-scope, non-frozen target; every
authorized dossier-07 candidate class was implemented against it and
measured.

## 2. Candidates, implementations, measurements, dispositions

### C-G: exact redundant guard/graph-preparation reuse — REJECTED pre-implementation
The A1 admission scan is O(V+E) ≈ 38 K iterations ≈ tens of µs per
engine call (≤0.03% of the kernel). No reusable redundant preparation
exists outside the admission scan; receipt stands from the stage
attribution.

### C5: owned per-origin/per-stripe label scratch with touched resets — REJECTED (measured regression)
Implemented as `_c_scope_search_scratch` (line-for-line legacy search +
flag/tlist recording, arena reuse, compiled touched-list reset inside
the prange driver). Compiled 4-thread walls: routed 0.341 s vs legacy
0.243 s (0.71×); thread scaling 1.81× vs legacy 2.27×; serial
composite ablation attributed the loss to recording + reset overhead
exceeding the removed init passes. The parallelism regression is an
allocator/cache effect of the per-thread arena layout (the legacy
fresh-alloc path scales better). This empirically re-confirms the
2026-09 register's C3 disposition (init share ~1%, attempt refused):
the init traffic is real (~0.45 GB) but its wall share is small and
the only exact removal mechanism costs more than it saves.

### C6: local destination enumeration from touched terminal reverse indexes — REJECTED (measured)
Implemented without C5 via a sequential touched-detection scan
(`label != init`) + two per-side reverse-index CSR marking passes.
Marking stage alone: **+81 ms serial** vs the **13 ms**
`adjust_destination_distances` pass it enables removing (6.2× at 11.9%
touched-node sparsity; `cpulayout_scaling_1t/4t.json`). With the
collect predicate required for every unmarked-but-touched-adjacent
destination anyway, marking is strictly additive work on this fixture.
Rejected; the V-scan variant also requires a finite-nonnegative
destination-terminal-weight guard condition (counterexample in §3).

### C4: producer/consumer fusion (single-pass ordered collect + kept fold) — REJECTED (wall-neutral)
Final implemented route: untouched legacy `_a3_scope_search_tailless`
per origin → one ordered collect recomputing the legacy
`min(start,end) ≤ cutoff` predicate for every destination into
compacted per-thread scratch → legacy fold arithmetic over the
compacted prefix (`_reach_gravity_knn_access_kept`, prologue-free body
copy). Bitwise-exact by construction (same search kernel, same
predicate expression incl. builtin-min NaN semantics, same ascending
order, same fold arithmetic; 22-test suite + engine-array parity +
public-path off/disk-cache parity all green).
Walls (`cpulayout_paired_*.json`, alternating A/B/A/B pairs in one
process): W3 4T paired speedup median 0.9895 (first paired window
0.998; mean deltas ±4 ms around zero); W3 1T 1.0029; W1 4T 0.8512
(noise-dominated, fixture kernel ≈ 1 ms); W1 1T ≈ 1.1 (fixture too
small to resolve). One clean-window scaling harness run showed
legacy 0.240 s vs routed 0.156 s (1.54×), attributed to a
cold-allocator transient, not reproduced in any steady-state paired
window. The route removes three d_count-sized temporaries per origin
(mask + two boolean gathers) and the 20 KB `d_distances` allocation —
strictly less allocation work at wall parity — but per BENCHMARKS.md a
default change requires a positive paired improvement. REJECTED as
default; implementation preserved for CPU_ALGORITHMS to consider when
freezing kernel contracts (the compacted-producer form is the natural
AOT/GPU port shape).

## 3. Deliberate counterexamples recorded (marking family)

Temptation: mark only from touched terminals (C6 without a weight-sign
precondition). Counterexample: destination i with terminals (t0, t1),
t0 untouched, `d_terminal_weights[i,0] = −2000`, cutoff 1200: legacy
`adjust_destination_distances` computes `dist_to_start = (1.0+1200) +
(−2000) = −798.0`, `min(−798.0, …) ≤ cutoff` ⇒ in scope with negative
distance; touched-only marking never visits t0 ⇒ reach differs ⇒
parity break. Hence the V-scan variant's guard refused
non-finite/negative destination terminal weights (widening the
fallback domain); the shipped C4 fusion needs no such condition (its
collect evaluates the predicate for every destination, on the full
legacy domain — NaN/negative/inf weights covered by dedicated parity
tests).

Second temptation (rejected without implementation): visiting
destinations in touched/heap order instead of ascending index order.
Float addition is not associative bit-wise; fold summation order is
part of the output contract. Any reordering breaks parity even with
identical value sets.

Third (bounded out by prior measurement + this run): replacing the
fold's `argsort` (full sort of ~306 in-scope distances to take the
knn_weights.shape[0] smallest) with a selection algorithm — tie order
is part of the observable output (equal distances, differing weights);
quicksort tie permutation cannot be reproduced by partial selection
without re-deriving introspect behavior. Exactness blocks it; recorded
so CPU_ALGORITHMS ports target the same contract.

## 4. What survives into the campaign

- The engine is byte-identical to the audited baseline
  (`AccessibilityWElevation.py` git-diff-empty). All legacy bitparty
  properties are untouched by construction.
- `rejected_implementation/` holds the complete C-route module, guard,
  22-test parity/unit suite (random, adversarial, dtype-refusal,
  interpreted-mode, public-path cache parity), and proof draft — a
  reviewed starting point for CPU_ALGORITHMS's contract freeze and the
  NATIVE_CORE/GPU ports, where the search kernel (83% of the kernel,
  29% of the job) is the designated target.
- Candidate register: `campaigns/una_platform/evidence/tasks/
  CPU_LAYOUT/cpul-20261002T2305Z/CANDIDATE_REGISTER.md`.

## 5. Limitations

- All walls measured on the shared login/analysis node; absolute walls
  drift ±40% between windows. Dispositions rest on paired in-process
  alternation (which cancels window drift) and stage attribution, not
  on absolute wall comparisons across processes.
- W1's kernel wall (~1 ms) cannot resolve single-digit-ms effects; W3
  is the decision fixture for this kernel.
- No speed claim is made for any candidate. The parent capability
  ("most valuable exact reduced-work/layout algorithms") is closed
  with the measured frontier: within dossier-07's authorized candidate
  classes, no exact transformation of this kernel yields a positive
  paired improvement; the remaining authorized levers are the AOT/GPU
  ports of the frozen contract.
