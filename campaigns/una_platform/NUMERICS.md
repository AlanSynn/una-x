# Bitwise parity and corrected science

## Three references, not one contradictory promise
- una_legacy: pinned UNA-X valid-input behavior and numerical environment.
- madina_legacy: pinned Madina valid-input behavior, including its selected num_cores and
  ordering. A minimal dependency-only bridge may enable reference execution, but its diff
  must be audited and numerical effects tested separately. Never silently patch the oracle.
- corrected_v1: independent reviewed scalar/graph reference for declared corrected semantics.

For each profile compare reference CPU -> optimized CPU -> native -> GPU -> cache hit ->
public parallel RunBatch. Require identical shape, dtype, endianness, categorical values,
finite bytes, signed-zero bytes and nonfinite masks; require NaN payload identity when a value
is copied/stored or the declared reference guarantees it. Where a producing math library
has nondeterministic NaN payloads, preserve the reference lane or define a separately reviewed
corrected canonical NaN policy. Do not blanket-exempt NaNs. Record bytes and first divergence.
Never pass through np.allclose, rounding exports, error-percent thresholds or fewer outputs.

## Floating-point implementation is part of the reference
Record Python, NumPy, Numba, LLVM, SciPy, Shapely/GEOS, compiler, CPU ISA, math library,
threading layer, fastmath flags, FMA policy, denormals, rounding mode and logical reductions.
Legacy fastmath=True does not authorize a new implementation's arbitrary reassociation.
It means the baseline compiled operation sequence must be characterized or retained.
A proof over real numbers cannot establish bitwise equivalence.

corrected_v1 uses binary64, explicit RN elementary operation order, no unsafe fast-math,
no implicit FMA contraction, and a documented deterministic transcendental implementation.
Start by retaining trusted CPU math-sensitive coefficients/decays in heterogeneous regions.
A portable deterministic implementation may replace them only after an independent bit-level
validation, licensing review, and complete-region benefit. New native/GPU paths that cannot
match a profile's arithmetic stay outside its admitted domain. They may support corrected_v1
without falsely claiming legacy coverage. Hardware lacking required FP64 does not get FP32.

## Ordered reductions and scheduling
Per origin: preserve destination, adjacency, heap tie, path and incidence order. No floating
atomics or completion-order accumulation. Parallelize independent outputs first.
Legacy flow: logical stripe count/membership and slot fold are part of the profile; physical
worker count may change only without changing these operations. For corrected_v1 use a
frozen canonical logical partition (initial specification: min(32, number_of_origins) stripes,
origin i -> i mod L, each stripe increasing origin order, final fold increasing stripe ID).
The scalar corrected oracle implements this same partition even with one physical worker.
Any future partition change versions the profile and invalidates numerical caches. Never
claim a chunk-total fold matches a serial-origin fold merely because both are deterministic.

## Validation procedure
First prove baseline-vs-baseline determinism, then compare candidate to immutable baseline.
Fixtures include parallel edges with the same neighbor, self-loops, tied heap priorities,
cutoff equality/nextafter, subnormals, +/-0, max finite values, fractional weights, disconnected
nodes, same-edge OD pairs, empty sets, zero-cost cycles and nonfinite rejection. Keep dangerous
legacy probes in a supervised subprocess with a hard memory/time bound.
Capture state at each pop, path emission, OD update, logical stripe completion and batch
commit where relevant. Collect debug traces outside final performance timing. Bounded hashes
may locate divergent blocks; retain the actual counterexample arrays, not just hashes.

## Two distinct scientific checks
(1) Bitwise implementation parity against the selected reference.
(2) Independent scientific correctness using analytically soluble graphs, high precision or
rational distance oracles where appropriate, mass conservation, dimension/CRS invariants,
and independently enumerated route sets. High-precision comparison validates intended math;
it is NOT the bitwise execution oracle. Both must pass on corrected domains.

Outputs changed by a bug fix are listed in bug_deltas.json. Tests must demonstrate the old
case fails its intended invariant, corrected reference passes, and every admitted backend
matches corrected bits. Never compare different model/profile outputs and call it speedup.
