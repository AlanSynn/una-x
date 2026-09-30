# D02: Scientific fixes and corrected reference

## Contract before code
Each correction needs a bug ID, exact failing input, current/Madina behavior, intended domain
rule, independent mathematical derivation, affected public fields, migration policy and tests.
Compatibility and correctness are separate claims. Never overwrite legacy expected results
with corrected outputs. Candidate performance is compared to the SAME corrected scalar reference.

## Priority hypotheses, not already-proven causes
1. Fractional destination weights assigned into integer reach arrays: tiny two-destination
   case with weights 0.25 and 0.50. Corrected weighted reach must follow the chosen float64
   reference rather than truncate; legacy output retains its recorded dtype and behavior.
2. Origins/destinations on the same host edge: compare direct partial travel against both
   endpoint routes. Direction, perceived length, elevation convention and obstacle crossings
   must be derived from the model, not Euclidean distance substituted for network impedance.
3. Coincident terminal IDs: seed overwrite versus minimum can change shortest-path labels.
   Corrected seeds combine by the declared minimum/tie rule; legacy reproduces old order.
4. Negative/nonfinite costs, finite radius sentinel overflow and zero-cost cycles: validation
   must run outside unsafe fastmath. Prove termination on admitted finite nonnegative graphs.
   Invalid inputs receive explicit typed errors, never unbounded queues or silent row drops.
5. Service-area degenerate hulls/GeometryCollection: preserve points/lines/polygons as geometry;
   derive exact intersection behavior. Do not fabricate buffer area or discard valid components.
6. Turn/elevation partial-edge formulas, KNN parameter selection, decay endpoints/overflow,
   demand normalization and conservation, parallel-edge attribution, empty arrays, stale
   obstacle/observer state and per-row workflow Network_File changes all require focused audit.
7. NUMBA_DISABLE_JIT Python guard stubs must be executable or explicitly take the reference
   route. An @overload-only `pass` returning None is not a valid public diagnostic implementation.

## Independent oracle construction
For tiny finite nonnegative graphs, enumerate simple paths independently and calculate costs
with explicit binary64 primitive operations in the declared order. For rational/integer fixture
weights also derive mathematical shortest paths without floating ambiguity. Turn-aware tests
use edge/state graphs with explicit transition costs. Same-edge and obstacle cases have
hand-computed answers. Use high precision to check scientific correctness, but expected binary64
bits come from the frozen ordered reference, not an arbitrary cast of a different formula.

Corrected shortest-path rule: deterministic priority tuples (distance, stable state ID, stable
insertion sequence where necessary), strict improvement plus declared tie policy; stale-entry
skips admitted only after proof. No tolerance changes radius membership. Route enumeration
states whether simple paths or walks are required; do not assume shortest-path termination
proves all-walk enumeration finite under zero-cost cycles.

## Numerical profile
corrected_v1 uses explicit primitive order, dtype and signed-zero policy from NUMERICS.md.
Use fixed logical reduction partitions independent of worker count. The reference and every
backend use the same partition and final fold. No accidental BLAS/tree reduction or libm swap.
Legacy safety fixes that replace a crash/hang with diagnostics are recorded as safety deltas,
not bitwise equality to a nonexistent result. Conflicting scientifically valid definitions
must be surfaced as model choices, not silently settled by the fastest formula.

## Acceptance
For each bug, demonstrate old failure, corrected success, unaffected-domain regression parity,
backend corrected-reference bit parity, documented output migration and cache-key invalidation.
Review mathematics independently of implementation. Fixes are mandatory engineering work,
not rejected merely because they increase computation. Optimize only after their reference is stable.
