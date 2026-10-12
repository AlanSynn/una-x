# P03B: typed per-metric reuse after compiled replay

## Admission is a dependency proof, not the name of a parameter
Acquire actual frozen cohort CSV and PRIMARY from the run environment; verify their committed
hashes. Do not recreate beta values from a guessed linspace. Produce a per-column difference
matrix. P00 records hashes, not enough data to assume every non-beta value is constant.
For each profile map ACTUAL consumer expressions and validated runtime scalar types.

The una_legacy full-fold dependency table at the pinned source is:
- reach: trace membership/order + destination weights + output dtype/cast.
- exponential gravity: those distances/weights + beta + plateau.
- logistic gravity: distances/weights + plateau + midpoint + actual growth_rate.
- KNN exponential: original argsort permutation, cropped k/coefficient/weights + beta + plateau.
- KNN logistic: same order/k/coefficient/weights + plateau + midpoint + growth_rate.
- KNN other: same order/k/coefficient/weights; no decay parameters consumed.
Common to all: profile, typed cutoff, fold producer/implementation environment, argument layout.
Changing beta alone does not change logistic gravity IF actual growth_rate/other inputs remain
identical. KNN is beta-invariant only in the appropriate branches. No fixed 'half-time' saving.

## MetricID v2 before any hit
The existing cohort result_key float64-coerces d_weights/knn_weights but replay sees raw arrays.
It is a reporting key today; do not promote it into an authoritative reuse key. Introduce a new
versioned MetricID schema whose fields are derived from the real argument bindings:
metric name; trace content+schema ID; original dtype/shape/logical-order bytes; meaningful layout,
writability/alignment specialization; scalar type AND bytes; decay branch; output dtype/cast;
profile; fold/helper code closure; compiler/math/ISA and logical reduction plan identity.
No tuple/object address, filename, .tolist() float coercion, NaN normalization or -0 collapse.
The supplied typed-frame model refuses types it cannot encode; production must enumerate actual
NumPy scalar/dtype support instead of casting to Python float as a convenience.
Version metric IDs without invalidating an unchanged valid trace producer. Record all old
reporting-key consumers before migration. Include legacy profile variants in rejection tests.

## Implementation order
1. Build and test MetricID v2 and a dependency perturbation matrix before storing any result.
2. Compute full fold once for first admitted metric group. Keep fields in immutable internal
   storage. Always copy to fresh per-row public output arrays, with independent write ownership.
3. For changed metrics, implement a separately reviewed compiled partial plan only when it is
   bit-identical to the corresponding component of the full compiled reference across its admitted
   signature. A source-expression match is not enough; splitting a fastmath function changes LLVM.
4. If a partial plan differs by even one ULP, keep the full fold for that signature. Trace sharing
   and compiled outer replay remain useful. Do not change the reference or round exported values.
5. Cross-worker reuse needs per-metric single-flight or worker-local reuse with measured repeated
   count. Never hold a trace-store lock while running the full cohort's fold. Bind immutable
   geometry and metric caches by explicit leases; do not pin full O*D arrays unnecessarily.

## Retain required observations
calculate_* flags may gate assignment only. Before skipping computation prove absence of required
warnings/errors/future mutations on the admitted finite domain. Validation still runs for every
row even if its metrics are all internally reusable. Destination weight mutation may not alter
routing only if connectors/model truly do not read it. Keep failed or incomplete outputs out of
cache. Hit/miss/produced, metric computations, copied result bytes and trace reuse get distinct
counters. Reused fields are not a reused complete final answer for a distinct-job claim.

## Evidence and boundaries
Test field-level independence by changing ONE actual dependency at a time, including dtype-only,
byte-order-only and layout-only changes, 0/-0, cutoff scalar type, compiler/profile and KNN branch.
Test collisions BEFORE benchmarking. Run full-fold vs split-plan parity at adversarial lengths
and ties, not just random finite values. Final validation requires all 64 artifacts unchanged
under the frozen clock/naming comparison policy and Settings/last-engine state parity.
Do not infer wall saving from number of saved metric fields; profile each admitted plan and
compare whole public call after P02/P03A. Keep C12/C31 historic receipts; append new resolution.
