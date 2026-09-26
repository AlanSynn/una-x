# Validation policy

Use progressively expensive tiers.

L0: exhaustive tiny graphs, algebra/state invariants, bit patterns, dtype/layout/ownership, guard/fallback behavior.
L1: immutable baseline versus candidate kernels/builders on adversarial random and crafted inputs.
L2: small genuine GIS public-API workloads including topology, snapping, chronology/state, flow gravity-cap dependency, CSV/GeoJSON/Feather, errors/warnings, and artifact schemas.
L3: medium genuine workload for profiling, CPU process/thread/queue/writer selection, memory and candidate selection.
L4: only the final immutable candidate on the unreduced actual target workload.

Exact transformations must verify finite bits where required, signed zero, nonfinite masks, categorical outputs, CSR row order, state at chronological boundaries, and artifacts. Final-output-only comparison is insufficient when later computation depends on intermediate state.

Preserve an independent oracle. Never regenerate the golden reference using candidate code. A changed shared helper invalidates descendant evidence. Docs-only changes do not invalidate numerical evidence.

If baseline artifacts are nondeterministic, first establish baseline-vs-baseline variability and define the smallest justified comparison rule before candidate results are observed.

Full release/CI matrix is near-merge evidence, not an exploration loop. Skipped or unavailable gates remain explicitly unavailable.