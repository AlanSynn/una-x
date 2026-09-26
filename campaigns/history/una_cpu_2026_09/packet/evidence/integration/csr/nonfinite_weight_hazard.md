# Discovered upstream hazard: NaN edge weights cause unbounded memory growth

**Evidence class:** measured (bounded probes, 2026-09-25)
**Applies to:** both the immutable baseline tree AND the candidate tree —
an inherited upstream property, NOT a candidate regression.

## Observation

With a single `NaN` entry in the edge-weight array, the baseline scope
search kernel `compact_vector_node_view_scope` (and the identical kernel in
the candidate tree) enters unbounded heap-queue growth in the stage-probe
`tests/perf_contract/_stage_probe.py`:

| probe | tree | last stage marker | RSS at kill | wall |
|-------|------|-------------------|-------------|------|
| nonfinite_weights origin 0 | baseline (`main/src`) | `STAGE scope origin=0 starting` (build had completed OK) | 4498 MB | ~45 s |
| nonfinite_weights origin 0 | candidate (`wt-integration/src`) | `STAGE scope origin=0 starting` (build had completed OK) | 4497 MB | ~45 s |

Growth continues until machine memory pressure triggers an external
SIGKILL. This is believed to explain two earlier unexplained `returncode=-9`
arm-runner terminations during L1 attempts (16:42 and ~17:30 local), where
the killed process was the one executing the NaN-pinned case.

## Mechanism (analysis, not measured)

The relaxation mask
`weights_neighbors <= cutoff & weights_neighbors < o_scope_weights[…]`
is compiled under `fastmath=True`; with a NaN lane present the compiled
mask can pass for entries where the IEEE comparison must fail, so the
queue keeps being refilled and RSS grows without bound. The kernels are
byte-identical between the two trees; the candidate only changes CSR
construction (pure NumPy, upstream of the kernel).

## Campaign handling

1. The L1 adversarial case formerly pinning `NaN`/`inf` edge weights
   (`nonfinite_weights`) was redesigned to pin **finite near-overflow**
   weights (`1e300`, `5e300`) and renamed `extreme_finite_weights`. This
   retains float64-boundary equality coverage (intermediate sums overflow
   to `+inf`, which the mask correctly rejects) while staying inside the
   engine's working domain.
2. Pure-CSR nonfinite behavior remains covered at L0: the CSR builder
   test feeds ±0.0, NaN, ±inf, and 1e-320 weight bits and requires
   bit-exact equality with the original builder (no arithmetic occurs in
   construction, so NaN/inf pass through identically). This gate is
   unaffected by the kernel hazard.
3. Recorded as a limitation in the final decision artifact: **NaN edge
   weights are outside the engine's safe operating domain in the upstream
   baseline; this campaign neither introduces, masks, nor attempts to fix
   that behavior.** Any NaN fix would be a semantic change outside the
   frozen contract and is out of scope.
