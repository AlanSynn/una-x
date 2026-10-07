# G11-Delta crossover preregistration (filed BEFORE any battery run)

Run dir: /u/dsynn/projects/una-x/runs/gpu-20261007T1326Z
Filed: 2026-10-07T13:26Z (UTC), before gate/battery execution.
Author: gpu-engineer agent (G11-Delta, task #37).

## Crossover arm choice (per work order, stated before the battery)

CPU arm: **Delta CPU arm, strongest matched configuration w4t1 q64 cache-off**
(4 workers x 1 thread, queue depth 64, engine cache off), executed on a Delta
compute node of the same node class as the GPU job's host node.

Why not the local-Mac Q10 walls (36.21-40.90 s): those figures come from a
different ISA and a different host. Using them needs a cross-ISA equivalence
argument, and matched-budget law forbids borrowing an arm the qualification
host does not own. The Delta CPU arm shares the engine source, the inputs, the
env python, and the node class with the GPU arm; the comparison is then a
device-boundary comparison on one host class, which is the claim G11 makes.

## Admission rule (frozen before execution)

- FULL-service crossover: all required outputs counted on both arms
  (preparation, first trace, integrity, durability, cleanup — same accounting
  frame as the campaign's PUBLIC 10x boundary).
- Per-cell admission into CapabilityRegistryEntry ONLY on a median wall win
  for the GPU arm vs the CPU arm, measured on the same node class, matched
  resource budgets (same core count billed class, same memory cap, same I/O
  root).
- The gate default stays CPU regardless of the crossover outcome: device
  dispatch requires the rc=0 bit-validation gate on the real device, and an
  admitted cell changes the DEFAULT only through an explicit, separately
  recorded decision.

## Gates this preregistration does not touch

- The rc=0 device bit-validation gate (una_legacy.fold.default) is a PARITY
  verdict, not a performance verdict; its outcome never depends on the
  crossover.
- tenx_status remains target_unmet; no 10x claim arises from this run dir.
  A future 10x claim needs its own preregistered gate.

## Amendment A1 — crossover design freeze (filed 2026-10-07, BEFORE crossover execution)

### Comparison boundary (stated why)

The crossover is at the FOLD-CELL service boundary, not the RunBatch pipeline
boundary. Reason: the GPU arm's production surface is `execute_fold_region`
(K0/K1 device + K2/K3 host tail); a pipeline-level (RunBatch) comparison would
measure GIS load, trace construction, and output writing on the CPU arm only,
which the GPU arm cannot perform — no production integration exists and none
will be simulated. Comparing fold-cell service to fold-cell service on
identical inputs is the matched-boundary comparison this run dir can support
honestly. A pipeline-level crossover remains possible only after a future
production integration and needs its own preregistration.

### CPU arm (strongest matched, per work order)

The installed engine fold `reach_gravity_knn_access`
(urban_network_analysis.Engines.AccessibilityWElevation — the same installed
route the wheel smoke bit-verified), driven exactly as RunBatch w4t1 q64
cache-off drives it: 4 worker PROCESSES (spawn context) x 1 numba thread
(NUMBA_NUM_THREADS=1 job-wide), tasks dispatched through a bounded queue of
depth 64, engine answer/trace cache OFF (nothing memoized across rows or
repetitions; per-leg fresh numba cache dir semantics carried over as
per-arm cache dirs created fresh at job start).

Scalar mirroring (bit-parity requirement): the engine call passes
cutoff=c, gravity_beta=0.001, gravity_plateau=0.0,
gravity_logistic_midpoint=500.0, knn_gravity_plateau=0.0, knn_weights=KNW,
knn_gravity_beta=0.001, knn_decay=decay, knn_gravity_logistic_midpoint=500.0,
gravity_growth_rate=1.0, knn_gravity_growth_rate=1.0 — mirroring
FoldRowScalars so both arms compute identical scalar semantics (the region's
knn kernels reuse the main gravity scalars by design).

### GPU arm

`execute_fold_region` through `gpu.kernels.device_exec.DeviceRuntime`, on the
A100, with the PTX REUSED from the rc=0 gate verdict
(fold_kernels_compute_80.ptx, sha256 9ca8f9a6... pinned in the run dir — a
durable gate artifact; reuse, not recompilation) and the durable gate store
`una_legacy.fold.default` (open; evidence re-verified per dispatch). Cell
carries association "seq" (the gate-certified tree). Context creation, module
load, all H2D/D2H, launches, host tail, and runtime close() are inside the
timed wall.

### Workload freeze (identical bytes on both arms)

- 32 rows per repetition; widths W in {64, 4096, 65536, 262144} destinations.
- Rows: np.random.default_rng([20261003, W]).uniform(1.0, 2000.0, (32, W))
  float64; weights: same generator .uniform(0.5, 2.0, W) float64.
- Row groups (decay, [start, stop), cutoff): exponential [0,11) c=400;
  logistic [11,22) c=800; linear [22,32) c=1600. knn_weights KNW =
  np.linspace(1.0, 0.25, 8).
- All other scalars as in the CPU arm mirroring above.

### Repetitions, timing, statistic

- 7 repetitions per (arm, W), one process per arm per W; repetition 1 is COLD
  (fresh numba cache dirs; first context/module load) and is INCLUDED in the
  median (never subtracted).
- Timed wall per repetition: [input preparation -> full cell service for all
  32 rows -> outputs resident and synchronized] (+ runtime close() on the GPU
  arm). Process import recorded separately, outside every wall.
- Statistic: median of the 7 walls per (arm, W). CPU spawn workers re-import
  and JIT inside rep 1 (4 independent JIT compilations; disclosed here).

### Integrity and validity

- Reference per W: serial engine fold over all 32 rows (timed separately,
  outside both arms' walls). BOTH arms' rep-1 outputs must equal the reference
  BIT-FOR-BIT per row (reach, gravity_exponential, gravity_logistic,
  knn_access). Any mismatch invalidates that (arm, W) comparison — recorded as
  invalid, never averaged in, never retried silently.
- Preflight (untimed): host twins (association seq) vs engine fold on one row
  per decay at W=4096; a mismatch is recorded before any timing happens.

### Admission rule (restated, frozen)

Per-cell admission into CapabilityRegistryEntry is PROPOSED to the campaign
register ONLY if the GPU arm's median-of-7 is strictly below the CPU arm's
median-of-7 at EVERY frozen width. Otherwise: no admission; the gate default
stays CPU. tenx_status remains target_unmet; no 10x claim arises from this
crossover.

### Matched resource budget

ONE sbatch job on gpuA100x4 (the same node class that passed the rc=0 gate):
--gpus=1, --cpus-per-task=4, both arms executed in that one job on that one
node (matched host, matched core allocation, same I/O root). The CPU arm
never touches the device; the GPU arm's host tail runs 1 numba thread.
GPU-hours consumed are charged to the run budget in RUN_PLAN.md.

## Amendment A2 — composition-parity finding and probe-first execution (filed 2026-10-07, BEFORE Delta crossover execution)

### Finding from the local rehearsal (macOS codegen; Delta probe is authoritative)

A1's integrity law (GPU arm bit-equal to the engine reference) was rehearsed
locally before burning SU. Result: the engine fold's per-output bits on
general rows are reproduced by reduction trees only per-output and per-row —
at W=64, 30/32 frozen rows fail full-seq parity, the reproducing tree varies
per output AND per row (engine numba/LLVM codegen), and linear-decay reach on
row 22 reproduces under NO tree in REDUCTION_TREES. The gate battery itself
records per-output certified trees (e.g. knn_access certified only under
vec2_tree* on big_at_* cases) — i.e. the rc=0 verdict's parity holds via
per-output tree selection on the battery's row set, not via the single
association the durable store records. The production region reduces ALL
outputs under the ONE certified association, so its engine-bit-parity domain
is the battery's certified row set, not a size class.

Consequence: A1 as written would run timed arms that are predetermined
invalid at every width on general frozen rows. Timing them would measure
nothing admissible.

### A2 execution shape (supersedes A1's execution order, not its admission law)

1. Per frozen width, UNTIMED, before any timed arm:
   a. serial engine reference over all 32 rows;
   b. parity probe: the region (device, gate store open) over all 32 rows,
      compared bit-for-bit per output per row -> in_domain true only on a
      perfect match across all rows/outputs;
   c. host tree sweep per output (one row per decay group) recorded as
      evidence of which trees reproduce engine bits on this host's codegen.
2. Timed arms (A1 walls, reps, medians) run ONLY at in-domain widths.
   Out-of-domain widths are recorded as skipped_by_parity_domain with their
   probe evidence — a DESIGNED refusal, not a fault.
3. Admission (restated): PROPOSED only if every frozen width is in domain
   AND the GPU median-of-7 beats the CPU median-of-7 at every in-domain
   width. Out-of-domain widths force no admission; the gate default stays
   CPU. tenx_status remains target_unmet; no 10x claim arises.
4. The Delta parity probe and tree sweep are the authoritative record of the
   cell's parity domain on the qualification host (engine codegen is
   host-specific; the local rehearsal is supporting evidence only).
5. Exit code 0 = job completed with a recorded verdict (admission false is a
   finding, not a fault). Exit 9 = driver-internal error.
