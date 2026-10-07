# G11-Delta run plan — gpu-20261007T1326Z

Qualification host: NCSA Delta GPU partition, device class A100 (partition
gpuA100x4, 1.0 SU/GPU-h). Device choice rationale: the gate requires a
CUDA-class FP64 device (hardware_fp64); A100 is FP64-strong, widely
available on Delta, and mid-cost. H200 (3.0 SU/GPU-h) adds no parity
signal for a bitwise gate; A40 (0.5) is FP64-crippled but would still be
bitwise-valid — A100 is the balanced qualification target. Device identity
is recorded by nvidia-smi -L INSIDE the job.

## Toolchain note (live-verified on Delta login, 2026-10-07)

Lead's stated module cudatoolkit/25.3_12.8 no longer exists in the live
module tree. Live modules: cudatoolkit/26.5_13.2 (default) and
cudatoolkit/26.5_12.9-unsupported. Node driver per lead: 570.148.08
(CUDA 12.8-era). PTX ISA check (live tool output, login node):
- nvcc 13.2 emits `.version 9.2` for -arch=compute_80 — driver r570 JIT
  will likely REFUSE this PTX.
- nvcc 12.9 emits `.version 8.8` for -arch=compute_80.
Primary gate attempt therefore loads cudatoolkit/26.5_12.9-unsupported.
If cuModuleLoadData still refuses PTX 8.8 (typed DeviceGateError -> exit 3),
ONE bounded fallback attempt runs under the default 13.2 toolkit (covers the
case where the node driver is newer than stated). Both attempts are recorded;
no further retries — exit 3 with both consoles is a reportable outcome.

## Stages

1. identity: hostname, date -u, nvidia-smi -L, nvidia-smi query, module list,
   nvcc --version (both toolkits probed).
2. host battery: validate_device_bits --emit-host-battery (rc 5 by design;
   build-bound expected-bits table computed ON the qualification engine
   build). Fresh NUMBA_CACHE_DIR root: /u/dsynn/projects/una-x/cache/numba_g11.
3. device gate (PRIMARY, toolkit 12.9): validate_device_bits --gate-store
   gate_store.json --evidence gate_evidence.json; target rc 0.
4. device gate (FALLBACK, toolkit 13.2) ONLY if stage 3 exits 3.
5. fresh gate-state read from disk (independent of the gate process).
6. tests/tenx/gpu suite on the GPU node (real-probe branch exercised).

## Budget accounting (filled at close)

| stage | job id | GPU-hours billed | wall | notes |
|-------|--------|------------------|------|-------|
| gate+battery (this job) | | | | gpuA100x4 @ 1.0 SU/GPU-h, --gpus=1 |
| CPU arm crossover | | 0 GPU-h | | cpu partition, billed in CPU-h |
| GPU arm crossover | | | | |

## Evidence law for this run dir

- Every figure in reports is a tool output from this run dir or explicitly
  marked unverified.
- Consoles are preserved sha-named (consoles/<sha256>.txt) at close.
- Parity is judged vs the host twins AND vs the real engine fold (leg B),
  never vs a re-implementation.
- rc=0 gate failure on real hardware is a reportable outcome (designed
  refusal preserved); no retry-forever.

## Budget accounting (filled 2026-10-07T14:05Z, tool outputs from sacct)

| stage | job id | elapsed | GPU-h billed |
|-------|--------|---------|--------------|
| gate attempt 1 (probe defect) | 22722619 | 00:00:27 | 0.0075 |
| gate attempt 2 (wrapper kwarg) | 22722669 | 00:00:25 | 0.0069 |
| gate attempt 3 (unsynced fix) | 22722764 | 00:00:16 | 0.0044 |
| gate attempt 4 (ctx diagnosis) | 22722795 | 00:00:21 | 0.0058 |
| ctx diag 1 (primary ctx) | 22722809 | 00:00:11 | 0.0031 |
| ctx diag 2 (explicit ctx OK) | 22722850 | 00:00:09 | 0.0025 |
| gate attempt 5 = PASSED rc=0 | 22722858 | 00:00:19 | 0.0053 |
| TOTAL | | 00:02:08 | 0.0356 GPU-h = ~0.036 SU (gpuA100x4 @ 1.0 SU/GPU-h) |

Consoles preserved sha-named under consoles/ (sha256 of each job_N.out).

## Wheel/artifact smoke (work order d) — 2026-10-07T14:2xZ

No .whl exists in the Delta project tree; the installed artifact on Delta
is urban-network-analysis 2.6.0 as an EDITABLE install of
/u/dsynn/projects/una-x/repo (pip metadata recorded in the smoke console).
The post-I00 wheel identity 9a84517b... refers to the Q10 wheel, which is
not the Delta-installed form; the smoke therefore binds to the installed
editable identity and states so.

Smoke (srun 22722942, CWD=run dir, no repo sys.path): installed engine
fold bits on the cohort case are BIT-IDENTICAL to the gate-certified
engine bits (4/4 outputs); availability through the installed python on
the GPU node names the A100 with FP64; the durable gate store opens the
cell through this import path. SMOKE_PASSED, rc 0.
Attempt ledger: 22722921 FAILED (list-typed inputs), 22722934 FAILED
(script crash after the bits check), 22722942 PASSED. Additional
GPU time: 14+14+14 s = 0.0117 GPU-h. Run total now 0.0472 GPU-h (~0.047 SU).

## Crossover job (planned before firing, per PREREGISTRATION A1+A2)

One job g11-xover on gpuA100x4, --gpus=1, --cpus-per-task=4, cap 00:40:00.
Both arms in the one job (matched host class/core allocation/I-O root).
A2 probe-first: timed arms run only at widths inside the cell's engine-bit
parity domain (recorded on the qualification host); out-of-domain widths are
a designed refusal recorded with probe + tree-sweep evidence. Planned
charging: <= 0.5 GPU-h cap; expected ~0.1-0.2 GPU-h. sacct figures to be
appended at close.

## Crossover job (executed 2026-10-7T14:47Z, job 22723334)

A2 probe-first execution. Console preserved sha-named (consoles/fd173c3c25b4895d0d3c2917bfa0a3d2c8baedbf298021900902e6884b082ae6.txt).
rc 0; results in crossover_results.json.

Outcome on the qualification host (gpua014, A100, repo head 7d0a946):
ALL FOUR frozen widths are OUTSIDE the cell's engine-bit-parity domain —
the region (single certified association seq) vs the installed engine fold
differs by 1-ulp reduction-order rounding on general rows at every width
(probe mismatch samples recorded per row/output), and the per-output tree
sweep shows rows/outputs with NO reproducing tree (e.g. W=64 linear reach,
W=64 exponential gravity_exponential, W=4096 logistic gravity_exponential,
W=262144 linear gravity_exponential). Timed arms therefore never ran (A2
designed refusal); admission is REFUSED on parity grounds regardless of
performance; the gate default stays CPU; tenx_status remains target_unmet.

The rc=0 gate verdict (battery rows, per-output certified trees) is
UNAFFECTED: the battery's row set is exactly the certified parity domain.

| stage | job id | elapsed | GPU-h billed |
|-------|--------|---------|--------------|
| fold-cell crossover (probe-first) | 22723334 | 00:01:09 | 0.0192 |
| RUN TOTAL | | | 0.0664 GPU-h (~0.066 SU) |
