# una_10x — next-phase plan: GPU optimization directions

Branch `perf/gpu-opt-plan` (from main `2d6142a`). Created 2026-10-12 under the user's
instruction to advance GPU optimization plans on a separate branch (sibling CPU/native plan:
`perf/una-10x-next`, `PLAN_CPU_NATIVE.md`). Region law: `campaigns/una_10x/dossiers/08_native_gpu_regions.md`;
engagement law: `src/urban_network_analysis/Engines/_gpu_fold.py` header. This is a plan, not
a claim: **no GPU acceleration, parity, or route claim exists** and none is created here.

## 1. Recorded measured state (the ground the plan stands on)

From `campaign_runs/una_10x/tenx-20261004T155640Z/g10_gpu/gpu_implementation.json` and
`g11_gpu/gpu_qualification.json` (V100 run) and the G10 module in tree:

- Hardware actually qualified: **Tesla V100-PCIE-16GB, sm_70**, partition `gpu-v100`,
  cuda/12.6.1, numba 0.67.0, numpy 2.4.6, QOS inferno, PACE salloc (user-authorized compute).
- The G10 bridge (`Engines/_gpu_fold.py`) is a numba.cuda port of the strict per-origin fold:
  association-free, per-origin thread-owned scratch (nnz total, never O×nnz), typed refusals
  before any device work, `GpuUnavailable` = recorded blocker (never simulated), production
  never imports it.
- Bit matrix (V100, vs the native strict fold): `reach` and `kept` bit-equal in EVERY case;
  `gl` diverges in every case (always carries exp); flat-decay `kn` bit-equal on every flat
  case; `ge`/`kn` divergences bounded ≤4 ULP, exp-sourced (libdevice exp ≠ glibc exp on
  specific arguments).
- **Binding fact vs the delivered authority: 0/4 outputs bit-equal to the delivered numba
  fastmath fold at rep4000 scale (every decay)** — the substitution domain is EMPTY. The
  bitwise parity law excludes the device fold regardless of speed.
- Crossover (warm full-service AB/BA ×6, same node, C=1): device 0.13535 s vs delivered numba
  driver 0.04291 s per 4000-origin structure ⇒ **device 3.15x slower**, transfer+launch+sync
  included; cold JIT 1.2955 s. Decision: `CPU_ROUTE_RETAINED`.
- Capability status: implemented + validated + **not admitted** (dossier 08's substantive
  real-device region requirement is MET as demonstrated capability; exact-domain admission is
  what is missing). Not claimed: acceleration, parity on any profile, route in the 10x claim.

## 2. ERRATA (found while grounding this plan; corrected on this branch)

README's GPU paragraph (since `288bd6a`) says the device kernel gate ran on "a real NVIDIA
**A100**". The recorded qualification receipt says **Tesla V100-PCIE-16GB (sm_70, partition
gpu-v100)**, and no tracked evidence anywhere records an A100 device run
(`gpu_qualification.json` `hardware_identity`; repo-wide a100 search finds only the g00
partition-landscape audit, which lists A100 as an *available* partition, not a run). The
README on this branch is corrected to the recorded device with the receipt cited. Dossier 08
law: record actual installed hardware; do not assume a different device is the target.

## 3. Direction 1 — enable the substitution domain: bit-exact device transcendental vs the DELIVERED authority

The binding blocker is not "libdevice ≠ glibc" per se; it is that the parity authority is the
**delivered numba fastmath host fold**, and the device matches neither that nor strict.
Steps, each falsifiable:

1. **Host authority census**: capture (arg → result-bit) pairs of the delivered fold's exp
   evaluations over representative domains — manifest-derived (per-row `gravity_beta` ×
   clipped distance ranges; logistic `-gr·(pos−mid)` args; knn decay branches) plus a dense
   logspace sweep. Probe: `probes/gpu_exp_ulp_census.py` (host side runs anywhere; device
   side device-gated).
2. **Device match, two roads**:
   - a device-side exp implementation reconstructed to reproduce the captured delivered bits
     exactly (gate: 0 differing arguments on the census domain), or
   - an **admission-domain classification**: the sub-domain of arguments where libdevice
     already equals the delivered bits (the census identifies it exactly); admission would be
     declared on that domain only, with the refusal matrix extended to enforce it.
3. **Gate**: extend the G11 battery matrix with the chosen domain; admission only after
   0-mismatch on the declared domain; contract-level creation of a NEW reviewed profile
   (accepting non-host bits) is a user-gated scientific decision, never a silent substitution.

## 4. Direction 2 — split-cell hybrid admission (admittable on today's evidence)

The matrix already shows a bitwise-safe device cell set: `reach`, `kept`, and flat-decay
`kn` are bit-equal vs strict in every case (no transcendental involved). A hybrid route —
device computes the safe cells, host computes the transcendental cells (`ge`, `gl`,
non-flat `kn`) — can be gated without Direction 1.

Honesty bound: the fold region is 10.9% of serial service today, so the region-sum law caps
this direction's standalone end-to-end value. It becomes material **after** the CPU plan's
R1 (shared preparation) lands, when the fold share rises. Sequence it behind R1.

## 5. Direction 3 — cohort residency amortization (the shape dossier 08 actually specifies)

G11 measured one structure per full-service call. The residency region is: upload immutable
prepared graph/trace arrays **once**, execute K independent row replays as K launches,
download only results. Amortization bound: per-row cost → launch + kernel + D2H only; the
0.0925 s/structure transfer+sync overhead measured in the crossover spreads over K rows;
cold JIT (1.30 s) amortizes per process, not per row.

Probe: `probes/gpu_phase_decomposition.py` measures per-phase walls (per-array H2D, launch,
sync, per-array D2H) at the rep4000 shape, plus a K=32 upload-once variant with per-replay
sync (region law: synchronize before timing ends or leases release). Gate (if pursued): paired
cohort-shape walls on-device vs the delivered CPU arm at C=1, hardware addition recorded
separately and never fed to the CPU 10x gate.

## 6. Direction 4 — parallelism shape, within the order law

The crossover loss is dominated by transfer+launch+sync and per-origin sequential
order-bearing loops (≈60 nnz/origin at rep4000). The order law forbids the tempting axes:
no atomics, no completion-order sums, no reduction reassociation — a warp cannot cooperate on
one origin's ordered sum without reassociating. Legal parallelism axes: across origins,
across rows (Direction 3 batch launches), across independent metric jobs; plus occupancy
(tpb tuning from 128) and tail effects. Re-measure the crossover only after Direction 3's
residency shape exists; do not tune against the one-shot shape.

## 7. What is NOT planned and why

- No FP32/TF32 substitution, no fastmath on device, no dense O×D×jobs tensors, no
  reduction changes because a host compiler used a different vector tree (dossier 08).
- No GPU-vs-baseline end-to-end multiple: none exists; measuring one requires an admitted
  bitwise domain first (Direction 1/2) and would still be a hardware addition outside the
  CPU gate.
- No claim changes to README beyond the §2 errata (device name), which alters no measurement.

## 8. Probes committed on this branch; execution status

- `probes/gpu_exp_ulp_census.py` — host+device exp bit census (Direction 1 step 1).
- `probes/gpu_phase_decomposition.py` — per-phase walls + K-replay residency variant
  (Direction 3 evidence).
- Device legs run only under a real salloc device (GpuUnavailable otherwise). Execution
  status and any results are recorded in `receipts/` on this branch; a probe that has not
  run on hardware is recorded as pending, never as passed.

## 9. Honesty / unrun gates

Nothing in this plan admits a GPU route or changes the CPU default. Every direction above
terminates in a named gate; until one passes, the standing verdict remains
`CPU_ROUTE_RETAINED` with GPU implemented+validated+not-admitted. No pushes from this branch;
publication requires a new explicit user instruction.
