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

1. **Host authority census** — EXECUTED 2026-10-12 on V100 (`receipts/gpuplane-20261012T1600Z.md`,
   `tenx-20261012T1600Z/gpu_probes/exp_ulp_census.json`): sweep n=308,384 args
   (manifest-derived ge 131,072 from the 32 cohort betas; gl 5,120; dense logspace 172,192).
   Device libdevice exp vs the **delivered fastmath authority**: 9,362 differ (3.04%),
   **max 1 ULP, 0 sign flips**; device == fastmath bit-exact on 299,022 args = **96.96% of
   the swept domain**. (Device vs `np.exp`: 14,103 differ; `np.exp` vs fastmath: 8,925 —
   the host authorities differ from each other too.) Dense-sweep exp overflow → +inf on
   both sides, recorded, ulp filter finite-only. Reading: the bit-equal subdomain EXISTS at
   the primitive level — road (b), admission-domain classification, is quantified as
   "exclude/arg-census the ~3% divergent region"; G11's fold-level 0/4 stands, because real
   fold arguments compose through the divergent region.
2. **Fold-relevant box census + the ACTUAL derived exp arguments** — EXECUTED 2026-10-12 on
   V100 (`receipts/gpuprobe2-20261012T1605Z.md`,
   `tenx-20261012T1600Z/gpu_probes/fold_box_census.json`):
   - Manifest-bounded boxes, device libdevice vs the delivered fastmath scalar authority:
     ge box 131,072 args → 8,071 differ (**6.16%**); gl box 20,480 args → 1,109 differ
     (**5.42%**); max 1 ULP, 0 sign flips everywhere.
   - The exp arguments the rep4000 structure ACTUALLY feeds per decay (derived in float64
     from the delivered fold source `Accessibility.py:83-106`; derived arg bits verified
     bit-identical to the fold's own inline fastmath construction): ge 178,777 args →
     10,649 divergent (**5.96%**); gl 178,777 → 11,087 (**6.20%**); kn-exponential 4,000 →
     45 (1.13%); kn-logistic 4,000 → 206 (5.15%). Divergent args SPREAD across each box's
     range — no clean safe sub-interval exists (ge divergents span |x| 0.011…1.59; gl
     0.010…10.0; kn-logistic divergents all sit at |x| 3.1…5.0, the nearest-neighbor crop
     region).
   - Derivation certification: the arg bits are certified (0 diffs), but fold-output
     reproduction is NOT: feeding the bit-verified derived args through verbatim
     re-implementations of the fold's exp-consuming statements diverges from the delivered
     fold by 1 ULP on one small case (rand96 exponential kn), while the same statements
     with args constructed INLINE reproduce the delivered outputs everywhere tested.
     Fold-level bit identity is a property of the delivered function's compiled reduction
     lowering, not of the argument set alone.
   - **No-go input for road (b) as a standalone mechanism:** ~6% of the actual cohort exp
     feeds diverge at the primitive level with no exclusion boundary, and even
     0-arg-divergence would not deliver fold-output parity. Arg-domain admission cannot
     establish fold-level bit parity vs the delivered authority. Remaining roads: a device
     exp reconstructed to the host bits on a declared domain, the Direction-2 exp-free
     hybrid, or a new reviewed profile accepting non-host bits (user-gated scientific
     decision, never a silent substitution).
3. **Device match, two roads**:
   - a device-side exp implementation reconstructed to reproduce the captured delivered bits
     exactly (gate: 0 differing arguments on the census domain), or
   - an **admission-domain classification**: the sub-domain of arguments where libdevice
     already equals the delivered bits (the census identifies it exactly); admission would be
     declared on that domain only, with the refusal matrix extended to enforce it.
     STATUS after the box census (item 2): this road alone cannot establish fold-level
     parity — the actual fold feeds diverge at ~6% density with no exclusion boundary, and
     fold outputs additionally depend on the delivered function's compiled reduction
     lowering (derivation-certification gap).
4. **Gate**: extend the G11 battery matrix with the chosen domain; admission only after
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
sync (region law: synchronize before timing ends or leases release). **EXECUTED 2026-10-12 on
V100** (`receipts/gpuplane-20261012T1600Z.md`, `.../gpu_probes/phase_decomposition.json`):
full-service bridge median 0.13718 s (cross-checks the G11 crossover 0.13535 s within ~1.4%,
different node/day); measured device phases of a call total 0.00345 s (per-array H2D ≈2.81 ms
incl. 0.90 ms allocation, launch 0.225 ms, sync-incl-kernel 0.146 ms, D2H outputs 0.242 ms);
residency variant upload-once + 32 replays: total 0.01789 s, **per-replay median 0.553 ms**.
Two readings: (i) the residency shape amortizes mechanically as dossier 08 specifies;
(ii) ~97% of the full-service wall is NOT device phases — the residual is unattributed
host-side bridge work, which re-attributes Direction 4's premise (see §6). Gate (if pursued):
paired cohort-shape walls on-device vs the delivered CPU arm at C=1, hardware addition recorded
separately and never fed to the CPU 10x gate.

**Residual attribution — EXECUTED 2026-10-12** (`receipts/gpuprobe2-20261012T1605Z.md`,
`.../gpu_probes/host_residual.json`): the full-service median re-measured 0.13836 s (third
independent cross-check of the 0.13535 G11 / 0.13718 phase-decomposition walls, different
node/day/device). Re-executing `gpu_replay_fold`'s per-call sequence section by section
(source order, 9 reps) attributes the wall: the **Phase-B host structural scan** — the
pure-Python O(origins+nnz) refusal loop at `_gpu_fold.py:185-199` — is **0.13259 s median =
95.8% of the full-service wall** (~0.56 µs per nnz element). Every other host section is
≤34.5 µs; the measured device sections total 3.51 ms (H2D 2.03 ms, scratch/output allocs
0.87 ms, launch 0.22 ms, sync 0.15 ms, D2H 0.24 ms); all sections sum to 0.13616 s, leaving
**2.20 ms (1.6%) unattributed** dispatch/allocator gap. Caveat recorded in the JSON: sections
are timed in isolated re-execution on warm state; the remainder quantifies exactly the
in-context gap. Note the chronology law requires the Phase-B refusal check to precede any
device work, so every one-shot call pays the scan; a residency batch pays it once per
structure, not per row.

## 6. Direction 4 — parallelism shape, within the order law

The crossover loss was attributed to transfer+launch+sync and per-origin sequential
order-bearing loops (≈60 nnz/origin at rep4000). The 2026-10-12 phase decomposition
revised the attribution to a ~133 ms host residual, and the same-day residual probe
**settles it** (§5): the wall is ~96% the bridge's own Phase-B Python scan
(`_gpu_fold.py:185-199`), with transfer (2.03 ms), scratch/output allocation (0.87 ms),
launch (0.22 ms), sync (0.15 ms) and D2H (0.24 ms) jointly ~2.5%. Consequences for
Direction 4: no device-side tuning axis (occupancy, tpb, tail effects) can move a wall
dominated by a host scan — those axes remain legal but are measured at ~3.5 ms of headroom,
not ~137 ms. The legal levers this evidence actually points to: (i) a bit-identical
vectorized re-expression of the Phase-B checks (same refusal codes, same chronology —
refusals still precede any device work), which is a change to the bridge instrument
`src/urban_network_analysis/Engines/_gpu_fold.py` and therefore needs its own task,
bit-identical-refusal gate and review; production never imports the bridge, so this moves
GPU-qualification walls, not the production route; (ii) the Direction-3 residency shape,
which pays the scan once per structure and so amortizes it per row (K rows share one
validated upload + one scan) but leaves the one-shot full-service cost unchanged. The
order law still forbids the
tempting device axes: no atomics, no completion-order sums, no reduction reassociation —
a warp cannot cooperate on one origin's ordered sum without reassociating. Legal
parallelism axes: across origins, across rows (Direction 3 batch launches), across
independent metric jobs. Re-measure the crossover only after a Direction-4 lever actually
lands; do not tune against the one-shot shape.

## 7. What is NOT planned and why

- No FP32/TF32 substitution, no fastmath on device, no dense O×D×jobs tensors, no
  reduction changes because a host compiler used a different vector tree (dossier 08).
- No GPU-vs-baseline end-to-end multiple: none exists; measuring one requires an admitted
  bitwise domain first (Direction 1/2) and would still be a hardware addition outside the
  CPU gate.
- No claim changes to README beyond the §2 errata (device name), which alters no measurement.

## 8. Probes committed on this branch; execution status

- `probes/gpu_exp_ulp_census.py` — host+device exp bit census (Direction 1 step 1).
  **EXECUTED 2026-10-12** on Tesla V100-PCIE-16GB (`atl1-1-02-006-31-0`, slurm job
  13983444, salloc `-A gts-hoh68 --qos=inferno`, gpu-v100, cuda/12.6.1) — results in §3
  and `receipts/gpuplane-20261012T1600Z.md`. One probe bug was found and fixed on this
  branch before the run (gl_args built per-scalar instead of per-array; local repro
  confirmed, then re-run).
- `probes/gpu_phase_decomposition.py` — per-phase walls + K-replay residency variant
  (Direction 3 evidence). **EXECUTED** same job — results in §5.
- `probes/gpu_host_residual.py` — section-by-section attribution of the full-service wall
  (Direction 4 prerequisite). **EXECUTED 2026-10-12** (slurm 13985066, node
  `atl1-1-02-005-30-0`, Tesla V100-PCIE-16GB UUID `GPU-f18ae2ec-bda3-dbe2-2f3a-
  9bdfcd428606`, cuda/12.6.1) — results in §5/§6, receipt
  `gpuprobe2-20261012T1605Z.md`.
- `probes/gpu_fold_box_census.py` — fold-relevant exp-arg boxes + the rep4000 structure's
  actual derived exp arguments + derivation certification (Direction 1 road-(b) input).
  **EXECUTED** same job — results in §3. The probe's certification leg decomposes
  failures (arg-bit identity vs statement-lowering identity); on this host CPU the arg
  bits certify and the fold-output reproduction does not (1 ULP, named gap).
- Device legs ran under a real salloc device; an earlier attempt without one returned the
  typed `device_unavailable` blocker (never simulated). Operational lessons recorded in
  the receipt: salloc executes its command on the LOCAL node (wrap in `srun`), bare
  `srun --gres` failed step creation on this cluster (allocate with `salloc`, step with
  `srun`), and `/tmp` is node-local (scripts must live on shared `/storage`).
- Compute accounting: first probe job ≈4 min wall (13983444); second probe job 35 s wall
  (13985066) — both 1-GPU allocations charged to the GPU probe scope under the standing
  salloc authorization; no route/parity claim created.

## 9. Honesty / unrun gates

Nothing in this plan admits a GPU route or changes the CPU default. Every direction above
terminates in a named gate; until one passes, the standing verdict remains
`CPU_ROUTE_RETAINED` with GPU implemented+validated+not-admitted. No pushes from this branch;
publication requires a new explicit user instruction.
