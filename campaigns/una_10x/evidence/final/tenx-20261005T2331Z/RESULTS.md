# RESULTS - una_10x final confirmation (run tenx-20261005T2331Z)

Date: 2026-10-07.  Branch perf/una-10x.  Every figure below is measured or labelled
derived_from_measured/unavailable; sources are the files in this directory (byte-identical
copies of the run evidence, sha-asserted by z00/build_final.py).

## Headline

**tenx_status = target_unmet.**  The fixed 10.0x gate was evaluated at full strength and NOT
reached.  The gate was not lowered.

- Primary gate (gate_10x.json): median paired ratio **2.3798**, one-sided
  95% bootstrap lower **2.2345** (7 paired windows, seed
  20261003, 20000 resamples, target 10.0,
  gate tool exit 2 = below target).
- Protected cells (raw_index.json): B/A median **0.8487**,
  one-sided 95% upper **0.9270** ->
  default_dispatch_ok = **True** (B never
  degrades protected cells; outputs bitwise equal across arms in all 5 cells, zero failed jobs).
- platform_status = **incomplete** (independent of tenx_status; see platform_obligations.json).
- merge_disposition = **do_not_merge** (gate unmet; no merge authorization exists).

## Claim scope

observed_proxy only: frozen distinct 32-job observed cohort (workload sha256
`235405de67cf22dc76c3e89bd0c3d198927d0ed0870172a0e86b942fecb8a60a`), installed wheels (A `395cdc5f683894b6f2ba460f7dbcefee99981ba3` ->
`f0504f886eb9976150a56dcfa818188cdedb172a13716206003b8b1e1a1dc4b1`; B `1e7237bebe8dd55e4a3b444f24ad9a255fa8644c` -> `9a84517b9a524d4632c990a2332f8ecc219d6d3dd844528e8dac9c8dafc58d97`), COLD public-call boundary
(`public_call_complete`, initial trace cost included).  NOT actual_target (no actual
unreduced deployment workload was supplied).  Per DECISION.md, observed_single/both/actual
classes are not inferred from this result.

## Measurement identity and fairness

- Boundary: timer brackets ONLY the public RunBatch call; import recorded separately
  (matched: A 2.5405-7.2248 s, B 2.6963-7.5062 s across the 7 pairs;
  both outside the timer).
- Resources matched: workers=4, threads=1, queue-depth=64, cache-mode off, fresh
  NUMBA_CACHE_DIR per leg, fresh outdir per leg; pre-leg loadavg gate < 8.0 (90-min bounded),
  in-run 60s sampler retained per leg (cadence limitation disclosed: legs shorter than 60s
  yield 1 sample).
- Attestations per window: installed (frozen venv + I00 identity chain + live wheel re-hash),
  bits (per-file sha census == I00 64-file reference map, both arms), state (zero failed +
  una_legacy effective snapshot), artifacts (64 files / 147,829,892 B per window),
  resource (q00 resource policy), boundary public_call_complete, COLD, distinct jobs.
- Invalidation: none fired (invalidation_flags empty); the preregistered one-FULL-repeat
  clause was never triggered.
- Paired walls (s): A 36.2146-40.8958, B 15.2087-17.8336;
  per-pair ratios 2.3388, 2.1823, 2.5399, 2.3798, 2.5119, 2.4777, 2.2345.

## Why not 10x (bottleneck report)

- measured: candidate public wall is 15.2087-17.8336 s/window under
  equal budget.  Imports are small and outside the timer.
- measured (p01m/phase_summary.json, instrumented smoke frame, 4 rows, workers=2):
  worker bootstrap_import 5.1974 s dominates worker-side wall
  vs run 0.7790 s total (per-row ~0.19 s); coordinator pool_start 0.2112 s +
  pool_close 0.3524 s.
- derived_from_measured: in the frozen COLD regime each window re-pays worker engine
  imports, pool lifecycle and first-trace JIT specializations inside the public call
  (initial_trace_cost_included; per-call pools close at call end).  These fixed costs cap
  the public-boundary ratio; per-row arithmetic is a small fraction of the wall.  Reaching
  10x would require reducing worker-side startup/trace cost or amortizing across calls -
  outside this regime's contract (and the gate forbids subtracting them).
- investigated and rejected routes: flow sparse-output producer + update tapes (F00
  not_admitted), P04 independent drain (residual backpressure never demonstrated), GPU
  region (hardware absent - qualification blocker), same-answer cache hits (law-excluded),
  unchecked queue enlargement (law-excluded).  Native region is qualified on CPU install
  (n10) but the frozen measured wheels are numba-only.

## Honesty ledger

- Fire history with 3 disclosed harness defects (no windows re-run for any defect; 24 legs
  total): raw_index.json fire_history + 4 consoles (sha-recorded in raw_index).
- Errata: q00 v1 protected inputs lacked the CSV header row -> v2 regeneration; v1 kept;
  primary unaffected (errata_pc_inputs.json).
- Sampler cadence limitation disclosed.  Co-tenancy handled by gate only.
- Legs executed at tree head ee5652a (dirty worktree evidence dirs); operative pins are
  wheel/harness/workload identities, all re-verified live (source_environment.json).
- Budget: cumulative ledger 4644.7692 s
  of 28800 s heavy budget (ledger only grows; untimed residue disclosed not merged).

## Files

PRIMARY.json, configuration.json (preregistration freeze); paired_samples.json,
gate_10x.json, raw_index.json (measurement + fire history); bitparity.json,
state_artifact_checks.json, backend_matrix.json (verification gauntlet);
source_environment.json, WORKLOADS.json, cache_matrix.json, commands.jsonl;
incident_updates.json, memory_lifetimes.json; candidate_register.json,
platform_obligations.json; merge_manifest.json, decision.json; review.json
(independent R00 review).  All copies byte-identical to run evidence.

Independent review: R00 verdict **approve_with_notes** (r00/review.json, sha 0cf3896b468c122d052c40f68eb7fcde1fcb1966f53e139ab4ad281ad7a22c0c; reviewer gate_10x re-run byte-identical to the committed result; 10/10 checks pass, 0 blocking findings; advisories non-blocking and disclosed in review.json).
