# Decision and completion contract

The gate is fixed before results: paired median ratio >=10.0 AND one-sided 95% lower bootstrap
bound >=10.0 on each claimed prespecified primary workload. Seven or more independent paired
windows, fixed sample count/seed/method, identical validated job counts, equal resource budget,
profile and scientific inputs. See BENCHMARKS.md; tooling computes, not certifies, the result.

Possible tenx_status:
- observed_cohort_10x: primary public RunBatch cohort passes, real observed input, installed,
  first-trace cost included, required artifacts/state parity and protected cases pass.
- observed_single_10x: independently prespecified public single-job case passes the same gates.
- observed_both_10x: both classes pass. Do not infer one from the other.
- actual_target_10x: actual supplied unreduced workload passes; identifies the exact deployment.
- improvement_only: useful reviewed improvements, less than 10x or scope gate incomplete.
- target_unmet: bounded investigations complete and target not achieved; report the bottleneck.
- blocked_external: an actual external requirement prevents required evidence; preserve progress.
No 'qualified' based on prior component ratios or unrun code. Confidence bounds are empirical
uncertainty summaries, not proofs that all future runs exceed 10x.

Compute platform_status separately. It stays incomplete/blocked unless every mandatory inherited
API/science/native/GPU/batch/cache/reliability/install gate is complete. No feature is silently
removed because the CPU throughput subgoal was met. Once 10x has sufficient margin, stop adding
unrelated speed variants but finish required capability qualification and regression checks.

Default promotion also requires exact admitted-domain behavior, correct refusal/fallback,
small/dense/cold protected cases, bounded measured host/device memory, recoverable outputs and
independent source/evidence review. A supported explicit fast option may be retained without a
blanket default if its domain is well specified; it still cannot claim 10x outside measured cells.

Required final directory evidence/final/<immutable-run>/:
RESULTS.md; decision.json; merge_manifest.json; candidate_register.json; platform_obligations.json;
source_environment.json; WORKLOADS.json; PRIMARY.json; configuration.json; commands.jsonl;
paired_samples.json; gate_10x.json; bitparity.json; state_artifact_checks.json; cache_matrix.json;
backend_matrix.json; incident_updates.json; memory_lifetimes.json; raw_index.json; review.json.
All evidence numbers identify measured, derived_from_measured, hypothetical, or unavailable.
Retain compressed raw records and counterexamples with verifiable paths, not dangling digests.

Each claimed cell records all original outputs, exact input byte/order fingerprints, matched
model/profile, wall boundaries, successful completed jobs, stage/crossover evidence, first-use
cost, peak memory, worker/thread/queue/writer/device configuration, baseline tuning and executed
wheel identities. Do not multiply sparse, reuse, backend and concurrency ratios with overlap.

Before merge-ready: disposable integration at recorded target main, full dependency closure
checks, installed-wheel equality, required tests and actual CI results, independent reviewer on
that SHA. Absent CI is unavailable, never passed. Packet publication did not authorize any later
code push/main merge. Return a truthful merge recommendation and exact source SHA.
