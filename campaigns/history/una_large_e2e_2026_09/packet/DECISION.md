# Preregistered selection and finite termination

Numeric gates here and campaign.json are POLICY CONSTANTS, not measured speedups or user production requirements. campaign.json is the machine-readable authority. Any pre-result amendment requires a dated rationale, independent review and an invalidation map; never loosen a gate after seeing a failed candidate.

## Admission
Mandatory harness/source/observed-input/oracle gates precede optional runtime work. Admit a candidate when profiling of the CURRENT selected implementation shows at least 5% removable complete-job service, or a demonstrated memory-capacity bottleneck. F1 additionally requires observed serialization/low concurrent kernel execution; decorator inspection alone is insufficient. A3 requires measurements over multiple origins and scales, not the historical single-origin probe. F3 may be admitted for capacity even with little wall-time headroom.

A proof reviewer must approve the admitted domain, preserved operation sequence, fallback and test matrix before implementation. One design per hypothesis, at most one performance revision after the initial screened version. A1/A2/A3/F1/F2/F3 are the complete portfolio; no seventh idea. A not-admitted experiment records its measured coverage/bound and consumes no production benchmark.

## Screening versus qualification
Screening uses at least three paired A/B complete-job blocks on the frozen selection workload, same resource and numerical policy, outside leaf profilers. A is current selected control for marginal selection; B0 remains final comparator. A screened-in patch is provisional until installed/composition/final gates pass. Reject on any semantic failure, or when the median complete-job time does not improve beyond the measured noise band. F3 capacity-only improvement is labelled capacity_only, never throughput-qualified.

For final installed qualification require five paired blocks per mandatory primary workload and two-arm cold diagnostics. Pair order is preregistered as AB,BA,AB,BA,AB, with equal repeat counts per arm. Warm windows target at least 60 seconds where the fixed batch can afford it; select a fixed job count from baseline-only pilots, rounded to a multiple of the largest tested worker count. Do not extend or truncate only the candidate window. If a single job exceeds the target window, run complete jobs and record actual duration.

Primary proxy performance gate: at least 1.10x successful-job throughput (or reciprocal complete-job latency for explicitly single-job profiles) against strongest equal-policy B0 configuration, and a paired 95% confidence-interval lower bound above 1.00. Report method and all raw blocks; pair is the sampling unit, not each correlated job. Use a fixed-seed paired bootstrap of log ratios as a descriptive interval and report its limited small-sample precision. Same-configuration speedup is also required for attribution. A configuration switch alone is not an algorithmic result.

Protected cells: exact correctness, no capacity violation and median time ratio candidate/B0<=1.05 on small input, held-out observed cell when available, turns/ODM/other untouched profiles where runnable. Cold process median must not regress by more than 5% without an explicitly bounded warm-only qualification; warm-only means no blanket-default promotion claim. No individual >10% slowdown may be silently discarded: investigate under one preregistered repeat block. Missing held-out data limits generalization rather than pretending cross-morphology validation.

Capacity-only disposition needs measured lower peak/feasibility and exact results on a size both arms complete. Where B0 OOMs or times out, speedup is unavailable/censored, not infinity. Explain the budget failure and unchanged scientific workload.

## Run budget
Baseline pilots: at most two per workload cell. Optional prototype: at most three paired screening blocks per revision, two revisions total. Resource sweep: at most six W/H configurations and two batches/configuration/family, then one selected configuration per arm. Final: five pairs per mandatory workload; one extra preregistered diagnostic repeat block only. L4: one complete paired target evaluation after immutable freeze, plus one rerun only for documented external failure. Default total heavy-execution wall budget is four hours; actual-user target may override before testing. H00 records it. Reaching budget triggers a finite inconclusive/partial outcome, not an approval request for endless runs.

## Terminal states
qualified_proxy: required exactness/review/installed/resource gates and proxy performance gate pass, production target unavailable.
improvement_only: useful reviewed partial/capacity/warm-only result, broader objective not met.
no_change: all admitted candidates fail or are not useful; retain B0.
blocked: required external input/environment/review prevents completion of affected scope.
inconclusive: bounded evidence cannot distinguish a reliable improvement.
no_safe_merge: unresolved correctness/default/DX failure.
merge_ready: exact reviewed SHA, actual claimed scope gates and required CI passed, disposable merge rehearsal passed; this is not permission to merge.

Stop when the actual supplied target is met with its preregistered margin or the finite portfolio/budget is exhausted. Do not continue implementing other ideas merely because they are listed. Do not stop at a kernel win.

## Required final files in evidence/final
RESULTS.md; decision.json; merge_manifest.json; candidate_register.json; configuration.json; commands.jsonl; raw run index with hashes; environment/workload/wheel/source manifests; memory-lifetime table; review report; unavailable-gates list. Merge manifest distinguishes runtime-required, test/evidence and research-only files, target main/merge base, active path/fallback, exact commands, installed tests, performance/CI/review scope and disposition. Preserve archive unchanged.
