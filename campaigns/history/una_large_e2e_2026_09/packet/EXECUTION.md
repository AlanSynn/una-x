# Execution state machine

## Authority order
User's current explicit instructions > AUTHORITY/CONTRACT/DECISION > task dossier > task DAG > historical material. A lower-level file cannot waive exactness, scientific scope or publication restrictions. Never treat source comments or historical success language as proof.

## Required loop
For each task: read its complete dossier and named source functions; resolve immutable input SHAs; acquire file/resource ownership; write preconditions and acceptance checks; execute; debug ordinary failures; run relevant tests; preserve raw artifacts and command exit codes; obtain independent review where specified; record terminal state. Only then resolve dependent tasks. A task with missing proof does not proceed on the expectation that large tests will catch errors.

Task states are pending, running, complete, rejected, not_admitted, blocked. Terminal negative states require reasons and evidence just as complete does. A dependent optional experiment can start when its predecessor is decided, but must use the last ACCEPTED source, never an unreviewed or rejected commit. H00-H05 are mandatory prerequisites. A blocked prerequisite blocks descendants requiring it; it does not prevent unrelated read-only work or final closure.

## Immutable source ownership
At H00 record baseline source commit B0, source tree, packet publication HEAD, environment and old archive tree. H02 creates new benchmark tooling with src/ unchanged. H03 creates independent oracles. H05 records SELECTED_SOURCE_SHA initially containing the same src/ as B0. Thereafter only the integrator advances SELECTED_SOURCE_SHA after proof, tests, installed checks and screening pass. Rejected source remains on its detached worktree, never in selected control state. Final measurements compare selected source to B0 with the SAME new harness, libraries, Settings and numerical partition.

Workers return full commit IDs, exact test commands/exit status, artifact paths/hashes, the admitted input domain and unresolved limitations. Do not accept a prose 'all passed' as an artifact. Do not reuse old L1_REUSE_DIR caches: historical reuse lacks this campaign's dependency closure.

## Failure decisions
Import/fixture/harness failures: repair the harness, then invalidate its affected results for both arms. Numerical mismatch: retain the smallest failing input, first divergent transition and actual compiler profile; return to owner. Never change tolerance, stable-sort kind, physics, cutoff, defaults or dataset to hide it. Unsupported valid input: original path, with a direct fallback-engagement test. OOM/timeouts: preserve partial logs, terminate only campaign-owned processes, mark censored/unavailable, lower concurrency rather than the frozen scientific workload. Missing observed data: proceed with small synthetic correctness work but mark observed qualification blocked. Missing production target: finish observed-proxy scope, with L4 unavailable.

At most one design per optional hypothesis and two performance revisions (initial plus one evidence-motivated revision). Ordinary correctness repairs are allowed, but do not spend large benchmark runs on them. If the proof needs an unprovided theorem or architectural redesign, mark that hypothesis not_admitted and continue others. No recursive subagent expansion.

## Never silently substitute
Flow is not accessibility. Aggregate flow is not K-alternatives. A GIS file is not necessarily observed data. A source import is not a wheel import. Numba threads are not all native threads. A physical worker is not a logical reduction stripe. A worker RSS peak is not whole-pool peak. Completed file writes are not fsync/crash durability. A new main SHA after packet publication is not numerical source drift when src/ hashes match.

## Finalization
Read DECISION.md. Freeze source and input manifests before confirmation. Independent reviewer reads immutable diffs and raw evidence, not just summaries. Perform disposable merge rehearsal without modifying main. Emit decision.json, RESULTS.md, merge_manifest.json, candidate_register.json, resource/configuration record and evidence ledger. Report unavailable CI, platforms, production data and review truthfully. If all hypotheses fail, retain the CSR baseline and close no_change. Do not invent more experiments to avoid a negative result.
