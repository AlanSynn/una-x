# Execution, ownership and task receipts

Perform implement -> focused tests -> debug -> repair -> evidence -> independent review for each
admitted task. Do not stop after writing a plan. Read the task dossier and its predecessor
receipts on the exact source tree. Resolve inputs from reviewed commits, not a branch label.

## Initial setup
Use S01's clean pinned baseline plus one integration branch perf/una-10x. Read git status before
any writes. Dirty user work is preserved; use a sibling worktree rather than stash/reset/clean.
If existing agents own batch/cache/native scopes, coordinate a single owner and import their
reviewed commit before modifying it. Fork alternatives from frozen dependency SHA in isolated
worktrees. Do not cherry-pick a full old campaign to get one helper.

## Ownership
Integrator alone edits public entry/dispatch/shared API files. Sparse engineer owns private sparse
implementation. Cache engineer owns trace/metric keys and resident store integration. Batch
engineer owns cohort plan/runtime. Native/GPU engineers own backend-specific code; kernels ABI
changes are integrated centrally. Independent reviewer does not author the candidate being
reviewed. Performance owner alone runs heavy benchmarks; other agents may read/reason during
that lease but not build, run broad tests or create I/O contention. Do not create uncontrolled
recursive agents or poll each worker repeatedly.

## Receipts and resumption
Each task gets an append-only run directory and a receipt from templates/task.json: task ID,
base/result full SHA, owner, read/write scopes, command/environment, fixture/profile identities,
verdict and next eligible tasks. Every status derives from real execution or an explicit reason.
Use pending/running/implemented/reviewed/qualified/rejected/blocked/not_admitted. Implemented is
not qualified. A rejection/blocked predecessor can release a decision-path task, never a test
requiring successful code. TASKS_CLAUDE.yaml distinguishes required_pass from resolved.

At a session boundary write current SHA, status ledger, outstanding review, last raw receipt,
resource consumption and next three tasks. Do not infer 'done' from a branch or missing worker.
No background promise. Resume from retained records and verify their dependency closure.

## Evidence custody
Place experiment output outside committed code; retain raw timings, logs, configuration,
engagement counts, minimal arrays and state traces for counterexamples. Hash-addressed final
bundles must remain available. Copies of the historical investigation remain byte-identical.
Never rerun run_sweep.py/reuse_probe.py/memo_probe.py in prior/: copy into a fresh run root first.
Proof/manifest fixes create a new generation and keep prior evidence. No self-referential hashes.

## Budget and failures
Use RESOURCES.md and the initial aggregate heavy budget in campaign.json. Missing external
hardware/data blocks only dependent tasks. Failed kernels run in isolated supervised processes;
release handles/workspaces and record outcomes. No swallowing failed rows in throughput reports.
Ordinary build or assertion failures are repaired locally, not escalated for approval. A semantic
contradiction is resolved with a minimal example/reference ruling or conservative refusal, not
with looser comparison. If evidence cannot support 10x, close target_unmet with the bottleneck.

## CI and publication
Focused local tests first; full installed/release checks near final integration. Discover actual
required checks from repository configuration at I00. New CI configuration may be prepared as a
reviewed platform deliverable, but never call a hosted check passed without execution. Use a
disposable merge rehearsal against recorded target main and recheck source/wheel identities.
No remote push/PR/main merge/release after this packet publication without separate authorization.
