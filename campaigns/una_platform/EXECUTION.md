# Execution protocol

## State machine for every task
pending -> admitted -> implementing -> local_validated -> independently_reviewed -> integrated.
Alternative terminal states are rejected, blocked_external or not_applicable_with_evidence.
Mandatory feature tasks cannot use not_applicable to evade a required capability.
A review rejection returns to the owner; a changed dependency invalidates descendants.

1. BOOT records checkout, runtime source digest, worktree locations, machine budget and remotes.
2. INVENTORY expands the seeded API matrix from pinned files, docs, examples and tests.
3. CONTRACT freezes semantic profiles, bug IDs, comparison fields and references.
4. HARNESS repairs test-frame/provenance weaknesses before evidence is generated.
5. FAILURES and scientific tasks minimize and reproduce incidents; compatibility tasks
   implement reference behavior without mapping different route models onto one another.
6. EXECUTION establishes backend-neutral typed inputs, ownership and cancellation contracts.
7. Caches, batch, native and GPU owners implement isolated scopes. They do not edit historical
   evidence. Corrected CPU semantics are the reference for corresponding accelerated paths.
8. PARITY and CITIES certify small genuine paths before PACKAGE and screening.
9. Screen equal-policy algorithms first, then jointly tune resources; select measured regions.
10. INTEGRATE freezes a complete composition; CONFIRM runs final installed multi-city tests;
    FAULTS verifies recovery; REVIEW and CLOSE emit the decision/merge manifest.

## No ambiguous completion
Every task returns: full base/head SHAs, changed files, commands/exit statuses, evidence paths,
proof or bug specification, numerical profile, reviewer identity, resource consumption,
limitations and next-state suggestion. “Works”, “tested” and “probably faster” are insufficient.

## Worktree rules
The baseline comes from source_baseline_sha in campaign.json, not the old pre-CSR SHA.
Each implementation worktree starts from the integrated reviewed dependency closure recorded
in that task's receipt. Do not resolve “main” again midway and silently change the oracle.
For parallel tasks editing the same engine file, use separate worktrees and produce patches;
only INTEGRATE's owner combines them after reviewing cross-task assumptions.

## Execution safeguards
Read the task's complete dossier, not only its title. Before modifying numerical code, write
its input/output/state dependency table and one deliberate counterexample to a tempting wrong
rewrite. Before accepting a test, make an intentionally broken candidate fail it. Before
claiming GPU/native execution, show that disabling that path causes the engagement test to fail.
Before accepting cached output, mutate each key dependency in isolation and test a miss.
Never copy candidate artifacts into the baseline directory to make tests pass.

## Review and resource separation
Use independent agents when available. Many agents may reason; only one holds the heavy-machine
lease. No competing builds or profiling during final timings. No recursive unbounded delegation.
If no independent reviewer exists, state review unavailable; do not self-approve under an alias.
Persist task state after each completed unit so interrupted sessions resume without repeating
all benchmarks. Use full SHAs and relative paths, not session names or private absolute paths.

## Publication
Work locally on perf/una-platform. This main-branch push publishes the packet only. Do not infer
permission for subsequent push/merge/release, credentials changes, paid GPU provisioning or
cleanup of historical evidence. Repository user changes and protected branches are preserved.

## Blocked dependency receipts
A dependency means its immutable outcome receipt is available, not that unavailable hardware
can veto all unrelated work. Implementation tasks requiring missing artifacts stop with a
blocked receipt. Aggregation tasks (PARITY, CITIES, PACKAGE, screens, BACKEND_POLICY, INTEGRATE,
FREEZE, CONFIRM, FAULTS, REVIEW, CLOSE) execute available arms and carry blocked cells forward.
A blocked mandatory feature never changes to passed merely because aggregation completes.
Review and final claim validators prohibit full qualification with those gaps. Partial source
compositions are labeled partial and never silently promoted to auto defaults.
