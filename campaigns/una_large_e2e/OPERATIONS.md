# Execution operations, roles and resumption

## Workspace
Operate only in this checkout or user-supplied paths. Before modifications: record git status --porcelain=v1, full HEAD, branch, remotes, git diff --stat, B0 source tree, Python and dependency versions. Do not search home directories for datasets or credentials. Dirty/untracked files are preserved; create a campaign-owned worktree instead of stashing or resetting them.

Create a new worktree/branch perf/una-large-e2e from the packet-publication main only if it does not already exist. If it exists, inspect ledger/SHAs and resume, do not recreate or reset it. Baseline is a detached worktree at B0. Build wheels there without editing baseline sources. Temporary environments/caches/results live in an explicitly recorded campaign workspace outside runtime imports. Do not check private inputs or large output geometries into Git.

At each implementation start, resolve the full selected control SHA from evidence/control.json, verify its source tree and accepted-track list, and write both to the worker record. Read-only proof tasks can share source; alternatives and reviewers use detached worktrees. Downstream work never silently starts from old main, pre-CSR or an unreviewed rejected patch.

## Roles
Coordinator: DAG, ownership map, stop budget and evidence ledger. Source auditor: source/oracle/provenance facts, no runtime writes. Harness owner: new runner and negative tests. Fixture/oracle owner: independent golden generation, small adversarial state tests. Implementation owner: one dossier and one writable scope. Numerical reviewer: independent source/proof/first-divergence review, not co-author under another name. Performance owner: exclusive machine lease for all heavy execution. Packaging owner: clean wheel/import/path proof. Integrator: sole shared-source/control-state owner and final merge rehearsal.

Use strongest available reasoning for proof and final review; bounded mechanical work can use other agents. Record actual provider/model identifiers if visible, otherwise unavailable. A model alias is not a verified model identity. Do not alter global routing or credentials. Lack of subagents permits sequential implementation but does not manufacture an independent review; affected promotion remains review_pending.

Workers do not recursively spawn uncontrolled agent trees. Return only completion or genuine blocker, with immutable commit, commands, exit codes, proof, tests and artifact hashes. No repeated status polling. Coordinator may reason/read while benchmarks run; no other compilation, profiling, large test or disk-heavy job under the performance lease.

## Restart protocol
Read registry, control.json, ledger and last worker artifacts. Verify files and git hashes before trusting complete statuses. A missing artifact or source/environment/fixture change invalidates affected descendants. Never overwrite earlier raw evidence: use a new run ID. A partial run is resumed only if the harness explicitly supports it; otherwise mark canceled and rerun within budget.

## Archival paths
Archive packet is immutable. Tests/perf_contract and benchmarks/una_cpu stay in their prior locations. New code goes under tests/large_e2e and benchmarks/large_e2e. Historical tests requiring UNA_BASELINE_SRC should point to B0 for current equivalence, with L1_REUSE_DIR unset and separate cache directories. For pre-CSR reproduction use the historical baseline in a separately labelled experiment only.

## Publication
This packet is being committed to main by explicit user request. Future execution has no standing push/merge/release permission. Create local reviewed commits and a disposable merge rehearsal, then deliver the merge manifest. Do not open a PR, change protected-branch rules, force push, rebase user work, or clean another workspace. If main advances during execution, record drift and review a new merge rehearsal; do not rewrite main.
