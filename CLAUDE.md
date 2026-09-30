# UNA-X active engineering campaign

The active campaign is `campaigns/una_platform/` (ID `una-platform-2026-09`). Read its
START_HERE.txt, EXECUTION.md, AUTHORITY.md, CONTRACTS.md, NUMERICS.md and TASKS_CLAUDE.yaml
before work. `/goal` starts this campaign. Execute the tasks rather than returning another plan.

Audited baseline: `16bf404d2cc5e776a78dc073c80405f6d186da29`, already containing the two prior
campaigns' improvements. Madina reference: `City-Form-Lab/madina@8b5c3bd3b1c0048ae8b04054daed92bfb201b9d6`.
Develop on `perf/una-platform` and isolated reviewed-dependency worktrees. Preserve dirty user files.

Required deliverables: complete supported Madina API parity, scientifically justified numerical
fixes, real AOT native CPU and GPU backends, public parallel RunBatch, safe caches, reliable
cancellation/recovery, installed multi-city tests and further end-to-end throughput optimization.
Old campaign bans on GPU/native/public batch/caching/science fixes are superseded. Existing
runtime functionality must remain intact while the new capability floor is implemented.

“Bitparty” means bitwise parity. Separate una_legacy, madina_legacy and corrected_v1. Intentional
bug-fix deltas use a reviewed corrected oracle; do not claim corrected outputs also reproduce
erroneous legacy bits. Backend/cache/parallel variants match their selected profile exactly.
No allclose, reduced precision, route truncation, changed reductions, or different model passed
as an exact optimization. Madina ALL alternatives are not UNA K-alternatives or aggregate flow.

GPU/native must execute substantive work. CPU fallback, stubs, simulated GPU runs and detection
alone do not satisfy those features. Hardware absence is a qualification blocker, not permission
to mark success. Continue unrelated tasks through explicit blocked receipts. Automatic backend
selection requires exactness and measured full-region crossover; explicit supported paths may
remain slower in some domains without being made default.

Use one owner per writable scope, separate independent review, and one exclusive heavy-machine
lease. Preserve decisive raw evidence, not only hashes. Never overwrite historical evidence or
load a candidate as its own oracle. The historical large-e2e snapshot is under
`campaigns/history/una_large_e2e_2026_09/packet/`; its old path remains for legacy links.

This publication authorizes the campaign packet on main only. Future implementation commits
stay local until separately authorized to push/merge/release. No paid provisioning, credential
changes, provider reconfiguration, destructive cleanup or historical evidence purge.
