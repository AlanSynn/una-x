# UNA-X Claude Code handoff

This repository is a development fork of `City-Form-Lab/urban_network_analysis`, seeded from upstream commit `c15ebda6981397f46eed5c2d55229f71e57d44fb`.

The active engineering campaign is the CPU throughput campaign in `optimization_una_cpu/`. Treat the current repository checkout as the target source tree. Upstream is the immutable semantic/oracle reference, not the place to write changes.

When the user invokes `/goal`:
1. Read `optimization_una_cpu/START_HERE.txt`.
2. Read the execution prompt, design authority, source audit, plan, task DAG, semantic/numerical contracts, benchmark protocol, promotion policy, branch/CI rules, and relevant dossiers.
3. Execute the campaign rather than returning another plan.
4. Preserve the frozen public API, numerical behavior, outputs, errors, state transitions, and fallback behavior.
5. Work on `perf/una-cpu-throughput` or isolated worktrees. Do not rewrite or force-update `main`.
6. Use the current `main` as the fork baseline while retaining upstream commit `c15ebda...` as provenance.
7. GPU work is out of scope for this campaign.
8. Do not push, merge, release, modify credentials, or change provider settings unless the user explicitly asks.
9. Keep measured, derived, hypothetical, and unavailable claims distinct.
10. Finish with an exact reviewed source SHA, raw evidence, installed-wheel proof, accepted/rejected register, selected CPU configuration, limitations, and merge manifest.

The supplied CSR candidate is not automatically accepted. Verify its exactness against the immutable baseline, then measure complete installed end-to-end regions under equal resources.