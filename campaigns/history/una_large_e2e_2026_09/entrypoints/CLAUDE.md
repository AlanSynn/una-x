# UNA-X: active execution authority

The active campaign is `campaigns/una_large_e2e/`. Read its START_HERE.txt and EXECUTION.md when the user invokes `/goal`. `campaigns/REGISTRY.json` is the routing authority. Do not resume the completed CSR campaign.

The previous packet and all its evidence are archived unchanged under `campaigns/history/una_cpu_2026_09/packet/`. Old benchmark scripts under `benchmarks/una_cpu/` and regression tests remain for provenance; they are not the new qualification harness. Do not run their hard-coded sweep scripts on a new laptop.

Current numerical baseline: source at commit `361928e4ba38f34622cafe065b0025244db61368`, which already includes ordered CSR. Upstream c15ebda and pre-CSR 98f498e are historical references, not the primary performance comparator.

Execute on `perf/una-large-e2e` and isolated worktrees. Preserve user files. Implementation, review, and performance ownership are separate. No GPU, native backend, relaxed math, broad cache, public RunBatch parallelization, or scientific bug fixing in this campaign.

The user's authorization to publish this packet to main is not standing authorization for future autonomous pushes, merges, releases, force-updates, credential changes, or permission changes. Produce a reviewed local candidate and merge manifest; await separate publication authorization.

A plan is not completion. Missing proof/evidence means not admitted, blocked, or rejected, never passed. Follow the exact task DAG, gate thresholds, benchmark boundaries, failure policy, and finite stop conditions in the active packet.
