# Delegation

Coordinator owns DAG, worktree identities, file-owner map, resource leases, and final integration.

Useful independent roles:
- source auditor,
- numerical proof reviewer,
- implementation worker,
- fixture worker,
- runtime/concurrency specialist,
- packaging specialist,
- exclusive performance owner,
- independent final reviewer,
- integrator.

One owner per writable code scope. Alternatives use isolated worktrees. Shared API/dispatch/package files are integrated by one owner. Workers start from immutable verified SHAs.

Workers execute implement -> test -> debug -> repair -> retest -> evidence -> completion report. Do not repeatedly poll for status. Workers should not recursively create uncontrolled agent trees.

The performance owner has exclusive rights to heavy benchmarks. Model/agent parallelism may continue for read-only reasoning, but no competing compilation, profiling or disk-heavy execution.

Independent review must not be performed by the implementation worker under another label. If independent agent capability is unavailable, record review unavailable rather than fabricating it.