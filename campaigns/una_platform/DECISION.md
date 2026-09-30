# Acceptance, completion and finite stop

Feature gates and performance gates are separate. A functioning explicit GPU backend that
loses to CPU in one cell is support, not a performance win. A CPU fallback is not GPU support.
Native here means an ahead-of-time compiled implementation doing actual core work, not a
renamed Numba dispatcher. Public parallel RunBatch requires observed overlap of independent
rows and deterministic ordered state/output commits. Cache support requires hit engagement,
complete invalidation tests, bounded storage and crash-safe publication.

Full qualification requires: complete reviewed Madina API matrix (no supported row missing),
existing UNA regression parity, independently validated corrected science, actual native and
GPU installed execution, bitwise matrix within each admitted semantic profile, public batch
concurrency/state/failure tests, cache correctness/recovery, multi-city performance, bounded
host/device resources, installed artifacts, independent review, and required CI/platform checks.
No mandatory feature may disappear as an “optional optimization”.

Numerical acceptance is exact under NUMERICS.md. Scientific corrections have documented
intentional old-vs-corrected deltas, not a claim that both old and corrected bits match.
Original reference hangs/crashes yield failed/censored rows plus corrected results, never
fake parity. A reproduced nontermination fix is important even when no original timing exists.

Performance objective: maximize the measured validated-throughput frontier after all mandatory
features are implemented, not merely exceed a fixed percentage. Apply BENCHMARKS.md gates to
automatic selection. Investigate meaningful regressions in protected small/cold/city/model
cells. Preserve slower validated expert routes only when they provide requested capability;
do not make them automatic. All accepted speedups compare identical scientific models/profiles,
outputs and resource policies. Record hardware-specific gains as such.

Terminal states:
- qualified: all mandatory technical gates passed for the explicitly claimed deployment scope.
- merge_ready: qualified plus exact reviewed merge rehearsal and required CI passed.
- partial: substantive safe deliverables exist, but one or more mandatory features/gates remain.
- blocked_external: remaining mandatory work needs absent device/data/reviewer/authorized resources.
- no_safe_merge: unresolved scientific/API/numerical/recovery defect in proposed default closure.
- no_performance_gain: capability work is correct but no validated throughput frontier improvement.
Each state records actual_target separately; proxy qualification never asserts production goals.

At most three bounded variants per measured bottleneck in each round; after a round, reprofile
the complete composed path and update the register. Do not stop solely because one kernel wins
or an old 1.10 threshold is met. Stop when required capability work is done and no remaining
admitted experiment has material measurable service to remove, or when an explicit resource
budget/external blocker prevents honest further qualification. Do not claim global optimality.

Final merge manifest names full source and packet SHAs, target main SHA/merge base, included
production files, optional expert backends, auto domains, fallback, profile/default migration,
exact wheel identities, tests, performance/raw evidence, bug deltas, incident dispositions,
review/CI status and remaining limitations. Rehearse in a disposable worktree. No actual push,
main merge or release without new explicit user authorization beyond this packet publication.
