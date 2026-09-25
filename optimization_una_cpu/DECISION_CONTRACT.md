# Decision contract

Per-track states: qualified, rejected, blocked, inconclusive, not_admitted.

Campaign states:
- merge_ready: exact reviewed SHA passes required correctness, installed-path, performance, resource, merge-rehearsal and required-check gates for the claimed scope.
- qualified: scoped candidate passes its technical gates but final merge/review/CI disposition is incomplete.
- improvement_only: useful reviewed improvement exists but actual production objective/target is unavailable or unmet.
- blocked: an external requirement prevents a substantive required gate and no safe admitted work remains.
- no_safe_merge: unresolved correctness/API/DX/fallback issue remains.
- no_change: bounded candidates rejected/inconclusive; keep original implementation.

Missing actual workload or throughput target prevents claiming the original production-wide goal, but does not prevent a truthful proxy-qualified improvement.

Stop after mandatory CSR qualification plus at most one admitted optional scratch experiment. Do not invent more backend/cache projects after bounded hypotheses are decided.

Final artifacts: `evidence/final/RESULTS.md`, `decision.json`, accepted/rejected registers, exact commands/environment, raw timings, memory record, installed proof, review report, actual-target status, and `merge_manifest.json`. Distinguish measured, derived, hypothetical, unavailable.