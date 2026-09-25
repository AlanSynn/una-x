# Evidence ledger

Store new campaign artifacts under:
`evidence/inventory`, `contract`, `probes`, `trials`, `reviews`, `selection`, `integration`, `installed`, `freeze`, `handover`, `final`.

Every record identifies immutable source SHA, workload/fixture hash, environment, resource policy, command, boundary, result, and evidence class.

Allowed number labels:
- measured: direct execution observation,
- derived from measured values: deterministic calculation from cited measured samples,
- hypothetical sensitivity analysis: model-based what-if,
- unavailable: required evidence not obtained.

Prior measurements under `prior/una_optimization/results` are historical synthetic/component evidence. They are not silently promoted to laptop or installed production evidence.

Rejected experiments remain in the ledger with reason so later agents do not rediscover them. Evidence may be reused only when its dependency closure is unchanged: source, fixture/reference, compiler/math profile, runtime/thread policy, backend and harness.