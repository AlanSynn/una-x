# UNA large end-to-end CPU campaign

Start with START_HERE.txt, then EXECUTION.md. This is an executable engineering specification, not a claim that any new optimization is already correct or faster.

Primary baseline: `361928e4ba38f34622cafe065b0025244db61368` including ordered CSR. Integration branch: `perf/una-large-e2e`. Six gated hypotheses, no GPU/native-backend work. Scope includes accessibility and aggregate flow; turn-aware and K-alternative paths are protected against unintended change, not interchangeable engines.

Read map:
- AUTHORITY.md: scope, source identities, permitted decisions.
- SOURCE_AUDIT.md: source facts and corrections to inherited claims.
- CONTRACT.md: exact observables, numerical ordering, fallback.
- MODEL.md: work/span/memory and candidate coverage.
- WORKLOADS.md and inputs_catalog.json: observed-data inputs, scale axes, honest missing-target policy.
- HARNESS.md: exact runner interface and negative tests to implement before benchmarking.
- VALIDATION.md, BENCHMARKS.md, RESOURCES.md, DECISION.md: hard gates and finite budget.
- DOSSIERS in dossiers/: executable algorithms, proof obligations, counterexamples, refusal rules.
- TASKS_CLAUDE.yaml: JSON-compatible YAML DAG with owners and artifacts.
- OPERATIONS.md: worktrees, model roles, resumption, integration, publication limits.
- tools/: packet/source/evidence validators and exact NPZ comparison helper.

From the repository root:

```bash
python3 campaigns/una_large_e2e/tools/validate_packet.py
python3 -m unittest discover -s campaigns/una_large_e2e/tests -v
python3 campaigns/una_large_e2e/tools/verify_source.py --repo .
```

These checks validate the packet and pinned source/archive bridge, not UNA correctness or performance. New executable benchmark commands are deliverables of H02; HARNESS.md freezes their interface. Do not pretend those future files already exist.

All numeric thresholds/workload sizes in policy are proposed campaign constants, not measurements. Historical values are labelled as such. Production dataset, SLA, and target throughput remain unavailable until supplied. No credentials, private inputs, generated result geometry, or local absolute paths should be published by default.
