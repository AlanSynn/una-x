# Writable scope map, proposed seams and integration order

Assign actual owners at resume; these roles are not active agents or completed reviews.
One owner per file scope, and exactly one heavy-machine lease. All source edits occur in isolated
worktrees based on reviewed current dependencies. Parent integration branch remains perf/una-10x.

| Owner | Intended code scope | Work and explicit non-work |
|---|---|---|
| Integrator | UNA.py, Execution/report shared API, versioned dispatch | Attach private prepare session before Add* consumption; preserve current signatures/profile defaults. No independent duplicate batch engine. |
| Preparation engineer | Topology.py, proposed batch/prepared.py adapter with cache owner coordination | Consumer audit, snapshot/reader equivalence, typed payload/reconstruction and row state. No unreviewed cleaning/CRS/ID changes. |
| Cache engineer | cache store adapter and identity/lease support | Version complete typed keys; single-flight, integrity, bounded leases. No second broad cache framework or default disk persistence under cache off. |
| Replay engineer | Engines/_trace_access_scratch.py and focused helper | Compiled outer replay, then separately admitted partial folds; exact output dtypes. No core search rewrite or relaxed arithmetic. |
| Batch engineer | batch/runtime.py, worker.py, report.py | Phase transport; later bounded receiver only if justified. One source owner integrates concurrent proposals. |
| Harness engineer | benchmarks/una_10x and tests/tenx/post_p00 | Installed identity assertions, public phase telemetry, fault/mutation fixtures and policy controls. No benchmark-only fast path. |
| Independent reviewer | append-only review evidence only | Inspect immutable changes, counterexamples, generated code/keys, raw timings and claims. Return required fixes to author. |
| Performance owner | fresh run evidence | One accepted hypothesis at a time, no competing builds/profiling, full budget ledger. |

## Proposed private APIs (design names, not new installed functions)
- prepare_request(stage, source_observation, consumed_settings, profile) -> request or refusal.
- acquire_prepared(request, budget, cancellation) -> lease(handle) or explicit pre-execution refusal.
- restore_network/restore_points(payload, row_context) -> original-class row-owned state.
- compiled_replay(validated typed arrays, exact scalars) -> fresh four result arrays.
- metric_identity_v2(metric, typed_binding, trace_id, producer) -> digest or refusal.
- receive_complete_results(channel, credits, ready, stop) -> bounded receipts, no public writes.

Before implementing, map names to real source and list EVERY read/write dependency. Review the
ABI and error boundaries; do not assume a proposed name exists or monkeypatch by string at
runtime. Scope ownership is recorded in resume.json; no independent agent edits shared files.

## No-bypass rules
Preparation must be consumed through the existing PUBLIC RunBatch/RunAccessibility path. Do not
call benchmark-only constructors before timing or special-case filenames/cohort beta values.
New schema does not justify discarding old committed caches/evidence: version keys and retain
incompatible payload as a miss. Do not suppress read/validation errors simply to hit the cache.
All new counters report actual engagement; checking an option exists is not execution proof.
