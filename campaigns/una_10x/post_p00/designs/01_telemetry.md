# P01: minimal causal telemetry and baseline-policy audit

## Implementation targets
Batch owner: RowResult.timings, worker._run_row/pool_main, runtime._Pool/run_batch_parallel,
report dataclasses. Preparation owner: opt-in phase hooks at existing Add* boundaries.
Integrator: diagnostic context transport through the public call. Normal output schema and
signatures remain compatible; any optional diagnostic report field must default absent.
Do not add per-node/edge events. At most a bounded number of events per row/stage.

## Event schema (proposal)
`schema, run_id, arm_sha, row_id, pid, tid, phase, start_mono_ns, end_mono_ns,
process_cpu_start_ns, process_cpu_end_ns, thread_cpu_start_ns, thread_cpu_end_ns,
bytes_in, bytes_out, object_key, parent_phase_id, status`.
All counters are nonnegative integers; unavailable CPU fields are null, never zero observations.
Clock domain is the same host's monotonic clock. Record process-start identity to avoid PID reuse.
Nested spans carry parent IDs. Sum exclusive same-thread spans only; process CPU from overlapping
threads must not be added twice. Report a whole-tree sampled CPU series separately.

Worker phases: bootstrap/import, row_validation, input_identity, source_decode, topology,
snap_origin, snap_destination, engine_build, prepare_restore, trace_key, trace_wait,
trace_build, trace_load, trace_integrity, replay, export, staged_hash, state_pack, result_put.
Coordinator phases: plan, dispatch, result_recv, staged_verify, file_fsync, rename, dir_fsync,
state_delta, composite, journal, rehydrate, pool_close. Record dispatch/ready/inbox-get times.
A result_put duration cannot be put into the result already sent. Emit a separate bounded
per-worker diagnostic log/event. Flush logs outside primary timing but preserve crash fragments.

Do not log through a second blocking result channel capable of creating the stall being studied.
Use bounded buffers with a dropped-event counter. A diagnostic run with dropped events cannot
prove interval completeness. Use fresh run roots; never overwrite P00 pstats or outputs.

## Three discriminating experiments
H1 load/parse contention: W1/W2/W4, identical original inputs, same outputs and total CPU budget.
Collect read/decode CPU vs wall, bytes, native thread counts, I/O wait and caller-attributed locks.
A Python GIL is not shared across spawn processes. A raster GDAL lock explanation is not proof
for GeoJSON vector loading. Separate disk access, decoding, conversion and allocation.

H2 commit backpressure: align worker end-of-export -> put start/end with coordinator receive,
hash/fsync and buffered-byte intervals. If puts block while commit owns the only reader, test
bounded independent drain AFTER P02/P03 to establish residual coverage. Do not infer it merely
from idle CPU. Add controlled slow-hash/slow-fsync injection in tiny diagnostics, never in the
performance comparator. Preserve the established synchronous completed-result ownership law.

H3 replay contention: after a trace is validated, compare the same total rows/origins at W1T4,
W2T2, W4T1 with active thread counts recorded. Also compare Python loop vs compiled outer driver
at IDENTICAL configuration. Measure Python dispatch/allocations and native CPU separately.
The 1.5 MiB trace fitting in cache does not prove an absence or presence of bandwidth saturation.

## Baseline-policy audit
Preserve PRIMARY as written. Inventory all allowed current input-stage cache regimes in both
arms; measure matched cold input-cache and matched warm input-cache policies as a separately
registered control when missing from the frozen primary. Final-result caching is forbidden for
the distinct-job primary; existing input preparation caching is not the same-answer shortcut.
Record fastest VALID baseline configuration and its exact regime. If it changes the denominator,
keep both outcomes and scope the fixed-policy result explicitly. No post-result threshold edits.

## Exit evidence
phase_events.jsonl, phase_summary.json, sampler_ablation.json, baseline_policy_audit.json,
primary_identity_check.json, overhead budget, hypotheses with supported/refuted/unresolved states.
A telemetry hypothesis is accepted only with matched causal evidence, not one screen sample.
