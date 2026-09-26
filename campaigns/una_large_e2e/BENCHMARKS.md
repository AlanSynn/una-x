# End-to-end experiment protocol

## Identities and primary comparison
B0=current CSR source; B=current selected candidate. Final headline compares B0 and B built/installed under identical dependency and resource policies. Historical pre-CSR arm is optional cumulative-context only and never replaces B0. For each individual experiment also report incremental gain relative to its selected parent, so overlapping savings are not added/multiplied.

Freeze actual workload manifest, selected origin set, required output formats, Settings, compiler/native versions, numerical stripes, application/session/verification boundaries and CPU/RAM budget before candidate performance results. Preserve original field/row order. Run every arm from a neutral cwd using its own clean noneditable installed environment and arm-qualified Numba cache. Cache identity includes Python/Numba/LLVM/ISA, source closure and kernel signatures; two wheels sharing package version cannot share cache identity by that version alone.

## Stages
Diagnostic call counts and exclusive timing cover Settings/validation, GIS reads/topology, point snapping/obstacles, directional cost/CSR, per-origin initialization/search/destination adjustment/metrics, gravity-cap derivation, flow gradients, OD loading, result assembly and existing synchronous exports. A diagnostic wrapper must not introduce extra constructors, cap passes or exporter calls. Verify wrapped and unwrapped outputs match; record profiling overhead. Final measurements use uninstrumented public APIs.

Probe searches across multiple sampled origins (policy default 32, or all if fewer), not a single first origin. Include interior/peripheral/unreachable/high-degree/duplicate-edge cases where present. Capture heap pops, neighbor incidences, labels written, touched unique nodes, candidate destinations, retained destinations, OD overlap sizes, gradient finite-entry counts and allocation counters when available. Instrumentation is explanatory and is not itself the final timing arm.

## Cold, warm and pool boundaries
Compile-cold: new process and empty arm-qualified JIT cache; include imports, compilation, preparation, outputs and exit. Cache-warm process: new process with existing verified compiled cache. Resident warm: pre-initialized worker pool with recorded warm-up. Report all separately. The primary batch throughput uses successfully validated completed jobs per full validated batch wall. Also report application-only span and verification cost, clearly labelled. Never subtract hashing from a window where it delays worker reuse.

Input checksum verification precedes session. Include checksum preflight in whole-session time, not inside just one arm's public job. Output verification occurs before the job is counted successful; required writes and returns complete before application end. File close does not mean fsync or power-loss durability. Do not disable outputs to improve final timing. Diagnostic compute-only measurements are allowed but cannot promote a candidate.

## Screening and sweep
H05 first produces baseline-only stage/memory profiles and admission decisions. Optional candidates use the current control and same configuration for attribution. Screen with three paired blocks per revision. Profile the post-candidate graph again so the next admission uses remaining, not removed, service.

S01 tests at most six feasible W/H profiles/family, including W1H1 and factor-like alternatives, bounded by C and RAM. Flow has fixed logical K; a different K is a separately labelled numerical profile and must not be silently mixed in a speedup table. Queue depth initially W, optional 2W only if memory and producer blocking justify it. Writer limit initially W; lower it only with a real implemented, reviewed ownership-preserving mechanism and explicit measurements. Include entire job setup/output and slowest-last-job effects. Choose fixed K_jobs divisible by the largest compared W when possible, without reducing work across arms.

Use sustained baseline-only pilots to set fixed batch size. Target >=60-second warm windows, but never truncate an individual scientific job. Longer inputs naturally test bandwidth, working set, thermal state and I/O; do not extrapolate 3-second grid batches. Record power/thermal indicators only when genuinely available; no invented P-core affinity.

## Final paired analysis
Five paired blocks, order AB/BA/AB/BA/AB. Keep same fixed workload/jobs within each pair. Store per-job records AND parent session records. Derive paired throughput ratios and report median ratio, absolute wall/success counts, raw distribution, paired fixed-seed bootstrap interval over log ratios and small-sample limitations. Do not treat jobs from one pool window as independent experiments. For each primary cell report same-configuration and strongest equal-policy arm-specific configurations; the latter includes scheduling gains but not fake multiplicative attribution.

Outliers: predeclare cancellation only for external load/resource/protocol failures with retained evidence. Do not remove a slow valid sample after seeing its arm. One bounded diagnostic repeat block may explain variance; preserve initial results too. Job failure invalidates that block and is not a timing outlier.

## Memory and capacity claims
Use simultaneous process-tree samples plus system pressure, with metric and sampling interval. Summed RSS can overcount shared pages; label it. Native/JIT caches, worker queues, outputs and gradient parts matter. Report measured payload counts separately from measured RSS and derived lifetime estimates. Baseline failure at a large cell yields censored time/unavailable speedup, not an infinite result. Compare exactness at a smaller common cell and report larger-cell capacity separately.

## Required per-run fields
run_id, arm, parent/source/full commit, src tree, wheel SHA-256 and installed file map, harness SHA, workload SHA/inputs, complete Settings, analysis/requested+actual engine, logical stripes and physical workers/pools, queue/writer limits and observed counts, CPU/RAM/admission, cache/JIT profile, app/validated/session/startup/compile/verify times, successful/failed/canceled counts, outputs/signatures, process-tree memory and pressure samples, return/exit statuses, environment, measurement class and limitations. A missing identity is invalid qualification data.
