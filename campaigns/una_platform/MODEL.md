# Cost model and experiment register

T >= max(W/C, S, Q_memory/B_memory, Q_IO/B_IO).
For a staged job batch: T_batch approximately sum_j ceil(K_j/N_j)*t_j(N_j,H_j)
+ T_serial + T_overhead. Work, critical path, RAM, bandwidth, serialization, queue/writer
concurrency and worker count are independent quantities. Never multiply two CPU speedups
that consume the same core budget.

Source-derived accessibility work after accepted CSR/search changes still includes graph
preparation, per-origin bounded search, all-destination adjustment/filtering, decay/KNN,
and outputs. Inspect current tailless admission before charging V+D state: legacy/public
routes and private admitted routes differ. Initialization O(V) and destination scanning O(D)
per origin may dominate for local radius on regional networks even when visited work is small.

Current flow already includes local overlap scratch and nogil work. Next candidates include
reusing exact predecessor/gradient facts within a job; avoiding repeated coefficient/decay
work across two passes; direct compact consumption of sparse gradients; allocation-lifetime
improvements; native irregular loops; device regions over many independent outputs; and
shared prepared-graph reuse across batch phases. Do not remove pass order just because a
contribution is recomputable. Analyze both recomputation and stored-q traffic at the active
memory level; q_sum must finish before scaled loading begins.

Memory live = persistent + max(non-overlapping stage workspace) + queues + runtime.
Account separately for simultaneous source/destination gradients, per-stripe accumulators,
retained sparse parts, concat copies, export frames, native arenas, device contexts and
in-flight host/device buffers. Read-only mmap is not zero resident memory. Output-volume
lower bounds cannot be optimized away under a materialized API.

GPU full time = prepare + upload + launch + kernel + download + sync + conversion + remaining.
For K repeats include one-time compile/residency setup J. Backend choice compares full regions,
including interaction with adjacent stages, not independent kernel estimates. Native SIMD
lane efficiency = sum_i L_i / (width * max_i L_i) is only a first-order divergence model.
Inspect actual compiler reports and gathers/spills; do not infer SIMD from source syntax.

Mandatory feature register: API parity, corrected science, native, GPU, public parallel batch,
cache, recoverability. Each requires implemented and validated evidence, even when its best
schedule falls back on some profiles. Optional optimization register is ranked after profiling
by removable end-to-end service / implementation+proof+validation cost. Consider in order:
remove dead/repeated work -> avoid materialization -> locality -> independent parallelism ->
CPU/GPU region -> backend specialization. No permanent refusal of a new idea solely because
an earlier small workload showed low coverage.

Use S_total=1/(1-f+f/s+delta). Distinguish measured f from source-derived work and hypothetical
s. Hardware bandwidth, GPU crossover and novel speedups are unavailable until measured.
Close each experiment with absolute timings, exactness, resources, accepted/rejected domain,
reason and remaining counterexamples. A faster but different scientific workload is rejected.
