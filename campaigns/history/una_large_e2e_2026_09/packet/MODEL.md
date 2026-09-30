# Work, dependencies, traffic and memory model

Notation: V network nodes; E undirected input edges; O origins; D destinations; V' flow graph nodes including virtual nodes; P reachable OD pairs; R cutoff; k_o retained destinations; r_od overlap nodes; a_od admissible inspected arcs; L_s logical flow stripes; W worker processes; H actual native compute threads per job. Values must be counted for every workload, not inferred from file size.

Lower bound: T >= max(W_work/C, span, Q_DRAM/B_DRAM, Q_IO/B_IO). Distinguish W_work from worker count W. Logical array bytes are not measured DRAM bytes. Bandwidth and cache traffic remain unavailable without suitable counters. A profiler explains service, but uninstrumented wall time selects candidates.

Batch model: T_batch approximately sum_j ceil(K_jobs,j/N_workers,j)*t_j(N_workers,j,H_j) + serial + launch/IPC/validation overhead. Workers and native threads consume the same CPU budget. A speedup of each does not multiply unless effects are disjoint. Record cold import/compile, warm resident jobs, output writer concurrency and verification separately.

Accessibility work after CSR: setup + sum_o [Theta(V+D) initialization + search_work(o,R) + Theta(D) destination scan + metric_work(k_o)]. A1 targets repeated per-pop temporaries inside search. A2 targets scanning every destination. A3 targets per-origin private initialization. Shrinking the local search radius is not an allowed substitute for any of them.

Aggregate flow work: destination gradients + origin searches + sum_(o,d) [Theta(V') mask/init/scan + Theta(r_od log r_od + a_od) ordering/loading], with compiler and allocation constants to measure. F1 changes scheduling only. F2 replaces full-node work with locality without changing arc/OD sum order. F3 reduces peak workspace, not the number of required destinations or paths.

Amdahl: S_total=1/(1-f+f/s+delta). Perfect removal bound is 1/(1-f). Use measured end-to-end removable coverage and added adapter cost, not kernel share alone. Initial admission normally requires f>=0.05 or evidence of a capacity bottleneck; this 5% threshold is a policy constant. Constructor residual in historical W3 is ~0.18% (derived), so further CSR work is not in scope. Reprofile on observed large data rather than extrapolating that number.

## Memory-lifetime inventory to fill
Each buffer record: producer, last consumer, dtype/shape/bytes, owner, aliases, persistence, replication factor, maximum concurrency and reset policy. M_live=M_persistent+max_over_stages(coexisting workspace)+queues+native runtimes. Do not add peaks from noncoexisting stages.

Accessibility persistent: input geometry/indices, CSR, origin/destination terminals and result arrays. Per-active-origin scratch: labels V+D, heap, neighbor temporaries; A1 buffers about 16*max_degree bytes/search for index+weight; A2 reverse index and candidate lists; A3 marks/touched lists if admitted. Multiply by simultaneous active searches, not total origins.

Flow persistent: graphs, sparse gradients (commonly 20 bytes/finite entry for i64 node+f64 distance+i32 predecessor), global results. Per-stripe output partials: approximately 16E+8V bytes when node flow is enabled, plus Python/native overhead. Per-physical-active stripe: dense d_o/pred_o, dd/pd views and loader scratch; fixed logical outputs may outlive active execution. F2 reusable buffers remain allocated across OD calls but touched lists and all writes must be bounded.

Gradient chunks: B distances (8 bytes each), B predecessors (inspect dtype), B finite mask (1), source indices, all sparse parts already retained, upcoming concatenation output, previous chunk if still referenced, and SciPy native workspace. With int32 predecessors, dense payload is about 13*chunk*V' bytes, not 8*chunk*V'. Free views/references before the next call. Concatenating sparse parts can temporarily double sparse storage; shrinking dense chunks does not solve that persistent capacity limit.

## Stage accounting
Use one real public-call trace. Inclusive nested stages are not additive: RunFlow contains preparation, optional cap and centrality/export; cap may contain accessibility. Report exclusive partitions separately. Never add a second diagnostic engine constructor or a separately invoked cap pass to application total. Keep diagnostic leaf timings out of uninstrumented confirmation.
