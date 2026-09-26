# Performance model

Model complete service time:
`T >= max(W/C, S, Q_memory/B_memory, Q_IO/B_IO)`.

For independent jobs:
`T_batch ≈ Σ_j ceil(K_j/N_j) t_j(N_j,H_j) + T_serial + T_overhead`.
Worker count W and per-worker thread count H consume the same CPU budget unless measured otherwise. Enforce admitted total execution <= C except when oversubscription itself is the experiment.

Primary removable work: baseline CSR setup performs full edge-array equality scans for every node in count and fill passes, creating O(VE) comparison work and Python list growth. Candidate stable incidence grouping changes construction to roughly O(E log E + V) using NumPy grouping and direct arrays while preserving row order. Measure, do not infer, actual memory traffic.

After CSR qualification, profile full public execution. Candidate priority follows removable end-to-end service divided by implementation/proof/validation cost. Use Amdahl:
`S_total = 1/(1-f+f/s+delta)`.
If residual CSR coverage is only a few percent, reject further CSR micro-optimization.

Memory accounting uses overlapping lifetimes:
`M_live = M_persistent + max_s M_workspace,s + M_queues + M_runtime`.
Measure whole process tree. Do not add non-overlapping scratch peaks.

Treat input load, topology/snapping, CSR, search, metric reduction, result assembly, serialization, filesystem writers, process startup, JIT/cache state, and verification as distinct boundaries. Final throughput uses required application boundary, not profiler/kernel time.