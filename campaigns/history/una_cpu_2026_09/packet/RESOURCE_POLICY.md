# Resource policy

Discover actual laptop limits before execution. Record physical/logical cores, CPU affinity/quota if any, RAM/swap, disk free space, platform/ISA, and Numba/native threading layers.

During correctness editing use small L0/L1 tests. CPU-heavy benchmarks, compilers, broad tests, and I/O-heavy jobs are globally bounded. The benchmark owner has exclusive performance-machine execution; other agents may reason/read but must not compile/profile concurrently.

For admitted CPU budget C, require sum(workers_i * effective_threads_i) <= C unless oversubscription is explicitly the experiment. Native BLAS/OpenMP pools must also be limited/recorded.

Do not start W concurrent jobs until W=1 memory is measured. Maintain a safety margin rather than filling RAM. Queue depth is bounded. Writers are bounded separately. A process owns its mutable UNA/Topology/Settings/output directory.

Use spawn-style process creation for portable independent jobs and guard the entry point. Do not fork an initialized threaded/JIT runtime.

No production-scale run after each edit. L4 only after final candidate freeze. If a resource cap makes the real workload impossible, report it unavailable; do not silently reduce scientific workload and call it L4.