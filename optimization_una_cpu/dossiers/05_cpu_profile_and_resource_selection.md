# Dossier 05: CPU profile and resource selection

After genuine L2 correctness, profile strongest baseline and candidate on the same frozen medium workload. Identify service fractions for load/topology, CSR, search/metrics, export, startup/JIT, and orchestration.

Measure single-job memory before concurrency. Then bounded sweep of (workers, threads) with total effective CPU <= admitted C, plus queue/writer limits. Include W1H1, W1HC, and factor pairs. Do not carry historical 2x2 settings over without measurement.

Use two screening batches per configuration, then freeze best arm-specific finalists and run alternating-order confirmation. Record absolute samples, throughput derived from them, CPU utilization when available, process-tree memory, output bytes, cold/warm state.

Choose configuration maximizing completed correct jobs per wall time under resource budget, not single kernel latency.