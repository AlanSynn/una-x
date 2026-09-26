# Benchmark protocol

Freeze commit SHA, workload manifest, dependency environment, resource budget, output policy, and timing boundary before candidate selection measurements.

Arms:
A = strongest verified original implementation.
B = same policy/resources with ordered-CSR algorithm/layout.
Optional B2 = B plus the single admitted scratch experiment.
No new backend arm.

Measure nested boundaries: CSR leaf; entire graph builder; public RunAccessibility; whole fresh job including required outputs; independent-job batch; cold process. Keep profiling separate from final uninstrumented timings.

Record cold process/JIT/cache and warm compiled execution separately. Synchronize/materialize before stopping timers. Include serialization and required output completion. Verification hashing may be outside primary timer but must have its own time.

Use separate clean baseline/candidate installs or immutable source worktrees. Final qualification uses installed wheels with import-path assertion proving no checkout/PYTHONPATH shadowing.

Tune at L3 only. Test bounded process/thread configurations under the same CPU/RAM limit. Begin W=1/H=1; include W=1,H=C and factor-pair alternatives where admissible. Record effective Numba/native thread masks, not just requested values. Use bounded queue and independent output directories.

For final confirmation, freeze finalists and run fresh alternating order samples (for example AB, BA, AB, BA, AB). Retain all valid raw samples and predeclare cancellation reasons. Report absolute times and throughput derived from those times; never multiply overlapping speedups.

Process-tree memory uses psutil RSS/PSS/USS where available, with sampling method and interval recorded. Hardware bandwidth/SIMD efficiency is unavailable unless directly measured.

L4 is only the unreduced actual target workload. If unavailable, report proxy qualification and actual_target=unavailable.