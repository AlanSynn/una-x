# Source audit and inherited-evidence reconciliation

Audit anchor: AlanSynn/una-x main 361928e4ba38f34622cafe065b0025244db61368. Root tree 089d29205918197c783e34c46eec76bf27090c10; src tree a5883dddcef3afb8debdb4438a6338da886add6f. Sources below are read at that immutable commit. These are source facts, not new benchmark results.

## Relevant source identities
- Engines/Accessibility.py: 473de486e5573992aba5079f3c8ac80668e61aa0.
- Engines/AccessibilityWElevation.py: dc75e48baf92817028a749a15778c6308e41c44f.
- Engines/_ordered_csr.py: 186e82c0ddb612eae0a5550e5e0b06cf4f16785d.
- Engines/AggregateFlow.py: 0114ce8f318a3247b62e3d468e32d54e692737a2.
- Topology.py: 26ca999f32f20bbb447a91ea51d60a3408ea6948.
- UNA.py: f803a536277c178f512e08243ac12ce18bf3de46.
- benchmarks/una_cpu/runner.py: a8bb62c859e62d385f3f892071fc0074332fb294.
- benchmarks/una_cpu/batch_worker.py: d5f947cccbcfb3f02068de775b2d987366f751fe.
- benchmarks/una_cpu/job.py: d28252a51c2970365aa9a848a29ef0fb071b7b06.
- benchmarks/una_cpu/profile_stages.py: 3d1986f217d39207b437a6e3c4fdc83e9d215025.
- benchmarks/una_cpu/workload_gen.py: fd37e67de72f869a155600befca97fb3b108c941.

## Source-derived findings to reproduce at H00/H02
1. Accessibility scope allocates V+D labels per origin. Every popped heap entry computes neighbor-distance arrays, snapshot comparisons and eligible offsets. There is no stale-entry skip. A scalar immediate-update loop is NOT equivalent for duplicate neighbors.
2. adjust_destination_distances scans every destination; reach_gravity_knn_access filters in original destination order and then applies existing reductions and argsort. All four metrics are calculated before Settings flags control storage. Do not assume unexported work is unobserved or error-free.
3. AggregateFlow uses a separate ThreadPoolExecutor and topology.num_threads. Static origin stripes and slot-ordered final sums make thread count part of the numerical profile. NUMBA_NUM_THREADS alone does not bound that pool.
4. _accumulate_od_flow and the turns counterpart omit nogil=True. This is a serialization hypothesis, not measured proof of GIL bottleneck. The official Numba jit documentation says GIL release requires the option and native compilation; verify actual targetoptions/signatures on the pinned laptop compiler.
5. Flow destination gradients are sparse in storage, but the OD loader builds/scans full-node masks and allocates full-node scratch per pair. Its predecessor accesses can extend outside the overlap set.
6. _precompute_dest_gradients bounds chunk*n_total near 1e8 elements, not a complete byte budget. Float64 distances, typical int32 predecessors and bool finite masks account for about 13 bytes/element, before sparse parts and overlapping allocations. Inspect actual dtypes.
7. Historical batch_worker._run_job_body always calls RunAccessibility despite requested analysis. Historical runner counts requested jobs in throughput even on failures, has an unbounded result queue and hard-coded cache paths, and does not rehash pre-existing input manifests on reuse.
8. Historical job sampler sums its own RSS plus immediate children, not a simultaneous recursive whole-batch peak. Import-path assertion exists in single-job mode but not equivalently in batch execution. Its threading-layer reporting conditional is inverted.
9. Historical stage profiler calls a second diagnostic constructor. Its flow mode performs extra accessibility and gravity-cap work before RunFlow. Neither diagnostic sequence may be summed as the normal public execution.
10. Historical t11_final.sh uses source arms. Installed wheel correctness elsewhere does not establish installed-wheel throughput. L1_REUSE_DIR can reuse artifact names without the full dependency identity; disable it in new qualification.

## Historical values, not new measurements
The archived stage model reports candidate centrality 1,007,720,625 ns, setup layers totaling 205,785,291 ns, export 47,192,042 ns, constructor 2,240,125 ns, validation 143,208 ns. Derived one-constructor total: 1,263,081,291 ns. The analogous baseline stage sum is 1,522,583,999 ns. Sums of medians are diagnostic reconstructions, NOT direct public-call timings; the diagnostic repeated constructor is excluded.

Archived final warm source-arm batches report approximately 4.44 candidate versus 3.33 baseline jobs/s at 8x1. Their ratio is derived and proxy-specific; 6.14x additionally changes concurrency and uses a different earlier reference. Do not use either as the new candidate target or a production result.

W1/W3 are generated jittered grids written as GIS files. They exercise real APIs but are synthetic inputs, with points deliberately placed on edges. Old 'genuine GIS' language does not establish observed-road representativeness. The scope-init probe chooses one origin and does not load the W3 obstacles; its ~0.98% finding is not a universal large-network rejection.

## Current uncertainties
No new large observed-network measurements, production target or laptop resource discovery have been performed by this packet. Existing tests record exactness on their domains, not every compiler/platform. Archive review records remain historical. No automatic default activation follows from this audit.

Source URLs are obtained by appending the named path to https://github.com/AlanSynn/una-x/blob/361928e4ba38f34622cafe065b0025244db61368/. External reference for F1: https://numba.readthedocs.io/en/stable/user/jit.html#nogil. Claude command compatibility: https://code.claude.com/docs/en/slash-commands. Record installed versions; live documentation is not an environment lock.
