# Source basis and P00 interpretation ledger

All repository observations below refer to 6130769a5751bf7446734dc39b9f8e3763461cfe.
Source paths and Git blobs are in sources.json. P00 candidate runtime was e7c9eda...;
its comparator was 395cdc5.... No new application experiment was run to author this amendment.

## Observed in committed records
- `stage_model.json`: baseline W4 public wall 92.1141 s; candidate W4 100.9639 s. The
  candidate is slower at this boundary. These are screen samples, not final paired estimates.
- `runs/e_cand_w4_c1.json`: one trace identity/build, 32 replays, 472032 origin folds;
  output comparison requested and recorded as 64 matching files; 147830672 bytes in p00_env.
- `candidate_register.json`: per-metric reuse is NOT implemented. Serial candidate has no
  CohortSession. No qualified native/GPU entry is registered in this measured build.
- `worker.py` constructs fresh UNA each row. `UNA.RunAccessibility` reloads three layers and
  builds its engine BEFORE installing CohortGraphEngine. Sharing begins too late for input work.
- `_trace_access_scratch.replay_trace` is Python: one origin loop, slicing, checks/copies and a
  compiled fold dispatch for every origin. The P00 replay timer includes this overhead.
- `runtime.py`: one synchronous result channel, one commit cursor; receive is not independently
  drained during `_commit_worker_row`. This is a backpressure HYPOTHESIS, not its measurement.
- `Topology.py`: existing snapshot/topology/snap serializers and restorers can be reused AFTER
  consumer audit. Their existence alone does not prove full state coverage for a new cache.

## Corrections to interpretation (do not rewrite original evidence)
E1. Serial diagnostic grouping: read_arrow self 74.169 + next self 66.8026 = 140.9716 s;
    adding from_wkb self 4.1087 gives 145.0803 s. Against 192.358 s profiled wall this is
    75.4220%; integrated_scope_access self 15.6162 is 8.1183%, not 10.9%. Verify caller
    attribution of builtins.next from pstats before assigning all of it to GIS. Summing
    cumulative and self values is prohibited; serial shares are not parallel span shares.
E2. 100.9639 vs 128.3811 changes CPU instrumentation as well as execution time. Co-tenant
    presence is observed, but does not establish that it caused all 27.2% spread. Measure
    sampler overhead with the same workload/order and retain co-tenant metadata.
E3. Whole-tree efficiency 0.459 is an outcome, not a disjoint 2x optimization stage. Shared
    preparation can remove the same waits/traffic attributed to idle. Never multiply those gains.
E4. Parent import_s 6.9522 is OUTSIDE candidate public wall. It cannot be subtracted again.
    Child spawn/import and pool teardown are INSIDE each call. Repeating calls does not
    retain the current pool; it closes per call. No off-clock preparation or JIT migration.
E5. writer_concurrency was not consumed. wc2 is a replicate, not a writer scalability test.
E6. Logistic gravity does not read beta directly in this engine. KNN's beta dependence is
    branch-specific. Obtain the actual frozen CSV before claiming which fields are invariant.
    'half the fold' is neither measured coverage nor a proven upper bound.
E7. P00 profile JSON sets bitwise_verify.requested=false in prof_cand_serial. Therefore do
    not repeat the blanket 'all 15 runs individually bit-verified' from the commit message.
    Separate timed correctness runs, warmups, diagnostics, and externally referenced evidence.
E8. C31 result key coerces destination/KNN weights to float64 while replay uses raw dtype.
    It is currently a reporting key; fix/version its closure BEFORE using it for reuse.

## Derived arithmetic, not achieved performance
92.1141 / 100.9639 = about 0.91235. The historical sample's 10x budget is 9.21141 s;
12x planning headroom is 7.676175 s. Final limits use the matched current comparator.
235.8 CPU-s / 4 = 58.95 s of service for the instrumented candidate if work is unchanged.
This makes work removal necessary; it does not predict new critical-path time.
Input files sum to 79940126 B; output payload is 147830672 B. Logical byte counts are not
measured physical DRAM/filesystem traffic. All new speed projections remain hypothetical.

Do not describe missing local out/ artifacts as reverified, source execution as an installed
wheel, opaque counters as CPU time, or a failed cold benchmark as a warm throughput win.
