# una_10x — next-phase plan: the remaining 10x gap on the CPU/native route

Branch `perf/una-10x-next` (from main `2d6142a`). Created 2026-10-12 under the user's
instruction to advance the 10x direction (CPU or native, whichever carries) alongside GPU
optimization plans (sibling branch `perf/gpu-opt-plan`). Scheduling authority:
`campaigns/una_10x/post_p00/AUTHORITY.md` (P01→P02→P03 sequence law unchanged). This document
is a plan, not a claim: only a tenx-law paired public-API gate establishes throughput.

## 1. Where the claim stands

| Claim | Value | Comparator / boundary | Evidence |
|---|---|---|---|
| Headline | **2.3798x median, one-sided 95% LB 2.2345x** | vs `395cdc5`, installed wheels, **cold public call boundary**, bitwise (64 files, 147,830,672 B) | `tenx-20261005T2331Z` |
| Merge-decision gate | 3.076x (LB 3.035x), 7/7 windows | vs pre-merge main `c1f8614`, warm legs, worktree context — scope law binds; not the headline | `g4rc-20261012T0512Z` |
| Lazy-import candidate | 1.10654x (LB 1.07908) | vs `f2ea398` tree, public API; **merged** | `exl-20261011T164228Z` |

10x target ⇒ a further **≈4.20x** on the headline boundary. Per the P00 brief (§4): from
arm-A W4 ≈ 92 s, G=10 means the whole cohort (32 rows + 64 exports) must complete in ≈9 s.

## 2. The measured residual (P00 stage model, e7c9eda serial cProfile, 32-row OBS_COHORT)

| Region | serial self | share | region wall (of 188 s serial) |
|---|---|---|---|
| per-row GIS load/parse (pyogrio `read_arrow` + `next` + `from_wkb`) | 141.0 s | **75.4%** | ≈141.8 s (`P_shared` serial share 145.1 s) |
| routing+metrics (`integrated_scope_access`) | 15.6 s | 10.9% | ≈20.5 s (warm fold 0.581 s/row × 32) |
| export write | 8.2 s | 4.4% | ≈8.3 s |
| topology + misc | — | 9.3% | ≈17.5 s |

Key structural fact: **every row re-reads the same three GIS inputs**; cohort trace sharing
alone removes only ~11%. Pool efficiency at W4 is 0.459–0.524 on both trees. Import was
3.42 s/process before the merged lazy-import work (now ≈1.5 s/worker).

### P01-refresh on current main (this branch, `tenx-20261012T1600Z/p01r_refresh`)

Same manifest (sha `1161f485…daaf` verified), serial public call, cProfile, mapping-v2
buckets (see `receipts/next-p01r-20261012T1600Z.md`; raw `prof_main_serial.pstats` committed):

| Region (self time, mapping v2) | current main | P00 (e7c9eda) |
|---|---|---|
| load_gis (pyogrio `read_arrow` 71.1 s + arrow `next` 64.2 s + `from_wkb` 3.8 s + …) | **148.6 s — 84.1%** | 141.0 s — 75.4% |
| metrics_fold (`integrated_scope_access`) | 7.8 s — 4.4% | 15.6 s — 8.1% |
| other (lock acquire 5.3 s, numpy `argsort` 5.3 s, tail) | 18.0 s — 10.2% | ~9.3% family |
| topology / export_io (self) | 1.5 s / 0.7 s | (in family) |

Mapping-v3 errata (consumer audit, same day): the load_gis bucket includes 9.0 s of
export write (`raw.py:write` matched "pyogrio" on the file path). **Function-level
input-read family = 139.1 s = 78.8%**; export ≥ 9.0 s = 5.1%. Raw pstats is the
authority; see `receipts/next-p01r-20261012T1600Z.md` ERRATA.

Reading: the merged engine work **halved fold self time** (15.6→7.8 s) but per-row GIS
load/parse is untouched and now an even larger share (84.1% of 176.6 s profiled serial wall).
**R1 (shared preparation) is the binding region on the current tree**; R3's absolute bound
shrank with the engine integration. Single-login-node sample, heuristic buckets — direction,
not a gate.

## 3. The four-region program (sum law: regions add on the wall; ratios are never multiplied)

The P00 brief already established these four regions are measured, disjoint, and **jointly
required** for G=10:

1. **R1 — shared input preparation (P02): LANDED — consumer-audited 2026-10-12.**
   `batch/preparation.py` (P02I `a71f0f5`, review-closed `fa92dca`, both ancestors of main
   `2d6142a`) implements the guarded PreparedBundle single-flight BEFORE decode: per-row
   snapshot-bytes sha256 + stage params + profile + geometry-env closure keys
   (`cache/identity.py:193-216`), cross-process file-lock single-flight over a bounded
   on-disk store (`.una-batch-prepared`, verification="on_read", quarantine+rebuild),
   content-derived warnings replayed from the payload, refusals loud (no_root / turns_row /
   obstacle_row / format_not_admitted / non-admitted stages). **Engage law (NR-6): the
   prepared route arms ONLY on `RunBatch(parallel=True, analysis='accessibility')` with
   dispatchable rows — the serial public call is structurally un-armed** (producer sets
   `una._prepared_session` in `batch/worker.py:375-395` only; serial `UNA.py:263-314` never
   arms). Consequences:
   - the P01-refresh 84.1% load_gis profile above measures the SERIAL boundary, where R1
     cannot engage by design; it does NOT describe the tenx gate boundary;
   - the parallel cohort gate boundary DOES consume it: G4RC INC01 records the candidate
     writing the 43-file/+18,046,104 B residency stores, and warm counted legs ran ~21 s
     (vs pre-merge main 63–65 s) — row 1 cold-builds, rows 2–32 hit the disk store;
   - **the published headline gate (tenx-20261005T2331Z, 2.3798x vs 395cdc5 installed
     wheels, COLD, macOS arm64) ran on treatment `1e7237b`, which does NOT contain P02I/
     P02R** (ancestry machine-checked 2026-10-12) — the landed R1 work is real on main but
     is NOT banked in the published headline number; a future headline on main's bytes
     requires a fresh tenx-law gate;
   - on prepared hits pyogrio is skipped but **from_wkb + GeoSeries + sindex rebuild still
     run per row** (decode-once NOT amortized) — that per-row restore is now a first-class
     residual region (see R5).
   Design obligations already enforced by the landed module (do not regress): bit law via
   observed-bytes digest keys; row validation/read/failure chronology preserved on hits
   (per-row snapshot read+digest gates at the seam BEFORE flight consult, TOCTOU-tested);
   immutable payloads with sindex rebuilt as derived state; no GDAL/GEOS handle sharing
   (flat WKB blob payloads + PROJJSON, WKT rejected for datum-ensemble loss).
2. **R2 — pool-efficiency repair (0.46→≥0.75 target).** P01 verdict H1: load/parse contention
   is **refuted as dominant** (per-row run wall flat at W2: ratio 0.9983; W4 stretch 1.1568
   with `peak_busy_procs=5 > cpuset 4`). Mechanism: coordinator oversubscription —
   amortize/batch coordinator work (dispatch/commit/journal/hashing) off the workers'
   critical resource; batched commits. Bound: ≈2x on every parallel stage at W4.
3. **R3 — per-metric reuse (C31-NOTE-1).** reach + knn are beta-invariant on this cohort
   (identity-table keyed, dtype-exact — see `metric_dependencies.gen2.json`); gravity folds
   must recompute. Upper bound ≈ half the fold region (20.5 s → ≈10–12 s). Typed identity
   reuse only; no identical-answer cache semantics.
4. **R4 — window amortization (partially banked).** Import 3.42→≈1.5 s/worker landed with the
   lazy-import merge; remaining overhead amortizes only via multi-cohort windows per
   BENCHMARKS.md window-composition law (bounds 10x on a single 92–118 s arm-A window).
5. **R5 — prepared-boundary residual (NEW binding region, 2026-10-12 audit).** On the
   parallel prepared boundary the remaining wall is: per-row restore on hits (snapshot
   read+sha, `_decode_snapshot_gdf`, `from_wkb` payload decode, GeoSeries+sindex rebuild,
   log replay — decode-once NOT amortized across rows), the metrics fold, exports,
   coordinator commit/journal/hashing, pool lifecycle and IPC. **The binding telemetry is a
   PARALLEL W4 prepared-store-warm profile of current main — the serial profile above
   cannot see this boundary.** Candidate levers, each needing its own design+review:
   in-worker decoded-stage lease reuse across rows (immutable internal leases, no public
   aliasing — the architecture's own "prepared input-keyed single-flight → independent row
   objects" step 3), R2 pool-efficiency repair, R3 metric reuse. Current warm reference:
   G4RC B arm 20.2–21.2 s (PACE, W4T1) vs pre-merge main 63–65 s; 10x from the P00 PACE
   baseline (92.1 s W4) needs ≈9.2 s.

**Sum-law projection (labeled projection, not a claim):** serial-equivalent cohort work
188 s → ≈40–55 s (R1+R3+banked R4); at W4 with R2 efficiency ≥0.75 ⇒ wall ≈ 13–20 s vs the
≈9 s needed for G=10 from arm-A W4. The gap closes only if R2 reaches ≥0.8 effective
efficiency or R1's decode-once slice also amortizes topology misc. **Binding uncertainties:
per-row validation duty on prepared hits, W4 coordinator cost after batching, co-tenant
variance (25–28% same-config spread — paired windows mandatory).**

## 4. Native track (carries R3/R1 kernelization; per dossier 08 native law)

- In tree: AOT C++17 `Engines/_native/metric_fold.cpp` (strict-IEEE ordered fold, both
  numerical profiles, bitwise vs compiled engines; identity pinned by loadable-section
  digests — never whole-file `.so` sha). Admission is per-profile and gated; CPU Python
  fallback must never break.
- P03C preconditions (from P01 H3, unresolved): W1T4 vs W2T2 vs W4T1 thread-config matrix and
  Python-loop vs compiled driver at identical configuration, before any receiver changes.
- SIMD characterization with compiler reports/assembly (gathers, lane divergence, alignment,
  FMA); no `-ffast-math`; parallelize origins or independent metric jobs, never reassociate.
- CPU-vs-native selection by measured crossover including build/install/startup costs in the
  declared boundary.

## 5. Closed by executed falsification — do not relitigate

- **M1 trace-shard budget law**: killed (split overflow leaves a residual parity band
  [8(N+1)+16·max(K_s)+64, 8(N+1)+16·K+64)); only a two-phase count pass or shared budget
  counter redesign reopens it — different design, new dossier.
- **Cross-worker build parallelization**: dead on PACE (row0 monadic build 0.5876 s;
  thread shard 2.4% of theoretical; internal prange w4 ≥ w1).
- **writer_concurrency**: measured NEGATIVE, power-stopped (7-window gate needs >9% log
  effect on this host; Settings.Validation rejects private attrs).
- Contract-forbidden regardless of speed: identical-answer cache hits that suppress row
  validation, weak-thread baselines, FP32 substitution, rounding/reassociation, hidden
  radius/data/output reduction, flow-model switch.
- **GPU route**: measured not-admitted (G11, V100: substitution domain vs delivered numba
  fastmath authority is EMPTY — 0/4 outputs bit-equal at rep4000; full-service crossover
  3.15x slower than the delivered arm on the same node). CPU route retained; GPU optimization
  plans live on `perf/gpu-opt-plan`.

## 6. Sequencing (post_p00 AUTHORITY law)

1. ~~P01-refresh (serial)~~ — DONE 2026-10-12 (`receipts/next-p01r-20261012T1600Z.md`), but
   boundary-scoped: the serial public call structurally never arms preparation (NR-6), so
   its 84.1% load_gis share describes the serial duty only.
2. ~~P02 shared-preparation single-flight~~ — **ALREADY LANDED on main** (P02I `a71f0f5` +
   P02R `fa92dca`; consumer audit 2026-10-12 confirms engage law, key composition, store
   laws). Sequence item closed; remaining R5/R2/R3 work builds ON it.
3. **P01-parallel (next telemetry)**: W4 prepared-store-warm profile of the CURRENT main on
   the parallel public boundary — per-row restore composition, coordinator share, fold
   share. This, not the serial profile, locates the residual ≈2.2x (G4RC B ≈21 s → ≈9.2 s
   needed for G=10 from the P00 PACE baseline).
4. **P03C preconditions**: thread-topology matrix on the current tree (probe
   `probes/replay_thread_matrix.py`, running 2026-10-12), then native fold admission per
   profile with crossover evidence.
5. Implementation increments, each island battery + independent review: R5 decoded-stage
   lease reuse; R2 pool-efficiency repair; R3 metric reuse (typed identity, gen2 law).
6. Then a fresh tenx-law paired gate of main's bytes vs the 395cdc5 installed wheel
   (cold and warm declared separately) — the published headline (2.3798x) binds treatment
   `1e7237b`, which is NOT main's lineage (ancestry machine-checked), so main's bytes have
   no qualifying gate yet.
7. R2/R4 receiver work only with measured residual backpressure (H2 stays refuted at this
   scale; P04 remains conditional).

## 7. Gate law (unchanged)

Paired AB/BA windows (sd from the steady warm pair; PACE same-config spread 25–28% ⇒ paired
ratios only), median + one-sided 95% LB over target, n=7 resolves ≥~3% ln effects, content
byte law under stamp normalization (64 exports / 147,830,672 B), all required outputs and
state, equal resources, timer brackets only the public call, cold and warm boundaries each
declared. Bitwise parity per una_legacy/madina_legacy/corrected_v1; 1-ULP mismatch = failure.

## 8. Honesty / unrun gates on this branch

- The probe result recorded here refreshes region shares; it does NOT establish any speedup.
- The sum-law projection in §3 is a model bound, not a measurement; the tenx headline is
  unchanged (2.38x; target_unmet stands until a qualifying gate says otherwise).
- **Boundary honesty (2026-10-12 consumer audit + ancestry checks):** the published
  headline gate tenx-20261005T2331Z ran on treatment wheel `1e7237b` (macOS arm64, COLD,
  workload sha `235405de…`, NOT an ancestor of main); the landed P02 preparation
  (`a71f0f5`/`fa92dca`) is in main's lineage but was NOT in that treatment — so the R1
  win is NOT banked in the headline number, and main's bytes have no qualifying gate vs
  395cdc5 on any boundary. The G4RC 3.076x record is the PACE WARM comparison vs
  pre-merge main `c1f8614` (scope law binds; not the headline).
- No pushes from this branch; publication requires a new explicit user instruction.
