# Optimization record: una-x vs vanilla UNA

This document compares the **current state** of una-x with **vanilla UNA** — the
MIT City Form Lab Urban Network Analysis package (Andres Sevtsuk, Raul Kalvo)
that una-x was seeded from. It is a state comparison, not a changelog: what the
package does and measures today, against what the original package did.

**What "vanilla" means here.** The seed commit `5a1927d` ("Seed una-x from
audited UNA baseline") contains only a README; the vanilla UNA source enters
the repository through the 22 "Import audited baseline" commits ending at
`98f498e`. `98f498e` is therefore the effective vanilla-source baseline: it
matches the upstream package's scope (topology/CSR loading, impedance with
elevation/turn/obstacle penalties, Dijkstra accessibility and flow engines,
serial batch, GeoJSON/Feather/CSV export, Numba-compiled hot loops).

**Honesty note on headline multiples.** No single measured
current-vs-vanilla throughput figure exists. The repository's measurement law
(`campaigns/una_platform/BENCHMARKS.md`, archived; see Provenance) forbids
composing one by multiplying overlapping segment ratios, and every figure
below is reported against the exact baseline it was measured against. The
baselines themselves are named commits, most of them *already optimized*
builds — not vanilla.

---

## 1. Measured performance

All figures below are preregistered, paired, bitwise-checked measurements from
the verification campaigns. Every number is traceable to a receipt (paths in
the Provenance section; files preserved in git history and in the off-repo
archive).

| # | Result | Candidate vs comparator | Boundary / scope | Status | Receipt |
|---|--------|--------------------------|------------------|--------|---------|
| 1 | Engine construction 245.4 ms → 2.2 ms (**109.5×**); other stages unchanged | ordered-CSR candidate `f57825d` vs `98f498e` (vanilla source + packet infra) | stage-wall medians, n=5/tree, alternating | official; **landed in main** | una-cpu campaign, `evidence/final/RESULTS.md` |
| 2 | Batch throughput 3.33 → 4.44 jobs/s (**1.336×**); **6.14×** vs the sequential baseline reference | same pair, 8 workers × 1 thread | public batch, final alternating confirmation | official; **landed in main** | same |
| 3 | O3_FLOW median **1.3464×** (CI95 1.342–1.455) | una_large_e2e selected line (published as `352aab0`) vs `361928e` (vanilla + CSR opt) | flow family, proxy-qualified | official (flow only) | una_large_e2e `CLOSEOUT.md`, `evidence/final/RESULTS.md` |
| 4 | A1 access family median **1.0970×** — `improvement_only`, radius-scoped, explicitly not throughput-qualified | same pair | access family, protected cells ≤ 1.05 | official (negative scope) | same |
| 5 | **2.3798×** median, one-sided 95% LB **2.2345×**, 7/7 paired windows, outputs bitwise identical (64 files/window), protected cells median 0.85 | una_10x headline candidate `1e7237b` vs frozen comparator `395cdc5` | **cold** public `RunBatch` boundary, installed wheels, 32-job Boston cohort (workload sha `235405de…`) | official gate, verdict **`target_unmet`** vs the 10× stretch goal | `campaigns/una_10x/evidence/final/tenx-20261005T2331Z/` (branch `perf/una-10x`); restated in README |
| 6 | Lazy-import **1.10654×** (95% LB 1.07908), byte-identical outputs | lazy `__init__` candidate vs `f2ea398` | import + public API | official, **merged** | `campaigns/una_10x/evidence/tasks/EXPL/exl-20261011T164228Z/receipt.json` |
| 7 | **3.076×** median (95% LB 3.035), 7/7 windows, exports byte-identical | G4RC candidate `9d82a0d` vs pre-merge main `c1f8614` | **warm** paired windows | official merge gate (**merged**); scope law: merge-decision evidence, not the headline | `campaigns/una_10x/evidence/tasks/G4RC/g4rc-20261012T0512Z/receipt.json` |
| 8 | **3.34791×** median (95% LB 2.92871), 7/7 windows | R5 decoded-stage lease reuse lineage (`631ebc6`) vs `395cdc5` | **warm** counted legs, w4 lanes | **diagnostic only** by binding scope law — not a headline gate | `campaigns/una_10x/next/receipts/next-diag10x-20261012T2213Z.md` |
| 9 | GPU: 0.13718 → 0.004929 s (**27.8×**) one-shot full-service; scan section 243× | GPU bridge internal (pre-fix vs post-fix scan) | instrument wall, rep4000 structure | **instrument-wall evidence only** — not a production or GPU-acceleration claim | `campaigns/una_10x/next/receipts/gpuprobe3-20261012T1630Z.md`, `next/PLAN_GPU.md` |
| 10 | GPU device fold **3.154× slower** than the delivered CPU driver per 4000-origin structure → `CPU_ROUTE_RETAINED` | device fold vs CPU driver, same V100 node | G11 crossover, C=1 warm | official negative result; README claims **no** GPU end-to-end multiple | `campaign_runs/una_10x/tenx-20261004T155640Z/g11_gpu/gpu_qualification.json` |
| 11 | Compacted-CPU comparison arm: paired medians 0.99–1.1 across worker/thread shapes — wall-neutral, not admitted as a route | una_platform CPU_LAYOUT C4 arm | batch shapes W1–W3 × 1–4 threads | comparison-arm characterization | `campaigns/una_platform/evidence/tasks/CPU_LAYOUT/cpul-20261002T2305Z/proof_c_route.md` |

Reading guide: #1–2 are the only rows whose comparator is (effectively)
vanilla. #5–7 compare against builds that already contain #1–4 and the
platform work below. The 10× program goal was evaluated at full strength and
closed `target_unmet` — it is not achieved, and no figure above should be read
as 10×.

## 2. Engine and runtime features absent from vanilla

All of the following are live code in `src/urban_network_analysis/`, exercised
by the in-repo test suite:

- **Ordered-CSR fast path** (`Engines/_ordered_csr.py`) — the landed #1–2
  optimization; falls back to the original construction (including its errors)
  for unsupported representations.
- **Sparse epoch-labeled scope search** (`Engines/_sparse_access_scratch.py`)
  — per-epoch touched-filter Dijkstra with no dense label materialization;
  narrow admission gate, refusals fall back to the original driver.
- **Snapshot-preserving two-phase search buffers**
  (`Engines/_large_access_scratch.py`) — reusable per-origin buffers replacing
  per-population temporaries; proven byte-identical.
- **Routing trace store** (`Engines/_trace_access_scratch.py`) — packed
  origin-major sparse trace with typed identities (GraphID → QueryID → DestID
  → TraceID); `replay_trace` re-runs the engine's own compiled fold; budget
  overflow marks records INCOMPLETE rather than serving a partial hit.
- **Typed metric identity + field reuse** (`Engines/_metric_identity.py`,
  `batch/metric_reuse.py`) — MetricID v2 derived from raw argument bits
  (dtype/byteorder/layout/payload; no float64 coercion, no `-0.0` collapse);
  run-scoped LRU-bounded store serving fresh writable copies. Measured engaged
  28/28 split folds, 0 fallbacks at close.
- **Cohort route** (`batch/cohort.py`) — batch rows with identical prepared
  inputs share one content-addressed trace; every refusal falls back to the
  inner engine (can only save work, never change a number).
- **Native AOT backend** (`backends/native/`) — C++17 scope-search core for
  both semantic profiles, ASan/UBSan-clean, typed loader with ABI/symbol
  checks; portable Python fallback byte-identical; explicit route selection,
  never a silent default.
- **CPU kernel freeze** (`kernels/`) — executable characterization of the
  legacy fastmath comparison/heap semantics plus canonical reduction plans; a
  compiler upgrade that shifts legacy rounding fails loudly instead of
  silently changing results.
- **GPU fold instrument** (`Engines/_gpu_fold.py`) — numba-CUDA port,
  device-gated, battery-domain qualification instrument only; CPU remains the
  default and only production route.

## 3. Batch, cache, telemetry, recovery (all absent from vanilla)

Vanilla batch execution is a serial loop, and "Topology is rebuilt at every
run — it is not cached" (DESIGN.md). una-x adds:

- **Parallel `RunBatch`** (`batch/plan.py`, `batch/runtime.py`,
  `batch/worker.py`) — pure planner with row-DAG input/output hazard analysis
  and ordered validation outcomes; coordinator with bounded reorder staging,
  deterministic caller-order commit cursor, atomic `os.replace` + fsync
  publication, finish-the-prefix failure semantics; spawn-safe workers
  reproducing serial row transitions. Engaged via
  `UNA.RunBatch(parallel=True, workers=...)`.
- **Durable checkpoints** (`batch/checkpoint.py`) — JSON journal + raw `.npy`
  captures, write-tmp → fsync → atomic replace → dir fsync, digest
  verification on read, quarantine-not-delete, no pickle; resume/cancel/tamper
  tests in `tests/batch/`.
- **Content-addressed cache** (`cache/store.py`, `cache/stages.py`) — bounded,
  checksum-verified, per-key single-flight, atomic publish, pickle-free; keys
  are sha256 over the full dependency closure (file bytes, parent stage keys,
  IEEE-hex scalars preserving `-0.0`, profile + environment fingerprint) —
  never paths or mtimes. Off by default (zero-cost inert).
- **Prepared-input single-flight** (`batch/preparation.py`) — topology and
  snapping prepared once per distinct input before decode, shared across a
  batch's rows.
- **Opt-in causal telemetry** (`batch/telemetry.py`) — bounded buffers with
  dropped counters, per-worker JSONL under `UNA_PHASE_TELEMETRY=1`; never part
  of any identity.
- **Public execution contract** (`Execution.py`) — immutable
  `ExecutionOptions`/`CacheOptions`, typed
  `CapabilityError`/`BackendNotAvailableError`/`ExecutionNotAdmittedError`,
  `admit_execution`; forced backends raise rather than silently falling back.
- **Lazy imports** (PEP 562 in `__init__.py`) — `import
  urban_network_analysis` no longer pays numba/sklearn import cost (the
  measured 1.11× gate).

## 4. Scientific corrections vs vanilla

Vanilla defects were audited, reproduced where real, and either fixed in the
`corrected_v1` reference or pinned as legacy behavior. Registry:
`campaigns/una_platform/evidence/contract/bug_registry.json`; framework:
`tests/science/test_bug_triples.py` ("OLD-FAILS / CORRECTED-PASSES /
LEGACY-RETAINS" — legacy bits are pinned, never silently overwritten).

| Bug | Vanilla behavior | una-x |
|-----|------------------|-------|
| FRAC-WEIGHT-TRUNC | reach accumulates destination weights in int64, truncating fractional weights | corrected reference keeps binary64 sums |
| MADINA-BTN-STATS-NAMEERROR | bare `except` swallows an `UnboundLocalError`, skipping `remove_node_to_graph` and poisoning shared state for later origins | corrected_v1 computes diagnostics and always removes the origin |
| COINCIDENT-SEEDS | seed overwrite vs declared minimum/tie rule at Dijkstra init | declared rule enforced |
| SAME-EDGE-OD | origin/destination on the same edge: undocumented substitution | explicit partial-edge impedance rule |
| NONFINITE-NOVALIDATION | negative/nonfinite costs, sentinel-overflow radii, zero-cost cycles reach the kernels unvalidated | typed refusals pre-search (`NegativeCostError`, `NonFiniteCostError`, `MalformedGraphError`, `OutOfRangeTerminalError`, `ParameterDomainError`) |
| TEA-KNN-TIE-ORDER | unstable introsort on tied distances; coefficient pairing on ties is an undeclared artifact | audited and characterized |
| TEA-NEGATIVE-DECAY-OVERFLOW | negative decay coefficients silently overflow accessibility to inf | refused with `ParameterDomainError` |
| TEA-KNN-DEAD-PARAMS | KNN silently reuses gravity parameters; its own parameters are never read | characterized; corrected path reads them |
| JIT-DISABLED-STUB | `NUMBA_DISABLE_JIT=1` crashed on an `@overload`-only guard stub | fixed, dispatch-parity regression test |
| LAYER-SET-STYLE / NETWORK-TO-LAYER / SVC-GEOMCOLL | pinned Madina broken paths | registered; `madina_legacy` reproduces the crash, corrected_v1 may repair |

Three numerical authorities are defined and pinned: `una_legacy` (pinned
compiled engine bits), `madina_legacy` (pinned Madina valid-input behavior,
including its defects), and `corrected_v1` (independently reviewed corrected
semantics: binary64, explicit round-to-nearest operation order, no
fast-math/FMA, frozen canonical reduction partition). Implementation:
`src/urban_network_analysis/reference/`. One registered candidate
(PREP-ZERO-COORD-DROP) was **refuted** on the pinned stack and recorded as a
refutation, not a fix.

## 5. Madina compatibility

- `compat/madina/` is an attributed verbatim port of City-Form-Lab/madina @
  `8b5c3bd3` (zonal + una namespaces) with a complete delta ledger for four
  upstream incompatibilities (GeoPandas ≥ 1.0, pandas 3.0, pygeos env write,
  lazy pydeck).
- Top-level namespace parity is machine-checked live: **61 of 62** upstream
  public names (missing: `pdk`, recorded in the attribution ledger); 131
  adversarially reviewed tests green (`tests/madina_api/`).
- Not done (recorded as platform obligations): the installable drop-in
  `madina` shim distribution is specified but not packaged; grouped-seed
  expansion policy and installed-facade checks remain open.

## 6. Correctness framework (absent from vanilla)

- **Bitwise equality per profile** — optimized CPU, native, GPU, cache hit,
  and parallel RunBatch outputs must match the selected reference in shape,
  dtype, endianness, finite bytes, signed-zero bytes, nonfinite masks, and
  categorical values. No `allclose`, no export rounding, no FP32 substitution.
- **Validation chronology survives caching** — a cache hit never suppresses
  the row's required validation, warnings, state transitions, or errors.
- **Two distinct scientific checks** — bitwise implementation parity, and
  independent mathematical correctness via analytically soluble graphs and
  rational-arithmetic oracles (exhaustive simple-path enumeration with
  `fractions.Fraction`).
- **Negative controls** — one-bit/zero-sign/NaN-payload flips, key reuse under
  changed parameters, worker death, tamper rejection, and restored-checkpoint
  mismatch must fail. Validation ladder L0–L4.

## 7. Test coverage today

57 test files under `tests/`: engine/trace/identity (52, incl. cache and
refusal gates), batch runtime/plan (18), cache store/invalidation (13),
execution contract (7), science/bug-triples/TEA (8), Madina API parity (32),
native core (6), CPU kernel conformance (5), perf contract (7), platform
geometry/harness (9), published-incident reproducers (9). The large-E2E
campaign suite (100 files) is archived; see Provenance.

## 8. Explicitly not claimed

- The **10× throughput goal is not met** — official verdict `target_unmet` at
  2.38× (cold) vs the frozen comparator; 3.35× is a warm-boundary diagnostic,
  not a headline.
- **No GPU end-to-end speedup is claimed**; the device fold lost its crossover
  and the 27.8× figure is an instrument-wall measurement.
- The **`madina` shim is not packaged**; native metric-fold production
  engagement, full L4 qualification, and installed/CI gates remain open
  platform obligations.
- No single current-vs-vanilla multiple exists or may be composed from the
  segments above.

## Provenance

Baseline chain (each verified ancestor/descendant relation in git):
`5a1927d` (seed, README only) → `98f498e` (vanilla UNA source complete) →
`f57825d` (CSR optimization, landed) → `361928e` (CSR campaign close, B0) →
`16bf404d` (large_e2e closeout) → `395cdc5` (una_platform close = vanilla +
CSR + large_e2e + platform features; the frozen comparator for all una_10x
gates) → una_10x line (PR #1 merge `2d6142a`) → HEAD.

All campaign packets, receipts, raw evidence, and run records referenced above
were archived off-repo on 2026-10-15 and removed from the working tree. The
bytes remain recoverable from this repository's git history (the REGISTRY-pinned
packet trees `917475d2` and `6ef3a4ff` stay reachable), and from the Dropbox
archive `Alan Synn/Research/una-x/` (manifest `MANIFEST.sha256` inside each
tarball; sha256 digests also recorded in this file's commit message).
