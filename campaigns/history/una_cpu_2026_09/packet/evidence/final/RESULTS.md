# UNA CPU-throughput campaign — RESULTS

**Campaign state: `qualified`** — all technical gates pass for the claimed proxy scope; `merge_ready` is not claimed (no CI, no actual production target, no merge authorization). Merge disposition: **do_not_merge** (clean-merge rehearsal complete; awaiting explicit authorization).

## What changed

Three `src/` files vs fork baseline `98f498e` (upstream provenance `c15ebda…`, blob-level bridge):

| file | change |
|---|---|
| `Engines/_ordered_csr.py` | **new** — guarded, order-preserving CSR construction (no weight arithmetic; owned int64/float64/bool arrays; returns `None` outside the admitted domain) |
| `Engines/Accessibility.py` | CSR block swapped for the guarded helper; original construction kept verbatim as fallback |
| `Engines/AccessibilityWElevation.py` | same |

One amendment during qualification (commit `f57825d`): the supplied candidate's int64-only guard silently fell back on **real** topologies, whose endpoint columns are **int32** (measured on the genuine W3 fixture). The guard now admits int32 endpoints via an exact value-preserving upcast (`node_count ≤ int32max`), justified because the original builder widens any signed endpoint input to int64 on output. Re-validated through the full gauntlet and independently re-reviewed.

**Final source SHA: `f57825d`** (branch `perf/una-cpu-throughput`; HEAD `2f4dae6` has identical `src/` blobs).

## Verification (all measured, evidence paths under `optimization_una_cpu/evidence/`)

- **L0 exactness** (`integration/csr/test_log.txt` + REVISION): 24 passed at the frozen blobs — exhaustive tiny-graph bit-exactness vs an independent oracle transcription, int32 up-cast exactness, read-only input immutability, NaN/±Inf/−0.0 bit patterns, 15+ crafted refusal domains → `None` → original path.
- **L1 whole-engine arms**: 6 seeded cases × 3 arms (baseline A/B, candidate) as fresh subprocesses; determinism gate then bit-identical candidate. NPZ artifact hashes per case identical across arms (preserved in `integration/csr/l1_arms/`).
- **L2 genuine GIS public API** (`trials/l2/`): ALL_MATCH across accessibility, no-elevation, flow/gravity-cap, ODM, and error-path legs.
- **T09/T04/T12 independent reviews**: approve_with_notes ×3; findings addressed (log refresh, artifact preservation, exec bits).
- **T12 final audit** (`reviews/final/`): recomputed stage medians, throughput, hashes — all exact.

## Performance (M1 Pro 8P+2E, budget C=8 excluding E-cores, exclusive lease per run)

Stage model (medians, n=5/tree, alternating): **engine construction 245.4 ms → 2.2 ms (109.5×)**; every other stage statistically unchanged. Single-process job ≈ 1.77 s → 1.27 s.

Batch throughput (12 jobs, W3 accessibility, queue 8; raw records in `selection/initial/runs/` and `trials/final/runs/`):

| config (W×H) | baseline jobs/s | candidate jobs/s |
|---|---|---|
| 1×1 | 0.724 | 0.925 |
| 1×8 | 1.232 | 2.105 |
| 2×4 | 1.912 | 3.292 |
| 4×2 | 2.496 | 3.580 |
| **8×1** | **2.848** | **3.915** |

Selected: **8 workers × 1 Numba thread, queue 8** (more processes beats threads at this kernel mix; wave-packing checked). Final alternating-order confirmation (cand, base, ×2, cand): **4.44 vs 3.33 jobs/s median (1.336×)**; candidate beat baseline in every run; **6.14× end-to-end vs the sequential baseline reference**. Memory: max worker RSS 352 MB — no regression at any concurrency.

Installed path: wheels built from both trees; candidate qualification in a clean venv proves site-packages import, fast-path engagement on genuine int32 input, guard-refused fallback on crafted int16 (monkeypatch counters, non-performance test), working CSV/GeoJSON/Feather writers, and zero packet imports. The wheel built from the rehearsed merge is **byte-identical** to the frozen candidate wheel.

## Rejected / not admitted

- **C3 scope-tail scratch-state reduction: not_admitted** — measured init share **0.98 %** of the Dijkstra scope kernel (3.54 µs alloc+init vs 361 µs full search), < ~1 % end-to-end headroom; dossier 06 materiality gate failed; zero implementation attempts (`trials/scope_tail/`).
- Design-rejected register (scalar relaxation, tree reductions, caches, RunBatch parallelization, GPU, etc.) — see `decision.json` and `CANDIDATE_REGISTER.md`.

## Explicitly unavailable / limitations

- **Actual production target (L4)**: unavailable; W3 is a genuine-API proxy recalibrated (46×40→92×80, 2560 origins/destinations, 1.64 s baseline job wall) *before* any recorded measurement. Transfer of gains to larger networks is hypothetical.
- **CI**: none exists in this fork.
- **Upstream object provenance**: commit `c15ebda…` absent from clone (content-seeded fork); T01 blob-level bridge stands in.
- Inherited, unchanged: **NaN edge-weight hazard** in the scope kernel (unbounded queue growth in *both* trees; `integration/csr/nonfinite_weight_hazard.md`); **undeclared pyarrow dependency** (`use_arrow=True` reader); shared-machine load variance (~13 % between sweep series).
- Flow analysis: stage-model + L2 coverage only; not batch-swept. Non-lp64 platforms and int32-boundary node counts: refusal-pinned, unmeasured at scale.

## Merge

Rehearsal (disposable worktree at target `main` SHA `98f498e`): `git merge --no-ff perf/una-cpu-throughput` → clean, zero conflicts (`a6f9a9a`); L0 24/24 in the merged tree; merged-tree wheel byte-identical to the frozen wheel. Worktree discarded; `main` untouched. Full manifest: `final/merge_manifest.json`.
