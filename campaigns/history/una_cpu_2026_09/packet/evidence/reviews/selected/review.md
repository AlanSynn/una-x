# T09 independent review — SELECTED candidate, pre-freeze

- Reviewer: independent-reviewer (T09; no shared authorship with T03/T04)
- Date: 2026-09-25
- Reviewed commit: `251c67012ea33c950833d273b0e7f365be85110e` (`perf/una-cpu-throughput`, worktree `/Users/alansynn/orca/workspaces/una-x/wt-integration`, clean)
- Fork baseline: `98f498e5b0ac97b05271d03812fa883b5e4afaba` (tree `/Users/alansynn/orca/workspaces/una-x/main`)
- **Verdict: approve_with_notes**

Every claim below was re-verified in this review; nothing was taken from the task briefing on faith. Evidence classes follow the campaign convention (measured / derived / hypothetical / unavailable).

## 1. Source identity (measured)

- HEAD is `251c670`; worktree clean; `98f498e` is an ancestor of HEAD (`git merge-base --is-ancestor`).
- `git diff f57825d..HEAD -- src/ tests/perf_contract/` is **empty**: the selected source is exactly the state that passed L0/L1/L2.
- `git diff --stat 98f498e..HEAD -- src/` lists **exactly three files**: `_ordered_csr.py` (new), `Accessibility.py`, `AccessibilityWElevation.py`. The rest of the baseline→HEAD delta is confined to `tests/perf_contract/`, `benchmarks/una_cpu/`, and `optimization_una_cpu/` (harness + evidence).

## 2. Blob hashes vs `source_manifest.json` (measured)

`git hash-object` on all seven manifest-pinned files — **all match, zero mismatches**:

| file | blob | manifest |
|---|---|---|
| `src/.../Engines/_ordered_csr.py` | `186e82c0ddb6…` | match |
| `src/.../Engines/Accessibility.py` | `473de486e557…` | match (baseline blob `246fe43…` = diff old-index) |
| `src/.../Engines/AccessibilityWElevation.py` | `dc75e48baf92…` | match (baseline blob `6906168…` = diff old-index) |
| `tests/perf_contract/test_l0_ordered_csr.py` | `90ba34f6153f…` | match |
| `tests/perf_contract/test_l1_engine_arms.py` | `8a059ad041c1…` | match |
| `tests/perf_contract/_arm_runner.py` | `192997d13459…` | match |
| `tests/perf_contract/ref_original_csr.py` | `84ba8320c120…` | match |

The manifest staleness that T04 flagged has been fixed; its `hash_revision_note` correctly describes the current values.

## 3. Exact diff review — builder purity (derived from diff + measured)

The candidate helper (`_try_build_ordered_csr`) is a **pure order-preserving regrouping**:

- **No floating-point arithmetic on weights**: weights pass only through `np.concatenate` (bit-copy) and a fancy-index gather. No add/multiply/cast.
- **No output dtype changes**: fast path emits `pointer` int64 (`np.cumsum(..., dtype=np.int64)`), `neighbors` int64, `weights` float64, `is_network` bool_ — identical to the baseline `np.zeros(...,int64)` / `np.array(...,int64)` / `np.array(...,float64)` / `np.array(...,bool_)`.
- **Order rule equivalence** — structural: the baseline visits nodes 0..V−1 and, per node, appends forward incidences (`start==node`) in edge order, then reverse incidences (`end==node`) in edge order. The candidate stable-sorts `[all forward incidences in edge order; all reverse incidences in edge order]` by source node; stability preserves input order among equal keys and every forward incidence precedes every reverse incidence, so each row is exactly forward-then-reverse in edge order.
- **Empirical**: 400 randomized adversarial graphs (V∈0..8, dense self-loops, parallel edges, isolated nodes, empty graphs; int64 and int32 endpoints; aliased ab/ba weight arrays; weights including −0.0, NaN, ±Inf, 5e-324, DBL_MAX) compared **bit-for-bit** (float64 via uint64 views; dtype and shape asserted) against a verbatim transcription of the baseline loops: **0 mismatches**. Hand-checked interleaving case: node-1 row `[0,2,1,0,0,2,1]` — the self-loop appears in both its forward and reverse position.
- **Parallel edges and both self-loop incidences retained**; no deduplication anywhere.

## 4. Guard / fallback structure (derived + measured)

- The fallback blocks in both engines are **byte-identical to the 98f498e originals** after removing only the 4-space indentation and stripped trailing whitespace (programmatic line-by-line comparison). Baseline exceptions are therefore baseline behavior.
- **Only `MemoryError` is caught** (single `except MemoryError: return None`); any other failure propagates.
- Admitted domain, verified in code and by a 15-case refusal sweep (all refused): `intp` itemsize 8; `node_count` exactly `int` (bool/np.integer/float refused) with `0 <= node_count < intp_max`; all four inputs exact 1-D `ndarray`; weights exactly float64; endpoints both int64, or both int32 with `node_count <= int32max` (exact upcast); equal lengths; `edge_count <= min(intp_max,int64_max)//2`; endpoints in `[0, node_count)`.
- **No silent domain-widening.** The int32 admission is exact (int32→int64 is bijective; baseline output is already int64 via list widening) and its `node_count <= int32max` bound is conservative — exactness would hold beyond it; the bound only routes extra cases to the identical fallback (performance, not behavior).
- Exception parity probe at absurd `node_count` (2⁶², 2⁶³−2; unreachable via public API since `node_count = node_points.shape[0]`): candidate and baseline transcription both raise `ValueError` — **parity holds** (measured). Note: the int32 upcast runs outside the try/except, so OOM there would propagate instead of falling back (≤16 B/edge transient vs a larger fallback footprint; note-only, inherited from T04's review).

## 5. Public API freeze (derived from diff)

Per engine the diff is exactly: one private import, the CSR construction block (original preserved verbatim under `if ordered_csr is None:`), and the three adjacency assignment expressions (fast path assigns the builder's already-correctly-typed arrays; fallback path uses the verbatim baseline conversions). The only new symbol is the underscore-private `_try_build_ordered_csr`, referenced nowhere else in `src/` (grep verified). No public class/function/signature/settings change; outputs, errors, state transitions on the fallback path are unchanged by construction.

## 6. L0 rerun (measured)

```
env -u UNA_CANDIDATE_SRC NUMBA_CACHE_DIR=.../nbc_cand \
  /Users/alansynn/orca/workspaces/una-x/venvs/campaign/bin/python \
  -m pytest tests/perf_contract/test_l0_ordered_csr.py -q
→ 24 passed, exit 0   (ran twice, 7.75 s / 7.80 s)
```
Suite composition at the pinned blob: 9 non-parametrized + 15 FALLBACK_CASES = 24. L1 was **not** re-run per brief.

## 7. Evidence-chain consistency spot-check

| item | status |
|---|---|
| `evidence/integration/csr/test_log.txt` L1 | consistent: 12/12 final gate, 6 cases × 3 arms, per-case NPZ SHA-256 triplets; arm artifacts present |
| `evidence/integration/csr/test_log.txt` L0 | **STALE — finding F1**: records "17 passed / 11 guard-fallback cases" (the pre-amendment suite); the pinned blob yields **24 passed** (measured here and by T04's amendment rerun) |
| `evidence/trials/l2/results.json` | consistent: `all_match: true`, 5 legs incl. error_paths raising identically |
| `evidence/selection/initial/*` | consistent: 8×1 selected, queue 8, budget Σ≤8, W3 (15,400 edges), 12/0 jobs everywhere, no RSS regression |
| `evidence/trials/scope_tail/decision.json` | consistent: `not_admitted`; ~1% init share fails the dossier-06 materiality gate; cross-checks against `stage_model.json` (centrality 1.0077 s ≈ 2560×361 µs kernel share) hold |
| `engine_diff.patch` | matches the actual two-engine diff (only git prefix cosmetics `i/w` vs `a/b`); intentionally omits the new helper — finding F3 |

## 8. Findings

- **F1 (minor, measured+derived)** — `test_log.txt` L0 line is stale (17 recorded vs 24 measured on the pinned blob `90ba34f6`). Evidence-hygiene only; recommend appending a one-line L0 refresh note before freeze.
- **F2 (informational, derived)** — manifest `hash_revision_utc` (19:30Z) and the test-log `recorded_utc` (21:36:30Z) precede the `f57825d` commit time (22:18:29Z) while citing its blobs: coherent only as pre-commit working-tree recording or as `-0400` labels written as UTC. Hash values themselves verified correct.
- **F3 (informational, derived)** — `engine_diff.patch` covers only the two engines and uses `i/w` prefixes; the helper's provenance is the blob hash. Document scope if the patch is cited as the full delta.

## 9. Explicitly unavailable

- Upstream provenance commit `c15ebda6981397f46eed5c2d55229f71e57d44fb` is **not an object in this clone** (`git cat-file` fails in worktree and main checkout); the claim that `98f498e` descends from it is accepted as a recorded claim only.
- L1, L2, L3 sweep, and scope-tail probe were **not re-executed** (beyond the L0-only brief); cited evidence was reviewed for internal consistency only.
- The int32 boundary at `node_count == int32max` and near `intp_max` is not executable here (tens-of-GB allocations in both paths); refusal side is L0-pinned, admission side is code-inspection only.
- Platforms with `np.intp` itemsize ≠ 8 fall back by construction; not measured (this review measured macOS arm64, itemsize 8).

## 10. Verdict

**approve_with_notes.** The selected source contains exactly the reviewed guarded order-preserving CSR change; hashes, fallback verbatim-ness, builder purity, guard domain, API freeze, and the L0 suite (24/24) all check out. The findings are evidence-hygiene items (stale L0 line in `test_log.txt`, timestamp labels, patch scope note) and do not touch source semantics. Fit for freeze provided F1's one-line log refresh is applied.
