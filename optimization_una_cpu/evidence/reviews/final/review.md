# T12 final evidence audit — verdict: approve_with_notes

Auditor: independent final evidence auditor (read-only; only this pair of files written).
Audit date: 2026-09-26. Worktree: `/Users/alansynn/orca/workspaces/una-x/wt-integration`, branch `perf/una-cpu-throughput`, HEAD `edd266b`. Machine-readable record: `review.json` (same directory).

## What was verified (all recomputed or re-inspected by this audit, not cited)

1. **Freeze blob hashes** — `git hash-object` on all six src files at HEAD matches `evidence/freeze/candidate.json` and `evidence/integration/csr/source_manifest.json` exactly (`_ordered_csr.py` 186e82c0, `Accessibility.py` 473de486, `AccessibilityWElevation.py` dc75e48b, `Base.py` 17442082, `Topology.py` 26ca999f, `UNA.py` f803a536). **PASS**
2. **Source frozen** — `git log 98f498e..HEAD -- src/` shows only 755c7b8 and f57825d; `git diff f57825d..HEAD -- src/` is empty; src identical across 251c670 / 743c760 / HEAD (0-byte diffs); `freeze/final_source_diff.patch` is byte-identical to `git diff 98f498e..HEAD -- src/` (331 lines, exactly 3 files). **PASS**
3. **Baseline untouched** — `main` checkout clean, HEAD == 98f498e. **PASS**
4. **Stage model** — all 9 stages x 2 arms (18 medians) recomputed from `selection/initial/stage_runs/*/stage_profile.json` match `stage_model.json` exactly; engine_ctor 245,379,333 ns -> 2,240,125 ns (saving 243,139,208 ns exact; 109.54x vs recorded 109.5x); workload counts identical in all 10 raws. **PASS**
5. **Final confirmation** — `selection/final/configuration.json` matches all five `trials/final/runs/*/session.json` exactly (throughput, wall, per-worker peak RSS, done/failed, W/T/Q, budget check, mem-available, loadavg). Medians recompute exactly: cand 4.4444, base 3.3270 jobs/s; speedup recomputed 1.3358 (recorded 1.336); every cand run beats every base run; 6.14x = 4.4444 / 0.7236 (base 1x1 raw 0.7236). **PASS**
6. **Wheels** — both files present in `/tmp/wheels/{base,cand}`; sha256 match `installed/manifest.json` and the freeze record; re-unzipped: content diff is exactly `Engines/_ordered_csr.py` added, Requires-Dist identical, version 2.6.0 both. **PASS**
7. **L1 arm hashes** — `l1_arms_hashes.txt` has exactly 18 lines, one SHA per case across the 3 arms; re-hashed live artifacts: post-amendment `/tmp/l1_arms` (incl. `int32_realdata` 180e4781) matches all six recorded SHAs; committed `arms/` matches the pre-amendment record (`int32_fallback` f98b15b7). **PASS** (see F1)
8. **Fixtures** — all 8 W1/W3 geojson sha256s match `probes/fixtures/manifest.json` and `trials/l2/manifest.json`. **PASS**
9. **L0 rerun (sanctioned)** — `24 passed in 7.66s` at the frozen blobs with `NUMBA_CACHE_DIR=venvs/nbc_cand`. **PASS**
10. **Initial sweep** — all 10 raw session throughputs match `followup_admission.json`; 8x1 top on both arms. **PASS**
11. **L2 / scope tail** — `all_match: true` (5 legs, error paths raise identically); scope-tail `not_admitted` arithmetic cross-checks (0.98% init share; 2560 x 361 us ~ 0.92 s vs 1.0077 s centrality median). Consistent; not re-executed.
12. **Overclaim scan** — no GPU claims, no L4 claims, no CI claims (none exist: no `.github`), upstream provenance consistently stated as a blob-level bridge with the upstream commit object explicitly absent from the fork ODB. **CLEAN**

## Findings

| ID | Severity | Class | Summary |
|----|----------|-------|---------|
| F1 | minor | measured | Post-amendment L1 raw npz artifacts (the frozen gate, incl. `int32_realdata`) live only in volatile `/tmp/l1_arms`; the committed `arms/` dir retains pre-amendment artifacts only. Hashes are committed and verified today. Copy `/tmp/l1_arms` artifacts into `evidence/` before the merge rehearsal. |
| F2 | informational | derived | `reviews/csr` cites "ctor_ab W3 E=15400: 239.9 ms -> 1.96 ms" with no committed raw artifact; committed `csr_microbench.json` instead shows E=15334: 258.95 ms -> 0.486 ms and E=62023: 3237.7 ms -> 1.968 ms. Non-blocking: the decision rests on the committed, recomputed stage model, not kernel-only numbers. |
| F3 | informational | derived | Frozen-SHA labeling: freeze record says 251c670, T11 evidence says 743c760 — correct only via blob identity (verified). decision.json / merge_manifest.json should carry the blob-hash set (or patch) as identity. |
| F4 | informational | measured | wt-integration tree not clean: uncommitted mode flip on `benchmarks/una_cpu/t11_final.sh` (100644 -> 100755, content identical). No src/tests impact. |
| F5 | informational | derived | `installed/manifest.json` "about() works" vs qualification record `"about": "None"`; version checks passed. Wording only. |
| F6 | informational | derived | `recorded_utc` labels remain pre-commit / possibly local-time (carries T09 F2). Hash-addressed content unaffected. |

## Unavailable gates the final decision.json must carry (verbatim reasons)

1. **actual_target / L4 — unavailable**: no unreduced production workload or throughput target supplied; all performance evidence is the W3_medium_proxy L3 proxy + installed-wheel qualification. Claim scope: proxy-qualified, not production-wide.
2. **Upstream commit-object provenance — unavailable**: c15ebda… is absent from the fork ODB; identity rests on the blob-level bridge (`contract/oracle_bridge.json`).
3. **CI / required checks — unavailable**: no CI configured, no runs exist; per DECISION_CONTRACT, `merge_ready` cannot be claimed — expect **qualified**.
4. **Merge rehearsal — pending** (T14 input).
5. **Non-lp64 / non-macOS-arm64 behavior — unmeasured** (guard falls back by construction; only macOS arm64 executed).
6. **int32max / 2^60-scale node_count boundaries — not executable here** (allocation scale); refusal side L0-pinned, admission side inspection-only.
7. **GPU — out of scope by charter** (never claimed — correct).
8. **Flow-path batch sweep — not performed** (flow covered by stage model + L2 only; recorded limitation).
9. **Sequential baseline reference is single-sample** — 6.14x divides by one base_1x1 run; the alternating 1.336x at-config figure is the measurement of record.

## Inherited limitations to carry

- NaN edge weights cause unbounded queue growth in the upstream scope kernel in **both** trees (`nonfinite_weight_hazard.md`); outside safe domain; not introduced or masked by the candidate.
- pyarrow used by `Topology._get_gdf` but undeclared in metadata; both wheels affected identically; provisioned explicitly in qualification venvs; upstream follow-up recommended.
- Laptop shared with unrelated load (swap 3.4/4 GiB, loadavg 5–7 at inventory); absolute numbers laptop-local; no cross-environment transfer claimed.

## Verdict

**approve_with_notes.** Every load-bearing claim recomputed from raw artifacts holds exactly; findings are evidence hygiene or wording, none touching source semantics, numerical behavior, or measured results. The evidence chain is fit for the merge rehearsal, conditional on the unavailable gates above being carried verbatim into `decision.json` (and F1's artifact retention being fixed before or during it).
