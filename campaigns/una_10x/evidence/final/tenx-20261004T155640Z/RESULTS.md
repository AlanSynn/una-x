# RESULTS — una-10x / run tenx-20261004T155640Z (Z00 final record)

Campaign `una_10x` · branch `perf/una-10x` · closed 2026-10-05 by the
integrator. Every figure below is loaded from a hash-pinned committed receipt or
recomputed live from raw bytes; nothing is transcribed from prose. The independent
final review (R00) recomputed the decisive numbers itself and found **zero
discrepancies**.

## 1. Terminal summary

**tenx_status: `target_unmet`** (DECISION.md taxonomy; inside the preregistered
outcome set of the q00r freeze). **platform_status: `incomplete`** (duties listed
in §6). **merge_disposition: `do_not_merge`** — publication remains a separate,
explicitly user-gated act; no push, merge or release was performed.

| Gate | Requirement | Measured | Verdict |
|---|---|---|---|
| Q10 primary (installed public UNA.RunBatch, frozen 32-job cohort, w4t1) | median paired ratio >= 10 AND one-sided 95% LB >= 10 | **median 2.6632x, LB95 2.5948x** | **TARGET_UNMET** |

The campaign delivers a real, tightly measured **~2.66x** public improvement
(window spread 2.594–2.710, ~4% relative), not the 10x claim. No gate was
weakened: no tolerance, no baseline change, no workload reduction, no kernel
extrapolation, no optional stopping.

## 2. Primary gate (Q10R, installed repaired wheels, preregistered)

7 paired AB,BA,AB,BA,AB,BA,AB windows; seed 20261003; 20000 bootstrap resamples
over paired ln(base/cand); timer brackets ONLY the public UNA.RunBatch call;
FIRST_COHORT cold stores; per-tree warm numba caches via two untimed warmups;
host atl1-1-02-003-4-2.pace.gatech.edu.

| window | order | base wall (s) | cand wall (s) | ratio base/cand |
|---|---|---|---|---|
| w01 | b->a->s->e | 66.8982 | 24.8718 | 2.6897 |
| w02 | c->a->n->d | 67.8346 | 25.1361 | 2.6987 |
| w03 | b->a->s->e | 66.2467 | 25.2829 | 2.6202 |
| w04 | c->a->n->d | 67.5362 | 26.0343 | 2.5941 |
| w05 | b->a->s->e | 65.9066 | 25.3994 | 2.5948 |
| w06 | c->a->n->d | 67.0173 | 24.7338 | 2.7095 |
| w07 | b->a->s->e | 67.9303 | 25.5069 | 2.6632 |

Every one of the 14 measured runs (both arms): cold store asserted
(`outdir_existed_before=false`), 32/32 rows, zero failed, engagement asserts
green, workers_effective=4, and **bitwise 64/64 sha256 equality vs the frozen
P00 reference (147830672 bytes) outside timing**. R00 independently corroborated all
14 runs' 64 artifacts byte-identical across arms/windows AND against the P00-era
anchor — the ratio is a TRUE same-bits comparison.

## 3. Why the target was missed (measured stage fact, not extrapolation)

The delivered candidate removes the per-row GIS load/parse rework that dominated
the v1 profile (~75% of row time): prepared input-keyed single-flight (P02I),
compiled outer replay (P03C), typed metric reuse (P03M). The residual
~25 s/wall is dominated by irreducible prepared-fold compute — the committed
stage model's demand floor is ~16.6 s of the ~66.5 s cohort demand at C=4,
bounding the cohort ceiling near 4x. Reaching 10x on this cohort would require
changing the numerical work itself, which the bitwise law
(una_legacy/madina_legacy/corrected_v1) forbids.

## 4. Defect found BY the gate, repaired under the recorded-invalidation law

Q10 attempt 1 died in its first UNTIMED warmup process: the installed wheels of
BOTH arms could never spawn a parallel pool — `una_batch_worker_main` (spawn
shim, inherited from 0acbb11, byte-identical at comparator pin 395cdc5 and
candidate tip) was not shipped by pyproject. No measured window ran; charged
0.0. Repair: packaging-only force-include (shim stdlib-only, arm-neutral),
both wheels rebuilt (+shim-only diffs proven), re-installed, V00R gauntlet
re-proven (368 behavioral passes; the v1 src-less twin failures WERE the
missing shim), installed parallel pool smoke green on both arms, identities
re-frozen (I00R manifest + Q00R re-freeze, protocol_delta NONE). R00 re-hashed
the wheels and byte-matched the installed targets.

## 5. Validation chain and budget

Chain: N00 → N10 (REJECT_DEFAULT) → G00 → G10 → G11 (V100 real device,
CPU_ROUTE_RETAINED) → X00 obligations ledger → I00 → V00 → Q00 → I00R fix →
I00R → Q00R re-freeze → Q10R gate → R00 review → Z00 close. Heavy budget: the
14 measured public walls charged **646.3 s**; cumulative **5100.6 s of
28800.0**, remaining before the 7200 s reserve **16499.4 s**. All other
closures were probe-scale evidence walls charged 0.0 per campaign precedent.

## 6. Platform status: incomplete (duties preserved, never waived)

- **MADINA_API_PARITY** — open_partial: pinned-Madina parity MATRIX re-run on
  the current installed tree still owed (X00/integrator).
- **SCIENCE_CORRECTED_V1** — open_partial: S02 distinct-reference closure owed;
  no model change was used for any speedup claim (bitwise law enforced).
- **CITY_FAILURES** — open_recorded: 9 incidents without reproducer, fix
  commit or verified root cause (H01/X00).
- **INSTALL_FINAL_GATES** — pending: this Z00 close records them honestly.
- Closed this campaign: NATIVE_SUPPORT and GPU_SUPPORT (real code, real
  hardware evidence, lawfully not production defaults); BATCH_RUNTIME covered;
  CACHE_STORE closed on perf/una-platform (merged locally, no push).

## 7. Identities

- Source tip (rehearsed + reviewed): `129c53f94e85210b1d4e5bbbc1a9d0d86b34e2e9`; src tree `0621e867990c3e623a95ea0991fa272ef931d935`.
- Comparator: pin `395cdc5f683894b6f2ba460f7dbcefee99981ba3`, repaired base
  wheel `f0504f886eb99761…`; candidate wheel `5a7017f71ed1a1d4…` (repair head
  2562689); spawn shim `037d6c47c7ec3a08…`.
- main `c1f8614a2733903b3fa9664579ba2fdce7ef7a74` is the merge base (fast-forward possible, 51 commits
  ahead); read-only `git merge-tree --write-tree main 129c53f` result tree ==
  tip tree (zero conflicts). Rehearsal: `129c53f94e85210b1d4e5bbbc1a9d0d86b34e2e9` — compileall rc0, import
  resolves to the tree, identity+public islands **50 passed**, worktree
  clean, disposable frame removed.
- CI: **unavailable** (none configured — absent CI is unavailable, never
  passed).

## 8. Claim scope and non-claims

The measured claim is the installed-proxy paired ratio above and nothing
else: no production workload was supplied; no 10x is claimed in any cell; no
component ratio is multiplied into a throughput claim; native/GPU routes are
explicit options, not defaults; unverified_claims: none — every claim here
carries a receipt, and the R00 reviewer lists its own checks-not-performed.

## 9. Merge recommendation

`do_not_merge` as delivered. Merging `perf/una-10x` (129c53f, 51 commits on
top of main c1f8614) is technically a clean fast-forward with a green
disposable rehearsal, but publication requires a new explicit user
instruction. The platform duties in §6 are not closeable by this merge.
