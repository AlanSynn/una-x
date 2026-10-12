# GOAL: UNA-X wrap-up execution (2026-10-12)

User authorization: "전부 실행. goal 로 만들어서 진행하기 위해서 진행할것." (2026-10-12) —
approves the wrap-up proposal of 2026-10-11/12: merge-readiness reconciliation, packaging fix,
evidence archive + session-worktree removal, branch-divergence recovery. Standing rules unchanged:
no pushes / origin merges / PRs / releases / paid provisioning / credential changes; PACE salloc
GPU compute stays user-authorized compute only; frozen-byte edits forbidden (fixes = new files +
errata); main-checkout staged user work (80A+1M) preserved untouched.

## Tasks (execution order; one heavy lane at a time)

- **T1 COMPAT_D3** — branch `perf/compat-d3-lazy` @ de2872f: lazy `.una` re-export (PEP 562)
  resolving the D3 env-probe collision (star-import pulled una.betweenness ->
  `USE_PYGEOS='0'` at import). Accept: platform_geometry 5/32; madina zonal 26/0 (with .refs);
  parity reflection 61/62 with delta {pdk}, una namespace 36/36, m.una chain + submodule
  identities pinned; env-write still fires exactly once on first `.una` load (upstream parity
  preserved after trigger).
- **T2 PKG_PYARROW** — branch `perf/pkg-pyarrow-dep` @ c1f8614: declare pyarrow in wheel
  metadata (+ conda yml if that matches convention; finding F1: METADATA omission breaks
  resolution-only installs at first file load). Accept: fresh-venv resolution-only install +
  public-API file-load smoke green with pyarrow pulled by resolution; METADATA diff shown;
  wheel sha256 recorded.
- **T3 EXP_BATCH_RECONCILE** — branch `perf/exp-batch-reconcile` (base be56059, merge 4658344):
  resolve the 4-file semantic conflict so BOTH survive — batchdef's reviewed resume/staging
  bytes (ExecutionOptions.checkpoint re-invoke entry; record_composite_finalized(files)
  signature; MAJOR-1 composite rebuild law) + exp-lazy's lazy-import transform and journal
  admission law (load() + verify_identity(), changed identity rejected never overwritten);
  test files take the union. Accept: batch island union green (counts explained per-file);
  cache 162 / execution 107 / cpu_kernels 97-4-0 exact; post-import sys.modules free of
  sklearn/networkx (M2 law); bootstrap import ~1.5 s class.
- **T4 GATE_RECONFIRM** (heavy lane, serialized, runs alone): 7-paired-window RunBatch gate on
  the reconciled tip using the exl-20261011T164228Z driver. Accept: one-sided 95% LB > 1
  (expect ~1.10-1.11 median); byte law 64 exports/window content-equal under stamp-prefix
  normalization. Result binds the reconciled bytes' speedup claim.
- **T5 TRAIN** — branch `perf/wrapup-train` @ c1f8614 (this file first): merge order
  T3 -> T1 -> perf/native-core-1 (69f9dd3) -> perf/readme-closure (288bd6a) -> T2. Resolve the
  expected pyproject.toml overlap (exp-lazy force-include shim vs pyarrow deps) by keeping both
  hunks. Accept: merged path set == union of branch diffs (mechanical); islands on the train:
  batch union, zonal 26, cpu_kernels 97-4, cache 162, execution 107, platform_geometry 5/32,
  native_core 110-4 after in-worktree build with loadable-section digests == recorded
  (cd404852.../64250dab.../2d68ee2e...), science standalone with the documented .refs-mode 1F,
  README renders. If green: update the README lazy-import sentence to merged-state + T4 receipt
  id (only if T4 confirms LB>1).
- **T6 MAIN_MERGE** — fast-forward local main c1f8614 -> train tip inside the main checkout.
  Abort on any refusal touching staged user paths; staged 80A+1M must survive byte-identical
  (porcelain before/after recorded). No push. Main thus carries the measured-claims README,
  reconciled perf+correctness branches, native backend, compat fix, packaging fix.
- **T7 DIVERGENCE** — preserve `perf/una-10x` @ f2ea398 (Z00 closure, earlier-gate lineage) as
  new ref `perf/una-10x-z00-local`; detach `.claude/worktrees/cpu-kernels` (untracked scratch
  stays); repoint `perf/una-10x` to origin 293730c (authoritative tenx-20261005T2331Z final,
  2.3798x / LB 2.2345). Fully reversible (z00-local ref). Receipt documents both refs.
- **T8 ARCHIVE** — before any removal: `git bundle` every branch (perf/*, mrehearse, train,
  main, z00-local); tar ALL untracked raw evidence from session worktrees (m2gate gate raws,
  native1 shipped binary/ + evidence, coldd probe, expw, g11b run dir incl.
  DECISION_MEMO/RUN_PLAN/results, batchfix, compat1, e1dec, expw-base, m2base, inspect-293730c,
  iwheel, mrehearse logs from /tmp/mrehearse-20261011) into
  `/storage/project/r-hoh68-0/dsynn6/una-x-archive-20261012/` with sha256 manifests; verify
  archive readability; THEN `git worktree remove --force` the session worktrees. KEEP forever:
  main checkout, `una-x-baseline-395cdc5r2` (삭제 금지), alan env, campaign_data.
- **T9 REPORT** — one final completion report: per-task change/verification status separated,
  receipts + shas, residual obligations (madina shim R3 packaging, parity matrix re-run,
  SCREEN_*/PARITY/CONFIRM frontier, city-failure reproducers, S02 closure), and what remains
  user-only (pushes, origin merges).

## Notes
- The tenx final evidence set lives on the perf/una-10x origin lineage; T6 does NOT merge that
  branch into main (not in the approved list) — main gets the wrap-up train only.
- Gate driver, cohort definition, and gate laws: campaigns/una_10x/evidence/tasks/EXPL/
  exl-20261011T164228Z (worktree una-x-m2gate-20261011). PACE paired sd ~0.027; n=7 moe ~2.5%.
- Receipts: campaigns/wrapup_20261012/tasks/<TASK>/receipt.md (tracked) + raw under
  evidence/tasks/<TASK>/ (untracked, archived at T8).
