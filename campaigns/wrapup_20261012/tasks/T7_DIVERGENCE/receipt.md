# T7 DIVERGENCE receipt — perf/una-10x recovery + repoint (2026-10-12)

Authorization: user "전부 실행" (2026-10-12) covering the wrap-up decision list item 4
(divergence recovery, non-destructive path chosen).

## Facts (from tool output, 2026-10-12)

- origin/perf/una-10x = 293730c9a53954ca6a846554567624880c1c4786 (authoritative tenx
  final tenx-20261005T2331Z: median 2.3798, LB 2.2345, target_unmet — the handoff-
  authoritative gate the README quotes).
- local perf/una-10x was f2ea398219111df11ba05839e617ecca5e14f2ed (Z00 closure, earlier
  final tenx-20261004T155640Z: median 2.6632, LB 2.5948, target_unmet; merge-base with
  origin 4a5e56f7 — the local lineage was superseded by the origin-side re-run).

## Actions (reversible, non-destructive)

1. `git branch perf/una-10x-z00-local f2ea398` — local Z00 lineage preserved under an
   explicit ref. NOTHING deleted; every commit reachable.
2. `.claude/worktrees/cpu-kernels` (was the only checkout of perf/una-10x, at f2ea398)
   detached at the same commit; its untracked campaign scratch (_c10_/_c11_/_c12_ scripts
   etc.) untouched on disk.
3. `git branch -f perf/una-10x origin/perf/una-10x` — branch now == origin 293730c
   (tracking established). This is the sole branch-ref mutation; reversal is
   `git branch -f perf/una-10x perf/una-10x-z00-local`.

## Verification

- `git rev-parse` after: perf/una-10x = 293730c..., perf/una-10x-z00-local = f2ea398...
  (both printed by the same tool invocation that performed the repoint).
- Worktree list post-action: cpu-kernels at f2ea398 (detached HEAD).

## Consequences

- Both tenx finals remain reachable: 20261005 (origin tip) and 20261004 (z00-local).
- perf/una-10x is NOT merged into main by this work order (not in the approved list);
  its evidence reaches main only on a future explicit user instruction.
