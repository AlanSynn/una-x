# T8_ARCHIVE receipt — 2026-10-12

Archived all session-worktree raw evidence to durable storage, verified it, and
only then removed the worktrees. Deletion order held: every artifact was written
AND verified (listing + sha256 + spot-extract compare) before any deletion.

- Executor: T8 agent (batchfix teammate session), on behalf of team-lead wrap-up GOAL.
- Destination: /storage/project/r-hoh68-0/dsynn6/una-x-archive-20261012/ (21.16 GB total)
- Full details: MANIFEST.md in that directory (authoritative; this receipt summarizes).

## Archive contents

1. Git bundle `bundles/una-x-all-refs-20261012.bundle` — 18,483,676 B,
   sha256 b75a8026a21753e903467ad52619e25103d056ad42847026bc2b13536cc07667.
   `git bundle verify`: okay; 43 refs; "The bundle records a complete history."
   Detached-pin audit (merge-base --is-ancestor vs durable refs): f2ea398 ->
   perf/una-10x-z00-local; 293730c -> perf/g11b-explicit-order (== tip); 395cdc5
   -> main; dd9c5ac -> perf/exp-batch-reconcile; c1f8614 -> mrehearse/integration
   (== tip); c8b19f2 -> main (verified specifically, contained by 8 further perf
   branches). All reachable: NO preservation branches needed, no re-bundle.

2. Per-worktree untracked-evidence tars (untracked non-ignored only; mandated
   regenerable exclusions `__pycache__`/`.pytest_cache`/numba caches applied —
   census: 0 files matched any exclusion pattern; `.so` artifacts and the
   iwheel venv deliberately kept). Each tar has `worktree-<name>.sha256`
   (whole-tar sha) + `worktree-<name>.files` (exact path list) sidecars.

| Worktree | Pin | Files | Tar bytes | sha256 (whole tar) | Spot-check |
|---|---|---|---|---|---|
| batchfix-20261011 | be56059 perf/batch-def24-major1 | 15 | 2529280 | 4958787818bfeae7c9d1f7ee023c09f1f61abb433bb45621fa58f4952bc785d2 | 3 MATCH |
| coldd-20261011 | efc33dc perf/exp-lazy-probe | 1084 | 1652736000 | 0544edcb3e938fc325430b51d58963924bb3d29980dc17b649f862fbfa12235b | 3 MATCH |
| compat1-20261011 | de2872f perf/compat-una-reexport | 1 | 10240 | 65e72add31cbb42b2efd2e179f7ae28ffa3c8154e9611e83ddf2f7ef10937242 | sole member `.refs` symlink; archived link text == live |
| compatfix-20261012 | 3bb3541 perf/compat-d3-lazy | 22 | 40960 | 669cd88c7399220b142bb6e5420188542f464185c953bbb97ac7ab94c20f2b4e | 3 MATCH |
| cpu-kernels | f2ea398 (det) | 5069 | 8171827200 | 9ec9430980fcff74a464b55d5fe38d937fada6c868d5099b1fc7ffddf91c15d3 | 5 MATCH; SPLIT 3 parts, cat-sha == sidecar |
| e1dec-20261011 | f2ea398 (det) | 523 | 580372480 | 303b41309e4086533e3872005284ba494477b121efc84a65aad22886c4e6f740 | 3 MATCH |
| expw-20261011 | bc001f3 perf/exp-writer-1 | 1289 | 1752176640 | 7bba55f465756b2e0684e271bd9521f12dd687f3d12cb9d4b51874c9322926a9 | 3 MATCH |
| g11b-20261011 | 293730c perf/g11b-explicit-order | 16 | 174080 | e778f6a4bf6a187fc2777d309fc03cd2031cb1c160178edacc0912b27ebb3a44 | 3 MATCH |
| g4cand-20261012 | dd9c5ac (det) | 1739 | 3123077120 | 07f80bab733e4c7f6b7d3326af2aaaf31bd72263252d1e873b7694ea902c9bb7 | 4 MATCH incl CRITICAL g4rc-20261012T0512Z/windows.json |
| iwheel-20261011 | c1f8614 (det) | 9061 | 782571520 | 26804ccee016c8982b986f3cd580b3aac22f52de1b0989a299045a73d00a1f38 | 3 MATCH |
| m2gate-20261011 | 4658344 perf/exp-lazy-1 | 2208 | 3322521600 | 8f92abc86e7b38dcf00896f1f0b1cad204842427959a1e92f66e93b1342c680e | 4 MATCH incl CRITICAL m2gate-20261011T164228Z/PREREGISTRATION.md + bytelaw_normalized.json |
| mrehearse-20261011 | 16467bc perf/wrapup-train | 1 | 20480 | e33f7d8c2b0860deb472945dd8f0a82502aa5cef771bd91230172f278c4c10d7 | sole member libuna_metric_fold.so sha256 1d9c3b4e...dae64 == live |
| native1-20261011 | 69f9dd3 perf/native-core-1 | 13 | 3778560 | 69031f598c5f6f5646732979f744ddcd5c67501c851b07c66a1fc7cc09b1d4e1 | 3 MATCH incl CRITICAL shipped _native_core.cpython-311-x86_64-linux-gnu.so |
| s12-review-r1 | c8b19f2 (det) | 27 | 6195200 | 81695b4d14c2c9194fa79c4b5e3330004054075521602556f4288a421822952b | 3 MATCH |

Zero-untracked worktrees (no tar; pins reachable via bundle branch refs):
expw-base-20261011 (f2ea398), g4main-20261012 (c1f8614), inspect-293730c
(293730c), m2base-20261011 (f2ea398), pkgarrow-20261012 (f23cfcd),
readme-20261011 (288bd6a), reconcile-20261012 (6c48b56).

3. tmp/ extras: `tmp/mrehearse-20261011-tmp.tar` (64 members, 3164160 B,
   sha256 2a62e329aa778e6fe47f0d5a5144500188ac993c081cab75d36fc11defad6f50,
   3 MATCH) and `tmp/main-evidence-PKG_PYARROW-EXP_BATCH_RECONCILE.tar`
   (29868 members = 29860 files + 8 symlinks, 1736079360 B, sha256
   3a91dad6c38fb3afe72e5a52f595fb05514f0d4f3011850bb251b4458eb42526, 3 MATCH).
   The latter is a COPY; originals remain in the main checkout (verified present
   post-copy). Note: no file named "receipt" exists in the m2gate raw run dir;
   its gate-verdict artifacts are PREREGISTRATION.md / bytelaw_normalized.json /
   guard+leg JSONs (first two spot-verified byte-equal).

Verification totals: 48 spot-extract sha256 comparisons, 48 MATCH, 0 mismatch;
every tar `tar -tf` clean; every sidecar sha re-measured equal. All checks ran
before any deletion.

## Removal log (only after verification was fully green)

`git worktree remove --force` (authorized: untracked files present), all 21:
batchfix-20261011, coldd-20261011, compat1-20261011, compatfix-20261012,
e1dec-20261011, expw-20261011, expw-base-20261011, g11b-20261011,
g4main-20261012, g4cand-20261012, inspect-293730c, iwheel-20261011,
m2base-20261011, m2gate-20261011, mrehearse-20261011, native1-20261011,
pkgarrow-20261012, readme-20261011, reconcile-20261012,
.claude/worktrees/cpu-kernels, .claude/worktrees/s12-review-r1.
Result: 21 REMOVED, 0 FAILED. Then `git worktree prune` (exit 0) and
`rm -rf /tmp/mrehearse-20261011` (confirmed gone).

Count note: the work order named 20 worktrees but stated "21"; the 21st is
una-x-readme-20261011 (present in `git worktree list`, zero untracked files,
archived-in-bundle pin 288bd6a = perf/readme-closure tip). It was archived
(recorded, zero-untracked) and removed to satisfy the required final state.

## Final state

`git worktree list` after prune (EXACTLY the required two):

    /storage/scratch1/1/dsynn6/una-x                     16467bc [main]
    /storage/scratch1/1/dsynn6/una-x-baseline-395cdc5r2  395cdc5 (detached HEAD)

Main checkout protection: `git status --porcelain` captured pre-removal (83
lines, sha256 7ba6559706e04a5abd7047b7e3abfb677b4aa98f595551c633d6274439a344df)
and re-captured post-removal: byte-identical (same sha). The user's staged work
and untracked files are untouched. Protected paths una-x-baseline-395cdc5r2,
/storage/scratch1/1/dsynn6/campaign_data and the alan env were never modified.
No pushes; this receipt is the only committed change (path-limited commit).

## Not verified / residual risks

- Restore path is documented but was NOT exercised end-to-end (no test clone of
  the bundle or test extraction into a fresh worktree was performed; the bundle
  verify + per-tar listing/spot checks are the evidence that the archive is
  sound).
- Split-tar restore of cpu-kernels requires `cat` of the 3 parts (documented in
  MANIFEST.md section 5).
