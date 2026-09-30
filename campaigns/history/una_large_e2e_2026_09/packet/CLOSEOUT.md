# CLOSEOUT — una-large-e2e-2026-09

Campaign CLOSED at the review layer (h04 final ack, msg 1c22ca11). Publication executed 2026-09-30 on explicit user instruction: push to origin, advance `main`, record all findings here, purge non-repo residue so that **this repository (main) is the single traceability point** for any future session.

## 1. Published state

- Campaign line tip at review closure: `352aab04dc4b137ab7fd75ee756121885151f452` (tree `c935e818130edb60ba96ca60b0d2470a01509eb4`) — named by the reviewer's final ack; all six final artifacts hash-pinned there.
- This closeout commit is documentation-only; it carries no measurement content. Runtime content is pinned by selected source `6b53a274` / src tree `3c59bb0d`, byte-unchanged from selection through tip (verified at every generation, including by the reviewer).
- Publication geometry (live-verified 2026-09-30): remote `refs/heads/main` was `ea08947` ("Archive CSR campaign and add detailed large end-to-end CPU campaign") — a verified ancestor of the campaign tip; the advance is a **fast-forward** (no merge commit, no force update). Local `main` (`98f498e`, itself an ancestor) fast-forwarded to the same tip.
- Probes on record: `git merge-tree --write-tree ea08947 352aab0` → rc=0, result tree == tip tree; earlier rehearsal probes `main(98f498e)←07fb3cc/992f4e5/52c3550/9091316` all rc=0 zero-conflict.

## 2. Terminal verdicts (full detail: `evidence/final/decision.json` + `RESULTS.md`)

| Scope | Verdict |
|---|---|
| Flow family (F1+F2+F3+A3) | **qualified_proxy** — O3_FLOW median 1.3463815459401314, CI95 [1.3418234340096669, 1.4549372249259942], both gate legs PASS |
| Access family (A1) | **improvement_only**, radius-scoped — median 1.0969951715019743 < 1.10 (miss 0.0030048284980257467); CI excludes 1.00 |
| Protected cells (small/cold) | PASS — all medians ≤ 1.05; no regression admitted |
| A2 | **not_admitted** (never integrated; I00 absence-verified) |
| F3 | **capacity_only** labeling |
| Merge disposition | was `do_not_merge` as delivered (CI unavailable → merge_ready NOT claimed); publication subsequently authorized and executed by the user directly |

Decisive evidence: Q10 summary `438bc69f…` (40/40 valid, 2,972 jobs, installed-qualified, harness digest `3b99e202…`), R00 audit `fbf30368…` (APPROVED QUALIFIED SCOPE, every gate figure reproduced bit-exact under frozen seed 20260929), M00 rehearsal v2 `d5d2c7fa…` at `992f4e5` (wheel byte-identical, install 17/17, FF merge zero-conflict).

## 3. Findings ledger — final dispositions

1. **M00-F1 figure erratum (OPEN, carried in-record)**: v1 rehearsal recorded `main..07fb3cc` as "104/1,073/+719,973" — actually described `main..53d70e1`. Corrected figures **105 commits / 1,076 files / +720,115 / −158** recorded in decision.json + configuration.json; committed M00 v2 evidence intentionally not rewritten (h04 "no v3" ruling).
2. **R00-F2 filename deltas**: A2I declared `proof.md`→delivered `result.json`; A3I the reverse. Non-required stages, substance complete. Recorded in configuration.json `r00_f2_filename_deltas`; sha-bound files not renamed.
3. **R00-F3 claim boundary**: access family never cited as throughput-qualified; pooled median never citable; enforced across all final artifacts.
4. **M00-F2 hardcoded worktree paths (RESOLVED-BY-RECORD; hardening parked)**: suite instruments reference the primary worktree — reconciled census **35 instrument files** (39 full-literal occurrences/33 files + 2 CAMPAIGN-variable f-string sites). Post-campaign hardening (WT-from-`__file__` for code sites; explicit exemption/regeneration for the 6 frozen-data sites) is parked, unstarted. h05's per-site semantics proposal was offered in-session and is not persisted beyond the ledger summary in configuration.json `post_campaign_hardening_notes`.
5. **M00-F3 evidence-rewrite on rerun (RESOLVED-BY-RECORD)**: F1/F3 harness sessionfinish flushes committed `test_artifacts.json` (F2 via `UNA_F2_ARTIFACTS` default); retain→restore mechanical in run scripts.
6. **Coordinator Z00-P1 census correction (accepted by reviewer)**: `windows[].extra.argv` exists in **16 of 40** Q10 windows (holdout 2, pair 4, smi w1–w2 4, cold 4, preflight 2); the other 24 carry the 6-key schema. Reviewer's original "no argv key" premise was a single-window generalization; final seq1 provenance in commands.jsonl states the true census and cites the 40 per-fire `session.json requested_config` records.
7. **Presentation-only riders (recorded here, no edit to acked artifacts)**: (a) RESULTS.md §2 lists AggregateFlow as "+606"; the numstat figure is **+586/−20** (606 total changed lines). (b) RESULTS.md §6/decision.json derived-claims bucket split ("campaigns packet/docs 38, other 22") is a presentation partition; a mechanical partition yields campaigns-non-evidence 36 / other 30, including 18 screen-era files at REPO-ROOT `evidence/F2R/` that ride the merge. Research-only.
8. **Min-pair reconciliation**: **1.0343748618764228** is authoritative (live summary bytes + raw A/B recompute agree); R00 `claim_audit.json`'s `…226` is an ulp-level arithmetic-path artifact. R00 bytes stay frozen.
9. **3 candidate-tip check failures** (dispositions in configuration.json): t7/t8 stale F1-era source-audit pins vs screened F2/F3 integration; diagnostic-mode `TypeError` at `AccessibilityWElevation.py:351` (`NUMBA_DISABLE_JIT` × A1 `@overload` pass-stub) — outside the qualified measurement frame; remediation notes recorded.
10. **Unavailable gates (recorded, never passed)**: production target (not supplied), CI (none configured), second observed morphology, turns/ODM real-engine, disposable-frame full suite, full-repo single-invocation pytest leg (withdrawn as frame-suspect).

## 4. Custody purge manifest (executed immediately after this commit)

Per the user's closeout instruction ("delete everything useless including worktrees; everything must remain on main"), the following out-of-repo residue was **deleted** after publication:

| Path | Size at closeout | Contents |
|---|---|---|
| `una-x/campaign_data/` | 32 GB, 246,658 files | raw session/console/lease records, arm wheels+venvs, identity manifests, incident custody, per-stage run trees (a1r…w00, q10_run, m00_rehearsal) |
| `una-x/venvs/` | 708 MB, 16,092 files | campaign supervision venv + arm venvs |
| `una-x/wt-large-e2e/` | 103 MB | integration worktree (branch `perf/una-large-e2e`; branch label deleted after merge) |
| `una-x/wt-b0/` | 78 MB | baseline worktree (detached at `361928e`; commit preserved in main history) |
| `/tmp/nbc_*` | 78 scratch roots | NUMBA_CACHE_DIR per-run roots + probe scratch |
| `una-x/.pytest_cache/` | — | campaign-created |

**Recovery story**: every figure and artifact identity cited above is hash-pinned in committed evidence (`evidence/**`, `raw_index.json` per-file sha256 index over the purged trees). Raw session bytes are intentionally NOT preserved. The qualified wheel is rebuildable **byte-identically** from this repository (M00-proven): `git archive <selected tree> | tar -x` + hatchling build per `evidence/final/commands.jsonl` seq3–seq4, expected sha256 `323c27f3ace8daafcc6d025596a605e8263cc14c8c4fe602c427d51069448ac4` (138,442 B). The b0 comparator wheel identity is recorded in `evidence/W00/wheels.json` (`c443bca9…` pin).

Retained on disk: the primary repo checkout (`/Users/alansynn/Workspace/una-x`, branch `main`), the user's `AlanSynn/main` worktree, and orca-managed session state. No branch history was lost: the campaign's 98 commits sit on `main`.

## 5. Fresh-session traceability map

1. This file → terminal verdicts, findings dispositions, purge manifest.
2. `evidence/final/` (6 required artifacts) → every number, hash, command, and limitation.
3. `evidence/<STAGE>/` → per-stage records, reviews, manifests (H00…Z00; `history/una_cpu_2026_09/` for the CSR predecessor campaign).
4. Key commits on main: `53d70e1` (Q10 evidence mirror) → `07fb3cc` (R00 close, rehearsed closure) → `10dca11`/`992f4e5` (M00 v1/v2) → `52c3550`/`9091316`/`352aab0` (Z00 v1/v2/v3) → this closeout.
5. Review authority chain lives in the commit messages of those commits and in `evidence/R00/review.json` + `evidence/M00/rehearsal.json`; the reviewer's final tip-naming ack is recorded in the `352aab0`-generation commit trail (message of the Z00 v3 commit and the reviewer's session ledger).
