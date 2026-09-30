# RESULTS — una-large-e2e-2026-09 (Z00 final record)

Campaign: `una-large-e2e-2026-09` · branch `perf/una-large-e2e` · recorded 2026-09-30 by the coordinator/integrator.
Every figure below is either quoted from a hash-pinned ruled record (pin noted) or computed in-session from live bytes on 2026-09-30; nothing is transcribed from prose alone.

## 1. Terminal summary

**Stop state:** DECISION.md **"Approved qualified scope" arm satisfied** (R00 `completion_condition_mapping`; `evidence/R00/review.json` sha256 `fbf30368b21470c6…`).

| Scope | Terminal state | Basis |
|---|---|---|
| Flow family (F1+F2+F3+A3 on the A1-advanced line) | **qualified_proxy** — primary gate PASS both legs | O3_FLOW 5-pair median **1.3463815459401314** ≥ 1.10 AND CI95-lo **1.3418234340096669** > 1.00 |
| Access family (A1) | **improvement_only** (preregistered space) | O3_ACCESS median **1.0969951715019743** < 1.10 (miss **0.0030048284980257467**); CI95 **[1.0518485379517482, 1.1577532365900536]** excludes 1.00 — statistically real, sub-threshold |
| Protected cells | **PASS** — no regression | small-input medians 1.0436491191232886 (access) / 0.9056450589414343 (flow) ≤ 1.05; cold medians 1.0223618310157438 / 1.0104819181552083 ≤ 1.05 |
| A2 | **not_admitted** | never integrated; absence-verified at I00 |
| F3 | **capacity_only labeling** governs (DECISION.md:11) | walls are capacity evidence, not promotion claims |

**Merge disposition: `do_not_merge` (as delivered).** The candidate is a reviewed, rehearsed local closure on `perf/una-large-e2e`; publication (merge/push/release) requires a separate explicit user instruction. DECISION.md `merge_ready` is **not claimed** because its "required CI passed" leg is unsatisfiable — CI is unavailable (none configured; recorded never passed).

**Production target: not supplied** (`evidence/Q10/actual_target.json`, `47b8f6d3…` — START_HERE.txt:9). All numbers are **installed-proxy** measurements on observed inputs; O3 is never claimed as L4.

## 2. Candidate composition

Selected source `6b53a274e8a8d3e7360ab5e1b38f741107ce3358`, src/ tree `3c59bb0d3f9bc1fba7547be47ec63add88dd3031` — byte-unchanged from selection through tip `992f4e565e1d46381750674b25d356e1abcfa530` (re-derived live 2026-09-30). Baseline→candidate src diff = **exactly 5 engine files, +1383/−20** (R00 source closure, re-derived bit-identically at tip):

| File | Change | Track |
|---|---|---|
| `Engines/Accessibility.py` | +171 | A1 snapshot-preserving search scratch |
| `Engines/AccessibilityWElevation.py` | +169 | A1 (+ A3 private scope init) |
| `Engines/AggregateFlow.py` | +606 | F2 local flow-overlap scratch (+F1/F3 hooks) |
| `Engines/_large_access_scratch.py` | +369 (new) | A1/A3 scratch kernels |
| `Engines/_large_flow_workspace.py` | +88 (new) | F1/F3 flow workspace |

No ODM or Turns engine file appears in any candidate diff (untouched-profile structural basis). `main..tip` additionally shows `_ordered_csr.py` +83 — present in baseline `361928e` but absent from main `98f498e` (provenance delta, campaign-neutral). A2 tokens: zero (I00 absence records). CONTRACT.md frozen; no tolerance relaxation anywhere in the final chain (R00).

## 3. Primary gate (Q10, installed, sustained)

Instrument: paired equal-resource windows, pair = sampling unit; bootstrap = paired fixed-seed **20260929**, 10,000 resamples over log ratios (seed pinned at Q00 freeze). All 40 windows rc=0, `session_status=valid`, `sourced_from_lease=true`; **2,972 validated jobs**; executed harness digest `3b99e202…` == all 40 session.json; 40/40 `installed_qualified=true`.

| Family | n | Pair ratios (blocks 1–5) | Median | CI95 | Verdict |
|---|---|---|---|---|---|
| flow (O3_FLOW, radius 1200) | 5 | 1.4931087845540902 · 1.3420674932729983 · 1.3463815459401314 · 1.3393065665622703 · 1.4593940412477087 | **1.3463815459401314** | [1.3418234340096669, 1.4549372249259942] | **PASS** |
| access (O3_ACCESS, radius 500) | 5 | 1.1841074182176383 · 1.0343748618764228 · 1.0969951715019743 · 1.0473762062400716 · 1.1629111900889264 | **1.0969951715019743** | [1.0518485379517482, 1.1577532365900536] | **FAIL** (median leg; CI leg passes) |

- Every one of the 10 pairs is candidate-favorable; min pair ratio **1.0343748618764228** — no individual slowdown anywhere near the 1.1 investigate line.
- Pooled 10-pair median 1.2617069923899544 (CI95 [1.146633544488414, 1.3445141946252872]) is **descriptive only** — never citable as qualification (R00-F3; gates are per-family per the A1R v2 `C1_gate_granularity` ruling).
- O3_HOLDOUT (same-network generalization): holdout arms ran valid (189 jobs each; warm 110.07 s / 105.16 s); A1's decisive screen benefit (1.2954) concentrates at radius 1200 — hence the **radius-scoped** claim language for A1.
- R00 independently **re-derived all 20 pair ratios from raw session bytes and reproduced both CIs bit-exact** under the frozen seed; preregistration traceability verified (freeze.json admission_gate == instrument GATE_PRIMARY).

## 4. Protected cells

- **Small input (O2-like):** wall-ratio medians — access **1.0436491191232886**, flow **0.9056450589414343**; both ≤ 1.05 → PASS. Recorded risk observation: candidate ~15% slower on small access windows (spread [0.9979, 1.1674]) — recorded, never hidden.
- **Cold process:** medians — access **1.0223618310157438**, flow **1.0104819181552083**; `regression_gt_5pct=false`; **warm-only qualification not required** — the claim shape carries no blanket-default constraint from the cold gate (A1R C2 machinery carried but not engaged).

## 5. Sustained qualification conditions

- **S01** (equal-resource CPU scheduling selection): 38/38 valid windows, receipts 38/38 + 1,014 jobs, per-window budgets 3,246–6,291 MiB, **CLEAN EXIT, key r9 retired** (`evidence/S01/review.json` `a347f6d3…`, verdict verbatim from ruled ec16d3f body).
- **Q10**: 40/40 valid windows, budgets 2,864–5,610 MiB (median 4,390.5), lease charge **3,431.0 s** (windows 3,430.9 s).
- **Wheel route**: single wheel `urban_network_analysis-2.6.0-py3-none-any.whl` sha256 `323c27f3ace8daafcc6d025596a605e8263cc14c8c4fe602c427d51069448ac4` (138,442 B); M00 disposable-frame rebuild **byte-identical** — the wheel main would ship equals the wheel that ran all qualified sessions. Fresh-venv install identity **17/17** module sha256 match; import smoke OK; build constraints `8c501f87…`.
- **Heavy-wall ledger**: 14,400 s budget; S01 close 3,846.8 s (26.7139%, ruled remaining 10,690.3 s); after Q10 lease, remaining **7,259.3 s** — the finite stop was never approached.

## 6. Integration and rehearsal

- main `98f498e5b0ac97b05271d03812fa883b5e4afaba` (tree `e9998400…`) is a **strict ancestor** (merge-base == main); `main..tip` = **107 commits, 1,078 files** (src 6, tests 124, campaigns evidence+history 857, campaigns packet/docs 38, benchmarks 31, other 22).
- **M00 disposable rehearsal PASSED AND ADMITTED** (`evidence/M00/rehearsal.json` v2, `d5d2c7fa37d74c1b…` at 992f4e5; h04 APPROVED + supplemental ack fixing the admitted M00 generation as d5d2c7fa@992f4e5, superseding v1 `6e981422`@10dca11 — v1 suite/full-repo legs withdrawn as frame-invalid, frame-free legs carried): FF `98f498e→07fb3cc` **0 conflicts**; no-ff probe merge `afa3c8ee` tree-diff vs candidate **EMPTY**; wheel byte-identical; install 17/17; main untouched before/during/after.
- **Tip drift probes** (read-only `git merge-tree --write-tree`, this session): `main←07fb3cc` → `e2b6fda9…` == closure tree; `main←992f4e5` → `2c960120…` == tip tree; both rc=0, zero conflicts. The tip differs from the rehearsed closure by **2 evidence-only commits** (+131 lines under `evidence/M00/`); src/ byte-unchanged.
- **R00 final audit: APPROVED QUALIFIED SCOPE** — "flow family + protected cells + untouched profiles approved as qualified; access family approved only as preregistered improvement (not throughput-qualified)"; no no-safe-merge condition found; unavailable evidence nowhere marked passed.

## 7. Known limitations (full carried set — all recorded, none passed)

1. **R00-F2 filename deltas**: A2I declared `proof.md` → delivered `result.json`; A3I declared `result.json` → delivered `proof.md` (non-required stages, substance complete; recorded in `configuration.json`, no renames of sha-bound files).
2. **R00-F3 claim boundary**: flow qualified both legs; access **not** throughput-qualified (improvement_only / radius-scoped language only); pooled figure never cited as qualification.
3. **M00-F1 erratum** (carried, committed evidence not rewritten): v1 rehearsal recorded `main..07fb3cc` as "104 commits, 1,073 files, +719,973/−158" — that describes `main..53d70e1`; corrected **105 / 1,076 / +720,115 / −158** (live-recomputed, exact; delta = the R00 record commit, 3 files +142).
4. **M00-F2 mixed frame** (resolved-by-record in the admitted v2): suite instruments hardcode primary-worktree paths — self-verified census 2026-09-30: 39 full-literal occurrences across 33 files (32 under tests/: 26 .py + 6 frozen-data manifests/golden hashes; + `compare_arms.py`) plus `w00_run.py:41` / `s01_run.py:37` via `REPO = f"{CAMPAIGN}/wt-large-e2e"` — 35 instrument files total; none in the wheel/install route. Suite runs from another frame import primary src and flush primary evidence. Designed deviation: primary-frame correctness suite + disposable-frame tree/wheel/install checks. Post-campaign hardening: WT-from-`__file__` for code sites; frozen-data sites (recorded identity in sha-referenced custody) need explicit exemption or regeneration, not a mechanical rewrite.
5. **M00-F3 + incident** (mechanism resolved-by-record): F1/F3 sessionfinish flushes committed `test_artifacts.json` with no path override (F2's `UNA_F2_ARTIFACTS` at `harness_f2.py:56` defaults to the committed path); v1 rehearsal-frame runs were frame-invalid and the coordinator's restore crossed h04's RETAIN instruction — transient generations lost (loss record `e06a8db0…`, 3,810 B; zero APFS snapshots; **no reviewed or ruled bytes lost**); v2 clean rerun with mechanical in-script retain→restore (interims `cceb3aa4`/`2993ff68`/`59c0981e`). Precision: the rerun's pre-run porcelain gate is script-internal (console shows POST-RUN-CLEAN + interim retention); the claim rests on the script assert plus reviewer clean-porcelain observations.
6. **3 candidate-tip check failures** (h04 dispositions — known-limitations, no fix-forward): `t7_targetoptions_census`, `t8_source_diff_single_decorator_line` (stale F1-era source-audit pins vs screened F2/F3 integration — t8 pinned 2 lines, actual 606); `test_diagnostic_mode_runs_source_and_labels_records` (NUMBA_DISABLE_JIT=1 × A1 `@overload` pass-stub → TypeError at `AccessibilityWElevation.py:351`; outside the qualified measurement frame; post-campaign remediation noted). All other suites green in the clean primary-frame rerun.
7. **Unavailable gates** (recorded, never passed): production target (not supplied); CI (none configured); second observed morphology; turns/ODM real-engine qualification (`not_run` with reason); disposable-frame full-suite execution; full-repo single-invocation pytest leg (withdrawn with the incident as unadjudicated frame-suspect data).

## 8. Memory-lifetime table

Policy (control.json): per-window budget = min(10,650 MiB, 0.80 × available-at-admission), re-measured at **every** heavy-lease admission; pressure stop = max(1 GiB, 0.10×16 GiB) = 1,717,986,918 B; H00 ambient (82.4% RAM used, 2.83 GiB available) is why admissions re-measure.

| Stage | Windows | Valid | Budget range (MiB) | Peak/charge | Custody |
|---|---|---|---|---|---|
| S01 sustained scheduling | 38 | 38/38 | 3,246 – 6,291 | CLEAN EXIT; charge 3,846.8 s | `evidence/S01/review.json` `a347f6d3…` + bound configs `ef670f21…` (27,660 B) / memory `e90bf330…` (3,753 B) |
| Q10 confirmation (pair/holdout/smi/cold/preflight) | 40 | 40/40 | 2,864 – 5,610 (median 4,390.5) | charge 3,431.0 s; zero admission refusals | `evidence/Q10/summary.json` `438bc69f…` (per-window `budget_mib` in `windows[].extra`) |
| Earlier R-screen stages | — | all conformant | per-stage | — | stage `evidence/*R/` records; arm wheels/venvs under `campaign_data/` |

Per-window lifetime detail (admission bytes, budget, child exit) lives in the stage session records under `campaign_data/q10_run/` and the S01 consoles — indexed by `evidence/Q10/raw_index.json` (`25ec966e…`).

## 9. Required-final-file index (DECISION.md:36)

| Required item | Delivered as |
|---|---|
| RESULTS.md | this file |
| decision.json | `evidence/final/decision.json` |
| merge_manifest.json | `evidence/final/merge_manifest.json` |
| candidate_register.json | `evidence/final/candidate_register.json` |
| configuration.json | `evidence/final/configuration.json` |
| commands.jsonl | `evidence/final/commands.jsonl` |
| raw run index with hashes | `evidence/Q10/raw_index.json` (`25ec966e…`) + stage raw records |
| environment manifest | `configuration.json` §environment (+ campaign_data/wheels.json, freeze-pinned) |
| workload manifests | `evidence/H01/workloads.json` + `evidence/Q00/q10_O3_*.run.manifest.json` |
| wheel manifest | `campaign_data/wheels.json` (freeze pin `c443bca9…`, R00 re-verified) + merge_manifest wheel_hashes |
| source manifest | SOURCE_AUDIT.md (audit anchor `361928e`, root tree `089d2920…`, src tree `a5883ddd…`) |
| memory-lifetime table | this file §8 + stage records |
| review report | `evidence/R00/review.json` + `claim_audit.json` (+ all stage reviews; h04-authored, independence verified) |
| unavailable-gates list | decision.json §unavailable + this file §7(7) |

## 10. Publication boundary

Delivered state: reviewed local commits `07fb3cc` → `992f4e5` on `perf/una-large-e2e`; clean FF-equivalent merge geometry into current main (probes above). **No push, no merge, no PR, no release has been performed or authorized.** Any publication is a separate explicit user act; if main advances first, record drift and review a new merge rehearsal (OPERATIONS.md:24). The `campaigns/history/` archive is preserved unchanged.
