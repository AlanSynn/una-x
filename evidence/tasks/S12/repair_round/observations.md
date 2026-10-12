# S12 review repair round (rank closure)

Reviewer: independent session, detached worktree s12-review-r1, verdict
ACCEPT_WITH_FINDINGS (review artifacts in ../: sparse_review.json,
sparse_L2.json, SHA256S.txt 26/26 verified in this tree, raw/ 24 files).

## Defects (all MINOR, admission-gate class, private module reachability)

The reviewer's counterexamples are in ../raw/hostile_fix_results.json
(H19 rows, input sha256s recorded there).  Classification probe (script:
_s12_rank_probe.py, preserved here; baseline first, then sparse):

- F1 `_sparse_access_scratch.py` gate: C-contiguous float64 (D,1)
  `d_weights` passed the gate, then the kernel failed numba typing at
  `cand_wts[j] = d_weights[d]` while the BASELINE RAN (broadcasting):
  gate-admit-then-raise on an input the baseline accepts.
- F2 `:176`: 0-d `d_weights` (np.array(0.5)) made the GATE raise
  IndexError at `.shape[0]`; the baseline raises TypingError.
- F3 `:174`: 1-D `d_terminal_weights` — same IndexError-vs-TypingError
  class.
- Same-law classes probed before patching: 1-D `d_terminal_idxs`,
  1-D `o_terminal_weights`, 1-D `o_terminal_idxs` were ALREADY
  parity-exact (engine gates raise TypingError both sides, fallback
  reproduces it) — pinned as regressions anyway.

## Repair (this round, one commit: kernel gate + regression tests)

`_sp_route_admits` now refuses every degenerate rank with reason
"shape" BEFORE any shape-element access (ndim checks only — the gate
still raises nothing), so the unchanged original-driver fallback carries
every such input: bitwise parity where the baseline runs (F1),
exception parity where it raises (F2/F3).  The block's identifiers were
spliced from the function signature via AST (_s12_fix_rank_block.py) —
a first hand-typed edit produced a NameError that the suite caught
(42 failed) and the AST rewrite replaced.

## Evidence

- rank_probe_postrepair.log — all six classes PARITY-OK post-repair
  (dwt_2d: refuse route compact_fallback / reason shape / bitwise
  parity; dwt_0d, dw_1d, d_1d, ow_1d, o_1d: both raise TypingError).
- suite_runB_postrepair.log — 74 passed / 1 skipped (68 pre-repair +
  6 new rank pins; first post-repair run had the same 74/1, second
  confirming run here).  Both runs twice-run discipline.
- Test pins: tests/tenx/sparse/test_sparse_refusal.py mutators
  dweights_2d_column (fallback/shape/bitwise), dweights_0d,
  dterminal_costs_1d, dterminal_ids_1d, oseed_weights_1d, oterms_1d
  (exception parity).

The S11 closure records (evidence/tasks/S11, commit c8b19f2) keep the
PRE-repair module sha d7b25ed5…b117 — historical, not rewritten.
Consumers (C11/C12) must use the post-repair sha recorded in the run
root obligations addendum.
