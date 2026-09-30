# F1I-N1 IR side-by-side (in-module, observed signature)

- B0 IR: `ir_b0_accumulate_od_flow.ll` (sha256 `328bb1d7111e6c95…`)
- candidate IR: `ir_cand_accumulate_od_flow.ll` (sha256 `072185a7d5471efa…`)
- FP sites: 92 in BOTH arms, identical order, identical fast flags (normalized site sequence hash `f443ec3ae89dfd5c…` both arms)
- fma/fmuladd/llvm.fma sites: 0 in both arms
- GIL handling: B0 0/0 save/restore; candidate 2/2 (PyEval_SaveThread / PyEval_RestoreThread around the compiled body)
- Full-IR residual: 168 mechanical pairs (SSA renumbering / label numbering / module-name length prefix only), 4 GIL lines (2 declares + 2 calls), 0 non-mechanical deltas
- Verdict: **PASS** — the ONLY semantic IR delta is the GIL save/restore pair; no new FP site, no reassociation, no fast-flag change (proof 5b; any other delta = rejection)
