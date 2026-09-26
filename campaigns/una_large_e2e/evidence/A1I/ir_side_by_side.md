# A1 IR side-by-side (proof.md section 7.3)

Candidate search scratch kernel `_a1_scope_search` vs B0 kernel
`compact_vector_node_view_scope`, compiled at the observed kernel
signature, full LLVM IR dumped for reviewer inspection. Per the proof
review note, this evidence is corroborating only: the load-bearing
equivalence argument is the byte-equality battery, not IR inspection.

## Provenance

- Captured by `capture_ir.py` (this directory) into `ir_b0_compact_vector_node_view_scope.ll` and `ir_cand_a1_scope_search.ll` (full unfiltered `Dispatcher.inspect_llvm` output).
- Environment: numba 0.67.0, llvmlite 0.49.0, Python 3.11.16, arm64 (darwin, Apple Silicon), `NUMBA_NUM_THREADS=2`, fresh `NUMBA_CACHE_DIR`, campaign venv.
- Decorator flags identical for both kernels: `parallel=False, cache=True, nogil=True, fastmath=True` (B0's module constants; the helper module pins the same values).
- Compiled signature (both kernels):
  `(int64[:] o_terminal_idxs, float64[:] o_terminal_weights, int64[:] adjacency_pointer, int64[:] adjacency_vector, float64[:] adjacency_vector_weights, boolean[:] adjacynct_vector_network_node, float64 cutoff, int64 d_count)`; the candidate takes two additional scratch arguments `(int64[:] eligible_offset, float64[:] eligible_weight)`. This is the observed O2 typing (int64/float64/bool CSR, float64 cutoff).
- Source SHA-256 at capture:
  - candidate `_large_access_scratch.py` `2d3b77363a5d42050024e335b5e2bedf8c8088a312d6e163666596ce42894e23`
  - candidate `AccessibilityWElevation.py` `83ed0bfdd0e2b0b7d413dc39a2e84899a1b145f6980ddd9a30243f6e76944a9c`
  - candidate `Accessibility.py` `5fec7ad5bf00b6b033747ae41ae2a7eec57f53748395ee7b5c0836536b805f75`
  - B0 `wt-b0/src/.../AccessibilityWElevation.py` `3471fdf9744d077f59e495ab8fd9d8a1ea14afbea6f409c430d2a4b44678c3e4`
  - B0 `wt-b0/src/.../Accessibility.py` `ea84560797989ab60f77315201308a8dc691abe49a04f8731986d1521ed80c5c`

## Floating-point instruction census (all sites, from both .ll files)

Extracted verbatim into `ir_fp_census.txt`. Counts:

| kernel | fadd | fmul | fsub | fdiv | fmuladd / llvm.fma | fcmp |
|---|---|---|---|---|---|---|
| B0 `compact_vector_node_view_scope` | 14 | 0 | 0 | 0 | **0** | 10 |
| candidate `_a1_scope_search` | 8 | 0 | 0 | 0 | **0** | 8 |

Every FP operation in both kernels carries only the `fast` fast-math
flag (both modules set `fastmath=True`), i.e. identical flag sets. The
fadd count differs because LLVM unrolls/vectorizes the two loop shapes
differently; every site is a SINGLE add (see regions below) — there is
no fadd-of-fadd chain in either kernel, hence no reassociation site.

## Region 1 — per-incidence cost add (`edge_weight + popped_weight`)

B0 (`ir_b0_compact_vector_node_view_scope.ll` around L645-660; the
`nonzero` mask loop unrolled by LLVM; `%.841.fca.0.load` is the popped
heap weight, `%.1235` the loaded incidence weight):

```llvm
%.1233 = getelementptr i8, ptr %.1061, i64 %.1231
%.1235 = load double, ptr %.1233, align 8
%.6.i181 = fadd fast double %.1235, %.841.fca.0.load
...
%.1235.1 = load double, ptr %.1233.1, align 8
%.6.i181.1 = fadd fast double %.1235.1, %.841.fca.0.load
```

Candidate (`ir_cand_a1_scope_search.ll` L159-190, phase-one eligibility
scan; `%.869.fca.0.load` is the popped heap weight, `%.1114` the loaded
incidence weight):

```llvm
%.1114 = load double, ptr %.1112, align 8
%.1116 = fadd fast double %.1114, %.869.fca.0.load
%.1159 = fcmp fast ugt double %.1116, %arg.cutoff
br i1 %.1159, label %B626, label %B566
...
%.1194 = load double, ptr %.1193, align 8
%.1198 = fcmp fast olt double %.1116, %.1194
br i1 %.1198, label %B590, label %B626
```

Same single `fadd fast` of the same two operands. The eligibility
conjunction compiles to the same two comparisons: B0 masks
`(cost <= cutoff) & (cost < S0[nbr])` (`%.7.i = fcmp fast ole ...
%arg.cutoff` + `%.8.i = fcmp fast olt ... <label>` + `and`); the
candidate short-circuits (`not (cost > cutoff)` as `fcmp fast ugt` with
inverted branch, then `fcmp fast olt` vs the snapshot label). Both read
the snapshot label BEFORE any write of this row — that ordering is the
snapshot property itself, and it is what phase two replays.

## Region 2 — sentinel materialization (`np.ones(nd) + cutoff`)

Identical form on both sides — scalar prologue plus an LLVM-vectorized
broadcast loop:

```llvm
# B0 L339, scalar prologue
%8 = fadd fast double %7, %arg.cutoff
# B0 L399-402, vector loop
%16 = fadd fast <2 x double> %wide.load, %broadcast.splat362
...
# candidate L356, scalar prologue
%11 = fadd fast double %10, %arg.cutoff
# candidate L416-419, vector loop
%19 = fadd fast <2 x double> %wide.load, %broadcast.splat97
...
```

## Region 3 — heap ordering (inlined numba heapq)

Both kernels contain the identical inlined sift-down comparison pairs
(same value names, both `fast`):

```llvm
%.119.i = fcmp fast une double %.53.unpack.i, %.106.unpack.i
%.121.i = fcmp fast olt double %.53.unpack.i, %.106.unpack.i
```

## Reviewer checklist (proof section 7.3 obligations)

1. **No FMA contraction**: `grep -E 'fmuladd|llvm\.fma'` over both .ll
   files returns nothing. The candidate adds no fused-multiply-add
   site; nor does B0.
2. **No reassociation site**: the cost expression is one `fadd` of two
   loaded operands in both kernels; no chained additions exist that
   `fast` reassociation could re-parenthesize differently between the
   arms. The `fast` flag SET is identical on every site in both.
3. **Same comparisons in the same order**: cutoff test, snapshot-label
   test, heap-order pair — all present in both with `fast` flags;
   branch orientation differs (mask-and vs short-circuit) as designed
   and does not affect the computed values.
4. **Phase two adds no arithmetic**: the stored-weight replay contains
   no FP ops at all (the census shows no fadd outside the prologue,
   phase-one scan, and inlined heapq shared with B0).

Captured values are facts from the probes, recorded verbatim; the
byte-equality battery (118 tests) remains the admission basis.
