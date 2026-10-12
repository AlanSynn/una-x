# P03A: remove the Python origin loop without changing the fold

## Why this comes before per-metric splitting
P00 counts 32*14751 = 472032 folds. Current replay_trace is Python and allocates/checks/calls
Numba once per origin. replay_s is therefore not a measurement of math-kernel time alone.
First remove dispatch/boxing overhead. Reuse the existing full fold and all scalar bindings.

## Proposed seam and phases
Keep replay_trace as a validating Python facade; introduce a PRIVATE compiled whole-row driver
in the same engine-owned scope (or one focused private module). Add no public API/selector.

Phase A: original type/schema/completeness/cutoff/destination weight/domain checks; verify bound
content digest and layout. Keep required corruption checks on every public replay initially.
Phase B: compiled structural scan of all origin offsets and ascending ID slices, with checked
integer bounds. A corrupt later slice must not expose partially computed earlier outputs.
Phase C: allocate fresh result arrays with trace.reach_dtype and trace.metric_dtype; call a
compiled serial origin loop once, using precisely the original full compiled fold.

```
for origin in range(origin_count):
    lo, hi = indptr[origin], indptr[origin+1]
    cand_dist = fresh writable C-order copy(dist_all[lo:hi])
    cand_wts = fresh writable C-order gather(d_weights, ids_all[lo:hi])
    outputs[origin] = ORIGINAL_FULL_FOLD(exact original arguments)
```

The initial driver is serial/nogil in each W4/T1 worker. Do not introduce nested prange or
increase total CPU as a claimed algorithmic gain. If independent-origin prange is later tested,
use exclusive scratch per lane, preserve fold internals, and budget native pools plus parent.
The fold count must still equal all requested origins, including empty retained slices.

## Arithmetic and compiler obligations
Preserve cutoff scalar type, raw weight and KNN dtype, contiguity, writable layout, order of
gathered IDs, first/second min behavior already captured in trace, output casts and argsort ties.
Do not change reach int output into float as an optimization. No tree reduction, reassociation,
fast-math expansion, nonfinite waiver, prefix-weight pre-sum, or full-metric pruning here.
Calling the old function from a compiled caller may still change inlining/specialization and
rounding. Inspect actual signatures/LLVM where useful; compare bits against BOTH current replay
and original integrated engine. The profile's compiled reference is the authority, not this
pseudocode. If outer compilation changes a result, repair the compilation boundary or refuse that
signature. Do not make a legacy allclose exception.

## Validation and error ordering
Use input/trace validation already promised. Moving an integrity check ahead of computation is
allowed only if all pre-check work was private and emitted no required warnings/errors or public
state. Otherwise preserve the observable sequence. The compiled driver returns a typed refusal
or raises through the declared facade; it cannot silently fall back after partial public writes.
Keep validation of forged offsets, negative/out-of-range IDs, wrong lengths, byte order and dtype
outside unsafe kernels. Integer offset products are checked before allocation/indexing.

The raw trace is immutable, but a Python ndarray writeable flag alone is not a security boundary.
Do not skip digest/identity verification across calls merely because the object address repeats.
Later lease-scoped validation elision requires a sealed ownership proof and separate measurement.

## Tests and evidence
Lengths 0/1/2 and SIMD boundaries; all empty and all kept; tied distances with unequal weights;
float32/int/float64 weight refusal/accepted specialization; +/-0; cutoff nextafter; malformed
trace at first/middle/last origin; unsupported layout; all decay branches; output mutation; source
and installed wheel. Debug traces retain first differing origin/destination, signatures and bits.
Run parity before timing. Compare current Python replay vs compiled driver at matched resources,
then the PUBLIC full batch. Record dispatch count, copies/bytes, per-origin count and compiler
warm-up separately. A replay-only win is not the 10x application claim.
