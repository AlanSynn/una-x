# F1: flow concurrency without changing arithmetic stripes

## Source and hypothesis
AggregateFlow._process_origins_aggregate and the turn-aware driver use a ThreadPoolExecutor and per-stripe output buffers. _accumulate_od_flow and _accumulate_od_flow_turns are njit(cache=True,fastmath=True) without nogil=True. A Python thread calling an njit function does not automatically release GIL. This suggests serialization, but measure actual compiled options and concurrent service first. Official reference: https://numba.readthedocs.io/en/stable/user/jit.html#nogil.

## Before implementation
Record Python implementation/GIL mode, Numba version/targetoptions/signatures, relevant compiled callees, per-kernel service share and a controlled concurrent independent-buffer probe. An extension may release GIL elsewhere; time spent in SciPy is not proof the Numba loader overlaps. If loader coverage is immaterial or execution already overlaps, stop not_admitted.

## Minimal candidate
Add nogil=True only to admitted flow-loading entry kernels whose complete read/write closure has been audited. Do not change fastmath, code arithmetic, helper implementations, arrays, stripe count, origin/destination ordering, graph or final reduction. Warm compile each actual specialization before concurrent execution. Avoid new JIT compilation races by deliberate non-performance warm-up in both arms, recording startup cost.

Read-only shared closure must include CSR arrays, gradients, geometry/node data, Settings-derived scalars and helper globals. Mutable closure must be stripe-owned AB/BA/node outputs, origin labels/predecessors, dd/pd scatter buffers, and all loader scratch. Check np.shares_memory or equivalent on actual arrays in tests. Empty shared node-output array is safe only because shape==0 prevents writes. Nothing may call Python/GDAL/loggers while GIL is released unless the compiled mechanism safely reacquires it; inspect native signatures and no object mode.

## Numerical partition invariant
Freeze K=baseline n_threads as a numerical dependency. Stripe s executes origins s,s+K,... sequentially; each origin visits destinations in original index order; loading updates remain ordered; final reduction adds stripe 0,1,...,K-1. Releasing GIL allows different stripes to overlap but does not change any stripe's arithmetic sequence. This is the proof route to exactness.

Changing K changes floating-point grouping. Do NOT compare baseline K=1 versus candidate K=8 as an exact scheduling improvement. Default-profile tests retain the baseline default K, which may exceed the campaign's chosen CPU slots. If it cannot fit, either mark default-profile timing unavailable or define an explicit K profile before timing and run it identically in both arms. A separate optional physical H<K scheduler must assign WHOLE fixed stripes and keep all K partials, then reduce by stripe index; it is not part of the minimal decorator experiment and needs its own proof/review within S01.

Progress logs are protected by the existing lock. Verify event counts/order and non-timing content; rate/ETA fields follow preregistered timing normalization. Exceptions still propagate and outstanding stripe work terminates by the original executor context semantics. Do not race logger state or add a shared scratch array.

## Tests
Run identical fixed-K workloads repeatedly and compare each stripe partial array, global AB/BA/node/undirected flow, delivered volume/cap/warnings and all artifacts. Exercise K=1 and at least two multistripe profiles as SEPARATE reference pairs. Include node flow disabled, zero-weight origins, unreachable destinations, obstacle/elevation cases, and small turn-aware path if modified. Existing K-alternatives code stays unchanged. Concurrent calls with independent engines must not share mutable state.

## Measurement and stop
Measure complete RunFlow with automatic cap and all outputs, separate origin-loop/loader diagnostics, total CPU use and actual overlapping kernels, startup/JIT and whole-process memory. The GIL may not dominate, memory bandwidth may saturate or Python OD orchestration may remain serial. If full installed job/throughput gain does not clear gates, reject; do not attribute a kernel-only ratio to application. Keep fixed logical K in every headline comparison. No hardware/GPU backend change.
