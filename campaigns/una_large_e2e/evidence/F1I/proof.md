# F1 proof: flow nogil at fixed arithmetic stripes (pre-implementation, for independent review)

Task F1I. Author: implementation-owner. Source identity at proof time:
worktree `/Users/alansynn/orca/workspaces/una-x/wt-large-e2e`, branch
`perf/una-large-e2e`, HEAD `25aef2dca9a7840e3f5a7c33d2784b3609bdc042`,
`git rev-parse HEAD:src` =
`b121ec0b4312f12caa98bc115243652e1564d3ec` (control.json selected
source, A1 screened in), working tree clean.
`src/urban_network_analysis/Engines/AggregateFlow.py` blob is
`0114ce8f318a3247b62e3d468e32d54e692737a2` at BOTH B0 (`361928e`) and
HEAD — the file this task changes is byte-identical to the source H05
profiled. Admission recheck: `admission.json` (all H05 F1 facts
re-verified at this source with line citations; the 1,080,036 B
per-call temporaries figure is reproduced exactly from the code).

Everything below is written against the ACTUAL source lines cited, not
a mental model. Dossier: `dossiers/F1_fixed_stripe_nogil.md`.
Reviewer approval of THIS document is required before implementation
starts (hard stop; implementation begins only after the coordinator
relays independent approval).

---

## 0. Scope

Writable: `src/urban_network_analysis/Engines/AggregateFlow.py`,
new `src/urban_network_analysis/Engines/_large_flow_*.py` (reserved;
the minimal candidate needs NONE),
`tests/large_e2e/F1/**`, `campaigns/una_large_e2e/evidence/F1I/**`.
Read-only: everything else — `Accessibility*.py`, `_large_access_scratch.py`
(A1's implementation; studied as house style, not touched),
`_ordered_csr.py`, `_betweenness_numba.py`, `Flow.py`, `UNA.py`,
`Settings.py`, the oracle package `tests/large_e2e/oracle/**`
(imported, never changed), the benchmark harness, evidence of other
tasks. NO git writes.

The candidate change surface is exactly ONE decorator kwarg:
`AggregateFlow.py:192` gains `nogil=True`:

```python
@nb.njit(cache=True, fastmath=True, nogil=True)   # was @nb.njit(cache=True, fastmath=True)
def _accumulate_od_flow(...):
```

No other byte of `src/` changes. In particular: the driver bodies of
`_process_origins_aggregate` (966-1178) and
`_process_origins_aggregate_turns` (1276-1461), the kernel body
(193-386), the helpers `_find_arc` (165-172) and `_decay` (175-189),
the turns kernel `_accumulate_od_flow_turns` (1672-1834), the
line-graph builder `_build_line_csr_turns` (1589-1669), and
`_compute_trip_volumes` (1468-1544) are untouched. Post-change
verifiable by `git diff` (one line) — the reviewer's primary evidence.

The change makes the ALREADY-EXISTING fixed arithmetic stripes
(`AggregateFlow.py:1044` `range(slot, n_origins, n_threads)`, slot
private buffers 1017-1025/1041-1042, slot-order reduction 1156-1160)
actually execute concurrently instead of GIL-serializing inside the
kernel. It changes NO stripe membership, NO stripe count K, NO
per-stripe arithmetic, NO reduction order. The EXACT ASSOCIATION of
the reduction is an exactness obligation, discharged by the induction
in section 5.

## 1. Baseline transition (actual code)

Dispatch (`Centrality`, `AggregateFlow.py:785-827`): `if ns["use_turns"]:`
→ turn-aware pipeline (line 811 `_process_origins_aggregate_turns`);
`else:` → node-graph pipeline (line 827
`_process_origins_aggregate`). The two pipelines are disjoint from
this point down.

Node-graph origin driver (`_process_origins_aggregate`, 966-1178):

* `n_threads = max(1, int(self.num_threads))` (1008);
  `self.num_threads = topology.num_threads` (`Base.py:125`), default
  `mp.cpu_count()-1` (`Topology.py:79`) = 9 on this laptop — matching
  the H05 probe's n_threads=9 and engine-level max_concurrent=9.
* Per-stripe private buffers: `local_AB`, `local_BA`, `local_node`
  lists indexed by slot (1017-1025); when node flow is disabled all
  slots alias ONE shared shape-0 array `_empty_node` (1022-1023).
* `_process_stripe(slot)` (1032): allocates slot-private
  `dd_buf = np.full(n_total_nodes, np.inf)` and
  `pd_buf = np.full(n_total_nodes, -9999)` (1041-1042); iterates
  `for o_pos in range(slot, n_origins, n_threads)` (1044) — stripe s
  owns origins s, s+K, s+2K, ... in ascending order. Per origin:
  scipy forward Dijkstra `d_o, pred_o` (1064-1068, fresh arrays),
  `d_shortest_arr = d_o[dest_node_ids]` (1070),
  `_compute_trip_volumes` (1072-1077), then
  `for d_idx in range(n_dest)` ascending (1079) with skip conditions
  `trip_vol <= 0.0 or not np.isfinite(d_shortest)` (1082) and
  `d_shortest > radius` (1084); per qualifying OD: scatter the
  destination's sparse gradient columns into dd_buf/pd_buf
  (1093-1096), call `_accumulate_od_flow` ONCE (1098-1109 — its only
  production call site), reset ONLY the touched columns
  (1112-1113), track gap statistics per slot (1115-1119).
* Progress: `n_done[0] += 1` and the periodic log line inside
  `with progress_lock:` (1123-1136). All logging in the stripe body
  happens under this lock.
* Execution: `n_threads == 1` → direct `_process_stripe(0)` (1144-1146);
  else `ThreadPoolExecutor(max_workers=n_threads)`, one future per
  slot, `f.result()` re-raises worker exceptions (1148-1153). The
  `with` block joins all stripes before the reduction.
* Final reduction (1156-1160), main thread, slot order 0..K-1:
  `self.edge_flow_AB += local_AB[slot]`,
  `self.edge_flow_BA += local_BA[slot]`,
  `self.node_flow += local_node[slot]`. Then
  `self.edge_flow = self.edge_flow_AB + self.edge_flow_BA` (830),
  `self.has_flow_results = True` (831), observer flows (834).

Kernel `_accumulate_od_flow` (192-386), njit fastmath no-nogil.
Preconditions/refusals/early exits — the COMPLETE set:
  * line 233-234: `if not reach[dest_virtual_node] or not
    reach[origin_virtual_node]: return 0.0`;
  * line 300-301: `if q_sum <= 0.0: return 0.0`.
  There are no other early exits, no exceptions raised in-kernel, no
  Python fallback (njit is nopython-only). Numba boundscheck is OFF as
  in B0. Two fixed full-V' scans (226-231 reach; 238-241 gather);
  per-call temporaries `reach` (224), `cont_o`/`cont_d` (252-253),
  `acc_o`/`acc_d` (308-309) = exactly 3*V'*1 + 2*V'*8 = 1,080,036 B at
  V'=56844 (H05 figure reproduced exactly; `admission.json`);
  off-envelope `acc_o[pv]`/`acc_d[pv]` writes (358, 377); +inf
  sentinels from the caller's `dd_buf` (1041) tested at 229; node-flow
  writes guarded by `out_node_flow.shape[0] > 0` (380) — the shared
  shape-0 `_empty_node` can never be written; returns
  `acc_d[dest_virtual_node]` (386).

## 2. Candidate transition

The single kwarg `nogil=True` on the decorator of
`_accumulate_od_flow`. Effect (numba documented semantics, dossier's
official reference): the compiled call is entered after
`PyEval_SaveThread` and left via `PyEval_RestoreThread`; the GIL is
released for the duration of the jitted call, so OTHER Python threads
(stripe workers in the existing executor) run concurrently with it.
Everything else is unchanged:

* D1. No new dispatch, no new guard, no new branch anywhere. Which
  inputs/configs take which path is byte-for-byte B0's routing:
  turns=False + n_threads>1 → executor stripes (1148); turns=False +
  n_threads==1 → fast path (1144); turns=True → turns pipeline
  (811). The "unchanged fallback" obligation is satisfied in the
  strongest form: every path IS B0's code, and no input that ran on
  any path runs anywhere else.
* D2. fastmath stays True; cache stays True. `nogil=True` adds no
  typing restriction for a function that already compiles nopython
  (njit is nopython-only; the GIL requirement is the ONLY thing nogil
  forbids, and the kernel uses no Python objects, no print, no
  reflection of non-arrays). Discharged empirically anyway: the
  candidate must compile and byte-match EVERY specialization in the
  battery, including the int32-CSR observed specialization captured
  in the O2 replay fixtures (T6/T7). A signature that compiled in B0
  but fails to compile under nogil is a compiler-equivalence failure
  = REJECTION, not a fallback.
* D3. Warm-up: unchanged engine code does no warm-up (as B0). The
  dossier's "warm compile each actual specialization before
  concurrent execution / record startup cost" is a harness/test
  obligation: F1 tests warm-compile every specialization serially
  before any concurrent run and record the warm-up wall time;
  qualification harness warm-up is HARNESS.md req 4 (already
  binding on the benchmark, unchanged).
* D4. K and stripe membership are frozen numerical identity
  (HARNESS.md req 3 "frozen flow stripe count is numerical identity";
  CONTRACT "Flow partition": K, per-stripe operations and final order
  are numerical dependencies). F1 does NOT implement any H<K physical
  scheduler — CONTRACT: "This scheduling extension is not required
  for F1". No work stealing, no atomics, no tree reduction, anywhere.
* D5. Screens: NONE in F1I. Performance screening belongs to F1R
  (≤3 paired blocks/revision per control_addendum_A1R). F1I delivers
  exactness evidence only.

## 3. Turns closure (H04-N1) — proof that F1's source closure excludes the turns path

H04-N1 requires, at F1 admission approval: "PROVE its source closure
excludes `_accumulate_od_flow_turns` and the shared flow driver, OR
commission an H03 turns supplement". Proof:

(a) Change surface. The Phase-2 diff touches exactly the decorator at
`AggregateFlow.py:192`. Line 192 is the decorator of
`_accumulate_od_flow` ONLY; `_accumulate_od_flow_turns`'s decorator is
line 1672, a different byte range, untouched.

(b) Call graph, node kernel. `_accumulate_od_flow` has exactly one
production reference: `AggregateFlow.py:1098`, inside the nested
`_process_stripe` of `_process_origins_aggregate` (1032). That
closure is invoked at 1146 (n_threads==1) and 1149-1153 (executor),
both inside `_process_origins_aggregate` (966), whose only caller is
`Centrality` line 827 — inside the `else` of `if ns["use_turns"]:` at
785, i.e. executed only when turns=False. A repo-wide grep confirms
no other reference in `src/`, `tests/`, or `benchmarks/` except
read-only oracle/instrument consumers that take the function OBJECT
(oracle traces/replay/compiler profile; h05 instrument wrapper) —
none is a call path into the engine.

(c) Call graph, turns kernel. `_accumulate_od_flow_turns` (1673) has
exactly one reference: line 1387, inside `_process_stripe` of
`_process_origins_aggregate_turns` (1337), invoked at 1431/1433-1437,
called only at line 811 (turns=True branch). The turns pipeline
(`_build_line_graph_turns` 1185-1232,
`_precompute_dest_gradients_turns` 1234-1274,
`_process_origins_aggregate_turns` 1276-1461,
`_build_line_csr_turns` 1589-1669, `_accumulate_od_flow_turns`
1672-1834) contains ZERO bytes inside F1's change surface.

(d) Recompilation closure. Numba compiles and caches per function
object, keyed by bytecode + targetoptions + signature types. The
turns kernel's bytecode is unchanged and its targetoptions
(`@nb.njit(cache=True, fastmath=True)`, line 1672) are unchanged, so
its specializations and on-disk cache entries are untouched by the
candidate — it is not merely semantically unaffected, it is not even
recompiled.

(e) Shared njit helpers. `_decay` (175-189) and `_find_arc` (165-172)
are `@nb.njit(cache=True, inline='always')`, unchanged. Both are
inlined into each caller's IR (`inline='always'`), so the node
kernel's compiled body embeds its own copies and the turns kernel
its own; changing one caller's flags cannot alter the other caller's
compiled code. `_find_arc` is in fact called ONLY by the node kernel
(262, 273, 350, 369 — the turns kernel is state-indexed and never
looks arcs up). `_build_line_csr_turns` is called only from
`_build_line_graph_turns` (1193). The F1 candidate modifies neither
helper and neither turns-only function.

(f) Shared Python-side code. Both pipelines share, before the branch
at 785: `_prepare_params` (437), `_build_digraph` (556), `_build_csr`
(657), `_gradient_limit*` (894-906), and after it the reduction/
observer/log tail (829-842); module-level `_normalize_method`,
`_cutoff_for_shortest` (144), `_compute_trip_volumes` (1468). NONE of
these bytes is touched by F1. The per-path drivers are NOT shared:
each pipeline has its own `_process_stripe` closure (1032 vs 1337),
its own buffers, its own lock. "The shared flow driver" in H04-N1's
sense — code that a turns run executes and F1 modifies — does not
exist for this change: the intersection of (F1 change surface ∪
recompilation closure of the changed function) with (any code
reachable when `ns["use_turns"]` is true) is EMPTY.

(g) Empirical pin (Phase 2, T8): a small turns=True engine case run
candidate-vs-B0 must be byte-identical end to end, and the turns
kernel's `targetoptions` census must equal B0's — the exclusion is
demonstrated on compiled behavior, not only by reading the diff.

CONCLUSION: F1's closure excludes `_accumulate_od_flow_turns` and all
turns-path code. No H03 turns supplement is required for F1. (The
turns coverage gap itself remains open for the coordinator exactly as
H04-N1 routed it; F2/A2/A3 owners must discharge their own N1
obligations if their closures touch shared code.)

## 4. Shared-closure audit under nogil (read/write sets of the released region)

The GIL is released ONLY inside `_accumulate_od_flow`. Safety requires:
everything the kernel reads is immutable during the parallel region,
and everything it writes is stripe-private.

READ-ONLY during the origin loop (writes all happen before it, in
`Centrality` up to line 827):
* `self._csr_indptr/_csr_indices/_csr_weights/_csr_edge_id/
  _csr_direction` — last written by `_build_csr` (657-699, including
  the obstacle penalty mutation at 675-693). The kernel body contains
  no write to any of its CSR arguments.
* Sparse gradients `g_indptr, g_nodes, g_dist, g_pred` — built by
  `_precompute_dest_gradients` (908-961) before the loop; only read
  (slices at 1093-1096).
* `dest_weights, dest_node_ids, dest_edge_ids` (995-997),
  `origins.node_weight` (read via `float()` at 1045),
  `ns`-derived scalars, `self._first_origin_node`.
All of the above are plain ndarrays that no stripe writes; concurrent
reads of ndarrays without GIL are safe (data is immutable; numpy
refcounts are untouched by in-kernel bare-pointer access).

STRIPE-PRIVATE (single writer = the stripe's thread):
* `local_AB[slot]`, `local_BA[slot]`, `local_node[slot]` (1017-1025);
* `dd_buf`, `pd_buf` — allocated INSIDE `_process_stripe` (1041-1042);
* `d_o`, `pred_o` — fresh scipy arrays per origin (1064-1068);
* every kernel temporary (`reach` 224, `reach_nodes` 236,
  `order_o`/`order_d` 243-244, `cont_o`/`cont_d` 252-253,
  `acc_o`/`acc_d` 308-309) — allocated in-kernel per call;
* `local_n_gap[slot]`, `local_worst[slot]` — each list slot written
  by exactly one thread (1115-1119).
Kernel writes go ONLY to `out_AB`, `out_BA`, `out_node_flow` and its
own temporaries; with node flow disabled `out_node_flow` is the
shared shape-0 array, unwritable (guard at 380) — the dossier's
"safe only because shape==0 prevents writes", preserved exactly.

SHARED MUTABLE, lock-protected (unchanged code): `n_done[0]` and the
progress log lines under `progress_lock` (1028-1029, 1123-1136).
The logger itself is unchanged and is called from worker threads only
under this lock, exactly as in B0.

MAIN-THREAD ONLY: `self.edge_flow_AB/BA`, `self.node_flow` reduction
(1156-1160) runs after the executor `with` block has joined all
stripes; `self.edge_flow`, observers, logs after that.

In-kernel library calls under nogil: `np.zeros/np.empty` (NRT —
numba's NRT allocator is thread-safe; this is the documented nogil
use case, dossier's official reference), `np.argsort` (nopython,
allocates via NRT), `np.exp/np.log` via inlined `_decay` (libm,
no shared mutable state). No Python C-API, no object mode, no
logger/GDAL call inside the jitted body — the dossier's "nothing may
call Python/GDAL/loggers while GIL is released" holds by
construction of the unchanged kernel body. `np.shares_memory` /
aliasing assertions on the actual arrays are pinned in tests (T13).

`_scipy_dijkstra` was ALREADY called concurrently in B0 (it releases
the GIL internally; H05 probe: buffers independent, CSR shared). F1
changes nothing about scipy's behavior; nogil only lets the NUMBA
portion overlap too.

## 5. Reduction-order / association exactness (the induction), and multi-thread determinism

B0's numerical association, per `Centrality` call, for K = n_threads
stripes and element e:

```
AB[e] = ((0 + L_AB[0][e]) + L_AB[1][e]) + ... + L_AB[K-1][e]
```

left-to-right over slot index 0..K-1 (lines 1156-1160), where
`L_AB[s]` is stripe s's private partial array, itself accumulated in
the kernel's fixed instruction order over stripe s's origins
(s, s+K, s+2K, ... ascending, one destination loop ascending per
origin, per-OD kernel calls sequential within the stripe).
B0 and the candidate agree on EVERY step:

Per-stripe claim (induction over stripe s's origin index j):
* Base: before its first origin, stripe s's buffers are freshly
  allocated by identical constructors (`np.zeros` 1017-1025,
  `np.full` 1041-1042) — identical initial state in both arms.
* Step: the j-th origin's computation consumes only (i) the origin's
  own scalars/weights, (ii) `d_o/pred_o` from scipy Dijkstra on the
  read-only forward CSR — a pure function of (CSR, origin, limit),
  proven bit-deterministic across thread counts by the H03 suite and
  by H04's observed-scale B0-vs-B0 double runs, (iii)
  `_compute_trip_volumes` — a pure numpy function of per-origin
  inputs, and (iv) the kernel — a pure single-threaded function of
  (read-only CSR/gradients, stripe-private arrays, per-call scalars).
  No other thread reads or writes any byte of stripe s's state:
  section 4's audit shows the read/write sets of distinct stripes are
  DISJOINT. Therefore the j-th kernel call writes the same values to
  the same cells in the same order regardless of what any other
  stripe is doing, and `dd_buf/pd_buf` scatter/reset (1093-1096,
  1112-1113) is self-contained within the stripe. The arithmetic
  SEQUENCE within a stripe is a property of the (single) executing
  thread alone; the GIL never mediated it, and releasing it cannot
  reorder it. Compiled-code identity (section 9: same IR modulo
  GIL-state calls, same fastmath flags) closes the last gap: the
  kernel's machine arithmetic is the same instruction stream.
* Hence `L_AB[s]`, `L_BA[s]`, `L_node[s]`, `local_n_gap[s]`,
  `local_worst[s]`, and every kernel return (`delivered`) are
  bit-identical between B0 and candidate for every s.

Reduction claim: the reduction loop (1156-1160) is unchanged
main-thread code executed after the executor join, in slot order
0..K-1, with the same `+=` elementwise association; applied to
bit-identical partials it yields bit-identical
`edge_flow_AB/edge_flow_BA/node_flow`, and the unchanged tail (830,
834) yields bit-identical `edge_flow` and observer outputs. Gap
aggregates (1162-1163) are sums/maxes over per-slot values —
deterministic.

Multi-thread determinism (stronger, run-to-run): the SAME argument
with B0 replaced by "candidate run i" shows every candidate run with
the same (inputs, n_origins, K) produces the same partials and hence
the same outputs — bit-identical regardless of thread timing. This is
the fixed-stripe design's whole point and is PINNED by a
repeated-concurrency test (T4): R independent runs at K>1 must be
byte-identical to each other and to the K-matched B0 run.

Logs: the progress counter takes each value 1..n_origins exactly once
under the lock, so the emitted "origin k/n" content sequence is
deterministic; only rate/ETA (timing fields, preregistered
structural comparison per CONTRACT) vary. No log line is added or
removed.

Exceptions: worker exceptions propagate via `f.result()` and the
executor context semantics terminate outstanding stripe work exactly
as B0 — the code is unchanged (1148-1153). On failure the reduction
is never reached, so engine arrays remain at their B0 post-failure
state (zeros); pinned by T10.

## 6. Admitted domain

The admitted domain is ALL inputs and configs on which B0 runs the
node-graph flow path — the change is semantics-preserving by the
section 5 argument, not domain-restricted by a guard, because there
is no new route to restrict. This differs from A1 (which replaced an
algorithm and needed a refusal guard); the dossier's F1 text likewise
requires only that the "complete read/write closure has been audited"
(section 4) — there is no refusal/fallback vocabulary in the F1
dossier beyond the unchanged-path obligations, which D1 discharges.
Consequences the reviewer should confirm:
* No input exists that "falls back" — and none is needed: no value
  can be mis-handled by a route that is byte-identical to B0's.
* The compile-domain argument in D2 (nogil adds no typing
  restriction; any specialization that fails to compile is a loud
  failure in tests and constitutes REJECTION).
* turns=True, node-flow on/off, K=1 and K>1, zero-weight origins,
  unreachable destinations, obstacles, elevation: all remain on
  their B0 code paths (T2/T8/T9 exercise each candidate-vs-B0).

## 7. fastmath, NaN/inf, sentinels, lifetimes, cleanup — preserved

* fastmath flags: IDENTICAL set (`fastmath=True` both arms); `nogil`
  does not alter fast-math IR flags. NaN/inf semantics of the kernel
  are the same compiled semantics both arms (IR side-by-side at
  Phase 2, section 9). The kernel's explicit `dov < np.inf and ddv <
  np.inf` test (229) operates on caller-provided arrays (+inf
  sentinels from 1041) — unchanged values, unchanged comparisons.
* Off-envelope `acc_o[pv]`/`acc_d[pv]` writes (358, 377): preserved
  verbatim (they are part of B0's arithmetic; their inertness is F2's
  subject, not F1's).
* Early exits (233-234, 300-301), `scale = trip_volume / q_sum`
  (302), per-pass orders (reach order_o/order_d 243-244, ascending
  pass 1/2 over `reach_nodes`, descending tree passes 342, 361),
  u-turn/snap-edge exclusions (261-275, 291-294): unchanged bytes.
* Temporaries lifetime: per-call NRT churn, alloc==free
  (H05 `memory_lifetimes.json` `accumulate_od_flow`, 48 allocs/call);
  F1 adds no allocation and removes none. Re-verified in a bounded
  child with NUMBA_NRT_STATS at Phase 2 (T11).
* Cleanup: no new scratch exists to leak; per-call temporaries die by
  refcount on the kernel's return path including exception unwind;
  `dd_buf/pd_buf` scatter/reset per OD is unchanged. F1 owns NONE of
  the scratch-reset changes (F2 later owns those; the obligation
  recorded here is only that F1 leaves the current reset semantics
  byte-identical).
* Return values: `delivered` (386) bit-identical (used only for gap
  stats — deterministic, section 5).

## 8. Inherited B0 behaviors preserved (H04-N9)

Not fixed, not counted as regressions, expected on BOTH arms:
`UNA.has_flow_results` never set True by RunFlow (engine-level
`has_flow_results` at 831 carries the state); RunAccessibility does
not clear a previous RunFlow's stored engine/results; RunFlow re-reads
the full network GeoJSON. F1 touches no engine attribute outside the
kernel decorator, no Settings field, no public surface
(`AggregateFlow` export via `UNA.py:10,500` unchanged), no defaults,
no warning/exception text.

## 9. Compiler obligations (delivered at implementation time, before promotion)

Source-level argument is NOT claimed sufficient. Phase 2 delivers:

1. **Targetoptions census**: candidate `_accumulate_od_flow` =
   {cache: True, fastmath: True, nogil: True}; `_decay`, `_find_arc`,
   `_accumulate_od_flow_turns`, `_build_line_csr_turns` census
   byte-equal to B0's. Note for the reviewer: the H03 frozen-fact
   tests pin `nogil is False` against the **B0 namespace only**
   (`test_l1_determinism.py:32-37` loads `b0()` = wt-b0/src; the
   golden-profile assertion at :66 binds the B0 golden record) —
   neither observes the candidate tree, so the oracle suite stays
   green; cross-arm comparison in `compare_arms.py` is output-bytes
   only.
2. **IR side-by-side**: `inspect_llvm` of `_accumulate_od_flow` at
   the observed signature(s), B0 tree vs candidate tree, recorded in
   `evidence/F1I/ir_*.{ll,md}` + an FP-op census (fadd/fmul/fcmp
   counts and order in the four passes identical; the ONLY expected
   differences are GIL save/restore calls; no new fastmath-affected
   instruction, no FMA introduction, no reassociation delta).
3. **Specialization census equality**: the candidate compiles the
   SAME set of `_accumulate_od_flow` signatures as B0 on the same
   battery (runner-style inventory), including the int32-CSR
   observed specialization (verified against the sidecar: 137
   captured records, 32 selected; indptr int32[55837], indices
   int32[165410], weights float64[165410], edge_id int32[165410],
   dir int8[165410], d_o/d_d float64[55836], pred_o/pred_d
   int32[55836]; outputs edge-indexed float64[69957] and
   out_node_flow float64[0] — the shape-0 node-flow guard is
   exercised by the observed replay itself).
4. **Behavioral byte equality** everywhere in section 10 via
   `comparator.assert_array_bytes_equal` (dtype+shape+bytes; first
   divergence named). NO tolerance anywhere. The H03 ULP<=4 rule
   applies ONLY to Python-trace-vs-compiled float sums and is NOT
   available to any candidate-vs-B0 comparison.
5. **Observed-scale replay** (the candidate-vs-B0 byte-equality
   obligation): `replay_od_flow` over the 32 selected captured calls
   from `golden/observed/` with the candidate namespace
   (`UNA_ORACLE_ARM_B` root), fresh zero buffers → byte-exact
   `out_AB/out_BA/out_node_flow` and bit-equal `delivered`
   (bounded, O2-scale, light-run class).
6. **NRT census**: 48 allocs/call, alloc==free preserved (bounded
   child, NUMBA_NRT_STATS, A1's `nrt_child.py` pattern).
7. Any compiler-equivalence failure in 1-6 = REJECTION of the
   candidate (task routing), regardless of the section 5 induction.

Per-import-identity cache roots (fleet finding): every candidate/B0
comparison runs with distinct `NUMBA_CACHE_DIR` per arm and import
alias; tests reuse A1's two-root load pattern; no cache root is ever
shared across import identities.

## 10. Test pins (tests/large_e2e/F1/**; each name maps to a proof obligation)

Fixtures reuse the A1 pattern (`cand_load.py` two-root loader,
`fixtures_f1.py`, `mt_child.py` bounded concurrency child,
`mutant_f1.py` first-divergence mutants); oracle package imported,
never modified. Engine cases include the dossier's required classes:
node flow disabled, zero-weight origins, unreachable destinations,
obstacle and elevation cases, K=1, and >=2 multistripe profiles
(baseline default K and a smaller explicit K) — each K profile is a
SEPARATE B0-vs-candidate reference pair; K is never compared across.

* **T1 kernel battery**: compiled candidate kernel vs compiled B0
  kernel on the flow kernel fixtures (small CSR graphs covering:
  early-exit envelope, q_sum==0, contamination marking, u-turn
  exclusion, snap-edge exclusion, empty reach, node flow on/off
  zero-shape buffer): `(out_AB, out_BA, out_node_flow, delivered)`
  byte equality.
* **T2 engine battery**: candidate `AggregateFlow` vs B0 engine on
  all engine cases (incl. the classes listed above): `edge_flow`,
  `edge_flow_AB`, `edge_flow_BA`, `node_flow`, observer flows byte
  equality; inputs unchanged; outputs freshly owned; sequential
  reuse on the same engine object byte-identical.
* **T3 fixed-stripe partials**: test-local serial replica of the
  striped driver (mirroring the oracle's `trace_origin_loop`
  approach) computes per-stripe partial AB/BA/node arrays BEFORE the
  reduction, for K=1 and two multistripe profiles; (a) replica
  reduction `((0+P_0)+P_1)+...` byte-equals the engine output of the
  SAME arm; (b) replica partials are arm-invariant (B0 replica ==
  candidate replica, byte-exact).
* **T4 repeated-concurrency determinism**: K>1 engine runs repeated
  R times (fresh engine objects, several in-process repeats plus
  bounded subprocess runs) — all runs byte-identical to each other
  and to the K-matched B0 run. The multi-thread determinism pin.
* **T5 concurrent independent engines**: two engines running
  simultaneously on independent topologies in one process — each
  byte-equal to its solo-run output (no shared mutable state).
* **T6 observed replay**: `replay_od_flow` on all 32 selected
  captured O2 calls, candidate namespace: zero-buffer outputs
  byte-equal to the stored fixture outputs, `delivered` bit-equal;
  plus `replay_trip_volumes` and the negative check
  (`budget_override`) proving the comparator discriminates.
* **T7 compiler census + specializations**: targetoptions census
  (obligation 9.1) and specialization-set equality (9.3), both arms.
* **T8 turns exclusion (H04-N1 empirical pin)**: small turns=True
  engine case, candidate vs B0 byte-identical end-to-end; turns
  kernel targetoptions census equal; `git diff` of src limited to
  the single decorator line (recorded in evidence).
* **T9 path routing**: n_threads==1 fast path and executor path
  byte-equal candidate-vs-B0 (the unchanged-fallback reachability
  obligation).
* **T10 exception semantics**: module-level interception raising on
  a chosen origin (test-local monkeypatch of the module attribute —
  no source change): exception propagates to the caller in both
  arms; post-failure engine arrays byte-equal (zeros; reduction not
  reached); second Centrality on the same object afterwards produces
  the correct full output in both arms (no contamination).
* **T11 NRT census**: per-call alloc==free and the 48-alloc rate at
  the observed fixture (bounded child, NUMBA_NRT_STATS).
* **T12 log content**: progress event content sequence (origin
  k/n values) deterministic and arm-invariant; rate/ETA compared
  only structurally (present, parseable) — never by value.
* **T13 aliasing/lifetime**: `np.shares_memory` assertions —
  outputs never alias inputs or scratch; per-stripe buffers are
  distinct objects at K>1; empty node array case: shape 0, shared,
  never written (byte-equality of outputs proves it).
* **First-divergence negatives**: mutant kernels under
  `tests/large_e2e/F1/` (never golden sources), each MUST be caught
  by the byte comparator with the first divergence recorded:
  M1 slot-order reduction reversed (K-1..0); M2 stripe decomposition
  shifted to contiguous blocks (breaks `range(slot, n_origins,
  n_threads)` association); M3 pass-3/pass-4 order swapped in the
  replica. A negative test that fails to fire on its mutant is itself
  a failure.
* Warm-up wall time of every specialization is recorded (dossier:
  record startup cost; no performance claims in F1I).

## 11. Explicit non-goals and bounds

No change to stripe count/membership, no H<K scheduler, no atomics,
no work stealing, no scratch reuse/reset change (F2), no gradient
workspace change (F3), no turns-path change, no `Flow.py` /
K-alternatives change, no public API/defaults/artifact-order change,
no GPU/native backend/relaxed math, no new warnings or log lines, no
benchmark or screening runs in F1I (F1R owns screening within the
3-paired-block discipline), no git writes. One design; at most one
performance revision after the initial screened version (DECISION.md).

## 12. Deviations from dossier text

None. Interpretations flagged for reviewer confirmation: D1 (no
guard/fallback for F1 — domain-wide semantics-preserving change),
D2/D7 (compile-failure = rejection, warm-up as harness/test
obligation), D4 (no scheduler work — CONTRACT excludes it for F1),
D5 (no screening in F1I). The turns-closure conclusion of section 3
is the H04-N1 "PROVE exclusion" branch — if the reviewer rejects it,
the fallback is commissioning the H03 turns supplement before
implementation (coordinator decision).
