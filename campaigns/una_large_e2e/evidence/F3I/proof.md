# F3 proof: byte-budgeted destination-gradient chunks (pre-implementation, for independent review)

Status: DRAFT REV 3 for h04 preimplementation approval. Superseded chain, filed per the sole-delta convention BEFORE each in-place revision: rev 1 at `campaign_data/f3i_proof_superseded_rev1_eb399d3d.md` (the coordinator's requested name `f3i_proof_superseded_eb399d3d.md` maps to this file), rev 2 at `campaign_data/f3i_proof_superseded_rev2_01047258.md`. REV 2 incorporated the shared-memory record of h04's ruling; REV 3 completes the coordinator's 9-point amendment list verbatim: the loop-body external-effect census (ruling 3(a)), single numerical execution with the where-it-fired log (3(c)), margin accounting with value and measured-derivation protocol (3(d)), fallback memory honesty (3(e)), and the base/implemented-at accounting from the routing ruling f1930690 (points 9a-9c). The relay confirmations have landed (ruling e0827fc0: model (b) + restart-fallback accepted with conditions; routing f1930690: author at the tip), so nothing in this proof is pending a ruling except h04's Phase-2 approval itself. No implementation byte exists; `git status` at authoring time shows only this file and `admission.json` (committed at 6525d2e) in `evidence/F3I/`.

Author: h05-profiler, 2026-09-27T02:54:20Z. Spec: `evidence/F3I/impl_spec.json` (blob `f6a7f927977deb65f9cb1ba85ac573418d63d6efa625873f37538f3ef18b1dd7`, commit 65a6729). Admission: `evidence/F3I/admission.json` (sha256 `e1f9e6bd015f42d5b26173d09ea2d8a4af0419bca939b18b8cf3c634355830d6`), verdict ADMITTED for this phase.

Base accounting (routing RULED, h04 f1930690: author at the tip, option (a); decisive ground: the committed spec's authored_against pin made option (b) self-contradictory): the F3I implementation is authored AT THE TIP. proof.md (and later result.json) states the implemented-at commit EXPLICITLY at Phase-3 time — it will be later than the admission commit 6525d2e; the src blob is expected to remain 455e92ad since intervening commits are evidence-only, but the hash is stated at result time, never assumed. The F3-only delta is presented as diff-vs-parent. The admission base (7a27866 / tree b121ec0b / blob 0114ce8f) and the implemented-at state are stated TOGETHER wherever either appears, so no base ambiguity survives into F3R. Doctrine on record: attribution lives in per-commit diffs, not base selection; accepted_tracks and physical lineage are separate ledgers (F1 rides the line per its disposition, throughput claim foreclosed; I00 integrates per evidence in hand at that time). Insensitivity of THIS proof is unchanged: it cites only `_precompute_dest_gradients` (:908-961) and its turns twin (:1234-1274), byte-identical between 0114ce8f and 455e92ad.

## 0. Scope

F3 changes ONE function's memory schedule: `_precompute_dest_gradients` (AggregateFlow.py:908-961) replaces the fixed element-count chunk rule (:928 `chunk = max(1, int(1e8 // max(n_total, 1)))`) with a byte-budgeted chunk chosen by a private pure selector, and deletes dense per-slice arrays before the next Dijkstra call. It adds one private module (`_large_flow_workspace.py`) and touches no other src byte. The turns twin (:1234-1274) is excluded byte-for-byte (spec C4, h04-confirmed); its honest cost is stated in section 8. F3 makes NO arithmetic change, NO dtype change, NO order change, NO thread-schedule change, and claims NO throughput (DECISION.md line 11: "F3 capacity-only improvement is labelled capacity_only, never throughput-qualified").

## 1. Baseline transition (actual code)

Walk-through at the committed pin, all lines printed and verified in Phase 0 (admission.json phase0_receipt):

- :921-925 bind `n_net`, `n_dest`, `n_total = self._csr_indptr.shape[0] - 1`, `limit`, `dest_nodes = np.arange(n_net, n_net + n_dest, dtype=np.int64)`. V' = network nodes + destination virtuals + origin virtuals (origin virtuals start at `n_net + n_dest`, :622). At the observed cell V' = 49159 + 6661 + 1024 = 56844.
- :928 fixes the slice width by ELEMENT count only: `int(1e8 // 56844)` = 1759. No byte accounting anywhere in the function; the :927 comment prices the intermediate at "~0.8 GB" counting float64 distance payload alone (H05's 12 B/elem derived convention reproduces 1,199,863,152 B = 1.11746 GiB exactly; the dossier's 13 B/elem convention gives 1,299,851,748 B; measured anchor is peak_rss 2489.125 MiB).
- :931 slices destinations ascending: `for s in range(0, n_dest, chunk)`.
- :934-939 one `_scipy_dijkstra(self._csr_rev, directed=True, indices=dest_nodes[s:e], limit=limit, return_predecessors=True)` per slice — the SAME native call F3 preserves verbatim.
- :940-942 shape handling (`dist[None, :]` when 1-D) and `finite = np.isfinite(dist)` (full c×V' bool mask).
- :943-948 per row k ascending: `cols = np.where(finite[k])[0]`; counts; appends of independent copies `cols.astype(np.int64)`, `dist[k, cols].astype(np.float64)`, `preds[k, cols].astype(np.int32)`.
- Lifetime defect the dossier names (line 7): `dist`, `preds`, `finite` stay live across the whole inner loop AND into the next `_scipy_dijkstra` call — they are rebound only after the next call returns (:934). During slice j's call, slice j-1's full dense triple (1759×56844×13 B at the observed cell) is still referenced.
- :951-953 three `np.concatenate` calls assemble `nodes`/`dist`/`pred` in ascending destination order; at this point the parts lists AND the new arrays coexist.
- :954-960 a v=2 log line pricing sparse storage at 20 B/entry (int64+float64+int32 per entry).
- :961 returns `(indptr, nodes, dist, pred)`.

H05 anchors (admission.json reverified_facts): chunk_chosen 1759 (reproduced exactly), sparse_grad_bytes 9,565,360 (9.122238 MiB, 478,268 implied entries at 20 B), gradient stage 1.73529525 s = 20.77% of the pilot window, sel-all capacity refusal caused by this dense transient (projected peak 2.43 GiB vs 0.9× ceiling 2.30 GiB).

## 2. Candidate transition

Three hunks, all inside F3-owned files:

1. NEW `src/urban_network_analysis/Engines/_large_flow_workspace.py`: the private pure selector (section 3) and the `NO_FIT` signal constant. Nothing else; PD3 conditions: stdlib arithmetic only, no I/O imports, no numba (cache surfaces untouched), imported module-privately into AggregateFlow.py, nothing exported through package `__init__`.
2. `_precompute_dest_gradients` (:908-961) rewritten: selector picks each slice width from measured bytes; per the dossier loop obligations (spec C3, quoted verbatim there): same scipy call/CSR/flag/indices/limit/return_predecessors; same shape handling and per-row `np.where(finite)` order; same gathered dtype conversions and independent copies; rows appended in exact destination order; **`del dist, preds, finite` (and any row temporaries) at the end of each slice iteration BEFORE the next call**; concatenation preserving identical indptr/nodes/dist/pred order and dtype.
3. The v=2 log line (:954-960) gains the actual schedule (full per-chunk width sequence, cap bytes, margin, chunk count, the PD2(a)/PD5 fallback marker when the original formula ran, and where the floor fired — slice index and chunks completed — so an aborted schedule is reconstructable from log + run record). Emitted ONCE at function end (ruling 3(c)). Append-only: every existing token stays prefix-stable. Artifacts carry no log text, so byte-for-byte artifact comparison is unaffected.

The superseded :927 comment ("~0.8 GB", distance-payload-only) is deleted and replaced by the byte-budget policy statement — the contradicting sentence must not survive, per the F1R framing precedent. No other comment, docstring token, or line outside the two functions and the new module changes; the Phase-2 blob diff must show exactly: new module + :908-961 rewrite + :927-928 comment/rule replacement + :954-960 log extension + one private import line. ZERO changed bytes at :1234-1274.

## 3. Selector contract and the item-5 PIN: per-chunk re-invocation (model b)

PIN: the selector is re-invoked at EVERY slice with the ACTUAL current fixed_live; this is h04's option (b), ACCEPTED with binding conditions (ruling e0827fc0, relayed and incorporated here in the ruling's form, not the original proposal's).

Selector inputs (all integers, measured by the caller from array nbytes — never modeled): `V_prime`, `remaining`, `dist_itemsize`, `pred_itemsize`, `mask_itemsize` (from returned dtypes at run time, never hardcoded), `fixed_live_base` (CSR data/indices/indptr bytes actually live both directions + retained result arrays + source/result arrays), `retained_parts_bytes` m_k (sum of appended part nbytes so far), `margin` (fixed constant 64 MiB — value, measured derivation and scope in the Margin accounting paragraph below), `cap_bytes` ∈ {64, 128, 256} MiB (PD1: default 256; the only admitted set).

Selection rule (exact integer arithmetic):

```
per_source = V_prime*(8 + pred_itemsize + mask_itemsize) + row_temporaries + source_index_cost
budget_k   = cap_bytes - fixed_live_base - m_k - margin
c_k        = min(remaining, budget_k // per_source)
if budget_k < per_source:  ->  NO_FIT
```

Purity, by ENUMERATION of the selector's external contacts (convention: a no-side-effect claim is an enumeration, not an assertion): the selector receives integers and returns an integer or the NO_FIT constant; it performs no attribute writes, no global reads or writes, no I/O, no import beyond arithmetic builtins, and no environment query of any kind (free-RAM, loadavg, cgroup, psutil or other); identical integers always yield the identical result. The caller-side measurements feeding it (`.nbytes` sums, counts) are pure arithmetic on lengths. Violation = implementation rejection (spec C2, dossier line 13: "Do not query changing free RAM every row and call it deterministic numerical identity").

**Max-transient theorem (h04's requested form).** For every slice k, charge the slice its full per_source (which by definition includes V'×(8+pred_itemsize+mask_itemsize) plus the row-temporaries and source-index terms — a conservative upper bound on that slice's dense cost). Then by construction of c_k = budget_k // per_source (integer floor), `m_k + c_k*per_source + margin <= cap_bytes` holds at every slice, and NO slice is entered when even c=1 cannot fit. The cap is therefore enforced on MEASURED bytes at every slice — no estimate can silently void it.

Why (b) and not (a): option (a) enforces the cap against an EXPECTED final retained size, which is unknowable at entry; its only sound value is the worst case n_dest×V'×20 B (prohibitive — forces NO_FIT on realistic inputs), and any smaller estimate can be wrong, and a wrong estimate is exactly the silent cap-void h04's catch names. Under (b) the theorem above needs no estimate.

**Mid-run NO_FIT — RULED (h04, e0827fc0, relayed and confirmed; conditions carried here in the ruling's form):** under (b), entry-time NO_FIT (m_1 = 0) sees only the first slice; as parts accumulate, budget_k can cross below per_source mid-schedule. Ruled semantics: mid-run NO_FIT frees all parts/temporaries and RE-EXECUTES THE ORIGINAL FORMULA LOOP from scratch. The hybrid (keep the prefix, run the original path for the remainder) is EXCLUDED: it executes a schedule neither dossier option ("preserve original behavior" / "report benchmark capacity unavailable") admits. This is the dossier stop condition's territory (line 36: sparse parts dominating means chunk reduction cannot solve the limit) — mid-run NO_FIT is expected to be the rare terminal, not a steady state.

**No-external-mutation census (ruling 3(a)) — the restart's byte-identity is an ENUMERATED claim, never an asserted construction:** h04 pre-verified at the pinned blob and this proof adopts the census: within :908-961 the loop body READS `self._*` only (`self._csr_rev` at :935; `limit` and `dest_nodes` are pre-loop locals), performs NO self/engine attribute writes, calls NO logger inside the loop (the v=2 line at :954 is post-loop), touches no module/global state, and writes no output buffer (`edge_flow_*`, observer counters, `has_flow_results` all live in other functions; this function's sole external effect is its returned tuple built from locals). The pre-abort mutation surface is therefore exactly the function's locals — `dist`, `preds`, `finite`, the parts lists, `counts`. Mid-run NO_FIT deletes exactly those (section 5) and the restart re-executes the original loop from the same untouched inputs; byte-identity to original output follows from this census plus the identity induction (section 4), and PD2(b) tests it empirically.

**Single numerical execution (ruling 3(c)).** Exactly ONE result-producing loop runs in any invocation — under fallback, the restart is that loop and the aborted schedule emits nothing numerical. The v=2 line is emitted ONCE, at function end, carrying: the full c_k sequence, cap bytes, margin, chunk count, the fallback marker (PD2(a)/PD5), AND where the floor fired (slice index and chunks completed before NO_FIT), so the aborted schedule is reconstructable from log + run record.

**Margin accounting (ruling 3(d)).** Value: margin = 64 MiB (67,108,864 B), a fixed constant for every selection call, recorded in the v=2 line. Measured derivation: (i) the loop's worst per-row temporary surge is bounded by 3×V'×8 B = 1,364,256 B = 1.301056 MiB at the observed cell (int64 `cols` index + fancy-index gather + astype copy) — the margin is 49.2× that surge; (ii) SciPy's internal workspace beyond the returned dist/pred/mask triple is INSIDE the margin by declaration, and its realized size is MEASURED at L0: test_f3_cleanup's process-tree/native instrumentation reports realized-transient-minus-payload per slice, pass requires the measured gap ≤ margin/2, and a violation is stop-and-re-derive the margin with reviewer approval before the implementation commit freezes the proof (the caps themselves never change); (iii) the residue reserves allocator granularity/fragmentation on the chunk arrays. Honesty statement, ruled form: the cap and all selector arithmetic govern PAYLOAD NBYTES, NOT RSS; the payload→RSS gap is covered by the margin + process-tree/native measurement (never tracemalloc alone) + F3R's screen discipline (per-window admission and watchdog at the real cell). At the frozen default the margin leaves 192 MiB of the 256 MiB cap to payload — c = 272 sources/slice at the 13 B/elem per_source of 738,972 B (base≈0 upper-bound illustration), a working schedule.

Schedule determinism: identical inputs produce identical finiteness patterns, hence identical m_k sequence, hence identical c_k schedule. The full c_k schedule is recorded (C5). Numerical behavior is independent of the schedule (section 4).

Endgame accounting, stated honestly: at the final `np.concatenate` the parts lists and the new arrays coexist, so the endgame holds ≈ 2×m_total (dossier line 7 names this coexistence). This term is NOT part of the per-chunk cap arithmetic (h04's requested theorem form is retained_so_far + c*per_source + margin) and is not silently waved away: m_total is the sparse payload (9.122238 MiB at the observed cell → endgame ≈ 18.244476 MiB, far under any cap), it is recorded per-stage by the tests plan's "explicit per-stage payload counts", and if 2×m_total + base approaches the envelope the dossier stop condition governs — record the remaining lower bound, declare the capacity path unavailable; no spill/checkpoint format enters scope.

## 4. Numerical identity (the induction)

Claim: for any admitted schedule (any c sequence, including the original formula and c=1 and c=whole-set), the returned `(indptr, nodes, dist, pred)` are bit-identical. SCOPE of every identity claim in this proof: data arrays and artifacts; log text is excluded by design wherever the schedule record or a fallback marker is mandated (sections 2, 7).

1. Per-source independence: each destination's row of `dist`/`preds` depends only on its own `_scipy_dijkstra` call with identical arguments (same CSR object, `directed=True`, its own index, same `limit`, `return_predecessors=True`) regardless of which other destinations share the slice — the same native search per source regardless of grouping (dossier line 27), so predecessor TIE behavior is identical across schedules; ties are tested bit-for-bit on `pred`, not inferred from distances (dossier line 28).
2. Per-row gather: `np.where(finite[k])[0]` and the fancy-index+astype copies inspect exactly row k; rows are appended in ascending destination order within a slice and slices partition destinations ascending — the global order is the destination order for every schedule (dossier line 21-22 obligations, spec C3 verbatim).
3. Assembly: `counts`/`indptr` via the same cumsum; three concatenates over identically-ordered part lists with identical dtypes.
4. Therefore any divergence between schedules is a bug, not a property — tested bit-for-bit across c ∈ {1, 2, 3, whole-set, original formula} on every fixture class of the dossier's counterexample list (section 10). If any schedule diverges on any fixture, the domain is rejected (dossier line 28: "reject that domain").

No FP surface changes: no numba kernel, no decorator, no arithmetic expression, no reduction is touched; F1's fastmath/nogil analysis (F1I proof secs 5/5b/5c) is untouched by construction — F3 adds no new reassociation site because it adds no new arithmetic.

## 5. Lifetime and cleanup audit

Every allocation in the candidate loop and its release point:

- `dist`, `preds` (c_k×V' dense), `finite` (c_k×V' bool): rebound each slice; **`del` at the end of each slice iteration, before the next call** (dossier line 22, spec C3). This is the load-bearing change: in the baseline they survive into the next call (section 1).
- Row temporaries (`cols`, the two gathered copies before astype): released by the existing rebinding each row iteration; F3 adds no new row-lifetime.
- Parts lists: necessary output, retained to the end by design; their bytes are the selector's measured `m_k`, not waste (dossier line 36 treats their dominance as the stop condition, not a defect to patch).
- No view aliasing: fancy indexing returns a copy and `astype` copies again (baseline behavior at :946-948, preserved); a test DEMONSTRATES non-aliasing of every appended part from deleted dense memory (write-after-free canary / base-object refcount instrumentation), per dossier line 28 "verify no appended part aliases deleted dense memory".
- Mid-run NO_FIT: `del` of parts lists and contents before the fallback re-execution; the fallback then reproduces the baseline loop exactly (section 3).
- Release verification is via allocation/lifetime instrumentation OUTSIDE final timing (dossier line 32) — weakref/refcount checks in test_f3_cleanup.py, never inside a measured window.

## 6. Admitted domain and warnings

Dossier lines 9-10 verbatim (spec admission_and_warnings). Concretely: admission covers validated CSR and normal finite nonnegative cost profiles where the original SciPy calls are warning-free. Negative/nonfinite costs, invalid sources/indices, unusual sparse subclasses, warning-producing conversions: the code KEEPS the original chunk formula AND original error/warning behavior — the fallback is not merely a memory policy but the domain guard. Warning/exception counts are recorded in B0/candidate L0/L1 tests; no warning is classified as harmless timing metadata. If native vectorized batching ever chose different ties under changed chunk size, the domain would be rejected (dossier line 28) — the tie fixtures exist precisely to detect this.

## 7. Refusal and fallback semantics (PD2)

- Entry NO_FIT (cap − base − margin < per_source at m=0): original formula runs as the ONLY result-producing loop; explicit fallback marker in run record + v=2 line.
- Mid-run NO_FIT (RULED, section 3): free-and-restart on the original formula — the restart is the SINGLE result-producing loop (3(c)); markers plus where-it-fired; arrays/artifacts byte-identical (scope per section 4), backed by the no-external-mutation census (3(a)).
- Fallback memory honesty (3(e)): the restart RE-OPENS the original unbounded transient BY DEFINITION — that IS "preserve original behavior"; the run record says so; NO new cap claim exists on the fallback path. Screen-cell expectation: at the frozen cell (default cap 256 MiB; base + retained far under it) mid-run NO_FIT should NEVER fire — its absence is expected, its possibility is tested.
- Both paths: arrays/artifacts byte-identical to the original formula's output AND the call sequence instrumented to assert the fallback execution's slice boundaries equal the original formula's (`max(1, int(1e8 // n_total))` throughout) — byte-identity alone cannot distinguish a restart from a hybrid (extension 4). The tiny-cap forcing is selector-level on small fixtures only; the main process is never driven toward OOM.

## 8. Inherited behaviors preserved / turns closure

- The turns twin :1234-1274 keeps the element-count rule and its large transient — Phase-2 blob diff must show zero changed bytes there (h04 PD4). Honest cost, stated: at any cell that selects turns, F3 delivers no memory relief.
- `_process_origins_aggregate` and every other function: untouched.
- The F1 decorator rides the line (routing RULED, h04 f1930690, option (a) author-at-tip; admission.json's open_routing_flag is resolved by that ruling and the committed admission stands as the state-at-its-time record): it is not part of any F3 hunk; the F3I commit's diff-vs-parent is exactly F3. F3R arm pairing stays the rev-5 rule: b0 = line-minus-F3 vs cand = line-plus-F3, same physical line.
- Docstring/comment hygiene: the superseded "~0.8 GB" comment is deleted (section 2); the function docstring gains only the byte-budget policy sentence and the fallback marker note.

## 9. Runtime/compiler obligations at implementation time

None new: no numba surface changes (PD3: no numba in the new module), no cache-directory implications (no compiled artifact added or changed), no thread pool change. Deliverables at implementation time remain the L0/L1 runs and their artifacts; heavy/gate-eligible execution stays out of F3I (spec tier_boundary; the 3-block screen is F3R's instrument).

## 10. Test pins (tests/large_e2e/F3/**; each name maps to a proof obligation)

- `test_f3_selector.py` → sec 3: contract bounds (1 ≤ c ≤ remaining; floor semantics; NO_FIT signal, never c=0); purity (forbidden-callable interposition + module source audit — no environment query); determinism (same integers → same c).
- `test_f3_chunk_equivalence.py` → sec 4: bit-for-bit indptr/nodes/dist/pred across c ∈ {1, 2, 3, whole-set, original formula} on: zero destinations; one destination; disconnected/+inf; TIED shortest paths (pred compared, not dist); duplicated source nodes if baseline allows; custom/elevation/obstacle costs; sorted input/parallel arcs derived from the actual flow CSR (sha-pinned observed inputs, sliced small); empty finite rows; uneven final tail.
- `test_f3_refusal.py` → sec 7: entry-NO_FIT and forced mid-run-NO_FIT; each proves output byte-identity AND (call-sequence instrumentation) that the fallback's slice boundaries equal the original formula's — the restart-vs-hybrid discriminator; markers present in captured log + record.
- `test_f3_cleanup.py` → sec 5: dense triple released before next call (refcount/weakref instrumentation, outside timing); appended parts demonstrably non-aliasing.
- `test_f3_observed.py` → sec 4/6: small observed-derived complete RunFlow; artifacts byte-identical across caps {64,128,256} and original formula; warning counts recorded (none swallowed); process-tree RSS + per-stage payload counts (not tracemalloc alone).

Fixture shas recorded in test artifacts. L0/L1 only; no lease window opens under F3I as specced.

## 11. Explicit non-goals and bounds

Dossier line 25 verbatim (spec non_goals): mask formation unchanged (per-row finite masks remain a possible SECOND subchange only if lifetime measurements justify and exactness is independently checked — not in this change); no min_only, no batched-nearest-source substitution, no sparse Dijkstra rewrite, no concurrent chunks, no changed limit, no compression, no new file cache. No public Settings field (cap is internal policy, PD1 default 256 MiB, set frozen {64,128,256}, no cap change after freeze). No source outside owned_files. No GPU, native backend, relaxed math, broad cache, public RunBatch parallelization, scientific bug fixing.

## 12. Deviations from dossier text

None in algorithm or obligations. Two elaborations, both reviewer-demanded or dossier-mandated, declared: (i) the invocation-model pin (section 3) exists because h04's review catch demanded the model be pinned, not silent; (ii) the mid-run NO_FIT restart-fallback resolves the dossier's "preserve original behavior or report benchmark capacity unavailable" for a case the dossier does not name — reviewer-ruled semantics (h04 e0827fc0, relayed and confirmed), carried with conditions 3(a)-(e) in sections 3 and 7. The v=2 log extension implements the dossier's "record the actual schedule" and changes no artifact byte.

Hard stop: no implementation byte until h04's Phase-2 approval is relayed by the coordinator. The base-routing ruling has landed (f1930690: author at the tip); Phase 3's only remaining gate is that approval.
