# Independent review: ordered-CSR integration (task T04)

- Reviewer: independent subagent (no prior exposure to the integration work)
- Reviewed commit: `755c7b8` on `perf/una-cpu-throughput` (worktree `wt-integration`)
- Candidate: `src/urban_network_analysis/Engines/_ordered_csr.py`
  (git blob `98504c2d...`), integrated into `Accessibility.py` (`473de48...`) and
  `AccessibilityWElevation.py` (`dc75e48...`)
- Baseline/oracle tree: `/Users/alansynn/orca/workspaces/una-x/main/src`
  (engine blobs `246fe439...` / `69061684...`, re-verified with `git hash-object`)
- Environment: Python 3.11.16, numpy 2.4.6, macOS arm64 (intp = 8 bytes)

## Verdict: approve_with_notes

No equivalence, robustness, fallback, ownership, or integration defect was found.
The notes below are provenance-hygiene and one benign exception-domain nuance.

## 1. Order equivalence — PASS

Analytical argument. Baseline row for node `n`:
`[end_nodes[i], w_ab[i] for i ascending, start_nodes[i]==n]` then
`[start_nodes[j], w_ba[j] for j ascending, end_nodes[j]==n]`. The candidate
concatenates forward incidences at indices `0..E-1` and reverse incidences at
`E..2E-1`, then stable-sorts by source. For fixed `n`, every forward index is
`< E` and every reverse index `>= E`, so stability reproduces
forward-block-then-reverse-block with ascending edge order inside each block —
exactly the baseline rule. The subtle interleaving case resolves identically:
for edges `i=(u,v)`, `j=(v,u)`, `u`'s row is `[v@forward-i, v@reverse-j]`
(because `i < E+j`) and `v`'s row is `[u@forward-j, u@reverse-i]` (because
`j < E+i`) in both schemes. `bincount` counts equal the mask sums; the
`cumsum(int64)` prefix equals the baseline's chained int64 adds, with the total
bounded to `2*edge_count <= intp_max-1` by the guard.

Empirical. 2661+ differential cases against
`tests/perf_contract/ref_original_csr.py` (which I independently checked
line-by-line against the actual baseline engines): all value-, dtype-, and
byte-exact.

- 1161 exhaustive directed multigraphs (2 nodes x <=4 edges, 3 nodes x <=3
  edges): every self-loop, parallel edge, duplicate, isolated node, empty and
  full edge sets.
- Hand-computed `(u,v),(v,u)` interleavings in both edge orders — attempt to
  break the row order FAILED.
- 300 bit-pattern trials and 1500 random graphs (up to 9 nodes / 59 edges,
  40 percent duplication) — FAILED to break equivalence.

## 2. Guard domain — PASS (with one benign note)

Verified effective: intp-itemsize gate; `type(node_count) is int` (rejects
`bool`, `np.integer`, `float`); `0 <= node_count < intp_max`; exact
`np.ndarray` (subclasses such as masked arrays refused) with `ndim == 1`;
exact native `int64`/`float64` dtypes (big-endian `>i8` refused — no silent
reinterpretation; strided/negative-stride/F-order/read-only inputs are safe
because all reads are copying ops); equal `.size` (catches weights-only
mismatches in both positions — probed); `edge_count <= intp_max//2` (bounds the
doubled arrays and prefix-sum maximum); endpoint range check that also makes
the `cumsum` out-slice shape exact and refuses `node_count == 0` with edges;
`np.argsort(kind="stable")` deterministic on numpy 2.4.6.

Note (benign). For `node_count >= 2**60` the pointer/bincount allocations
exceed numpy's size limit and raise `ValueError("array is too big")`, which the
`except MemoryError` does not catch. Measured: the fallback's first statement
`np.zeros(node_count+1, int64)` raises the identical ValueError type and
message at the same call depth, so end-to-end observable error behavior is
unchanged; the domain is unreachable from real flows (`node_count` is
`node_points.shape[0]`, so a real array that size must already exist). For
allocations within numpy's limit but beyond RAM, the fast path can die by OS
OOM rather than MemoryError — the baseline is strictly slower and heavier
there. Optional cosmetic hardening: reject `node_count > intp_max//8`.

## 3. Fallback — PASS

Both `if ordered_csr is None:` blocks are the baseline count+fill loops
verbatim: indent-normalized line-by-line equality against the actual baseline
files, plus whole-file AST equality of the committed engines against baseline
copies patched with the supplied candidate packet. Baseline dtype/casting
behavior is therefore preserved for int32 endpoints, float32 weights,
big-endian arrays, subclasses, out-of-range endpoints, etc. (all probed to
return `None`). The recorded L1 `int32_fallback` case (cited, not re-run)
exercised this route through both engines with bit-identical arm artifacts.

## 4. Ownership and bit guarantees — PASS

All four returned arrays are freshly allocated (`owndata`, writable,
C-contiguous); `np.shares_memory` is false against every input and among the
outputs — probed with `ab_weights is ba_weights` aliasing, which is exactly
what the elevation engine passes in symmetric mode. Inputs are never mutated
(byte-compared). No arithmetic touches the weights: +0.0/-0.0, +/-inf, 1e-320
denormals, float64 max, quiet NaNs with distinct payloads, negative NaNs, and
sNaN-pattern bits all land in rows byte-identically to the oracle, with NaN
payload placement hand-verified per row. Field dtypes match the baseline
graph_engine attributes (int64/int64/float64/bool_), so the numba kernels
compile identical signatures — consistent with the recorded SHA-identical L1
arm artifacts.

## 5. Integration soundness — PASS

`diff -rq` over the whole package: the candidate tree differs from baseline by
exactly the two engines plus the new helper; `AccessibilityWTurns.py` is
byte-identical (turn flow intentionally untouched). Each engine has exactly one
CSR construction site, now routed through the helper. All three conditional
field assignments test the same `ordered_csr is not None` predicate and all
four names are bound in both branches, so no stale array can be silently
picked up. Call arguments match baseline gather semantics
(`weights, weights` vs `ab_weights, ba_weights`). Real GIS flows take the fast
path (Topology casts endpoints to int64; engines cast weights to float64);
everything else falls back.

## Adversarial attempts summary

Attempts that FAILED to break the candidate: interleaved reciprocal edges in
both orders; 1161 exhaustive tiny multigraphs; NaN-payload/sNaN/signed-zero/
denormal bit patterns; guard-escape probes (bool/np.integer/float node_count,
subclasses, masked arrays, byte-order, weights-only mismatches, endpoint ==
node_count, node_count == 0 with edges, empty wrong-dtype arrays); aliasing and
mutation probes; ownership/flags probes.

Attempt that produced an observation (benign): the `ValueError`-escapes-
`except MemoryError` case at `node_count >= 2**60` — identical error from the
fallback path, unreachable in practice.

Harness false alarms (transparency, resolved): `np.array_equal` without
`equal_nan=True` rejects NaN rows, and one hand-computed NaN row expectation
was misderived; byte-level and corrected checks all pass. Harness defects, not
candidate defects.

## L0 rerun

```
cd /Users/alansynn/orca/workspaces/una-x/wt-integration && \
/Users/alansynn/orca/workspaces/una-x/venvs/campaign/bin/python -m pytest \
  tests/perf_contract/test_l0_ordered_csr.py -v
=> 17 passed in 0.14s  (matches the recorded evidence)
```

L1 (`test_log.txt`, `arms/*.npz`) cited, not re-executed, per review brief. The
`nonfinite_weight_hazard.md` finding was reviewed and accepted: NaN edge-weight
queue growth is an inherited upstream kernel property present identically in
both trees; the candidate only changes construction upstream of the kernel.

## Notes (non-blocking)

1. Stale manifest hashes. `source_manifest.json` `verification_files` entries
   for `test_l1_engine_arms.py` (recorded `aa37daba...` vs committed blob
   `188f332e...`) and `_arm_runner.py` (recorded `47993bc2...` vs committed
   `eddc8f34...`) predate the post-SIGKILL per-case subprocess redesign
   described in `test_log.txt`. All other entries verified correct (3 engine
   blobs, `ref_original_csr.py`, `stub_topology.py`, `l2_trial.py`,
   `l2_error_probe.py`). Recommend regenerating the manifest or adding an
   addendum with the two current blobs.
2. Packet patch mechanics. The supplied packet patch
   (`prior/una_optimization/patches/ordered_csr.patch`) is rejected by strict
   `git apply` (missing final newline on its last line; LF-normalized against
   this CRLF repository; applies only with `--ignore-whitespace`). The
   committed `_ordered_csr.py` matches the patch's new-file content byte-for-
   byte except one trailing newline, and both engines are AST-identical to the
   packet-patched baseline; byte differences are line-ending/trailing-
   whitespace normalization only. "Verbatim" in the commit message is accurate
   semantically, not byte-for-byte.
3. Worktree hygiene. The wt-integration working tree carries uncommitted,
   out-of-scope benchmark experiments (`benchmarks/una_cpu/workload_gen.py`
   modified; `csr_microbench.py`, `profile_stages.py` untracked). All reviewed
   source under `src/` and `tests/` is clean at `755c7b8`.

## Files

- Machine-readable: `optimization_una_cpu/evidence/reviews/csr/review.json`
- This summary: `optimization_una_cpu/evidence/reviews/csr/review.md`
- Reviewer differential harness (throwaway, outside both trees):
  `/tmp/csr_review_pkt/adv_diff.py`
