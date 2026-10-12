# R5 — in-worker decoded-stage lease reuse: design note (2026-10-12)

Status: DESIGN — implementation gated on the p02-audit mutation report (delegated
2026-10-12; resend requested after the compaction boundary). This note records the
measured mechanism and the smallest-lawful shape so implementation can start the
moment the verdict lands. Branch `perf/una-10x-next`.

## Problem (measured)

On the parallel prepared boundary (P01P/P01E/P01F), prepared-store hits still pay a
per-row restore: `_restore_network_from_cache` (Topology.py:462-543) rebuilds
`from_wkb` → GeoSeries → STRtree (`net.geometry.sindex`) on EVERY row, and
`_restore_access_points_from_cache` (Topology.py:774-814) repeats the from_wkb +
GeoSeries pair for snap stages. Decode-once is NOT amortized.

Micro-benchmark (login node, alan interpreter, 2026-10-12, 100k short LineStrings —
cohort scale): full restore path (from_wkb + GeoSeries + sindex) = **505.8 ms**;
lease path (fresh GeoSeries wrapper over a resident GeometryArray + shared cached
STRtree) = **33.4 ms** ⇒ **≈472 ms/row save at this scale**, ≈3.8 s per 8-row
worker pass — the P01F projection's worker-side cut, now measured rather than
inferred.

## Mechanism (verified against installed geopandas 1.2.0)

- The STRtree caches on the **GeometryArray** (`GeometryArray._sindex`, built from
  `_data`; `GeoSeries.sindex` → `geometry.values.sindex`).
- `gpd.GeoSeries(arr, copy=False)` shares the array AND its cached tree
  (verified: `gs.values is arr`, `gs.sindex is arr._sindex`).
- Passing `crs=` forces an array copy (pandas crs semantics) → tree lost.
  **Attach sequence for the lease**: construct the fresh wrapper with
  `copy=False` + the payload CRS, then `gs.values._sindex = resident_tree`;
  verified `gs.sindex is tree`, element identity preserved, query answers
  identical. If a future geopandas changes `_sindex`, the attach must degrade
  LOUDLY (diagnostic counter + full rebuild correctness path) — never silently.

## Smallest-lawful shape (pending audit confirmation)

- Worker-scoped decoded residency inside the existing P02 arena/registry
  (cap-bounded by `per_worker_residency_cap`; `close_registry()` on worker exit
  already releases it — worker.py:628-632).
- Resident per (stage_key, worker): the decoded GeometryArray + its cached tree +
  the immutable pyproj CRS (pyproj CRS is immutable) — keyed by the SAME
  identity-closed stage key the store uses (snapshot-bytes sha256 + params +
  profile + geometry_env; identity.py:193-216).
- Per row on a verified hit: fresh GeoSeries wrapper (copy=False + post-attach),
  **fresh copies of all numeric fields** (lengths/weights/node_points/degrees/…
  — the flight's `_acopy` discipline unchanged, E10 preserved), fresh pandas
  Series where the payload carries one (uid), full per-row log replay unchanged.
- Chronology preserved: per-row snapshot read + sha256 gates at the seam BEFORE
  any consult (TOCTOU law); the lease only replaces DECODE work after a verified
  hit — validation, failure taxonomy, and log streams are untouched.
- No public mutable aliasing beyond what the audit certifies: if the audit finds
  ANY site that mutates restored state in place, that state stays copy-per-row
  (copy-on-write per field), or R5 scopes down to the fields that are truly
  read-only.

## Law tests (extend tests/tenx/post_p00/test_preparation.py)

1. Lease identity: second same-key serve in one worker shares the decoded
   GeometryArray/tree (`is` checks); numeric arrays are fresh per serve.
2. E10 continues to pass unchanged (flight-level fresh copies).
3. Mutation isolation of the lease tier: a row that mutates what it received
   must not corrupt the next serve (this test's feasibility depends on the
   audit verdict — if geometry mutation is possible by engine code, R5 scopes
   down and this test asserts the copy-per-field law instead).
4. Key invalidation: a byte change (same-size, mtime-touched) between rows
   produces a DIFFERENT stage key → no lease attach (TOCTOU tests extended).
5. Byte parity: lease-path exports bitwise equal to the current prepared-hit
   path (island battery, stamp-normalized).
6. Log stream: `test_prepared_hit_replays_log_stream` passes unchanged.
7. Cap/eviction: decoded residents respect the residency cap; no-fit serves
   without retention (E8 analog).
8. Cross-process: residents never pickle (detach law unchanged).

## Non-goals

- No change to the serial public call (NR-6 engage law).
- No change to store format, keys, or the single-flight protocol.
- No GPU/native coupling; no claim creation — a fresh tenx-law paired gate is
  the only throughput authority (plan §6.6).
