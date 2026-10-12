# P02D — Prepared-input consumer audit (preparation side verified line-by-line)

Campaign una_10x, task P02D, run root `campaign_runs/una_10x/tenx-20261004T155640Z/`.
Commit under review: `050b316` (P01) on `perf/una-10x`. Every claim below carries a
`file:line` reference into that tree; laws proven by existing tests are cited by test id.

Method note (per AGENTS.md §3): this audit reports observed facts, not proposals.
Design dossiers: `campaigns/una_10x/post_p00/designs/02_prepared_bundle.md` (P02),
`01_telemetry.md` (P01 evidence), `07_measurement_and_gates.md`. Sections 7–9
(downstream enumeration) were cross-checked by an independent read-only audit agent
against this same tree; this document integrates both reads.

---

## 1. The five preparation stages of one accessibility row

Pinned identically by the CACHE_GRAPH invalidation suite
(`tests/cache/invalidation/conftest.py:12-19`):

| # | stage id | producer | per-row cost class (P00: ~75% of row service) |
|---|----------|----------|-----------------------------------------------|
| 1 | `una:topology:v1` | `Topology.AddNetwork` → `_get_gdf`/snapshot decode → `Network.BuildTopology` | dominant (decode + topology build) |
| 2 | `una:snap:v1` (origins) | `AddOrigins` → `BuildAccessPoints` | decode + snap |
| 3 | `una:snap:v1` (destinations) | `AddDestinations` → `BuildAccessPoints` | decode + snap |
| 4 | `una:snap:v1` (obstacles) | `AddObstacles` → `BuildAccessPoints` | not in the frozen cohort (no obstacle file) |
| 5 | `una:awe:o_access:v1` | engine `Centrality` (accessibility results) | NOT preparation; P03/M03 scope |

The frozen 32-job cohort (`baseline_policy_audit.json`) exercises stages 1–3 only:
turns FALSE, elevation FALSE, no obstacle/observer files, search_radius 400,
all four metric families on, single-file `.geojson` sources.

## 2. Entry chain and error-order inventory (stages 1–3)

`UNA.RunAccessibility` (`src/urban_network_analysis/UNA.py:316-407`) is the only
integration seam; its order is:

1. `self.has_centrality_results = False` (UNA.py:320) — the reset flag.
2. `self.settings.Validation()` (UNA.py:322) — full settings validation; **never
   skippable on a prepared hit** (design line 26). Contents: unexpected-attribute
   gate (Settings.py:487-506), `knn_weights` coercion (509-521), Literal checks via
   type hints (527-534), engine numeric checks (537-583), `search_radius > 0`
   (585-586), OD-id pairing rule (588-594), `flow_engine` literal (596-599), plus
   the remainder of the method (through ~line 660).
3. `topology.AddNetwork(settings)` → `AddOrigins` → `AddDestinations` (324-326).
4. Optional `AddObstacles` when `obstacle_points_file` set (333-334).
5. Engine construction — `AccessibilityWTurns` (turns) or `AccessibilityWElevation`
   (else) (338-379).
6. Cohort wrapper install — only when a runtime session is attached AND
   `not settings.turns` (392-397; the non-turns clause is load-bearing,
   C31-NOTE-3).
7. `Centrality` → `ExportAccessibilityResults` (399-407).

Error gates a prepared fast-path must preserve **in order**, per stage:

- `AddNetwork` (Topology.py:195-325):
  - `FileNotFoundError` when `os.path.isfile` fails (210-211);
  - non-LineString geometry `ValueError` with null-geometry naming (255-271);
  - CRS-None warning log branch (273-276) — content-derived, must replay;
  - custom-weight NaN/non-positive guard with logged warning (291-301).
- `BuildAccessPoints` (Topology.py:985-1200):
  - non-Point geometry `ValueError` (1041-1047);
  - network-CRS-set + layer-CRS-None `ValueError` (1048-1053);
  - layer-CRS ≠ network-CRS `ValueError` (1054-1058);
  - weight resolution: `"Count"` sentinel → unit weights (1072-1073); configured
    column NaN warning, recorded for replay (1074-1092); missing-column fallback
    warning, recorded for replay (1093-1104);
  - `uid` = `np.arange(n)` when no uid column else the Series (1105).
- `AddObstacles` (Topology.py:640-770; outside initial admitted domain but
  enumerated for completeness):
  - `snap_to` literal check (669-672, 668);
  - penalty-column presence `ValueError` (681-685);
  - negative-penalty `ValueError` (687-692);
  - direction-value `ValueError` (693-703).

Law: a prepared hit must raise the SAME exception classes for the SAME row
conditions. File existence is observed per row even on a hit (the row's own
`isfile` gate runs before any session consult — design lines 66-84).

## 3. TOCTOU / snapshot semantics (existing, proven)

- `_read_snapshot` (Topology.py:55-60): one `open(rb).read()`; the SAME bytes
  are hashed and decoded. "Hash one version, decode another" is the forbidden
  class (dossier 06, referenced at Topology.py:675-678).
- `_decode_snapshot_gdf` (64-78): the same reader stack as the path reader —
  feather→`gpd.read_feather(pa.BufferReader, columns)`, parquet→
  `gpd.read_parquet(pa.BufferReader, columns)`, everything else→
  `gpd.read_file(io.BytesIO(data), engine='pyogrio', use_arrow=True)` — vs
  `_get_gdf` path dispatch (1493-1498). `keep_columns=None` on all prepare-side
  callers (251, 254, 680, 1037, 1040).
- Equivalence evidence: `tests/cache/invalidation/test_parity_safety_engagement.py:39`
  (`test_warm_cold_off_artifacts_bitwise_equal`) proves the buffer-decoded warm
  path produces bitwise-equal artifacts to the path-decoded cold/off path on the
  suite's workload (geojson network/origins/destinations/obstacles with elevation).
- Stat-before/after alone is NOT proof against concurrent overwrite (design
  lines 70-73); the snapshot-read rule is the enforced mechanism.
- Snapshot bytes are bounded by input file size and released immediately after
  decode (Topology.py:250-252, 1036-1038: `snapshot = None`).

## 4. Existing storage/restore helpers (the reuse surface)

The design's "reuse existing storage/restore helpers" is concrete:

- **Topology stage** (`una:topology:v1`):
  - store: `_store_network_to_cache` (329-372) — WKB payload
    (`_geoms_to_wkb_payload`, 81-101: `output_dimension=coord_dim,
    byte_order=1`; mixed-dimension layers return `None` → **skip, never
    truncate**), flat-blob + int64 lens, arrays {lengths, weights, node_points,
    start_nodes, end_nodes, degrees, z, start/end/min/max_degrees}, metadata
    {n_edges, n_nodes, n_input_features, n_empty_removed, `crs_projjson`
    (**PROJJSON law**: WKT drops datum-ensemble member ids and the loss survives
    into exported feather geoarrow metadata — observed artifact-hash delta,
    Topology.py:363-366), geom_coord_dim, weight_warning, no_crs}.
  - restore: `_restore_network_from_cache` (374-433) — GeoSeries rebuilt with
    `name="geometry"`, crs from PROJJSON, arrays attached verbatim,
    **`edge_sindex` rebuilt from restored geometries** (derived state, 400-403),
    `_stage_key` set, content-derived log lines replayed (no-CRS warning,
    edges count, weight warning, empty-removed, topology counts, hit
    announcement at v=2).
- **Snap stage** (`una:snap:v1`):
  - store: `_store_access_points_to_cache` (437-497) — WKB geometry +
    node_weight + uid (**uid dtype law**: only `ifubU` kinds storable; object
    dtype → skip, identity still recorded, 448-462), snapped payload
    {nearest_edge_point WKB, nearest_edge_id, edge_start_node, edge_end_node,
    weight_to_start, weight_to_end}, metadata {n_points, snapped, uid_is_series,
    crs_projjson, geom_coord_dim, log_extras}.
  - restore: `_restore_access_points_from_cache` (499-539) — AccessPoints with
    GeoSeries(crs), uid Series/array per stored law, snapped arrays;
    `snap_to`/`snapped_node_id`/`direction`/`penalty` deliberately NOT restored
    (502-505) — callers re-derive them (their inputs are inside the stage key).
- **Key closure** (`cache/identity.py`): `stage_key` = sha256 over canonical
  (stage, schema, deps) (135-140); `file_digest` over snapshot bytes (145-147);
  `arrays_digest` dtype+shape+content with None-presence semantics (150-164);
  `geometry_env` = python/platform/machine/profile/package_version +
  {numpy, shapely, geos, geopandas, pyogrio, pyarrow} (186-209) — over-closure
  by design ("under-closure could serve bits a different environment would not
  have produced").
- **Store + single-flight** (`cache/stages.py`): `StageCache.arrays_once`
  (140-187) — hit→serve; miss→cross-process `store.singleflight(key)`; owner
  produces once and commits; loser re-gets; bounded spins (≥3) → local compute;
  per-key wait timeout → local compute; "a cold cached run is the owner's
  compute, not an uncached fallback" (review MAJOR-1, 152-156). A second writer
  can only publish an identical generation — the key closure pins every
  numerical input (158-162).
- **Cohort transport pattern** (`batch/cohort.py`): per-process
  (ContentStore(disk, root), residency-cap) pair keyed by run root
  (131-168); `CohortSession` counters {rows_attempted, builds, faultins,
  resident_hits, singleflight_lost, replays, folds_done, refusals, build_s,
  faultin_s, replay_s} with `mark`/`delta` per-row attribution (171-235);
  RowJob transports `cohort_root` + `cohort_residency_cap_bytes` (runtime-only,
  `batch/worker.py:351-363`); the session is attached as `una._cohort_session`,
  consulted by the RunAccessibility dispatch, and **detached before any state
  transport** (worker.py:401-411).

### Known constraint discovered (integration seam)

`Topology._apply_stage_cache_policy` (Topology.py:188-193) **overwrites**
`self.cache_options` from `settings.execution.cache` at the head of EVERY Add*
call ("the policy follows the row"). A session-injected CacheOptions would be
clobbered. The P02 seam must therefore resolve the stage facade per call — e.g.
the Add* bodies' `_stage_cache_for(self.cache_options)` expression (226, 1013)
resolves through a session-aware resolver that returns the session's StageCache
when a session is attached and admits the stage, and the existing expression
otherwise. With no session attached the expression must be byte-identical to
today's (mode-off law, Topology.py:50-53).

## 5. Producer-state inventory (what preparation must produce)

Per row (fresh UNA per row, `batch/worker.py:324-341`), preparation leaves:

On `Topology`:
- `crs` (network CRS, `self.crs`, 272) — consumed by snap CRS gates and exports;
- `network` (Network): geometry GeoSeries(name="geometry", crs), lengths, weights,
  node_points, start_nodes, end_nodes, degrees, z, start_degrees, end_degrees,
  min_degrees, max_degrees (store set, 342-356), `edge_sindex` (STRtree, derived),
  `_stage_key`;
- `origins` / `destinations` (AccessPoints): geometry GeoSeries(crs), node_weight,
  uid, nearest_edge_point, nearest_edge_id, edge_start_node, edge_end_node,
  weight_to_start, weight_to_end, `_stage_key`;
- `has_clusters = False` reset on every AddOrigins/AddDestinations (560, 576) —
  **must be preserved by any fast path** (stale-cluster KeyError law,
  comment 555-559);
- obstacles state (only when obstacle file configured — outside initial domain).

Derived on demand (not stored): `get_obstacle_arc_penalties()` (776-793; returns
`(None, None)` when no obstacles — 786-787), `get_partial_edge_corrections()`
(795+, computed from obstacle arrays + snap state).

Snap computation itself (ThreadPoolExecutor over chunks; sindex nearest,
`line_locate_point(normalized=True)`, `interpolate(normalized=True)`,
`weights * p` / `weights * (1-p)`) (1116-1183) — deterministic given
(network geometries, weights, points); this is what the payload replaces.

## 6. Identity closure per layer (design lines 40-58, grounded)

- **SourceSnapshot**: exact bytes via `_read_snapshot`; digest `file_digest`;
  reader = the extension-dispatched stack of `_decode_snapshot_gdf`; format from
  extension; original path retained for diagnostics/logs ONLY (restores log the
  source path, 416, 530) — never identity (`test_same_bytes_different_path_still_hits`,
  `test_same_bytes_new_mtime_still_hits`).
- **DecodedLayer**: covered transitively — preparation stores POST-build arrays,
  not decoded frames; decoded-frame identity is pinned by (bytes, reader env).
  No nullable-mask/category-order exposure exists in the prepare chain (the
  chain consumes geometry + named columns only: 286, 306-310, 1063-1105).
- **NetworkPrepared**: file digest + {network_weight_column, network_precision,
  network_load_nodes} + profile + geometry_env (233-244). All three params are
  read by the producer (217-218, 305-315).
- **AccessPrepared(role)**: file digest + parent network stage key + {cost_attribute,
  default_cost, uid_attribute} + profile + env (1019-1030). Beta/radius are NOT
  read by any producer (verified: no beta/search_radius reference in
  AddNetwork/AddOrigins/AddDestinations/BuildAccessPoints/BuildTopology call
  arguments) — matching design line 50 ("Do NOT include beta if no producer
  reads it").
- **EnginePrepared**: out of P02 scope (stage 5 belongs to the cohort/P03 owners).

## 7. Downstream consumers (what a restored bundle must satisfy)

(Status: this section records DIRECT verified reads made for this audit. The
independent read-only enumeration has since completed — its verdict: no
contradiction with §7; extensions recorded in §13. Combined with §13 this
section is final.)

### 7.1 Engine construction (`AccessibilityWElevation.__init__`, Engines/AccessibilityWElevation.py:600-738)

Reads: `topology.network.start_nodes/end_nodes/weights/node_points/z`
(635-644), `topology.get_obstacle_arc_penalties()` (681),
`topology.origins.edge_start_node/edge_end_node/weight_to_start/weight_to_end`
(693-696), `topology.destinations.{node_weight,weight_to_start,weight_to_end}`
(699-703), `topology.get_partial_edge_corrections(...)` for both roles (719,
732), `topology.logger` (796). The CSR graph-view init reads
`topology.network._stage_key`, `topology.origins._stage_key`,
`topology.destinations._stage_key` (854-856) — **a prepared fast-path must set
the SAME `_stage_key` values the CACHE_GRAPH store path sets (the closure
hash), because these values are composed into the engine's RESULT-stage key.**
(Correction from the P02D independent review: the C30 cohort TRACE keys do NOT
chain on `_stage_key` — they hash the prepared CSR array bytes directly,
_trace_access_scratch.py:169-218; the set-same-`_stage_key` requirement
survives via the result-key composition above.) `Centrality` re-reads
`topology.destinations.node_weight` (901).

### 7.2 Exports (Engines/Base.py)

`write_gdf_outputs` / accessibility export reads results plus
`topology.origins.geometry.crs` / `net.geometry.crs` (179, 297, 347, 372, 392),
`obs.uid`/`obst.uid` when present (379-381, 401-403), geometry arrays
(`net.geometry.values`). ODM export reads `topology.origins.uid` /
`topology.destinations.uid` as arrays (237-238). (Correction from the P02D
independent review: Base.py:179 reads `topology.origins.crs` under `hasattr`
/ getattr, but AccessPoints has no `crs` attribute — it yields None on BOTH
paths and is not a fidelity surface.) CRS metadata therefore flows
into exported files from the restored GeoSeries — pinned by
`test_restored_crs_matches_uncached_crs`.

### 7.3 Batch state transport

The worker packs a state bundle for the plan's expected last row
(`batch/worker.py:423-431`: topology + engine pickled). The rehydrated
coordinator state must be the PLAIN engine — the cohort wrapper is detached
before packing (401-411). A prepared session is likewise runtime-only: attach,
use, detach; nothing session-owned may ride the RowResult or the state bundle.

### 7.4 Serial route

`RunBatch(serial)` calls `una.RunAccessibility()` on the instance with no
session attached (`batch/worker.py` attaches only in spawn workers). The
prepared route is engaged only by explicit runtime transport; serial and
session-less runs execute the current code byte-for-byte.

## 8. Mutability classification (initial domain)

| state | kind | shared-payload risk |
|---|---|---|
| Network.geometry / AccessPoints.geometry | geopandas GeoSeries (shapely 2 array) | container mutable; WKB-restore produces FRESH objects per restore — required law (`test_hit_payloads_are_fresh_copies`, `test_mutating_restored_payloads_cannot_poison_later_runs`) |
| network arrays (lengths/weights/node_points/start_nodes/end_nodes/degrees/z/degrees) | numpy arrays | engines `.astype(np.float64)` COPY at engine init (637, 695-703); but defensive copies at restore are the standing law — initial phase COPIES into worker-local arrays |
| uid | numpy array or pandas Series | Series construction at restore (516); fresh |
| edge_sindex | pygeos.STRtree native handle | NEVER transported (derived, rebuilt per restore, 400-403); never shared across processes (design line 95) |
| crs (pyproj.CRS) | immutable value object | restorable from PROJJSON |
| has_clusters | bool flag | per-instance, reset per Add* |
| logger | per-UNA object | never shared; log replay emits through the row's own logger |

## 9. Settings consumed

- During preparation: `data_folder`, `network_file`, `origins_file`,
  `destinations_file` (+ `obstacle_points_file` when set),
  `network_weight_column`, `network_weight_default` (read at 218 but consumed
  only via `network_cost` — actually compared against `"Geometric"` at 283;
  `default_cost`/`network_weight_default` is NOT read in stages 1–3 — verified:
  no reference outside the assignment), `network_precision`,
  `network_load_nodes`, `origin_weight_column`, `destination_weight_column`,
  `origin_uid_column`, `destination_id_column` (via Add* kwargs 545-547, 566-568),
  `execution.cache`, `execution.semantic_profile` (192-193).
- NOT consumed by preparation (must stay out of stage keys): `search_radius`,
  `gravity_beta`, knn/gravity params, output naming, `turns`/`turn_threshold`/
  `turn_penalty` (BuildTurnPenalties call is commented out, 317),
  `elevation` (z comes from the network file itself, 643).
- Post-preparation (engines/exports): `search_radius`, beta/decay/knn/gravity
  families, output folder/naming/delimiter, csv/format flags — all OUTSIDE the
  prepared identity by construction.

## 10. Log/warning replay inventory (content-derived, must replay on hits)

Network stage: no-CRS warning (v1, 274/409-412), "CRS set to {crs}." (v2, 276 —
present when the CRS IS defined; the current CACHE_GRAPH restore does NOT
replay it: NF-1, the prepared restore must replay it — inherited divergence,
amended per the P02D review), "Network added from source …
with N edges" (v1, 278/413-416), weight warning (v1, 296-301/417-418),
empty-removed (v1, 419-423), topology counts (v2, 424-427),
"edge_sindex created" (v2, emitted inside `Network.BuildTopology`,
1778 — AMENDED per the P02R review: this line belongs to the inventory;
the prepared restore replays it (prepared path only — the CACHE_GRAPH
restore keeps its documented inherited divergence), and the prepared
OWNER branch must emit it naturally via its `_build_network_from_gdf`
call), "edge_list_dict created" (v2, 428 — AMENDED per the P02R
review: the prepared owner branch sits above AddNetwork's unconditional
tail line, so it must emit this line explicitly before its return to
keep the produce stream identical to the uncached path).
Snap stage: "Access points are built from source … with N points" (v1,
528-531/1061), stored `log_extras` (NaN weight warning v0, fallback warning v1;
replayed via `_cache_log_replays`, 532-533), hit announcement (v2, 534-538).
The P01M telemetry spans and the batch report remain untouched by replays
(loggers are per-row).

## 11. Failure-order and fallback law (from the design, grounded above)

1. Row validation and error observation are NEVER suppressed by a hit:
   `isfile` gates (210, 552, 571) and `Validation()` run per row before any
   session consult.
2. A preparation failure attaches to the logical row/stage the original
   execution would fail (design line 36) — no eager preparation of future rows
   ahead of earlier commits.
3. Any refusal (unsupported format, mixed dims, uid dtype, store failure,
   owner failure, memory NO_FIT, turns, obstacles in the initial domain) →
   the row takes the CURRENT path with zero observable difference; refusals
   are counted, never silent (cohort precedent: zero-refusal engagement
   asserts in P00).
4. Owner failure → waiters re-elect or compute locally (`arrays_once` loser
   path, bounded); a failed producer never makes waiters skip their own
   validation.

## 12. Memory law (design lines 116-121, grounded in existing sizing)

Residency sizing is ENFORCED law from C31-MINOR-1: per-worker cap
`max(64 MiB, allowance // workers)`; when the floor binds, caps sum to
`64 MiB × workers` (documented exception, pinned by unit tests in
tests/tenx/cache). Admission must reserve build scratch + retained payload +
decoded worker copies + current row + receiver bytes before building
(NO_FIT = explicit pre-execution refusal). The per-process registry pattern
(`_PROCESS_STORES`, cohort.py:158-168) is keyed by root and unbounded across
distinct roots — the prepared session must bound its registry (fixed small
LRU or single-entry-per-root with close) per design line 121.

## 13. Downstream-enumeration appendix (independent cross-check — COMPLETE)

An independent read-only audit enumerated all five engine modules, Base
exports, all seven kernel/scratch modules, the run paths, Settings, the batch
layer, cache/stages and Logger on this same tree, with file:line per claim.
Its four most load-bearing citations were spot-verified against bytes in this
session (Flow.py:1994 clasters_byId write; AccessibilityWTurns.py:501
BuildClusters call; _sparse_access_scratch.py:509-512 dtype flow;
AggregateFlow.py:782 copy=False view) — all confirmed exactly. Findings
integrated above; the deltas beyond §7:

- **State surface oracle**: the batch's own rehydration contract pickles the
  live Topology+engine (`batch/worker.py:427-436`) and restores by attribute
  assignment (`batch/runtime.py:838-846`); the only non-picklable state it
  handles is `edge_sindex` (rebuilt from `geometry.sindex`, Topology.py:403)
  and locks (worker.py:248-268). A restored bundle must match this roundtrip's
  state surface.
- **Dtype/layout are behavior-affecting**: reach output dtype flows from
  `o_terminal_idxs.dtype`; gravity/knn from `adjacency_vector_weights.dtype`
  (_sparse_access_scratch.py:509-512; _trace_access_scratch.py:552-553,
  785-788). Ordered-CSR/sparse/trace admission accepts only exact ndarray
  int64/int32/float64/bool_ C-contiguous representations (_ordered_csr.py:29-47,
  _sparse_access_scratch.py:139-238) — a bundle that changes dtype INT changes
  observable output dtype.
- **Exactly two downstream write-sites to topology-carried containers**, both
  OUTSIDE the v1 admitted domain (flow/turns): Flow.Centrality writes
  `origins.clasters_byId[0]` (Flow.py:1990-1995); AccessibilityWTurns
  triggers `topology.BuildClusters` which overwrites cluster_masks, all three
  clasters_byId dicts, num_clusters, has_clusters (Topology.py:886-960) —
  per-run derivative state, never shared-immutable state.
- **Dead defensive reads** (no assignment sites anywhere): `assigned_dest_id`,
  `id_to_dest_indices`, `cluster_buffer_radius` — a restored bundle need not
  provide them.
- **No input sidecars**: the reader chain opens no companion files (no
  .prj/.qpj/proj-db reads in src/); CRS travels in-payload; batch runtime
  companions (journal/captures/cohort store) are outputs, not inputs.
- **Kernels are clean**: all seven kernel/scratch modules consume arrays
  passed by engines; none reads topology state directly.
- **batch/checkpoint.py and batch/plan.py** read no topology state (journal +
  raw .npy captures + Settings fields only); **batch/runtime.py** calls
  `una.topology.logger.log` (678, 696, 1082) — the logger is live observable
  state — and applies the settings delta at commit (611-620).
- **Composite path re-reads input FILES** via `gpd.read_file(join_path)`
  (UNA.py:873) rather than consuming prepared state; `_build_composite_nodes_gdf`
  reads `topology.network.node_points` + `net.geometry.crs` onto a freshly
  built gdf (UNA.py:899-908).
- **Trace identity already pins the prepared domain by bytes**: trace ID
  hashes prepared CSR array bytes with fixed semantics strings
  ("not-present-in-prepared-domain" for turns/elevation,
  _trace_access_scratch.py:169-218), and replay refuses cutoff bit/dtype,
  dest-weight layout/count/value and structure-digest mismatches (770-824).
- **z extraction is unconditional** in BuildTopology (Topology.py:1604-1608);
  `settings.elevation` gates only whether engines USE z — refining §9's
  "z comes from the network file".

Audit sections §1–12 above were written from direct reads BEFORE the agent
reported (§7's status note records this); nothing in the agent's report
contradicts them, and the items above extend them.
