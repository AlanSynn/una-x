# P02D — PreparedBundle consumer/producer audit (pre-code)

```json
{"schema": "una-10x-p02d-audit-1", "task": "P02D", "generated": "2026-10-06",
 "tree": "perf/una-10x worktree main @ P01M post-commit (cb023d4)",
 "status": "DRAFT_FOR_PROOF_REVIEW"}
```

## 0. Method and verification legend

Three independent read-only audits were commissioned (Explore agents `store-audit`,
`topo-audit`, `integ-audit`) over the current tree, and the coordinator personally
re-verified every load-bearing claim against source bytes before it entered this
document. Provenance tags:

- **[V]** — coordinator-verified against file bytes this session (Read output on record).
- **[A]** — auditor-reported with file:line; not personally re-verified unless also tagged [V].
- **[F]** — frozen P00 evidence (`campaign_runs/una_10x/tenx-20261003T1212Z/p00_profile/`).

All paths relative to `src/urban_network_analysis/` unless noted. This audit is
pre-code: P02D's completion criterion is an independent proof review accepting the
admitted domain and the state/reader/alias/TOCTOU rules BEFORE any implementation.

## 1. Scope

P02D must enumerate every source observation, sidecar, reader parameter and piece of
mutable state on the per-row preparation path, then specify the private
admission/identity/lease/restore seams a shared PreparedBundle layer would use.
This document records the measured consumer/producer surface. The contract itself is
`prepared_contract.json` (companion file). Design authority: `designs/02_prepared_bundle.md`,
`IMPLEMENTATION_MAP.md` ownership table, `SOURCE_AND_ERRATA.md` E1–E8, CLAUDE.md parity laws.

The performance target this serves (P00 facts, [F]): the observed 32-row cohort
re-reads and re-parses all three admitted inputs in every row — 79,940,126 input
bytes × 32 rows — and serial service is 75.4220% `read_arrow`+`next`+`from_wkb`;
the trace already shares topology build (builds=1, identities=1, replays=32,
singleflight_lost=31) but per-row load/parse is not shared. That load/parse sharing
is exactly what P02 must add, guarded.

## 2. Producer surface — Topology.py Add* (what a prepared layer must reproduce)

### 2.1 AddNetwork(settings) — Topology.py:195-325

Settings fields read, in order [V]:

1. `settings.data_folder`, `settings.network_file` → `source_file` join (:208);
   `FileNotFoundError` if missing (:210-211).
2. `settings.network_weight_column` (:217; default `"Geometric"`, Settings.py:43).
3. `settings.network_weight_default` (:218) — **read but never used** [V: read site;
   A: no-use claim, consistent with :295 substituting geometric length and the key
   at :233-244 omitting it]. Dead parameter; excluded from identity inputs.
4. `settings.execution.cache` + `settings.execution.semantic_profile` via
   `_apply_stage_cache_policy` (:225; :188-193 — "the policy follows the row, not
   the instance") refreshing `self.cache_options` / `self._stage_profile` per call [V].
5. `settings.network_precision`, `settings.network_load_nodes` (:237-238 stage key;
   :315 BuildTopology call) [V].

First side effect: `self.has_clusters = False` (:215 — "A reloaded network invalidates
previously built clusters") [V].

Read/decode [V]:

- Cache path: `snapshot = _read_snapshot(source_file)` (:232; :55-61 — reads the
  whole file's bytes once: "Snapshot the exact bytes a stage will consume, for
  hashing AND decoding (dossier 06 TOCTOU: never hash one version and decode
  another)"), then `gdf = _decode_snapshot_gdf(source_file, snapshot, None)` (:251;
  :64-78 — `.feather` → `gpd.read_feather(pa.BufferReader(data), …)`, `.parquet` →
  `read_parquet`, else `gpd.read_file(io.BytesIO(data), engine='pyogrio',
  use_arrow=True)`).
- Uncached path: `self._get_gdf(source_file, keep_columns=None)` (:254; static
  :1467-1501, same extension dispatch). Two latent defects, neither triggered by
  Add* (keep_columns always None from Add*) [A→V at proof review]: the
  GeoDataFrame-source branch assigns without `return` (:1477-1479) so a GeoDataFrame
  input then raises `ValueError` via the `Path(source)` fallback (:1477-1483) —
  stronger than "no return"; `keep_columns += ['geometry']` mutates a caller's list
  (:1474-1475).

Existing input identity [V]: `net_key = _ckid.stage_key("una:topology:v1", …)`
(:233-244) over `{file_sha256: file_digest(snapshot), params: {network_weight_column,
network_precision, network_load_nodes}, profile, env: geometry_env(profile,
_PACKAGE_VERSION)}`. Helpers: `cache/identity.py:135 stage_key` ("sha256 over the
canonical (stage, schema, deps) closure"), `:145 file_digest`, `:186 geometry_env`
(python/impl/platform/machine/profile/package_version + libs shapely/GEOS/geopandas/
pyogrio/pyarrow). On hit: `_restore_network_from_cache(source_file, hit)` + return (:245-248).

Validation [V]: geometry types (:255-271) — non-LineString (null geometry yields NaN
`geom_type`, cast to str, named "missing/null geometry") → `ValueError` "Network file
must contain LineString (line/polyline) geometries". `self.crs = gdf.crs` (:272); no
CRS → logger warning v=1 (:274); `reset_index(drop=True)` (:277).

State written, in order [V]: `self.network.geometry` (:280);
`self.network.lengths = …astype(np.float64)` (:281); weights (:282-302 —
`"Geometric"` → `self.network.lengths.copy()` :284 explicit defensive copy; custom
column → `gdf[network_cost].values.astype(np.float64)` :286, missing column = raw
KeyError; guard `bad = ~np.isfinite(weights) | (weights <= 0.0)` :291 → rowwise
geometric-length substitution + warning pinned in `cache_weight_warning` :296-301);
node-ID load (:303-313 — if `network_load_nodes` (default True, Settings.py:47),
reads `_node_start_id`/`_node_end_id` into `network.start_nodes/end_nodes` int64) —
**dead effect** [V]: `BuildTopology` re-derives `node_points, start_nodes, end_nodes,
degrees` unconditionally from geometry (:1590-1596), overwriting any loaded columns.

### 2.2 AddOrigins(settings) — Topology.py:541-561 [V]

`source_file` join (:544); `cost_attribute` = `settings.origin_weight_column` else
`"Count"` (:545); `default_cost` (:546); `uid_attribute` = `settings.origin_uid_column`
else None (:547); `_apply_stage_cache_policy(settings)` (:548); `os.path.isfile` gate
(:552-553, `FileNotFoundError`); `self.origins = self.BuildAccessPoints(source_file,
cost_attribute, default_cost, uid_attribute, label="Origins")` (:554);
`self.has_clusters = False` (:560, comment: fresh origins object has no
clasters_byId; without the reset, row 2+ of a clustered batch raises KeyError).

### 2.3 AddDestinations(settings) — Topology.py:563-577 [V]

Mirror of AddOrigins: join (:565), `destination_weight_column`/`Count` (:566),
`default_cost` (:567), `destination_id_column` (:568), cache policy (:569),
isfile gate (:571-572), `BuildAccessPoints(…, label="Destinations")` (:573),
`has_clusters = False` (:576).

### 2.4 AddObstacles(settings) — Topology.py:640+ [V]

Gated by `settings.obstacle_points_file` truthiness at the UNA.py:333-334 call site
[V] and again inside (:654-660 sets `self.obstacles = None` + penalty arrays when
falsy). Reads penalty column, direction column, snap target. **Not in the observed
cohort** ([F]: p00_env.json workload defines network/origins/destinations only), so
the initial admission domain excludes it; it remains on the public path untouched.

### 2.5 BuildAccessPoints — the origins/destinations producer

Snap mechanics [A + partial V]: per-point `sindex.nearest` + `line_locate_point`
snap (:1129, :1141 region); weights from the cost attribute, with the missing-column
fallback being **unit weights `np.ones(len(gdf))`** (:1073/:1104 — corrected at proof
review; `default_cost` is read :546/:567 and threaded into the snap key :1024 but is
NEVER used arithmetically — same read-never-used class as `network_weight_default`);
UID handling with dtype guard — uid dtype must be in `{"i","f","u","b","U"}` else the
key is attached and the store is SKIPPED ("miss never wrong hit", :448-462) [V of the
guard]. CRS gate (:1048-1058): no reprojection ever; if the layer CRS is None the
gate is skipped entirely [V]. Weight-column check (:1072-1104) [V at proof review].
Empty-layer path:
`max_workers=0` → else-branch → snap fields None (:1116-1119, :1193-1194) [V of the
dispatch shape]. Stage-cache hit path logs and returns the restored points layer
(:528-539) [V].

Stage-cache keys for these layers compose the same closure family as the network key
(file sha + reader params + profile + geometry_env) with `has_clusters=False` state
semantics [A].

### 2.6 BuildTopology — derived arrays

`_edge_start_and_end_points(geometry, precision)` (:1590-1593) then unconditional
`node_points/start_nodes/end_nodes/degrees` derivation (:1595-1596), degree products
(:1598-1602), optional z extraction (:1604-1608), optional redundant-edge discard
(:1612-1619), `edge_sindex = geometry.sindex` (:1623) [V]. int32 start/end node
arrays and coordinate rounding at `network_precision` [A].

### 2.7 Serializers and their coverage limits

Topology stage-cache serializers exist for the network layer and the snapped
access-point layers and are asserted reusable by the errata only AFTER this consumer
audit ("existence ≠ full state coverage", SOURCE_AND_ERRATA) [A]. `to_pickle` legacy
path exists [A] but persistent-cache pickling is banned by designs/02 ("no pickle
from persistent cache"). CRS is stored as PROJJSON, not WKT (:362-366 — "to_wkt()
drops datum-ensemble member ids … observed as an artifact-hash delta") [V].

## 3. Consumer surface — RunAccessibility ordered sequence

`UNA.RunAccessibility` — UNA.py:316-409 [A, spot-verified V at :320/:322/:324-326/
:333-334/:336/:338/:352-360/:379/:388-397/:399/:401/:403-407]. Serial batch loop
calls it once per row (UNA.py:269-297: `self.settings = s` then
`self.RunAccessibility()`) [A].

| # | Step | Location | Notes |
|---|------|----------|-------|
| 1 | `self.has_centrality_results = False` | UNA.py:320 | result-flag reset, first statement |
| 2 | `self.settings.Validation()` | UNA.py:322 | pure attribute check/coercion; **re-reads no files** [V: Settings.py:480 has no file I/O; only open() at :432/:476 in Load/SaveSettings] |
| 3 | `self.topology.AddNetwork(self.settings)` | UNA.py:324 | §2.1 |
| 4 | `self.topology.AddOrigins(self.settings)` | UNA.py:325 | §2.2 |
| 5 | `self.topology.AddDestinations(self.settings)` | UNA.py:326 | §2.3 |
| 6 | obstacles if `obstacle_points_file` set | UNA.py:333-334 | §2.4; absent in cohort |
| — | `has_model = False` | UNA.py:336 | dead — assigned, never read [A] |
| 7 | engine branch on `self.settings.turns` | UNA.py:338 | turns=True forces `cluster_parallel=True` (:352), `cluster_workers=NUM_THREADS` (:357), `clasters=max(1,min(NUM_THREADS,n_origins))` (:358) → `AccessibilityWTurns` (:360); turns=False → `AccessibilityWElevation` (:379) for BOTH elevation True/False (comment :362-366) |
| 8 | cohort wrap: `session` getattr; `if session is not None and not self.settings.turns:` → `CohortGraphEngine(graph_engine, accessibility, session)` | UNA.py:392-397 | ONLY consumer of `_cohort_session`; "not turns" clause LOAD-BEARING (C31-NOTE-3); wrapper batch/cohort.py:452-477 — `o_access` offers cohort route then falls back to inner view on refusal, `o_scope` delegates, `__getattr__` passes compact producer-kernel arrays straight through |
| 9 | `self.accessibility.Centrality(self.settings)` | UNA.py:399 | AWE:913; WTurns:559; no OD matrix on this path |
| 10 | `self.has_centrality_results = True` | UNA.py:401 | |
| 11 | default output_folder if None | UNA.py:403-404 | usually already set by Validation (Settings.py:653-654) or RunBatch override (UNA.py:272-274) |
| 12 | `ExportAccessibilityResults(settings, folder_prefix="accessibility_", file_name=output_file_name)` | UNA.py:406-407 | Base.py:137 |

No cleanup after export (UNA.py:409 "should i clean up … maybe not") and no reuse
check: every call re-runs Add* unconditionally, even for byte-identical inputs [A].
That unconditional re-run is the measured waste P02 targets.

## 4. Post-construction engine reads (restore-validity constraints) — COMPLETE [V]
*(for the admitted domain; see the AWE:681 caveat below)*

Coordinator-verified read of `engines/AccessibilityWElevation.py` (grep over all
topology/settings/logger attribute reads + direct Read of :700-744, :893-942).
Independently confirmed by the proof review with ONE omission recorded: the census
below does not list **`topology.get_obstacle_arc_penalties()` (AWE:681)**, which
reads `topology.obstacles` (None-check, Topology.py:786-787) and, WHEN obstacles are
present, further reads `network.start_nodes/end_nodes` (Topology.py:791-792) and the
penalty arrays. Benign in the admitted domain because obstacle rows are excluded
from admission (`obstacles` stays None on any prepared hit, so no live field is
skipped) — **recorded forward constraint: any future obstacle-domain extension
RE-OPENS restore validity and must re-run this census.**

Ctor-copy vs live-read split (integ-audit full report, coordinator-checked): the
AWE constructor COPIES its inputs into `CompactNodeView` (adjacency arrays
:798-810, o/d terminal arrays :812-817) and never re-reads them; live object reads
after construction are only `topology.destinations.node_weight` (:901/:945),
`topology.logger`, `topology.cache_options` (:931), the `_stage_key` identity
stamps + `_stage_profile` (:854-881), export-time `origins.geometry`/`origins.crs`
(Base.py:155-156/179), and ctor-time `num_threads`/`num_clusters` (Base.py:125-126).

A
prepared/restore path must leave EVERY surface below populated and semantically
identical to the produced path — a restore that skips a field the engine later
reads is a correctness defect, not an optimization. This is the "existence ≠ full
state coverage" errata constraint in operative form.

Graph-engine construction / o_access compute (AccessibilityWElevation.py:613-744):

- guards: `topology.network is None` (:627), `topology.origins is None` (:629),
  `topology.destinations is None` (:631)
- `topology.network.start_nodes` / `.end_nodes` (:635-636), `.weights` →
  `.astype(np.float64)` (:637), `.node_points.shape[0]` (:638), `.z` (:643-644,
  elevation directional weights — None allowed only when engine built with
  elevation off)
- `topology.origins.edge_start_node` / `.edge_end_node` / `.weight_to_start` /
  `.weight_to_end` (:693-696)
- `topology.destinations.edge_start_node` / `.edge_end_node` / `.node_weight` /
  `.weight_to_start` / `.weight_to_end` (:699-703)
- `topology.get_partial_edge_corrections(topology.origins|destinations,
  for_origins=…)` (:719-721, :732-734) — obstacle corrections; reads
  `topology.logger` (:725, :738 — the TOPOLOGY's logger, not the engine's)

o_access result stage cache (AWE:837-897, :899-942):

- `_stage_key` via getattr on `topology.network`, `topology.origins`,
  `topology.destinations`, and obstacles (:854-862)
- `topology._stage_profile` (:881)
- `topology.cache_options` (:931) gating `cache.arrays_once(result_key, …)` (:937-938)
- `self.logger` (:921+), `settings.search_radius` and gravity params at Centrality
  (:913-942); `topology.destinations.node_weight` re-read in the uncached producer
  (:901)

Per-row input re-reads on the produced path occur at pyogrio `use_arrow`
(Topology.py:1498) / `read_feather` (:1494); snap cost sits in `sindex.nearest`
(:1129) + `line_locate_point` (:1141); network build at :315/:1590+.

**Restated contract consequence:** restore must populate
network.{start_nodes, end_nodes, weights, node_points, z},
origins.{edge_start_node, edge_end_node, weight_to_start, weight_to_end,
geometry, crs}, destinations.{…, node_weight, …}, each layer's `_stage_key`,
topology.{_stage_profile, cache_options, logger}, and the obstacles identity —
or fall back to the produced path.

## 5. Export surface (what must not change)

- Exporter columns are ONLY origins geometry + engine arrays — no UID, no original
  input attributes [A, headline]. Any preparation that changes origin row order,
  count, dtype, or geometry values therefore changes public bytes.
- Reach dtype differs by engine: int64 on WElevation vs float64 on WTurns [A,
  headline]. Bitwise parity laws (CLAUDE.md) forbid normalization.
- CRS: PROJJSON on export (:362-366) [V]; no reprojection anywhere (:1048-1058) [V].
- Naming/timestamps: unchanged duty (CLAUDE.md: actual clock/naming policy preserved).

## 6. Storage primitives available for reuse (store-audit verdicts) [A, spot-verified]

`cache/store.py` ContentStore: payload-agnostic primitive; caller-owned 64-hex
sha256 keys; tmp→rename atomic publish with fsync; singleflight (default 60 s;
P00 cohort configured 600 s); corruption → miss + quarantine; `schema_version`
mismatch = miss; re-put creates a new generation. It does **NOT** enforce
content-address truth — a caller may put payload X under key K derived from Y.
The refuse-different-bytes law therefore lives with the caller; the in-tree pattern
is `cache/trace_residency.py:257-283` (admitted bytes pinned; a residency restore
refuses bytes whose hash differs from the identity it claims). StageCache +
`cache/identity.py` provide the stage-key closure (§2.1).

Design consequences for P02 (contract inputs):
1. Reuse ContentStore as the durable prepared-payload home; do not fork a new store.
2. Prepared identity must compose `file_sha256 + reader params + profile +
   geometry_env(+ package closure)` exactly like :233-244, extended with the
   preparation-stage name/schema — NOT path/mtime/ndarray identity (CLAUDE.md).
3. The prepared admission must verify bytes-against-identity at lease time
   (refuse-different-bytes), because the store will not do it.

## 7. Admitted domain — frozen P00 observed facts [F]

From `p00_profile/p00_env.json` (workload.data_inputs) and
`configuration_screen.json`:

| Input | Bytes | sha256 (prefix) |
|-------|-------|-----------------|
| 20260703_PercLenNetwork_InnerCore.geojson | 71,290,787 | 83ec3e724693bab4… |
| Cambridge_building_centroids.geojson | 5,982,662 | 14999756f30cc0f3… |
| MA_transit_stations.geojson | 2,666,677 | 52080688bb2678e1… |

All three are `.geojson` (pyogrio `use_arrow` decode path), single-file, same
`data_folder`. Reference outputs: 32 rows × {feather + geojson} = 64 files,
147,830,672 B. Cohort row settings vary per-row parameters (search radius etc.);
`turns=False` throughout (engine = AccessibilityWElevation everywhere; cohort wrap
active). Trace facts [F]: builds=1, identities=1, replays=32, folds=472,032,
singleflight_lost=31.

**Admission domain (per designs/02:82-85):** single-file `.geojson`/`.feather`/
`.parquet` network/origins/destinations inputs, admitted ONLY where reader-
equivalence fixtures prove each format; engagement begins with the observed
`.geojson` cohort shape and extends only on new conformance evidence. Over-key
initially (full per-role preparation parameters, designs/02:11); lazy
first-consumer single-flight AFTER original validation; mixed-dimension inputs
fall back, not truncate (designs/02:101). Shapefile sidecars, archives, GDAL
virtual files, mutable GeoDataFrames and custom reader hooks stay on the produced
path pending a complete multi-resource identity and reader proof.

## 8. Errata constraints binding the contract

- **E8 (C31-NOTE-1):** `result_key` float64 coercion must be fixed BEFORE typed
  metric reuse keys on it (P03 scope; noted here because prepared identity must not
  inherit the coercion).
- **E6:** logistic gravity does not read beta; KNN weights are branch-specific —
  per-row parameter identity must not assume beta participates uniformly.
- Topology serializers reusable only after this consumer audit (this document is
  that audit input; coverage limits in §2.7).
- Read/validation/failure chronology is still required on cache/prepared hits
  (CLAUDE.md): a hit may not skip the validations the produced path performs,
  and failure order must match §10.
- Immutable internal leases do not permit public mutable aliasing (CLAUDE.md):
  prepared payloads may be shared immutably; anything handed to engines/consumers
  that they could mutate must be per-consumer.

## 9. Mutable-state inventory and alias hazards (preparation path)

| State | Site | Behavior | P02 rule |
|-------|------|----------|----------|
| `topology.has_clusters` | :215, :560, :576 | reset False by every Add* | prepared path must preserve reset semantics per row |
| `topology.cache_options`, `_stage_profile` | :188-193 via :225/:548/:569 | refreshed per row from settings.execution | prepared admission reads the ROW's policy, never a stale instance field |
| `has_centrality_results` | UNA.py:320/:401 | reset at entry, set after Centrality | untouched by preparation |
| `network.weights` | :284 | defensive copy from lengths | prepared payload must keep per-consumer mutable copies where produced path copies |
| `network.start_nodes/end_nodes` (from columns) | :303-313 | dead — overwritten by BuildTopology :1595-1596 | prepared network payload may omit column-loaded node ids; derived arrays are the live state [V] |
| `_get_gdf` keep_columns list | :1474-1475 | mutates caller's list | not triggered by Add* (None); preparation must not start passing a list |
| Validation mutations | Settings.py:480-654 [V — full read] | ZERO file I/O. Mutates exactly: `knn_weights` (coerce to 1-D ndarray :509-521), `flow_gravity_cap` (string spec normalized :566), `origin_destination_id_column` + `destination_id_column` (both overwritten with `flow_route_id_column` :618-619), `output_folder` (default :653-654). Raises on unexpected attrs (:504-506) and accumulated errors (:650-651) | prepared admission binds POST-Validation settings only — concrete [V] case: `destination_id_column` is Validation-mutated (:619) AND AddDestinations-read (:568); Validation runs first (UNA.py:322 before :326), so the producer-read value is the post-Validation value and that is what identity keys |
| origins/destinations objects | :554/:573 | fresh per row | per-row fresh mutable outputs preserved; only immutable prepared inputs shared |
| `_cohort_session` | UNA.py:392-397 | per-row wrap after engine ctor | preparation sits strictly BEFORE decode; cohort seam untouched |

## 10. Failure order (current; prepared path must reproduce it on hits)

Per row, observed failure precedence [V unless noted]:

1. `Settings.Validation()` — pure; raises on unexpected attrs / bad values
   (Settings.py:480-654; no file I/O) [V].
2. `AddNetwork` file existence (:210-211) [V] → snapshot read → decode errors →
   geometry-type validation (:255-271) [V].
3. `AddOrigins` existence (:552-553) [V] → `AddDestinations` existence (:571-572) [V].
4. `BuildAccessPoints` CRS/weight checks (:1048-1058, :1072-1104) and snap-stage
   failures [A].
5. Engine ctor / Centrality / export failures [A].

Integrity gates are keyed to fixed-input deterministic returns; run-parameter probe
returns are data, not gate failures (campaign data-vs-integrity gate class). On a
prepared hit, steps 1-4 must still execute their validation/failure chronology
against the SAME bytes (re-hash or lease-verified bytes), in the same order —
hitting the prepared layer may not reorder or skip a failure the produced path
would have raised.

## 11. Design implications feeding prepared_contract.json

- **I1** Prepared identity = stage_key closure over {file sha256 of admitted bytes,
  reader params (extension-dispatched), cost/uid/default attrs actually read by the
  Add* producer, profile, geometry_env(package closure)} + preparation schema tag.
- **I2** TOCTOU law inherited verbatim: hash the SAME bytes that are decoded
  (:55-61); prepared admission re-verifies bytes against identity at lease time
  (refuse-different-bytes per trace_residency.py:257-283 pattern).
- **I3** State machine ABSENT→BUILDING(owner)→COMPLETE→LEASED with crash-orphaned
  BUILDING treated as absent (store singleflight already covers concurrent build;
  cohort budget 600 s).
- **I4** Restore validity: every post-construction engine read (§4) and every §10
  validation must hold on the restored path; restore populates the same attribute
  surfaces the produced path populates (topology serializers only where their
  coverage is proven by fixtures).
- **I5** Per-consumer mutability: arrays the produced path defensively copies
  (:284) or engines may mutate are copied per consumer from the immutable prepared
  payload; no public aliasing of internals.
- **I6** Validation chronology preserved: prepared lookup happens only AFTER the
  produced-path validations that would have run at that point (lazy first-consumer
  single-flight AFTER original validation, designs/02).
- **I7** Initial domain: single-file inputs with `.geojson`-first engagement (the
  contract's admission_domain defines the conditional three-format domain); anything
  else falls back to the produced path (mixed dimensions, feather/parquet until
  tested, obstacles, multi-file).
- **I8** Instrumentation seam: preparation increments add opt-in Add*-boundary
  phase hooks (designs/01 deferred scope) so H1 decomposition can see
  source_decode/topology/snap without a second channel.

## 12. Post-verdict amendment (2026-10-06)

Independent proof review (`prepared_proof_review.json`, fresh reviewer): verdict
**ACCEPT_WITH_NOTES**, zero blocking defects, reviewed against
`prepared_contract.json` sha256 `047e507970346c39d92199af45cd147e878ca6df216be5e15e089b79dc0204e5`
and this file's sha256 `221baa83c424ea286465d94b1d2822c6f53e589ae05bd026c1c0bf9dd738060c`
(both re-hashed unchanged on disk before this amendment; those pinned bytes are the
verdict's binding identity). This section and the annotations above record the
reviewer's requested remedies and post-verdict evidence folds. Rule content of
`prepared_contract.json` is unchanged except its `post_review` provenance record.

Remedies applied (reviewer-requested, pre-implementation):

1. §2.5 weight-fallback prose corrected: missing-column fallback is unit weights
   `np.ones` (:1073/:1104); `default_cost` is read and keyed (:546/:567/:1024) but
   never used arithmetically — keep it in the prepared identity (conservative,
   matches the existing snap key). Refuted claim was non-load-bearing [A] prose.
2. §4 census annotated with the `get_obstacle_arc_penalties()` (AWE:681) omission
   and the forward constraint that an obstacle-domain extension re-opens restore
   validity.
3. Citation imprecisions corrected: uid guard :448-462 (was :452-462); snap
   restore log-return :528-539 (was :535-539); `_get_gdf` GeoDataFrame branch
   raises `ValueError` via the `Path()` fallback (:1477-1483), stronger than
   "no return".

Post-verdict evidence folds from the integ-audit full report (additive precision,
no rule change; P02I consumers read both):

- Engine-level stage-cache key surfaces already exist and close over exactly the
  identity surfaces this contract specifies: `una:awe:engine:v1` (AWE:869-883 —
  network/origins/destinations stage keys + obstacle identity + elevation) and
  `una:awe:o_access:v1` (AWE:884-893 — gravity_beta/plateau/logistic_midpoint/
  decay_constant/knn_decay/knn_weights/search_radius). The prepared key composes
  with, never duplicates, these; beta lives ONLY in the engine result key
  (open-item #3 confirmation, AWE:886).
- WTurns differs structurally: Centrality re-reads snapped arrays from topology
  EVERY call (WT:567-575), builds its state graph at ctor (2·n_edges states,
  :250-264), lazily rebuilds clusters (WT:500-501), and has NO stage-cache path —
  consistent with turns rows being outside the initial admission domain.
- Exporter precision: `ExportAccessibilityResults` (Base.py:137-204, distinct from
  `write_gdf_outputs`); default `output_wStamp=True` creates a fresh
  `accessibility_YYYY-MM-DD_HHMM` folder per call (Base.py:36-39); format flags
  output_feather/output_csv/output_geojson default True/True/False
  (Settings.py:106-108); to_feather :181-182 / to_csv :194 (sep from
  settings.csv_delimiter) / to_file fiona-GDAL :201 with NO pyogrio
  COORDINATE_PRECISION pinning on this path; column construction order :155-173
  (geometry, reach, gravity_logistic, gravity_exponential, knn+decay-suffix);
  `RunBatch` renames `Results`→row name at UNA.py:287-288 before the call.
- Cost-site precision: `_get_gdf` :1468-1501 (:1494 read_feather, :1496
  read_parquet, :1498 pyogrio `use_arrow`); in-repo `shapely.from_wkb` appears
  ONLY on cache-restore paths (:104-109, used :386/:512/:520) — the P00 profile's
  `from_wkb`+`next` cost is pyogrio's arrow WKB decode INSIDE
  `gpd.read_file(use_arrow=True)`, external to this repo; `BuildTopology` re-runs
  per AddNetwork (:315) with `edge_sindex` rebuild (:1623/:1764); snap uses
  ThreadPoolExecutor (:1121), nearest (:1129), line_locate_point (:1141-1147),
  interpolate (:1167); an obstacle layer adds a second raw decode (:679-680).
- calculate_* flags gate only result ASSIGNMENT, not the arithmetic
  (AWE:851-852); assignments :952-966; `od_matrix` stays None on this path
  (class attr AWE:591); `resolved_gravity_cap` is RunFlow-path only (UNA.py:411-524).
- Logger is cumulative append-only across rows (`total_time +=` Logger.py:33,
  `log_list.append` Logger.py:43) with verbosity fixed at ctor; cache hits replay
  pinned lines (:112-118) — preparation must preserve this chronology surface.
