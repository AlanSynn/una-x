# P02: shared preparation before decode

## Selected first implementation
Implement a PRIVATE run-scoped PreparedSession transported beside CohortSession, not a new
standalone public analysis API. Install it from the existing integrator dispatch. The existing
RunAccessibility call, validation sequence, Add* state transitions and exporters still execute.
Add* hooks may replace only their pure prepared payload production and must replay every
observable transition. Serial/mixed/unsupported cases keep the current trusted implementation.

Start with non-turn accessibility and single-file local sources whose reader/state round trips
are proven. Initially over-key full per-role preparation parameters. Broad format/turn/profile
support comes only with new conformance evidence, not because the tiny fixture passed.
API support is not reduced: unadmitted inputs continue down the existing route.

### Integration seam
Candidate module: batch/prepared.py (private PreparedSession/descriptor/lease types), with
cache/prepared_payload.py only if a distinct reusable storage adapter is justified. Reuse the
existing ContentStore ownership/publication primitives after auditing their semantics. Avoid a
second broad cache framework. UNA integration is owned by one integrator; Topology owns its
state-rehydration adapter. No runtime import of campaign prototypes or test modules.

```
RunAccessibility():
    reset the same result flag
    settings.Validation()                 # never skipped on a hit
    topology.AddNetwork(settings)         # hook before _get_gdf/decode
    topology.AddOrigins(settings)
    topology.AddDestinations(settings)
    original obstacle handling
    construct or restore admitted engine payload
    install existing cohort wrapper
    original Centrality/ODM/export sequence
```

A preparation failure is attached to the logical row/stage that original execution would fail.
Do not eagerly prepare every future row and raise its error before earlier commits. Use lazy
first-consumer single-flight AFTER that row's original validation; workers already admitted by
the planner may speculate privately, but ordered failure/publication remains coordinator-owned.

## Layers, dependency identities and descriptors
1. SourceSnapshot: ordered source bytes and sidecars, source content digest, byte length,
   reader/driver/open options, format/extension, encoding, projection/column selection, producing
   library environment. Retain original source path for diagnostics, not as sufficient identity.
2. DecodedLayer: values/index/schema/CRS and geometries in exact input order. Include nullable
   masks, category order, string encoding, UID Series name/index, coordinate dimensions and NaNs.
3. NetworkPrepared: decoded identity + all consumed network weight/precision/load-node options,
   topology algorithm/profile identity; arrays and original order after BuildTopology.
4. AccessPrepared(role): network identity + decoded point identity + role/snap/UID/weight/default
   selections and all actual role-dependent logic. Do NOT include beta if no producer reads it.
   Destination metric weights can later be separated only after proof they do not affect snaps.
5. EnginePrepared: network/terminal payloads + prepared obstacle/elevation/directional costs and
   ordered-CSR producer identity. Metric parameters and output naming remain separate.
6. PreparedBundle: handles for the above and complete payload manifest, never live worker objects.

The complete key includes profile and CODE CLOSURE, not just package version. A changed reader,
geometry library, rounding behavior, topology helper or default invalidates dependent payloads.
Record graph/terminal value bits and order. Reader equality is not implied by a file digest alone.
No normalized floats, retyped weight keys or content-identical but differently interpreted data.

Descriptor fields: schema_version, kind, dependency_keys, producer_digest, profile,
array_specs(name,dtype,shape,layout,nbytes,digest), geometry_specs(WKB offsets and dimensions),
metadata_digest, payload_digest, storage locator, generation, complete, admitted_domain.
A locator is not identity. Reject absolute/traversing paths in untrusted cache manifests. No
pickle/object-array decode from persistent cache. Runtime-only state transport stays separate.

## Source observation and TOCTOU
The hash must identify the SAME bytes the producer decodes. Do not hash a path then separately
read the mutable file. Reuse _read_snapshot/_decode_snapshot_gdf only after proving the path vs
buffer reader retains driver inference, FIDs, CRS metadata and error behavior on the domain.
Stat-before/after detects some changes; it is NOT proof against concurrent overwrite/ABA.
An open file descriptor is NOT an immutable snapshot. A file lock works only with cooperating
writers. Hard-linking a mutable source does not freeze it.

For mutable path inputs, preserve the established per-row/stage observation policy and re-read/
hash at the necessary boundary, then reuse PREPARATION only for the verified byte identity.
This still removes expensive repeated parsing/snap work. Permission/existence/errors required by
that row must still be observed. Snapshot bytes must be bounded/spooled with the copy charged.
For explicitly immutable/snapshot-owned inputs, one hash/decode per identity per call is admitted.
Do not silently impose cohort-wide snapshot semantics on mutable callers. A freeze manifest is
not itself immutability enforcement. Unknown concurrent mutation cases use the original path.

First admit single-file .geojson/.feather/.parquet only where tested. Shapefile/prj/dbf/shx,
archives, GDAL virtual files, mutable GeoDataFrames and custom reader hooks need a complete
multi-resource identity and reader proof; otherwise fallback. Missing columns or nonfinite data
keep current validation/failure order rather than being silently cleaned during preparation.

## Rehydrate without mutation leaks
Audit EVERY attribute read by engines, later API calls, observers/compositors, exports and final
state. Existing topology/snap serializers are a STARTING POINT, not a complete-state theorem.
Preserve original class, UID/index name and dtype, crs_projjson, geometry ordering/dimensions,
network arrays, terminal costs, reset flags and content-derived warnings. Existing WKT conversion
previously changed Feather metadata, so retain PROJJSON and exact exporter schema.

Publish internal payload once. Reconstruct a worker-local geometry/STRtree once per dependency
identity when proof permits. Never transport GDALDataset, GEOS/STRtree pointers or open native
handles across processes. WKB/typed arrays + metadata are the first portable transport.
Shapely 2 geometries can be immutable, but their NumPy/GeoSeries/container references are not.
Fresh row wrappers must not share mutable result columns, Settings, locks, loggers or writable
arrays with other rows. Initially COPY public writable arrays; later read-only internal aliasing
requires lease ownership and consumer audit. WKB reconstruction must preserve the baseline
representation; unsupported mixed dimensions/extension arrays fall back, not truncate.

On final state rehydration, materialize independent ordinary objects before closing preparation
leases. Later user mutation of the returned topology must not alter shared payloads or another
batch. Preserve original per-row flag reset and obstacle clearing/retention semantics; fixing a
scientific stale-state bug is a separately versioned corrected-profile task, not incidental cleanup.

## Single-flight and memory state machine
ABSENT -> BUILDING(owner token) -> COMPLETE(manifest-bound immutable payload) -> LEASED.
Owner failure/cancellation discards incomplete bytes; waiter wakes and either safely re-elects
or takes original compute path before public mutation. Never spin forever or accept partial data.
Cross-process store lock covers publication, not metric replay. Independent keys can progress.
A failed producer may not make waiters suppress their own validation/errors. Record cold build,
verified hit, contention wait, corruption rebuild and refusal as DIFFERENT counters.

Reserve build scratch + retained payload + decoded worker copies + current row + receiver bytes
before admission. Do not let a 64 MiB minimum allocation exceed the real budget. NO_FIT is an
explicit pre-execution refusal, not permission to use the old large fallback inside a cap claim.
Memory margin and total CPU budget are shared across preparation, metrics, writers, parent and
native pools. Release leases explicitly; owner-only unlink after all readers finish. Refuse new
leases during shutdown. No unlimited process-global registry keyed by distinct batch roots.

## Implementation increments and acceptance
A. Reader/state equivalence fixtures and serializers; no dispatch change.
B. Worker-local prepared reuse across rows: at most W first builds, useful low-risk ablation.
C. Run-wide content single-flight: one decode/build per stable identity plus at most W geometry
   reconstructions; move no work outside RunBatch. Count actual preparations, not only trace hits.
D. Engine payload reuse only if its measured cost justifies it; no scope expansion by default.

Every increment compares source/installed public outputs, Settings after each commit, aliases,
error prefixes, cancellation/resume and exact input changes. Mandatory invalidations: changed
bytes with unchanged size/mtime; changed weight/precision/UID/CRS/reader/profile; mixed rows; stale
manifest; corrupted WKB; owner death; memory refusal; final object mutation; file disappearance.
Acceptance needs a measured public benefit. No new speedup is established by this design.
