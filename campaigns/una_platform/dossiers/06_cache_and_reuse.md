# D06: Cache identity, safe reuse, and duplicate-work removal

## Numerical identity is not export identity
Define typed content-addressed keys for decoded inputs, cleaned geometry/topology, ordered CSR,
point snapping, directional costs/turn graph, destination gradients, reusable coefficient
blocks and complete numerical results. Cache only stages measured to repay hashing/store/load.
A public cache API exposes configuration, clear/inspect/stats, bytes, generation and hit reasons.
Reused computations still execute required validation and publish requested formats/metadata.

## Required numerical key closure
Include ordered input content and index IDs, geometry coordinates/CRS/precision, snapping and
redundant-edge policy, source column values, dtype/shape/byte-order, weights/alpha/beta/plateaus,
elevation/turn/obstacle/observer dependencies as applicable, OD order/radius/detour/route model,
semantic profile + bug-fix version, arithmetic/reduction partition, compiler/math fingerprint
when bits may differ, and output-dependent numerical demand. No path+mtime-only key. No backend
name-only key. Two backends share results only when certified numerical identity is the same.

Separate publication key: filename/folder, timestamp/provenance, output format/schema options,
compression, geometry rounding, licensing/provenance attribution. Changing this should not
force a numerical miss if the numerical closure is identical. Changing a numerical dependency
must never hit because filenames happen to match.

## Mutable objects and TOCTOU
Zonal gdf, network tables, Settings and input files are mutable. Either snapshot and hash
owned immutable data, or rehash the relevant bytes before every reuse. Object ID or user
promise of immutability is insufficient as the automatic correctness boundary. Snapshot file
bytes consistently while hashing/decoding; detect concurrent replacement/write and retry before
scientific execution or fail explicitly. Do not hash one version and decode another.

## Store protocol
Read-only immutable values, explicit copy-on-use for publicly mutable outputs, per-key single
flight, bounded memory LRU plus bounded disk quota, no unbounded retained gradients. Writers
store schema/dtypes/shape/payload checksums in a temporary generation, fsync as promised, then
publish a commit manifest atomically. Readers verify lengths/checksums/schema before use.
On corruption treat as miss/quarantine, never silently accept partially finite arrays. Use
non-executable formats with allow_pickle=False and path traversal checks. Ejecting cache data
must not invalidate in-flight readers; lifetime references/pins are explicit.

## Workflow reuse targets
Audit repeated topology and snapping across pairing rows, accessibility-derived gravity caps,
flow setup, exports and benchmark diagnostics. Reuse only complete identical dependencies.
Flow's distances/model are not automatically identical to accessibility's; compare actual
turn/elevation/connector/cutoff semantics before sharing. Do not expose cached mutable arrays
as shared live topology. Keep validation/logging side effects ordered at the coordinator.

## Tests
Warm/cold/cache-off bit parity; mutate each key dependency separately; reorder equal rows;
change dtype/signbit/NaN payload, same path different bytes, restored mtime, weights only,
obstacle clear, model/profile/compiler version, output-only changes. Concurrency: identical
requests compute once; cancellation releases lock; crash midwrite never creates a hit; corrupt
entry and stale lock recover; eviction under use safe; returned arrays do not alias cache.
Benchmark hot and cold cache including hashing/I/O. A hit counter without avoided producer work
fails engagement. Negative-control incomplete keys must fail the invalidation suite.
