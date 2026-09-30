# Workloads and city-failure investigation design

Keep separate: adversarial synthetic graphs, synthetic GIS grids, observed-city proxies,
user production workloads, and incident reproducers. Do not call a generated grid real city
performance. Do not label an observed proxy as the user's production target.

Reuse the pinned Boston/Cambridge input catalog from the historical large-e2e packet, not its
purged runtime directories. Verify content hashes and licenses before reading. Preserve a full
network as path context when selecting origins/destinations for a small test; do not crop roads
and silently change reachability. Test 2D and 3D, geometric and perceived costs, obstacles,
turns, observers and non-network access points. Required output formats remain enabled.

INVENTORY/FAILURES must select at least three observed morphology strata before performance
selection: dense cyclic urban network, irregular/disconnected network, and regional sparse
network; prefer user-reported failing city/data when available. Pin the exact authorized
source files, preprocessing, CRS, ordering, ID mapping and license. The current audit located
Cambridge/Boston references but no authenticated failing-city dataset or second/third city.
Discover public or user-authorized data without inventing incidents or buying services.
Missing strata block a broad multi-city claim, not useful other implementation work.

Freeze independent scaling axes: nodes/edges, origins, destinations, radius, detour ratio,
turn-state density, nearest-vs-all destinations, exposure/elasticity, repeated batch settings,
output bytes, cache identity diversity and sparse-gradient occupancy. Only mutate an axis
between named workloads, never between baseline/candidate arms. A larger detour changes the
scientific workload and often the route count; it is not just a performance tuning knob.

Minimum workload families:
W0 tiny exhaustive/adversarial graph and bit-pattern corpus.
W1 real file I/O plus small full public workflow for every API family.
W2 current observed baseline workflow including the earlier radius 500/1200 distinctions.
W3 medium observed sensitivity experiments and same-profile native/GPU/batch/cache screens.
W4 held-out full observed cities and supervised reported failures, immutable finalist only.
W5 actual production input/settings/hardware/target, unavailable until supplied.

Cache matrix: off, cold miss, memory hit, disk hit, invalidation after one dependency change,
content-identical relocation, concurrent same-key producers, corrupted/partial files and quota
pressure. Publication-only metadata changes may hit numerical caches but produce new artifacts.
Never count cache-only runs against an uncached baseline without labeling both warm/cold scope.

Failure corpus: import/dependency drift; empty/invalid/GeometryCollection/multipart geometries;
CRS units and degree coordinates; invalid weights, zero cycles, parallel edges and collapsed
nodes; same-edge OD; no reachable destinations; coincident origin/destination; high detour/path
combinatorics; full result queues; dead/slow workers; interrupted writers; disk full; native
bounds errors; GPU OOM/context loss; stale/malformed caches and checkpoints. Diagnostic time
and memory caps produce explicit incomplete status, not silently partial “successful” results.
