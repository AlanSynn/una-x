# Public and scientific contract

## Existing UNA API
Preserve existing imports, call signatures/positional parameters, Settings persistence,
result attribute names, row association, CRS and formats on the legacy route. Snapshot all
public methods/attributes and default values from the baseline before extending them.
New parameters are keyword-only or new settings with explicit serialized defaults. Settings
round-trips CSV/JSON/TSV, project cloning and Reset must include or default them consistently.

## Madina parity surface
The primary namespace is urban_network_analysis.compat.madina with zonal and una submodules.
Provide an explicitly installed optional drop-in shim distribution that supplies the same
madina import paths in an isolated environment. Never overwrite an existing madina installation
or inject sys.modules aliases globally. The facade's signatures/defaults/returns/mutations
match the pinned API. A one-line import migration alone is not drop-in import parity; report
facade parity and shim parity separately. No stub or silently ignored parameter counts.

Preserve Zonal.layers and indexing, Layer/Network state, source_id/parent_street_id lineage,
input preparation and snapping conventions, graph insertion/clear lifecycle, custom weights,
turn parameters, result joins, diagnostics/exposure columns, geometry types, order and CRS.
Make visualization optional at installation but implement it; missing optional dependencies
must produce actionable errors. Seed any random display colors identically in comparison
fixtures; never globally seed the user's RNG as an optimization.

## Models are not interchangeable
Implement Madina's all-path enumeration/assignment on its declared domain. Do not substitute
K-alternatives, shortest-only, aggregate flow, sampled paths or a capped route set under the
same method. Their conservation/decay/attribution equations differ. Keep separate engine IDs
and fingerprints. API names do not prove mathematical equivalence. Build a model crosswalk
with each equation, default, output and intended interpretation.

## New public execution contract
Add immutable ExecutionOptions, CacheOptions and capability/report types through documented
UNA imports. Proposed stable RunBatch extension:
RunBatch(analysis, pairing_file=None, *, parallel=False, workers=None, execution=None)
Existing calls keep serial behavior and return None; new diagnostics live in batch_report.
The integrator reconciles exact typing/naming once at CONTRACT and records the schema before
workers code against it. parallel=True must actually overlap independent rows when admitted.
Uncloneable custom subclasses/callbacks must use a reported serial compatibility route or
raise a documented explicit-mode error, not drop their behavior.

ExecutionOptions includes semantic_profile, backend (auto/reference/native/gpu), device,
CPU budget, threads per worker, logical reduction policy, memory/workspace limits, queue/writer
limits, timeout/cancellation, cache and checkpoint policy. backend='native' or 'gpu' must
raise a capability/admission error instead of silently falling back; auto may fall back with
a reason. No automatic remote provisioning. Semantic identity is separate from scheduling.

## Scientific fixes
Every proposed correction has a bug ID, minimal failing input, intended equation/invariant,
independent corrected oracle, affected domain and artifacts, migration note, and rejection
of alternative explanations. Implement fixed behavior in corrected_v1 and maintain legacy
valid-input reproduction. A documented fix is not a numerical tolerance waiver. Invalid
input hangs, unsafe casts and unbounded allocation need diagnosed bounded failure, not a
promise to reproduce a crash. Document these safety corrections explicitly.

## Geometry and outputs
Do not silently snap disconnected components, discard islands, split crossings, project CRS,
change endpoint precision, buffer degenerate hulls, deduplicate edges or trim path geometry
unless the selected model specifies it. Preserve origin/destination membership and full
required output extents. Cache/parallel/backend metadata must not corrupt legacy schemas;
put execution provenance in a separate manifest unless the user opts into new metadata.
Timestamp elapsed-time fields are nondeterministic; test format/order/association under a
controlled clock. Binary numerical arrays remain bit-exact, not globally normalized.
