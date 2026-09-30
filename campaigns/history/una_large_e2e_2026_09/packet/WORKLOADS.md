# Observed inputs, scientific workload identity and scale

Numbers specifying sample counts, radii and budgets below are POLICY CONSTANTS or explicit proposed workload definitions, not measurements. Count actual rows/nodes/arcs after loading. Do not infer counts from GeoJSON size or synthetic grid dimensions.

## Input acquisition
Use only a user-supplied path, a file in the current checkout, or the exact public catalog in inputs_catalog.json. Default observed-data source is the Boston tutorial at upstream c15ebda6981397f46eed5c2d55229f71e57d44fb: full InnerCore network, Cambridge building centroids and MA bus stops. These are real-world tutorial inputs, not the user's production target. Verify source byte length and Git blob hash, then record SHA-256. Do not download mutable main or assume code's MIT license settles redistribution of all datasets. Keep downloaded/derived inputs local; record attribution and unresolved dataset-license restrictions, do not republish data.

H01 implements a download/acquisition script under benchmarks/large_e2e, with normal TLS, a size limit from the catalog, temporary file, hash verification before atomic rename, no overwrite of mismatched existing files, bounded retries and disk preflight. Existing files are rehashed on every new frozen session. Failed download or schema validation does not fall back to synthetic data under the same label.

Inspect CRS, geometry types/validity, Z availability, custom columns, missing/nonfinite/nonpositive costs, weights and IDs before generating Settings. Do not clean/reproject/simplify input secretly. Keep baseline's existing warnings/fallback behavior and record it. If chosen input cannot run unchanged, report the specific problem and define a separate explicitly transformed diagnostic fixture; it is not the original observed workload.

## Named workload classes
S0: tiny deterministic synthetic graphs and W1/W3 generated GIS regressions. Label synthetic even when calling real public APIs.
O2: observed L2 fixture, full observed network and destinations with 16 deterministically selected origin rows. Intended to test actual loading/snapping/geometry/chronology cheaply, not to claim full-scale performance.
O3_ACCESS: full observed network, all original Cambridge origins and MA bus stops, radius 500, all four metrics, no turns/elevation, original geometric network cost. This is the default primary large observed accessibility proxy.
O3_FLOW: same full network/destination layer and a preregistered origin selection from {256,1024,all}, radius 500, AggregateFlow, automatic gravity cap p95. Choose the largest selection fitting baseline-only resource/time pilots before candidate timings; record exact selection and do not describe a subset as full-origin flow. Full-origin feasibility is separately reported. K-alternatives is a protected small test, not replaced by aggregate flow.
O3_HOLDOUT: radius 1200 on the same observed network, and a 3D/elevation profile using the catalog 3D network when valid. This checks settings generalization, NOT independent network morphology. A user-supplied second observed morphology can be added before preregistration; otherwise cross-morphology generalization remains unavailable.
U4: the user's actual production dataset/settings/outputs/job volume/throughput target. Not supplied. Leave unavailable instead of calling O3 an L4 success.

## Frozen settings procedure
Start from B0 Settings, call ToDict(compact=False) to capture ALL resolved defaults, then apply explicit profile overrides. For accessibility use network_weight_column=Geometric, origin_weight_column=Count, destination_weight_column=weekly_departures, search_radius=500, turns=False, elevation=False, all calculate_* metrics True, output_csv/output_geojson/output_feather=True. Verify that Count behavior and weekly_departures column resolve correctly from actual baseline source/data; missing columns are errors, not guessed substitutions. Keep remaining defaults identical and recorded, including knn weights/decay, gravity parameters and precision.

Flow additionally sets flow_engine=aggregate_flow, flow_decay=True, flow_decay_method=gravity_cap, flow_gravity_cap=p95 and records all effective defaults. Before/after Settings are captured because cap becomes numeric. Explicit thread profile is part of the manifest; do not call a nondefault stripe count the default API profile. Initial observed cases use no invented obstacle layers. Synthetic regression fixtures cover obstacles; use catalog observers for optional observed coverage only when that baseline path is valid. Do not compare two different input files as baseline/candidate.

Timestamp policy: capture it unchanged for public compatibility tests. For deterministic perf output directories each job receives a unique campaign-owned output root; any output_wStamp override is applied equally and declared in that workload, not inferred from results. Preserve every enabled format and required companion output. Hashes plus semantic signatures are outside application timing but inside validated batch accounting as BENCHMARKS specifies.

Origin selection: seed 20260925; select row indices without replacement using a pinned NumPy RNG implementation, then sort selected indices into ORIGINAL row order before writing a derived fixture. Persist the explicit index list so future environments do not regenerate a different set. Do not reset/rewrite IDs to make comparisons pass. Each derived fixture has its own hash and selection manifest.

## Scale axes
Network extent (V,E), destination count D, origin count O, radius/detour, and independent job count K are separate axes. Initially test baseline-only pilot cells: O2, O3_ACCESS, O3_FLOW and one O3_HOLDOUT. Expand at most one axis for each admitted bottleneck, not a Cartesian product. Artificial disconnected extensions and generated density sweeps are useful diagnostics but remain synthetic. Any observed network subset is a distinct scientific workload, never an execution tile with lost neighborhood context.

No code transformation may crop physical network support, shrink radius, drop output formats, reduce destinations, alter impedance or snap thresholds within a frozen A/B cell. Cache warming is an execution profile, not permission to skip application work. Repeated jobs use new UNA instances and fresh output directories unless the explicit user workflow requires otherwise.

## Manifest contents
Source URL/commit/blob; SHA-256/bytes for every input; dataset class and licensing/attribution note; original and selected row IDs/order; CRS; actual V/E/O/D counts; complete Settings before run; expected engine; required output set; numerical stripe/thread profile; output/checkpoint policy; source/wheel/native environment; preprocessing script/hash; target status. A null required field means not ready for qualification.
