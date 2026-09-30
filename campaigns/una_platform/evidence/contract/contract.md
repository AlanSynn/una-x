# CONTRACT — frozen semantic profiles, references, folds, comparison fields

Campaign una-platform-2026-09. Frozen 2026-09-30 on branch perf/una-platform.
This document is the binding resolution for all numerical and API comparisons downstream.
Changing anything recorded here requires a new versioned contract and cache invalidation.

## 1. Immutable references

| profile | definition | execution environment |
|---|---|---|
| `una_legacy` | UNA-X source at `16bf404d2cc5e776a78dc073c80405f6d186da29` (src tree `3c59bb0d3f9bc1fba7547be47ec63add88dd3031`), installed as wheel `urban_network_analysis-2.6.0-py3-none-any.whl` (sha256 `323c27f3ace8daafcc6d025596a605e8263cc14c8c4fe602c427d51069448ac4`); all 16 pinned files verified equal to `source_pins.json` after install | `.refs/venv_una_legacy` (see §2) |
| `madina_legacy` | Madina at `8b5c3bd3b1c0048ae8b04054daed92bfb201b9d6` + audited dependency bridge `dependency_bridge.patch` (sha256 below). Bridge edits are dependency-only: (a) `geometry.values.data` → `np.asarray(geometry.values, dtype=object)` (GeoPandas ≥1.0 accessor removal, upstream Issue #8); (b) removal of the private `pd.Series(..., fastpath=True)` kwarg (pandas ≥2 removal, upstream Issue #12). No algorithm, ordering, or arithmetic edits. Numerical effect of the bridge is tested separately (contract test CT-BRIDGE-1: bridged-vs-pristine comparison on the legacy environment is impossible because pristine does not import; instead dtypes/shapes of all constructed gdfs are asserted equal to the recorded schema table) | `.refs/venv_madina_legacy`, source root `.refs/madina_ref/src` (copy; pristine pinned clone `.refs/madina` is never modified) |
| `corrected_v1` | To be implemented by SCIENCE as `urban_network_analysis.reference` (pure Python, binary64, explicit operation order, NUMERICS.md profile). Until SCIENCE's receipt lands, corrected_v1 has NO reference and no parity claims are admissible | same as una_legacy env unless stated |

Oracles are never modified to fit candidate outputs. A reference rerun always reconstructs
outputs from the immutable sources above; no stored golden may substitute for a rerun when
the golden's producing run cannot be reproduced from recorded inputs.

## 2. Frozen numerical environment (recorded 2026-09-30)

- CPython 3.11.4 (micromamba `alan` env and both reference venvs, python.org 3.11.4 build)
- numpy 2.4.6, numba 0.67.0, scipy 1.17.1, pandas 3.0.6, geopandas 1.2.0,
  shapely 2.1.2 (+GEOS via wheels), networkx 3.6.1, scikit-learn 1.9.1, psutil 7.2.2,
  pydeck (madina_legacy venv only, latest at install), pytest 9.1.1
- CPU: Intel Xeon Gold 6226 @ 2.70GHz, cpus 20-23 (4 effective), numba threads default 1
  unless a receipt records otherwise; Linux 5.14.0-570.128.1.el9_6.x86_64
- fastmath flags: whatever the baseline source pins (una_legacy); recorded per-kernel at
  HARNESS/PARITY time from the compiled signatures — never assumed

## 2.1 madina_legacy determinism contract (executed evidence, 2026-09-30)

Pinned Madina calls `origins.sample(frac=1)` (unseeded numpy global RNG) before origin
processing and uses unseeded `random` for layer colors; origin processing order changes
accumulation order, and (reproduced, see bug registry BUG-M01-STATS-NAMEERROR) the default
`closest_destination=True` path skips `remove_node_to_graph`, so results depend on order.

madina_legacy reference runs are therefore pinned to:
1. `num_cores=1` (single process; the multi-core path's as_completed merge is order-nondeterministic and is NOT a bitwise-comparable profile arm),
2. `random.seed(S)` and `numpy.random.seed(S)` with the seed S recorded per fixture before zonal construction,
3. timestamps excluded from comparison fields (format/order tested under a controlled clock).

Cross-process bitwise determinism under this pin was verified on the 3x3 grid smoke
fixture (identical sha256 of betweenness bytes across 3 separate processes).

## 3. Canonical logical reduction (corrected_v1 and all corrected backends)

Per NUMERICS.md, frozen: `L = min(32, n_origins)` stripes; origin `i` → stripe `i mod L`;
each stripe accumulates its origins in increasing origin order; final fold adds stripes in
increasing stripe ID. The scalar corrected oracle implements the same partition even with
one physical worker. Any partition change versions corrected_v1 → `corrected_v2`, … and
invalidates numerical caches. una_legacy's internal fold is whatever baseline `16bf404`
executes; it is characterized by HARNESS traces, never re-associated.

## 4. Comparison fields per profile

For every profile arm pair (reference → optimized CPU → native → GPU → cache hit →
parallel RunBatch), compare and require identity of:
1. pandas/numpy dtypes, shapes, column order, index (name, dtype, values, order);
2. numeric array bytes (bit-exact; signed-zero bytes; NaN payloads where stored/copied);
3. categorical/object values and ordering;
4. geometry: WKB bytes + CRS; geometries are never normalized;
5. exceptions: type and message for invalid-input cases;
6. madina_legacy additionally: layer/network gdf state after the call (mutation parity),
   `source_id`/`parent_street_id` lineage, `nearest_edge_id`, partial weights, node/edge ids;
7. nondeterministic fields (elapsed times, timestamps, logger seconds) are compared for
   format/presence/order only under a controlled clock.
`np.allclose`, rounding, or error-percentage comparisons are never acceptable evidence.

## 5. Bug fix vs legacy parity contradiction — resolution

Contract per AUTHORITY.md §2.2: legacy profiles reproduce pinned historical behavior on
valid inputs, including its defects, except where a defect makes execution unsafe
(nontermination, unbounded memory) — those become declared safety corrections with
bounded rejection/diagnosed failure, recorded as safety deltas, never as silent legacy
parity. Corrected outputs never masquerade as legacy outputs. Each corrected behavior
carries a bug ID from `bug_registry.json`, a minimal failing input, an independent
oracle, and `bug_deltas.json` rows listing changed outputs.

## 6. Public schema approval (integrator-owned files)

Approved for EXECUTION to implement exactly (from INTERFACES.md):
- `urban_network_analysis.ExecutionOptions` (frozen dataclass, keyword-only, immutable):
  `semantic_profile: str = "una_legacy"`, `backend: Literal["auto","reference","native","gpu"] = "auto"`,
  `device: str | None = None`, `cpu_budget: int | None = None`, `threads_per_worker: int = 1`,
  `logical_reduction_plan: str = "canonical"` (frozen §3 partition),
  `memory_limit_bytes: int | None = None`, `workspace_limit_bytes: int | None = None`,
  `queue_depth: int = 64`, `writer_concurrency: int = 1`, `timeout_s: float | None = None`,
  `cache: CacheOptions = CacheOptions()`, `checkpoint: str | None = None`.
- `urban_network_analysis.CacheOptions`: `mode: Literal["off","memory","disk"] = "off"`,
  `directory: str = ".una-cache"`, `max_memory_bytes: int | None = None`,
  `max_disk_bytes: int | None = None`, `verification: Literal["never","on_write","on_read","always"] = "on_read"`,
  `schema_version: int = 1`.
- `UNA.execution` attribute (default `ExecutionOptions()` → una_legacy) and
  `UNA.RunBatch(analysis, pairing_file=None, *, parallel=False, workers=None, execution=None)`
  with existing serial behavior and `None` return preserved; diagnostics in
  `UNA.batch_report`. Serial prefix invariant per dossier 04.
- `urban_network_analysis.compat.madina` facade: `Zonal.execution` optional control
  (default madina_legacy). Existing Madina-signature positional parameters and defaults
  remain intact and visible; additive keyword-only controls only.
- Settings persistence: `execution.semantic_profile`, `execution.backend`,
  `execution.cache.mode`, `execution.cache.directory` serialize into project
  CSV/JSON/TSV with explicit defaults; Reset restores those defaults; cloning preserves
  explicitly chosen values. Runtime-only fields (device handles, cancellation tokens)
  are not serialized.

backend='native'/'gpu' without a qualified capability raises a typed admission error
before scientific execution; auto may fall back with a recorded reason. No silent fallback.

## 7. Dependency bridge policy

The bridge exists only in the madina_legacy reference environment. It is never merged
into UNA-X runtime code (compat implementations are written natively). The patch file is
hashed here and stored twice (`.refs/madina_ref/dependency_bridge.patch` and
`evidence/contract/madina_bridge.patch`); any difference invalidates madina_legacy
evidence produced with it.

sha256(dependency_bridge.patch) recorded in `profiles.json`.
