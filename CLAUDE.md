# UNA-X

Urban Network Analysis Python package — accessibility and pedestrian/bicycle
flow on spatial networks. Fork of the MIT City Form Lab UNA toolkit, hardened
and optimized (see `docs/optimization.md` for the current-state vs vanilla
comparison and every measured figure with its exact baseline).

## Layout

- `src/urban_network_analysis/` — the package (engines, batch runtime, cache,
  native backend, Madina compatibility facade, reference profiles).
- `tests/` — live test suites. Run per subdirectory (`pytest tests/<dir>`);
  a whole-repo single-process run has known conftest module-name collisions.
- `benchmarks/platform/`, `benchmarks/una_10x/` — measurement harnesses.
- `docs/optimization.md` — what changed vs vanilla UNA.

## Laws that stay binding

- Bitwise parity per numerical profile (`una_legacy` / `madina_legacy` /
  `corrected_v1`) is mandatory: no tolerance, no output rounding, no
  reduction reassociation, no float64 key coercion. A cache hit never
  suppresses a row's validation, warnings, state transitions, or errors.
- No performance claim without a preregistered, paired, bitwise-checked
  measurement against a named baseline commit; never compose multiples
  across different baselines or boundaries.
- Historical campaign evidence (receipts, raw runs, packets) is archived
  off-repository at Dropbox `Alan Synn/Research/una-x/` (MANIFEST.sha256 per
  tarball; digests also in commit `e7b29f1`) and remains recoverable from
  git history (REGISTRY-pinned packet trees `917475d2` / `6ef3a4ff`).
- Do not recreate `campaigns/` or campaign apparatus without new explicit
  authorization. Pushes, merges, PRs, releases, and destructive cleanup need
  explicit user authorization.
