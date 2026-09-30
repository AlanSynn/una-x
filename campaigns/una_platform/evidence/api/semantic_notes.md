# INVENTORY semantic notes (una-platform-2026-09)

Scope: pinned Madina `8b5c3bd3b1c0048ae8b04054daed92bfb201b9d6` (clone at `.refs/madina`,
all 12 pinned file hashes and the `src` tree hash verified equal to `source_pins.json`)
and UNA baseline `16bf404d2cc5e776a78dc073c80405f6d186da29` (src tree `3c59bb0d`).

Outputs:
- `inventory_madina_pinned.json`, `inventory_una_baseline.json` — raw `tools/inventory_api.py` AST censuses.
- `api_matrix.json` — expansion of the 18 seed rows into every public symbol of both trees
  (76 Madina records, 112 UNA baseline records), with exact signatures/defaults and
  family-level semantic annotations transcribed from the source reading below.
- `generate_api_matrix.py` — deterministic generator (rerunnable; prints content hash).

## Method
Full read of all 12 pinned Madina files (~4,945 lines) and the UNA public surface
(`UNA.py`, `Settings.py`, `Topology.py`, `Logger.py`, `Engines/*`). Semantic notes in
`api_matrix.json` (`families[].semantic_notes`) record observed source behavior including
quirks and suspected defects. Everything there is a **source observation**, not an executed
result; executed references come under CONTRACT/FAILURES.

## Registered observations requiring owned dispositions (not silent choices)
1. `Layer.set_style`/`Layer.color_layer` reference `self.default_colors`, which `Layer`
   never defines → `AttributeError` on any call (pinned bug). `utils.color_layer` is the
   module-level variant that works against a `Zonal`-like `self`.
2. `Network.network_to_layer()` constructs `Layer` with 2 args (requires 6) → `TypeError`.
3. `Network.add_node_to_graph` swallows all exceptions via print-and-continue; its epsilon
   perturbation is `1e-7*(weight_sec+1)` while `update_light_graph` uses flat `1e-7`.
4. `Network.update_light_graph` mutates `add_nodes` while iterating it.
5. `betweenness_exposure`: with `closest_destination=True` (the `tools.betweenness`
   default), `eligible_destinations_shortest_distance` is undefined in the stats block →
   `NameError` → bare `except` → diagnostics after `gravity` are skipped AND
   `remove_node_to_graph` is skipped (suspected d_graph poisoning). Also all-zero Huff
   gravities `continue` without graph cleanup. These must be confirmed by executing the
   pinned reference before any corrected_v1 disposition (FAILURES/SCIENCE).
6. `paralell_betweenness_exposure`/`parallel_access` call `origins.sample(frac=1)`
   unseeded; accumulation order over origins is therefore nondeterministic run-to-run
   (single-core included). madina_legacy fixtures must seed the RNG (pandas/numpy global)
   and record it in the reference environment.
7. `prepare_geometry` drops Z via `tuple(filter(None, [x, y]))` — a coordinate 0.0 is
   dropped along with Z (pinned quirk; corrected_v1 candidate BUG-geometry-z).
8. `create_street_network` source default is `redundant_edge_treatment='discard'`;
   the docstring prose claims "Default 'split'". Source default wins (SOURCE_AUDIT).
9. `Layers.__next__` yields label strings, not `Layer` objects; `Zonal.describe`'s
   `geo_center is None` check compares a tuple to None and never fires.
10. `node_edge_builder` uses `geometry.values.data` (GeoPandas ≥1.0 removal, Issue #8/PR10)
    and `pd.Series(..., fastpath=True)` (pandas ≥2 removal, Issue #12/PR13). Pinned Madina
    cannot run on the recorded 2026-09 environment without the audited dependency bridge
    (CONTRACT defines the bridge; FAILURES executes it).
11. Documented unsupported stubs (raise NotImplementedError): `Network.visualize_graph`,
    `Network.scan_for_intersections`, `Network.fuse_degree_2_nodes`,
    `Network._get_nodes_at_geometric_distance`, `..._network_distance`, `..._bf_distance`
    (the last three are private; recorded for completeness).
12. Madina `madina.una.__init__` star-imports only `betweenness` and `paths`;
    `tools`/`workflows` are submodule imports. Facade and shim must reproduce this shape.

## Facade vs shim coverage
- Facade: `urban_network_analysis.compat.madina` namespace (always available with UNA-X).
- Shim: separately installed optional distribution providing `import madina` paths for an
  environment WITHOUT upstream Madina. Never overlaps an existing install, never injects
  `sys.modules` aliases. Both surfaces tracked as separate `required_checks` rows
  (`installed_facade`, `installed_shim`) in `api_matrix.json`/API_PARITY.json.

## UNSUPPORTED upstream stubs (recorded, not invented)
`Network.visualize_graph`, `Network.scan_for_intersections`,
`Network.fuse_degree_2_nodes`. Parity claims must not silently include them.
