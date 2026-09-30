# TOPOLOGY independent review report (received post-receipt, 2026-09-30)

Reviewer: topology-reviewer agent (fresh context, read-only, numerical-architect role).
Scope: claim that src/urban_network_analysis/compat/madina/zonal/ is a verbatim attributed
port of .refs/madina/src/madina/zonal/ @ 8b5c3bd3 with exactly D1–D4 + adapter.py.
All 5 review tasks executed (full per-module diffs, bridge provenance, adapter copy
semantics, import-order hazards incl. empirical venv runs, comparator blind-spot skim).

## VERIFIED CLAIMS

1. Facade vs pinned diffs: zero undocumented behavior deltas. utils.py = D4 only
   (facade utils.py:119-127 lazy pydeck import); layer.py = docstring + trailing blank
   lines; network.py = docstring + trailing-whitespace-only edits, zero code-token
   deltas; network_utils.py = D1 single site (facade network_utils.py:28
   `GeoPandaExtractor(np.asarray(geometry_gdf.geometry.values, dtype=object))`) + D2 all
   34 pinned `fastpath` occurrences removed (facade's 1 remaining hit is the docstring);
   zonal.py = D3 (os.environ['USE_PYGEOS'] + `import os` removed, no residual `os.` use)
   + D4 (lazy create_deckGL_map binding, zonal.py:431-433; `-> "pdk.Deck"`). Error
   messages byte-identical incl. pinned quirk
   `node_snapping_tolerance={redundant_edge_treatment}`; RNG consumption order
   preserved; no np.random anywhere.
2. Bridge provenance intact: .refs/madina_ref differs from pinned in exactly one source
   file; that diff is byte-identical to
   campaigns/una_platform/evidence/contract/madina_bridge.patch (sha256 d47fbd6dc07580e…
   = profiles.json:29); second copy .refs/madina_ref/dependency_bridge.patch has the
   same sha; .refs/madina confirmed a real git clone at 8b5c3bd.
3. Adapter: no `.tables` attribute — nodes/edges ARE the live tables (network.py:31-32).
   All four arrays in network_state() are private copies (column_stack + .copy();
   to_numpy dtype-conversion + .copy()); GraphState.from_nx only reads; writeable=False
   verified.
4. Import order safe (empirical in both legacy venvs): deep submodule-first import works
   in a fresh process (parents always execute first, rebinding line
   compat/madina/__init__.py:25 always precedes); madina.zonal resolves to the package,
   .zonal.zonal to the module; root PEP 562 lazy exports load no numba/sklearn on plain
   import, from-imports work, UNA.__module__ unchanged (pickle-safe).
5. Comparator: sabotage tests non-vacuous (honest==ref premise then broken!=ref;
   3 dimensions incl. 1-ulp weight kill); reference arm run twice (determinism gate);
   digests unlinked per run (no stale-artifact pass); PYTHONHASHSEED=0, same interpreter
   both arms; all Zonal/Layers/Layer/Network state covered by the digest; test count
   36 = 14 parity + 14 error + 1 unsupported + 1 lazy-import + 3 sabotage + 3 adapter
   (matches claim).

## FINDINGS (no BLOCKER, no MAJOR)

1. MINOR — src/urban_network_analysis/compat/madina/zonal/zonal.py:431-433: D4
   late-binds create_deckGL_map via `.utils`; monkeypatching `zonal.create_deckGL_map`
   no longer affects create_map (must patch `utils.create_deckGL_map`). Inherent to D4,
   parity arms unaffected; optional ledger note.
2. MINOR — tests/platform_geometry/test_adapter.py:73-75: in-place mutation immunity of
   SNAPSHOTTED fields not directly tested (test mutates non-snapshotted `degree`, then
   clear_nodes rebind); immunity rests on code inspection of the .copy() calls at
   adapter.py:100-111. Follow-up: mutate `z.network.edges.at[id,'weight']` in place
   post-snapshot, assert snap unchanged.
3. MINOR — tests/platform_geometry/_scenario.py:261-269: _graph_digest records only edge
   (weight,id), node (type), graph.graph["added_nodes"]; a delta confined to any other
   graph/node/edge attribute would be invisible. Mitigated by verified source-level
   identity.
4. MINOR — tests/platform_geometry/conftest.py:33-48: no digest scenario exercises
   set_node_value / add_node_to_graph / remove_node_to_graph / update_light_graph / the
   chain-sort epsilon tie-break path (coincident_od covers same-node only). Add one
   multi-node-per-edge scenario before MADINA_* engines consume insert_node outputs.
5. INFO — src/urban_network_analysis/compat/madina/zonal/adapter.py:76-93: GraphState
   captures no node set/attrs; unreachable in the ported surface (nodes exist only as
   add_edge endpoints) but relevant if GraphState becomes a cache fingerprint.
6. INFO — src/urban_network_analysis/compat/madina/__init__.py:25: rebinding makes
   compat.madina.zonal the PACKAGE whereas pinned upstream leaves madina.zonal = the
   zonal.py MODULE; surface-compatible for everything reviewed, but the future drop-in
   `madina` shim must decide which identity it promises.
7. INFO — tests/platform_geometry/_scenario.py:250-251: object-dtype columns compared
   via str(v) (dtype recorded separately); NaN payloads collapse to None; -0.0
   preserved as distinct bits.

## BOTTOM LINE

The "verbatim attributed port with exactly D1–D4 + adapter" claim is confirmed at
source level; bridge provenance chain (git commit + patch sha + dual copies) intact;
adapter copy semantics correct; import-order changes safe.

## VERDICT

APPROVED
