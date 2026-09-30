# D01: API census, references, and Madina compatibility

## Required deliverable
A working compatibility namespace `urban_network_analysis.compat.madina` with `zonal`,
`una.tools`, `una.paths`, `una.betweenness`, `una.workflows` and the supported reexports.
Provide a separately installed drop-in shim distribution for unchanged `import madina` code;
never install overlapping `madina` files into an environment containing upstream Madina.
Document safe separate-environment migration and preserve upstream MIT attribution.
Existing UNA methods remain available. Do not call parity complete with only a mapping table.

## Census procedure
1. Obtain pinned Madina source from source_pins.json in a dedicated read-only checkout.
2. Run tools/inventory_api.py. Inspect __init__ reexports, module public definitions, class
   methods/properties/dunders, public data fields, documentation snippets and test notebooks.
3. Expand the immutable API_PARITY.json seed into evidence/api/api_matrix.json, one record
   per supported symbol and parameter family. Do not overwrite the packet seed. Include
   defaults, positional/keyword behavior, return type, mutation targets, errors, schemas,
   ordering, geometry, optional dependencies, representative scenarios and oracle source.
4. Public-looking unsupported stubs are documented as such; they are not features to invent.
   Network.visualize_graph raising NotImplementedError is not evidence of working visualization.
   Working Zonal.create_map is required, including its pydeck.Deck return.
5. Run original examples in a pinned upstream-compatible environment. If modern dependencies
   prevent execution, retain original failure and create a minimal dependency-bridge patch in
   the reference environment only. Hash it; forbid scientific algorithm edits in that bridge.
6. Run each scenario baseline twice before comparing the facade. Capture full frame schemas,
   index/order, dtype, numeric bytes, geometries/CRS, mutable fields and exceptions.

## Minimum supported behavior to implement
Zonal load_layer/create_street_network/set_turn_parameters/insert_node/create_graph/clear_nodes,
layer indexing/ordering/style/map behavior; supported Network graph mutations; accessibility
(alpha, beta, KNN plateau, closest facility, turns, named saved columns); service_area;
all alternative_paths; betweenness with exponent and power decay, closest destination or Huff
competition, elastic KNN demand, diagnostics and path exposure; the two pairing workflows.
Inspect public helper exports and direct-call contracts rather than guessing they are private.

## Non-equivalent shortcuts forbidden
- AggregateFlow is not Madina betweenness, and a finite K-alternatives engine is not ALL paths.
- closest_facility assigns each destination to its closest ORIGIN. It is not each origin's
  closest destination. Test two origins equidistant from one destination and source-ID ties.
- Preserve actual redundant_edge_treatment='discard' legacy default, despite a conflicting
  docstring saying split. corrected_v1 may repair documentation, not silently change defaults.
- Madina mutates Zonal layers and network tables. Returning arrays without updating those
  tables is not parity. Preserve source_id/parent_street_id mapping after split/discard.
- Random default map colors and timestamps need controlled test RNG/clock, not dropped fields.

## Implementation architecture
Use a facade owning the layer/graph state and an explicit model adapter into typed core data.
Maintain source row -> network entity maps and mutation generations. Reuse verified common
algorithms only after matching graph construction, route set, ties and arithmetic. A private
attributed port of a Madina reference algorithm is an acceptable correctness-first floor;
then remove its work using the same optimization gates. Do not depend on upstream Madina at
runtime or invoke it as the supposed optimized implementation.

## Tests and stop
API reflection is necessary but not sufficient. Deliberately swap alpha=1 with alpha=2, reverse
closest-facility direction, drop a split edge, and omit an exposure column; each mutant must fail.
Required test combinations include repeated calls on the same Zonal, changed layer weights,
clear/reinsert, mixed IDs, no reachable destination, empty output, all output names and invalid
arguments. A symbol not implemented is a mandatory open task, never a silent exclusion.
