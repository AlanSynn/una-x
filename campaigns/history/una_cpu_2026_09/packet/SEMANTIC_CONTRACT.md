# Semantic contract

## API and DX
Package/import name, public classes/functions, method signatures, Settings fields/defaults/serialization, examples' calling pattern, and ordinary Python installation behavior are frozen unless a pre-existing baseline bug prevents execution and the repair is independently justified outside the optimization claim.

## Inputs
Preserve supported network/origin/destination/observer/obstacle file formats, CRS handling, geometry interpretation, custom weights, IDs, precision, edge/node snapping, and missing/invalid-input error behavior.

## Outputs
Preserve result attributes, array lengths/order, result column names, directional flow fields, node/edge/origin association, output filename policy, CSV delimiter/index behavior, GeoJSON geometry/CRS, Feather schema/metadata, ODM columns, logger-visible behavior that tests establish as contractual, and batch composite semantics.

## State and chronology
Preserve mutation order for Settings and UNA flags, topology reload behavior, gravity-cap derivation/writeback, per-run result ownership, and any chronological dependency between accessibility and flow. Do not parallelize a dimension that shares mutable UNA/Topology/Settings state.

## Operational guarantees
Do not silently skip writers/checkpoints/validation, shrink workload, change output extent, or mask failure behind fallback after an admitted optimized execution begins.

## Equivalence classes
Track separately: mathematical equivalence, floating-point execution equivalence, scientific/domain equivalence, API/artifact equivalence, and performance equivalence under equal resources. A proof in one class does not imply the others.