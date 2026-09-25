# Execution architecture

Current public accessibility path:
Settings validation -> network file load -> topology/node construction -> origin/destination load+snap -> optional obstacles -> directional/terminal costs -> CSR build -> bounded per-origin search -> destination adjustment -> reach/gravity/KNN reductions -> engine state assignment -> existing exporters.

Primary changed region is only CSR build after all numerical cost preparation and before search. The candidate helper is private to Engines and returns ordered pointer/vector/weight arrays or signals fallback. Engines and topology public contracts remain unchanged.

External throughput schedule treats each independent analysis job as owning a fresh UNA object, Settings, mutable topology, outputs, and output directory. Pass descriptors/paths rather than serializing large GeoDataFrames/CSR unless measurement proves otherwise. Bound queue and writers. Preserve required publication ordering at coordinator if applicable.

Public RunBatch contains shared mutable state and composite capture; do not make it concurrently execute rows without a separate full dependency/state proof.

No GPU residency/backend abstraction in this campaign. No graph cache. No runtime dependency on `optimization_una_cpu` packet files.