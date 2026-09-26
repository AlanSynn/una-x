# Dossier 06: optional private scratch/state reduction

This experiment is NOT admitted by default.

Admission requires post-CSR profile showing material end-to-end service in per-origin allocation/initialization of search state, plus a concrete proof target. Candidate idea may reduce private scratch initialization or represent unvisited state more compactly only when every consumer observes an equivalent value.

Proof obligation: define projection pi from old state to new state and show pi(F(s,u)) = F'(pi(s),u) for every transition, and show every future observation/output depends only on pi(s). Preserve duplicate-neighbor/vector-update behavior and heap ordering.

Do not change Dijkstra relaxation order, tree-reduce, use generation counters that alter overflow/lifetime semantics without proof, or expose new public state.

Stop immediately on state/bit mismatch, low Amdahl coverage, memory regression, added synchronization, or maintenance cost disproportionate to complete-region gain. At most one implementation attempt.