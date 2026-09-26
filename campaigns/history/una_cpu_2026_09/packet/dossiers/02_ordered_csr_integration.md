# Dossier 02: ordered CSR integration

Problem: baseline CSR construction repeatedly computes `start_nodes == node` and `end_nodes == node` for every node in count and fill passes, then extends Python lists.

Candidate: form incidence arrays in baseline logical order: all forward incidences (source=start, neighbor=end, weight=AB) in original edge order followed by all reverse incidences (source=end, neighbor=start, weight=BA) in original edge order. Stable-sort/group by source node, compute counts/pointers, and gather neighbor/weight arrays.

Equivalence: stable grouping preserves order among equal source nodes, therefore each row exactly matches baseline forward-then-reverse edge order. No arithmetic changes. Parallel edges and self-loops remain separate.

Admitted fast domain: 1D NumPy arrays, consistent lengths, native int64 endpoints, float64 directional weights, valid endpoint range, safe ownership/layout. Anything else returns to original builder.

Tests: exhaustive tiny graphs, random graphs, duplicates, self-loops, isolated nodes, empty edges, signed zero/nonfinites in weights, read-only/strided/wrong dtype fallback, baseline search state and metric arrays.

Promotion: complete builder/public API must win, not only helper. Reject if stable ordering or artifact/state equality fails.