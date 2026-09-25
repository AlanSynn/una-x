# Numerical contract

The compatibility profile preserves baseline dtypes, casts, operation order, comparison semantics, heap tie behavior, reduction order, fastmath configuration already present in source, NaN/Inf handling, signed zero where observable, integer overflow behavior, and emitted arrays/artifacts.

For CSR optimization, no floating arithmetic is introduced: directional/terminal weights are computed by the unchanged baseline expressions first, then copied into stable grouped adjacency storage. Required proof: for each node n, the candidate adjacency row equals the baseline concatenation of forward incidences with start==n in original edge order followed by reverse incidences with end==n in original edge order. Parallel edges and self-loops remain distinct.

Do not rewrite vectorized neighbor relaxation into scalar immediate relaxation. Historical counterexample shows duplicate-neighbor ordering can change labels. Do not tree-reduce, reassociate, change transcendentals, enable new fast-math/contraction, or change precision under the compatibility profile.

L0/L1 exact checks should compare bit patterns for finite values when baseline determinism permits, signbit for zero, NaN/+Inf/-Inf masks, categorical outputs, row pointers/vectors/weights, and per-step state on adversarial tiny cases.

A relaxed numerical profile is not authorized in this campaign.