# 05: per-metric dependency reuse without a new floating-point computation

Start from the actual compiled reach_gravity_knn_access, not symbolic formulas in prose. Freeze
a machine-readable dependency table per profile/plan. Tests mutate each dependency and verify
miss/recompute; poison unused parameters only as a characterization test, not broad inference.

For the current legacy expression with a fixed ordered kept sequence:
- reach uses kept destination weights and the driver's final output cast.
- exponential gravity uses kept distances/weights, gravity_beta and gravity_plateau.
- logistic gravity uses kept distances/weights, gravity_growth_rate, gravity_plateau and midpoint.
- exponential KNN uses the original argsort/crop, coefficients, beta and plateau.
- logistic KNN uses original argsort/crop, coefficients, growth, plateau and midpoint.
- other KNN branch uses original argsort/crop, coefficients and weights without decay.
Every plan also depends on membership/cutoff/extraction/ordering/dtypes and compiler identity.
The current o_access wrapper derives growth from decay_constant/midpoint; include the actual
result bit pattern or both inputs with identical typed evaluation. Do not use generic 'beta' or
'KNN parameters' labels to omit real dependencies. Changing profile invalidates this table.

For distinct beta rows with logistic KNN, reach/logistic/KNN may be identical. Compute once and
copy into FRESH correctly typed arrays per public job. Exponential stays per-row. Shared immutable
storage is private only; tests mutate a completed row's outputs and ensure all others unchanged.
Changed destination weights generally require ALL weighted metrics again but may share routing.

Critical compiler gate: extracting ge_only can produce different reduction bits than the original
combined fold. The prototype's 170 random expression cases do not prove portable equivalence.
Use actual full fold on selection length boundaries, ties, extremes, every known specialization,
and inspect IR/assembly when needed. Keep a proven same ordered plan, or reuse only the whole
combined metric bundle for identical metric inputs. A failed standalone fold does not block safe
routing reuse; it blocks that finer reuse profile. Never fix by allclose or 1-ulp tolerance.

Do not transform exp(beta*d) to pow(exp(d),beta), shift decay across a sum, factor trip weights
outside repeated RN updates, change sum layout, or select top-k with a different tie permutation.
Do not skip fields just because calculate_* prevents export: trace backward from public result
attributes, future state, warnings/errors and later consumers. Existing legacy exceptions may be
observable even for an unassigned result; only proven dead computation is removable.

Deliver metric_dependency_table.json, compiled-parity evidence per plan, alias tests, work counters
(folds done/reused), cache invalidation tests and public per-row equality. Screen C trace-only vs
C+metric reuse as separate arms; their gains overlap and cannot be multiplied.
