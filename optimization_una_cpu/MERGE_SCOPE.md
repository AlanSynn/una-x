# Merge scope

Production candidate should contain only code required for the accepted optimization, focused regression/equivalence tests, and maintainable benchmark/qualification helpers that are justified for the repository.

Expected runtime delta for the mandatory track:
- private ordered-CSR helper in the Engines package,
- minimal call-site changes in Accessibility.py and AccessibilityWElevation.py,
- exact original fallback.

Do not leave historical oracle copies, repository-path loaders, experimental dispatchers, inactive backends, benchmark output artifacts, qualification registries, or packet research machinery on the installed runtime path.

Before merge disposition classify every new file as production-required, test/evidence-required, documentation, or research-only. Research history may remain under `optimization_una_cpu/` in this fork, but package imports must not depend on it.

Merge manifest records source branch, reviewed SHA, target main SHA, merge base, files included/excluded, active/default path, fallback, tests, wheel result, performance evidence, memory, actual-target status, review/CI status, limitations, and recommended disposition.